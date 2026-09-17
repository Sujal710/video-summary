"""
Auto-enrollment for the FR pipeline: gives every unique unmatched face its own
persistent identity (Person_0001, Person_0002, ...) instead of always logging
"Unknown", so the same stranger is recognized as the same person on their next
appearance.

Storage is a dedicated MongoDB collection, separate from the curated
'embeddings' collection so hand-labelled identities are never touched by
auto-enrollment:
    DB:         arcis
    Collection: embeddings-new

Concurrency: post_processor_worker runs as a *pool* of processes, so two
workers can see the same brand-new face in the same instant. If each worker
minted an ID independently, one person would get two IDs. To prevent that,
no post-processor ever writes to MongoDB or decides on a name directly -
it only enqueues the raw embedding here. enrollment_manager() runs as a
single dedicated process and is the sole reader/writer of the enrolled
identity set, so ID minting is fully serialized.
"""
import argparse
import concurrent.futures
import csv
import datetime
import glob
import json
import logging
import os
import subprocess
import threading
import time
from queue import Empty

import cv2
import numpy as np
import onnxruntime as _ort
from azure.storage.blob import BlobServiceClient
from pymongo import MongoClient

_ORIGINAL_ORT_SESSION_INIT = _ort.InferenceSession.__init__


def _capped_ort_session_init(self, *args, sess_options=None, **kwargs):
    """InsightFace's model_zoo.get_model() never exposes sess_options, so every
    ONNX session it creates uses onnxruntime's uncapped default thread pool
    (~1 per physical core, per session - 5 sessions x N concurrent camera
    workers can exhaust a container's shared cgroup pids limit). Force a
    capped default here instead, applied once at import time so it covers
    every FaceAnalysis load in this process (including face_search_api.py,
    which imports this module)."""
    if sess_options is None:
        sess_options = _ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
    _ORIGINAL_ORT_SESSION_INIT(self, *args, sess_options=sess_options, **kwargs)


_ort.InferenceSession.__init__ = _capped_ort_session_init

# --------------------- Configuration ---------------------

MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING", "")
MONGO_DATABASE = 'arcis'
MONGO_COLLECTION_EMBEDDINGS_NEW = 'embeddings-new'

EMBEDDING_DIM = 512  # InsightFace (buffalo_l) embedding size
DET_SIZE = (640, 640)

AUTO_ENROLL = True
ENROLL_THRESHOLD = 0.45      # must be HIGHER than SIMILARITY_THRESHOLD - "confidently nobody"
MIN_FACE_SIZE = 10           # px - reject tiny faces
MIN_DET_SCORE = 0.2          # reject low-confidence detections
MAX_ENROLL_PER_PERSON = 10   # cap embeddings stored per auto-enrolled identity

IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
VIDEO_URL_PREFIXES = ('rtsp://', 'rtmp://', 'http://', 'https://')
VIDEO_FILE_EXTENSIONS = ('.mp4', '.avi')

DEFAULT_SAMPLE_INTERVAL_S = 1.0  # run face-detection on one frame per this many seconds
RECONNECT_DELAY_S = 10
FFPROBE_TIMEOUT_S = 30
MAX_CONCURRENT_CAMERAS = 10  # at most this many ffmpeg processes/camera workers run at once
HEARTBEAT_EVERY_N_FRAMES = 10  # proof-of-life log when a camera sees no qualifying faces

# Azure Blob Storage - stores a snapshot image for every detected/enrolled
# face; the resulting URL is linked onto the person's MongoDB document.
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING", "")
AZURE_CONTAINER_NAME = "nvrdatashinobi"
AZURE_BLOB_PREFIX = "live-record/frimages"

# --------------------- End Configuration ---------------------


def _is_video_source(source):
    lower = source.lower()
    return lower.startswith(VIDEO_URL_PREFIXES) or lower.endswith(VIDEO_FILE_EXTENSIONS)


def _read_stream_list_csv(csv_path):
    """Reads a streams.csv: one row per camera, "<url>,<location>" - the
    location column is optional for backward compatibility with a plain
    single-column streams.csv (defaults to "Unknown"). Returns a list of
    (url, location) tuples, in order."""
    entries = []
    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if not row:
                continue
            url = row[0].strip()
            if not url:
                continue
            location = row[1].strip() if len(row) > 1 and row[1].strip() else "Unknown"
            entries.append((url, location))
    return entries


def _probe_dimensions(source):
    """Uses ffprobe to get the source's frame width/height, needed to know how
    many bytes make up one rawvideo frame read from the ffmpeg pipe."""
    command = [
        'ffprobe', '-v', 'error', '-print_format', 'json',
        '-show_streams', '-select_streams', 'v:0', source,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_S)

        data = {}
        if result.stdout.strip():
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                pass

        if isinstance(data, dict) and 'error' in data:
            # ffprobe reached the source but couldn't open it - e.g. offline
            # camera, wrong URL, auth failure. This is ffprobe's own
            # diagnostic for *why*, not a Python-side error.
            logging.error(f"[Enroll] ffprobe could not open {source}: {data['error'].get('string', data['error'])}")
            return None, None

        streams = data.get('streams') if isinstance(data, dict) else None
        if not streams:
            # No JSON at all, or JSON with no video stream - stderr (from
            # '-v error') usually has the real reason (connection refused,
            # 404, protocol not supported, ...).
            detail = result.stderr.strip() or f"no video stream in ffprobe output (exit code {result.returncode})"
            logging.error(f"[Enroll] ffprobe failed to read dimensions for {source}: {detail}")
            return None, None

        stream = streams[0]
        return int(stream['width']), int(stream['height'])
    except subprocess.TimeoutExpired:
        logging.error(f"[Enroll] ffprobe timed out after {FFPROBE_TIMEOUT_S}s probing {source}")
        return None, None
    except Exception as e:
        logging.error(f"[Enroll] ffprobe failed to read dimensions for {source}: {e}", exc_info=True)
        return None, None


def _connect():
    # tz_aware=True so datetimes read back from Mongo carry explicit UTC
    # tzinfo instead of a naive value - paired with writing timezone-aware
    # UTC datetimes below, this makes stored timestamps unambiguous and
    # correct in any standard BSON viewer (e.g. MongoDB Compass), not just
    # when read back through this same script.
    client = MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=30000, tz_aware=True)
    return client, client[MONGO_DATABASE][MONGO_COLLECTION_EMBEDDINGS_NEW]


def _upload_snapshot(blob_service_client, image_bgr, name):
    """Uploads one detection snapshot to Azure Blob (same container the live
    pipeline uses) and returns its public URL."""
    file_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
    file_name = f"{name}_{file_timestamp}.png"
    _, buf = cv2.imencode('.png', image_bgr)
    blob_client = blob_service_client.get_blob_client(
        container=AZURE_CONTAINER_NAME, blob=f"{AZURE_BLOB_PREFIX}/{file_name}"
    )
    blob_client.upload_blob(buf.tobytes(), overwrite=True)
    return blob_client.url


def _store_snapshot(coll, blob_service_client, image_bgr, name, location):
    """Uploads the snapshot and records the sighting (image + location +
    timestamp) on the person's document. Failures here are logged and
    swallowed - a missing snapshot shouldn't block enrollment, which has
    already been committed to MongoDB.

    Stored under "Sightings" (new field) rather than the older flat
    "ImageUrls" list, since location/timestamp need to travel with each
    image - "ImageUrls" is left untouched on documents that already have it,
    so nothing written before this changed."""
    try:
        image_url = _upload_snapshot(blob_service_client, image_bgr, name)
        now = datetime.datetime.now(datetime.timezone.utc)
        coll.update_one(
            {"PersonName": name},
            {
                "$push": {"Sightings": {"ImageUrl": image_url, "Location": location, "Timestamp": now}},
                "$set": {"LastSeen": now, "LastLocation": location},
            },
        )
    except Exception as e:
        logging.error(f"[Enroll] Snapshot upload/store failed for {name}: {e}")


def _annotate_frame(frame, faces_info):
    """faces_info: list of (bbox, label). Draws a simple box + name label
    onto the stored detection snapshot."""
    annotated = frame.copy()
    for bbox, label in faces_info:
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated


def load_auto_enrolled_faces():
    """Loads every {PersonName: (K, D) L2-normalized embeddings} from embeddings-new."""
    known = {}
    client, coll = _connect()
    try:
        for doc in coll.find({}):
            name = doc.get("PersonName")
            emb = doc.get("Embeddings") or doc.get("Embedding")
            if not (name and emb):
                continue
            vectors = np.array(emb, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors[np.newaxis, :]
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            vectors /= norms
            known[name] = vectors
    finally:
        client.close()
    return known


def build_matrix(known_faces_db):
    """Same layout as fr-railway-facseresolver.build_known_face_matrix: a flat
    (N, D) matrix plus a parallel names[] list, one row per embedding."""
    names, vectors = [], []
    for name, embeddings in known_faces_db.items():
        for vector in np.atleast_2d(embeddings):
            names.append(name)
            vectors.append(vector)
    if not vectors:
        return [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    matrix = np.stack(vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    matrix /= norms
    return names, matrix


def publish_matrix(shared_state, names, matrix):
    """Writes the current names/matrix into the cross-process shared dict.
    Post-processor workers re-read this every loop iteration to pick up
    newly-minted identities."""
    shared_state['names'] = list(names)
    shared_state['matrix_bytes'] = matrix.astype(np.float32).tobytes()
    shared_state['shape'] = matrix.shape


def read_matrix(shared_state):
    """Counterpart to publish_matrix() - reconstructs (names, matrix) from the
    shared dict. Call once per post-processor loop iteration."""
    names = list(shared_state.get('names', []))
    shape = shared_state.get('shape', (0, EMBEDDING_DIM))
    matrix_bytes = shared_state.get('matrix_bytes', b'')
    if shape[0] == 0:
        return names, np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    return names, np.frombuffer(matrix_bytes, dtype=np.float32).reshape(shape)


def _next_counter(names):
    counter = 0
    for n in names:
        if n.startswith("Person_"):
            try:
                counter = max(counter, int(n.split("_")[1]))
            except ValueError:
                pass
    return counter


def enroll_embedding(v, names, matrix, coll, counter):
    """Matches-or-creates a single L2-normalized embedding against the given
    (names, matrix) state, writing the result to `coll` (the embeddings-new
    collection). Returns (name, names, matrix, counter, is_new, stored) with
    the updated in-memory state - shared by enrollment_manager() (queue-driven,
    for the live pipeline) and the __main__ batch-image CLI below, so both
    follow the exact same match-or-create rule.

    `stored` is False once a matched identity has hit MAX_ENROLL_PER_PERSON -
    callers should treat that as "no new information for this person" and
    skip anything else keyed to this sighting (e.g. a snapshot upload), or a
    person seen every sampled frame would keep accumulating those forever
    long after their embeddings capped out.
    """
    if matrix.shape[0]:
        sims = matrix @ v
        best = int(np.argmax(sims))
        if sims[best] >= ENROLL_THRESHOLD:
            name = names[best]
            if sum(1 for n in names if n == name) < MAX_ENROLL_PER_PERSON:
                coll.update_one(
                    {"PersonName": name},
                    {"$push": {"Embeddings": v.tolist()}},
                    upsert=True,
                )
                names = names + [name]
                matrix = np.vstack([matrix, v])
                return name, names, matrix, counter, False, True
            return name, names, matrix, counter, False, False

    counter += 1
    name = f"Person_{counter:04d}"
    coll.insert_one({
        "PersonName": name,
        "Embeddings": [v.tolist()],
        "AutoEnrolled": True,
        "FirstSeen": datetime.datetime.now(datetime.timezone.utc),
    })
    names = names + [name]
    matrix = np.vstack([matrix, v]) if matrix.shape[0] else v[np.newaxis, :]
    return name, names, matrix, counter, True, True


def enrollment_manager(enroll_queue, shared_state):
    """
    Sole owner of identity minting for embeddings-new. Consumes raw face
    embeddings from enroll_queue; for each one:
      - Re-checks it against the CURRENT matrix (this closes the race: another
        worker may have enrolled this same face microseconds ago, so by the
        time this one is processed it might no longer be a stranger).
      - If it now matches an existing identity above ENROLL_THRESHOLD, stores
        it as another angle for that identity (capped at MAX_ENROLL_PER_PERSON).
      - Otherwise mints a new Person_XXXX identity.
    Publishes the updated names/matrix to shared_state after every change.
    """
    threading.current_thread().name = "EnrollmentManager"
    client, coll = _connect()

    names, matrix = read_matrix(shared_state)
    counter = _next_counter(names)

    logging.info(f"[EnrollmentManager] Started. {len(set(names))} identities, "
                 f"{matrix.shape[0]} embeddings loaded from '{MONGO_COLLECTION_EMBEDDINGS_NEW}'.")

    try:
        while True:
            try:
                emb = enroll_queue.get()
                if emb is None:
                    break

                v = np.asarray(emb, dtype=np.float32)
                norm = np.linalg.norm(v)
                if norm == 0:
                    continue
                v = v / norm

                name, names, matrix, counter, is_new, _ = enroll_embedding(v, names, matrix, coll, counter)
                publish_matrix(shared_state, names, matrix)
                if is_new:
                    logging.info(f"[EnrollmentManager] New identity: {name} "
                                 f"({matrix.shape[0]} embeddings, {len(set(names))} people)")

            except Empty:
                continue
            except Exception as e:
                logging.error(f"[EnrollmentManager] {e}", exc_info=True)
    finally:
        client.close()


def _run_on_image_folder(images_dir, location):
    """Standalone batch mode: detects every face in every image under
    images_dir with InsightFace, and enrolls each one against embeddings-new
    (matching existing identities, minting new Person_XXXX ones otherwise)."""
    from insightface.app import FaceAnalysis

    if not os.path.isdir(images_dir):
        logging.critical(f"[Enroll] Not a directory: {images_dir}")
        return

    image_paths = []
    for pattern in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(images_dir, pattern)))
    image_paths.sort()

    if not image_paths:
        logging.warning(f"[Enroll] No images found in {images_dir}")
        return

    logging.info(f"[Enroll] Loading FaceAnalysis model...")
    face_app = FaceAnalysis(det_size=DET_SIZE)
    face_app.prepare(ctx_id=0)

    logging.info(f"[Enroll] Loading existing identities from '{MONGO_COLLECTION_EMBEDDINGS_NEW}'...")
    known_faces_db = load_auto_enrolled_faces()
    names, matrix = build_matrix(known_faces_db)
    counter = _next_counter(names)
    logging.info(f"[Enroll] {len(set(names))} identities, {matrix.shape[0]} embeddings loaded.")

    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    client, coll = _connect()
    try:
        for path in image_paths:
            frame = cv2.imread(path)
            if frame is None:
                logging.warning(f"[Enroll] Could not read image: {path}")
                continue

            faces = face_app.get(frame)
            if not faces:
                logging.info(f"[Enroll] {os.path.basename(path)}: no face detected")
                continue

            for face in faces:
                bbox = face.bbox.astype(int)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                det_score = float(getattr(face, 'det_score', 1.0))

                if min(w, h) < MIN_FACE_SIZE or det_score < MIN_DET_SCORE:
                    logging.info(f"[Enroll] {os.path.basename(path)}: skipped low-quality face "
                                 f"(size={min(w, h)}px, det_score={det_score:.2f})")
                    continue

                v = np.asarray(face.embedding, dtype=np.float32)
                norm = np.linalg.norm(v)
                if norm == 0:
                    continue
                v = v / norm

                name, names, matrix, counter, is_new, stored = enroll_embedding(v, names, matrix, coll, counter)
                tag = "NEW" if is_new else ("matched" if stored else "matched (cap reached, skipped)")
                logging.info(f"[Enroll] {os.path.basename(path)}: {tag} -> {name}")

                if stored:
                    snapshot = _annotate_frame(frame, [(bbox, name)])
                    _store_snapshot(coll, blob_service_client, snapshot, name, location)
    finally:
        client.close()

    logging.info(f"[Enroll] Done. {len(set(names))} identities, {matrix.shape[0]} embeddings total.")


class _SharedIdentityState:
    """Names/matrix/counter shared by every camera thread in one run, guarded
    by a single lock. Without this, two cameras seeing the same brand-new
    face at the same instant could each mint a different Person_XXXX for it -
    the exact race enrollment_manager() prevents for the live-pipeline path,
    reproduced here for the multi-camera CLI path."""

    def __init__(self, names, matrix, counter):
        self.names = names
        self.matrix = matrix
        self.counter = counter
        self._lock = threading.Lock()

    def enroll(self, v, coll):
        with self._lock:
            name, self.names, self.matrix, self.counter, is_new, stored = enroll_embedding(
                v, self.names, self.matrix, coll, self.counter
            )
        return name, is_new, stored

    def people_count(self):
        return len(set(self.names))

    def embedding_count(self):
        return self.matrix.shape[0]


def _camera_worker(source, location, sample_interval_s, face_app, face_lock, coll, blob_service_client, state, stop_event):
    """Reads frames from one rtsp://, rtmp://, http(s)://, or local .mp4/.avi
    source via an ffmpeg subprocess (not cv2.VideoCapture), running
    face-detection on each frame ffmpeg hands back and enrolling each valid
    face against the shared identity `state`.

    ffmpeg does the frame-rate sampling itself via '-vf fps=...' - there's no
    point decoding every frame when the same face sits in front of the camera
    for many consecutive ones, and letting ffmpeg drop frames before decode is
    far cheaper than decoding-then-discarding in Python. Live streams
    (rtsp/rtmp) are reconnected if the ffmpeg process dies; a file or http
    source simply ends this worker when it runs out of frames.

    The FaceAnalysis model is shared across all camera threads (loading one
    per camera would multiply GPU memory use) - face_lock serializes access
    to it, since concurrent inference calls into the same onnxruntime session
    aren't guaranteed safe.
    """
    tag_prefix = f"[Enroll:{location}:{source}]"
    # Every network source (rtsp/rtmp/http/https) is a continuous live DVR
    # feed in this project - including the https://....flv ones, whose
    # "RTSP-" prefix is just part of the stream key, not the transport. Only
    # a local .mp4/.avi file is actually finite and should be allowed to end.
    is_live = source.lower().startswith(VIDEO_URL_PREFIXES)

    # A transient ffprobe failure (timeout, brief network hiccup) must not
    # permanently drop this camera - it's a live feed like any other, so keep
    # retrying at the same cadence as the ffmpeg reconnect logic below,
    # instead of giving up after one failed attempt.
    width = height = None
    while not stop_event.is_set():
        width, height = _probe_dimensions(source)
        if width and height:
            break
        if not is_live:
            logging.critical(f"{tag_prefix} Could not determine frame size, skipping this source.")
            return
        logging.warning(f"{tag_prefix} Could not determine frame size, retrying in {RECONNECT_DELAY_S}s...")
        time.sleep(RECONNECT_DELAY_S)

    if stop_event.is_set():
        return
    frame_bytes = width * height * 3  # bgr24 = 3 bytes/pixel

    def _open_ffmpeg():
        command = [
            'ffmpeg', '-nostdin', '-loglevel', 'error',
            '-hwaccel', 'cuda',
            '-i', source,
            '-vf', f'fps={1.0 / sample_interval_s}',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-',
        ]
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 bufsize=frame_bytes * 4)

    logging.info(f"{tag_prefix} {width}x{height}, reading via ffmpeg, ~1 frame every {sample_interval_s}s.")
    proc = _open_ffmpeg()
    sampled = 0
    try:
        while not stop_event.is_set():
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                proc.stdout.close()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()

                if stop_event.is_set():
                    break
                if is_live:
                    logging.warning(f"{tag_prefix} ffmpeg stream ended/failed, reconnecting...")
                    time.sleep(RECONNECT_DELAY_S)
                    proc = _open_ffmpeg()
                    continue
                logging.info(f"{tag_prefix} End of video source.")
                break

            sampled += 1
            try:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

                with face_lock:
                    faces = face_app.get(frame)

                qualifying = 0
                for face in faces:
                    bbox = face.bbox.astype(int)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    det_score = float(getattr(face, 'det_score', 1.0))

                    if min(w, h) < MIN_FACE_SIZE or det_score < MIN_DET_SCORE:
                        continue
                    qualifying += 1

                    v = np.asarray(face.embedding, dtype=np.float32)
                    norm = np.linalg.norm(v)
                    if norm == 0:
                        continue
                    v = v / norm

                    name, is_new, stored = state.enroll(v, coll)
                    tag = "NEW" if is_new else ("matched" if stored else "matched (cap reached, skipped)")
                    logging.info(f"{tag_prefix} sampled frame {sampled}: {tag} -> {name}")

                    if stored:
                        snapshot = _annotate_frame(frame, [(bbox, name)])
                        _store_snapshot(coll, blob_service_client, snapshot, name, location)

                # A camera that never sees a qualifying face never logs
                # anything above, which is indistinguishable from a stuck
                # worker - this proves it's alive either way.
                if qualifying == 0 and sampled % HEARTBEAT_EVERY_N_FRAMES == 0:
                    logging.info(f"{tag_prefix} heartbeat: {sampled} frames sampled, "
                                 f"{len(faces)} face(s) detected in latest frame, none qualifying.")
            except Exception as e:
                # A single bad frame (or a transient Mongo/Blob error) must not
                # kill this camera's worker thread - log it and keep reading.
                logging.error(f"{tag_prefix} Error processing sampled frame {sampled}: {e}", exc_info=True)
    except Exception as e:
        # Anything else (reconnect logic, ffmpeg process handling, ...) -
        # without this, an uncaught exception here silently ends this thread
        # with no log line at all, and the other cameras keep running with no
        # visible sign that this one died.
        logging.error(f"{tag_prefix} Worker crashed: {e}", exc_info=True)
    finally:
        try:
            proc.kill()
        except Exception:
            pass

    logging.info(f"{tag_prefix} Done. {sampled} frames sampled.")


def _run_on_sources(sources, sample_interval_s):
    """Runs _camera_worker over every (source, location) pair through a
    bounded thread pool - at most MAX_CONCURRENT_CAMERAS run (and so have an
    ffmpeg process open) at once; once a pool slot frees up, the next queued
    source starts. All workers enroll against one shared identity state and
    one shared FaceAnalysis model/MongoDB connection/Blob client."""
    from insightface.app import FaceAnalysis

    logging.info("[Enroll] Loading FaceAnalysis model...")
    face_app = FaceAnalysis(det_size=DET_SIZE)
    face_app.prepare(ctx_id=0)
    face_lock = threading.Lock()

    logging.info(f"[Enroll] Loading existing identities from '{MONGO_COLLECTION_EMBEDDINGS_NEW}'...")
    known_faces_db = load_auto_enrolled_faces()
    names, matrix = build_matrix(known_faces_db)
    counter = _next_counter(names)
    logging.info(f"[Enroll] {len(set(names))} identities, {matrix.shape[0]} embeddings loaded.")
    state = _SharedIdentityState(names, matrix, counter)

    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    client, coll = _connect()

    stop_event = threading.Event()
    pool_size = min(MAX_CONCURRENT_CAMERAS, len(sources))
    logging.info(f"[Enroll] Starting {len(sources)} camera source(s), "
                 f"{pool_size} running at a time...")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="Enroll")
    futures = {
        executor.submit(_camera_worker, source, location, sample_interval_s, face_app, face_lock,
                         coll, blob_service_client, state, stop_event): (source, location)
        for source, location in sources
    }
    reported = set()

    def _report_failures():
        # _camera_worker catches its own exceptions and logs them, so this is
        # a fallback for anything that goes wrong before its own try block -
        # without it a dead worker could go unreported until every other
        # camera also stops, which for a live stream may be never.
        for f, (src, loc) in futures.items():
            if f.done() and f not in reported:
                reported.add(f)
                exc = f.exception()
                if exc:
                    logging.error(f"[Enroll] Camera worker for {loc}:{src} failed: {exc}", exc_info=exc)

    try:
        while not all(f.done() for f in futures):
            _report_failures()
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("[Enroll] Stopped by user, waiting for camera workers to exit...")
    finally:
        stop_event.set()
        executor.shutdown(wait=True)
        _report_failures()
        client.close()

    logging.info(f"[Enroll] Done. {state.people_count()} identities, {state.embedding_count()} embeddings total.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(
        description="Detect faces and enroll each unique face into MongoDB "
                     "(arcis.embeddings-new) under its own Person_XXXX identity, "
                     "matching repeats to the same identity. Source can be a "
                     "folder of face images, an rtsp://, rtmp://, or http(s):// "
                     "video stream, a local .mp4/.avi file, or a .csv file "
                     "listing multiple stream URLs (same format as streams.csv) "
                     "to enroll from concurrently."
    )
    parser.add_argument("source", help="Image folder, stream URL, video file, or .csv of stream URLs")
    parser.add_argument("--interval", type=float, default=DEFAULT_SAMPLE_INTERVAL_S,
                         help="Seconds between sampled frames for video sources (default: %(default)s)")
    parser.add_argument("--location", default="Unknown",
                         help="Location label to record with every sighting (single source/folder modes only - "
                              "a .csv gets its location per-row from its second column)")
    args = parser.parse_args()

    if os.path.isdir(args.source):
        _run_on_image_folder(args.source, args.location)
    elif args.source.lower().endswith('.csv'):
        csv_sources = _read_stream_list_csv(args.source)
        if not csv_sources:
            parser.error(f"No stream URLs found in {args.source}")
        _run_on_sources(csv_sources, args.interval)
    elif _is_video_source(args.source):
        _run_on_sources([(args.source, args.location)], args.interval)
    else:
        parser.error(f"Not a directory, .csv, or a recognized video source "
                      f"(rtsp/rtmp/http/https/.mp4/.avi): {args.source}")
