"""
Live CCTV stream processor - prose tier (real-time counterpart to merge-final-many.py).

Reads camera_id/rtmp_url(/location) rows from a CSV (default live_cameras.csv),
where rtmp_url is a live http(s)/rtsp/rtmp stream URL (e.g. an .flv DVR relay
like https://mediatesting.vmukti.com/hdl/DVR/Cam-11.flv), and continuously
decodes each stream, samples frames every FRAME_INTERVAL_SECONDS within each
rolling 60-second segment, and analyzes each segment with a VLM exactly the way
merge-final-many.py does for recorded files. Cameras run concurrently via a
ThreadPoolExecutor, capped at MAX_CONCURRENT_VIDEOS, each in its own infinite
read loop that reconnects on stream drop.

    python merge-final-many-live.py                      # run with live_cameras.csv
    python merge-final-many-live.py --config other.csv    # a different camera list
    python merge-final-many-live.py --workers 3           # cap concurrent cameras

The one thing that fundamentally differs from the batch version:

TIMESTAMPS here are simply wall-clock datetime.now(), captured at the instant
each sampled frame is read off the live pipe. There is no video_anchors.json,
no anchor arithmetic, and no burnt-in-overlay OCR fallback: none of those exist
to solve a problem a live source has in the first place. A frame read from a
live stream right now unambiguously happened right now, which is what makes
this simpler than the recorded-file case merge-final-many.py has to handle.

FRAME ACCESS is sequential decode (this is the one thing a live source cannot
do that a file can: there is nothing to seek to). ffmpeg is opened once per
camera as a persistent pipe and every frame is read off its stdout in order;
a segment boundary is a frame-count boundary (SEGMENT_DURATION * fps), not a
byte offset. If the pipe drops (network hiccup, DVR restart) the worker
reconnects after RECONNECT_DELAY_S and starts a fresh segment - a live feed
has no "resume from this exact frame" concept, unlike a file.

Note on GPU decode: nvidia-smi working does not mean ffmpeg can use NVDEC. It is
probed once (gpu_decode_available) and the result cached, because retrying a
doomed GPU open per frame costs a wasted process spawn every time. The cuvid
decoder is also chosen from the stream's real codec, same as the batch version.
"""

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtmp_live;1|rtmp_buffer;100"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(1)
import numpy as np
import time
import datetime
from datetime import timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import subprocess
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from azure.storage.blob import BlobServiceClient, ContentSettings
import requests
import base64
from typing import Any, List, Optional, Tuple, Dict
from sentence_transformers import SentenceTransformer
import pickle
import re
import faiss
import warnings
import csv
from ollama import chat
import tempfile
from pathlib import Path
import pandas as pd
import GPUtil
import gc
import psutil

warnings.filterwarnings("ignore", category=FutureWarning)

def emergency_memory_check():
    """Emergency check - force GC if memory too high"""
    try:
        mem = psutil.virtual_memory()
        if mem.percent > 90:
            logger.warning(f"⚠️ High memory: {mem.percent:.1f}% - forcing GC")
            gc.collect()
            time.sleep(0.5)
            mem = psutil.virtual_memory()
            if mem.percent > 95:
                logger.critical(f"🚨 CRITICAL: RAM at {mem.percent:.1f}%")
    except Exception as e:
        logger.error(f"Memory check error: {e}")

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# The Azure blob SDK logs a full HTTP request+response dump per upload at INFO -
# roughly 40 lines per segment, which buries the progress and timing lines.
for _noisy in ("azure", "azure.core.pipeline.policies.http_logging_policy",
               "urllib3", "httpx", "httpcore", "pymongo"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

SEGMENT_DURATION = 60  # seconds

# How many live cameras decode + analyze concurrently in the thread pool.
MAX_CONCURRENT_VIDEOS = 4

# Live camera CSV: columns camera_id,rtmp_url(,location)(,enabled) - rtmp_url is
# a live stream URL here (http/https/rtsp/rtmp), unlike the batch script where
# it is a local file path. See load_camera_config().
LIVE_CAMERAS_CSV_PATH = os.getenv("LIVE_CAMERAS_CSV_PATH", "live_cameras.csv")

# A dropped/ended stream (DVR restart, brief network hiccup) is retried after
# this many seconds rather than ending that camera's worker for good - a live
# feed is expected to be interrupted occasionally and should just resume.
RECONNECT_DELAY_S = int(os.getenv("RECONNECT_DELAY_S", "10"))
FFPROBE_TIMEOUT_S = int(os.getenv("FFPROBE_TIMEOUT_S", "30"))

# Skip segments already present in MongoDB for this camera, so a restarted
# worker continues the segment-id sequence instead of starting back at 0 and
# colliding with (upserting over) segments it already stored.
RESUME_FROM_MONGO = True

# Optional cap, for a quick smoke test of one camera without leaving it running
# forever. None = run indefinitely (the normal, live mode of operation).
MAX_SEGMENTS_PER_CAMERA: Optional[int] = None

# Stream Settings
STREAM_FPS = 10

# Azure Configuration
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING", "")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN", "")
AZURE_CONTAINER_NAME = "nvrdatashinobi"
AZURE_BLOB_PREFIX = "live-record/frimages"
STATIC_IMAGE_URL = f"https://nvrdatashinobi.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{AZURE_BLOB_PREFIX}"

# MongoDB Configuration
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING", "")
MONGO_DATABASE = 'arcis'
# Overridable so a run can be isolated from unrelated data already in the
# database: arcis.activities1 holds 1048 segments from a previous indoor-camera
# deployment (Reception, Canteen, B1-STairs...) which would otherwise be
# returned by every search alongside this dataset's footage.
#
# Deliberately NOT named MONGO_COLLECTION_ACTIVITIES: the checked-in .env sets
# that to 'activities' (and MONGO_DATABASE to 'arcis-railway'), which is stale -
# arcis-railway.activities holds 33 documents while arcis.activities1 holds 1048.
# Reading that variable would silently repoint the whole system at the wrong
# collection for anything that calls load_dotenv().
MONGO_COLLECTION_ACTIVITIES = os.getenv('ACTIVITIES_COLLECTION', 'activities1')

# The vision model that writes the per-segment prose. Served locally by Ollama.
VLM_MODEL = os.getenv('VLM_MODEL', 'qwen2.5vl:7b')

# Which service actually runs that per-segment vision call.
#   'ollama'     (default) local qwen2.5vl - no frame ever leaves the box.
#   'openrouter' GLM 5.3 Flash, which accepts ['text','image','video'].
#
# This is the ONE switch that changes what leaves the machine. SUMMARY_BACKEND
# only ships prose that is already in MongoDB; VLM_BACKEND=openrouter uploads
# the FRAMES THEMSELVES (12 per segment, base64 JPEG) to a third party. For
# police CCTV that is a disclosure decision, not a performance one - set it
# deliberately.
VLM_BACKEND = os.getenv('VLM_BACKEND', 'ollama').lower()
OPENROUTER_VLM_MODEL = os.getenv('OPENROUTER_VLM_MODEL', 'z-ai/glm-5.3-flash')

# Roll-up summariser: a text model reads a video's segment descriptions and
# writes one video-level summary. Local Ollama, same as the VLM.
SUMMARY_MODEL = os.getenv('SUMMARY_MODEL', 'qwen3-vl:latest')

# How densely each 60s segment is sampled for the VLM. One frame every 5s = 12
# frames per segment. Measured on this dataset: 12 frames at 1280px cost the
# same VLM time as 2 frames at native resolution (23.4s vs 23.9s) because output
# tokens dominate and prefill is cheap - so the extra temporal coverage is
# effectively free. What is NOT free is resolution: 12 frames at native
# 2560x1440 is 21,664 prompt tokens and blows the 16k context outright.
FRAME_INTERVAL_SECONDS = float(os.getenv('FRAME_INTERVAL_SECONDS', '5'))
VLM_FRAME_LONGEST_SIDE = int(os.getenv('VLM_FRAME_LONGEST_SIDE', '1280'))
FRAME_EXTRACT_WORKERS = int(os.getenv('FRAME_EXTRACT_WORKERS', '4'))

# How many SEGMENTS of one video run at once. --workers parallelises across
# videos, which does nothing for a single-video run - the segment loop inside a
# video was strictly sequential.
#
# Worth having because of where the time actually goes. Measured on cam16 with
# VLM_BACKEND=openrouter, per 49.8s segment:
#     VLM (network wait)  37.9s  76.2%
#     frame seek           4.2s   8.4%
#     azure upload         1.1s   2.2%
#     embed (GPU)          0.4s   0.9%
# Three quarters of the run is a thread blocked on an HTTPS response, with the
# GPU at 0% and 956 MiB of 24 GB in use. That is latency, not compute, and the
# only thing that removes latency is overlapping it.
#
# Keep at 1 for VLM_BACKEND=ollama: those calls are GPU-bound and queue behind
# each other in Ollama anyway, so concurrency buys nothing and costs VRAM.
SEGMENT_WORKERS = int(os.getenv('SEGMENT_WORKERS', '1'))

# Each frame seek is a separate ffmpeg process. Unbounded, SEGMENT_WORKERS x
# FRAME_EXTRACT_WORKERS of them run at once (6 x 4 = 24), which thrashes the
# disk and starves the very calls it is trying to feed. One global ceiling.
MAX_CONCURRENT_SEEKS = int(os.getenv('MAX_CONCURRENT_SEEKS',
                                     str(max(FRAME_EXTRACT_WORKERS, 6))))
_SEEK_SLOTS = threading.Semaphore(MAX_CONCURRENT_SEEKS)
GENERATE_VIDEO_SUMMARY = os.getenv('GENERATE_VIDEO_SUMMARY', 'true').lower() == 'true'

# ── speed controls ───────────────────────────────────────────────────────────
# The VLM is ~83% of per-segment cost and is bound by OUTPUT tokens (~45 tok/s
# measured), so the two things that actually move the needle are writing less
# and not writing at all when nothing happened.
#
# MOTION_GATE: compare the sampled frames to each other; if nothing moved, the
# segment gets a templated "no activity" line instead of a 50-second VLM call.
# On night footage most segments are empty, so this is the single biggest win
# and it costs no accuracy - there is nothing in the frame to describe.
MOTION_GATE = os.getenv('MOTION_GATE', 'true').lower() == 'true'
# Upload one labelled montage of all sampled frames alongside the two evidence
# thumbnails, so the stored images cover everything the description refers to.
CONTACT_SHEET = os.getenv('CONTACT_SHEET', 'true').lower() == 'true'
# Let the VLM read the burnt-in clock when OCR cannot. Costs ~2s per attempt,
# and only runs on frames where tesseract already failed.
OVERLAY_VLM_FALLBACK = os.getenv('OVERLAY_VLM_FALLBACK', 'true').lower() == 'true'

MOTION_MIN_CHANGED_PX = float(os.getenv('MOTION_MIN_CHANGED_PX', '0.0015'))  # 0.15% of pixels
MOTION_PIXEL_THRESHOLD = int(os.getenv('MOTION_PIXEL_THRESHOLD', '25'))      # 0-255 per-pixel delta

# FAST_MODE: a compact single-paragraph description instead of the 8-section
# report. Roughly 3x faster because it emits ~1/3 the tokens. This DOES trade
# detail for speed - it keeps who/what/colour/action and drops the exhaustive
# appearance and spatial-relationship prose.
FAST_MODE = os.getenv('FAST_MODE', 'false').lower() == 'true'

# SEGMENT_STYLE: which report shape each 60s segment is written in.
#   'default' the eight-section observation report (ALERTS / CHANGES / SCENE /
#             VEHICLES / PEOPLE / PERSON-OBJECT / PERSON-PERSON / OBJECTS)
#   'police'  the same nine-section police report summarise_video() writes for a
#             whole video, scoped to one clip. Costs more output tokens per
#             segment, and takes precedence over FAST_MODE.
SEGMENT_STYLE = os.getenv('SEGMENT_STYLE', 'default').lower()

# num_predict caps thinking + answer TOGETHER, and qwen3-vl's thinking varies
# widely run to run (measured 1,800-2,700 chars, i.e. 500-750 tokens, on
# identical input). A 900 budget left no room for the report after a long think,
# so ~half of segments came back with an empty `content` and were failed. The
# cap is not a target - the model stops when done - so a generous ceiling costs
# nothing on the good runs and removes the wasted retry on the bad ones.
# A reasoning model needs headroom for thinking AND the answer; a plain one
# needs only the answer, and a tight cap is what stops it rambling (qwen2.5vl
# produced 9,000-character loops uncapped, 250 characters at 300 tokens).
_REASONING_VLM = 'qwen3' in VLM_MODEL.lower()
if _REASONING_VLM:
    _DEFAULT_PREDICT = '2200' if FAST_MODE else '4000'
else:
    # Fast mode now answers nine labels (vehicles, people, crowd, objects, text,
    # violations, incidents, anomaly) rather than writing one summary sentence,
    # so 300 tokens would truncate it before the incident lines - which are the
    # ones police read. Full mode has eight sections to reach for the same reason.
    _DEFAULT_PREDICT = '750' if FAST_MODE else '1800'
# Nine sections with per-second detail runs longer than the eight-section report:
# measured 3,347 chars / 794 tokens on a two-frame cam14 clip, and a twelve-frame
# clip has more to say. 1800 truncates it mid-section, which loses sections 7-9 -
# the violations and the analyst's read, i.e. exactly the part police act on.
if SEGMENT_STYLE == 'police':
    _DEFAULT_PREDICT = '4000' if _REASONING_VLM else '2600'
VLM_NUM_PREDICT = int(os.getenv('VLM_NUM_PREDICT', _DEFAULT_PREDICT))

# Temperatures to re-roll at when a call returns reasoning and no answer.
# Escalating slightly each time; identical settings tend to reproduce the loop.
VLM_REPEAT_PENALTY = float(os.getenv('VLM_REPEAT_PENALTY', '1.15'))
VLM_RETRY_TEMPERATURES = ([float(x) for x in
                           os.getenv('VLM_RETRY_TEMPERATURES', '0.45,0.8').split(',')]
                          if _REASONING_VLM else [])
VLM_NUM_CTX = int(os.getenv('VLM_NUM_CTX', '16000'))

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Memory management
MEMORY_CHECK_INTERVAL = 30
MEMORY_WARNING_THRESHOLD = 80
GC_INTERVAL = 60

# Paths
DB_PATH = "./video_segments_db"
#IST_OFFSET = timedelta(hours=5, minutes=30)

def frames_have_motion(frames: List[np.ndarray]) -> Tuple[bool, float]:
    """True if anything moved across the sampled frames.

    Downscales to a small grayscale thumbnail and diffs consecutive frames, so
    the check costs ~1ms against a 50-second VLM call. Compression noise and IR
    sensor grain are rejected by requiring a per-pixel delta above
    MOTION_PIXEL_THRESHOLD across at least MOTION_MIN_CHANGED_PX of the frame -
    a car crossing the scene changes far more than that, while an empty street
    at night changes almost nothing.
    """
    if len(frames) < 2:
        return True, 1.0

    small = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        small.append(cv2.GaussianBlur(cv2.resize(g, (160, 90),
                                                 interpolation=cv2.INTER_AREA), (5, 5), 0))

    peak = 0.0
    for a, b in zip(small, small[1:]):
        diff = cv2.absdiff(a, b)
        changed = float(np.count_nonzero(diff > MOTION_PIXEL_THRESHOLD)) / diff.size
        peak = max(peak, changed)
        if peak >= MOTION_MIN_CHANGED_PX:
            return True, peak
    return False, peak


def build_contact_sheet(frames: List[np.ndarray], offsets: List[float],
                        segment_start: float = 0.0, cols: int = 4,
                        tile_width: int = 480) -> Optional[np.ndarray]:
    """Tile every sampled frame into one image, each labelled with its offset.

    The description is written from all N frames but only two are uploaded, so
    without this the stored evidence cannot support what the text claims. One
    montage costs a single upload instead of N.
    """
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    tw = tile_width
    th = max(1, int(round(h * tw / float(w))))
    rows = (len(frames) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)

    for i, f in enumerate(frames):
        tile = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
        rel = offsets[i] - segment_start if i < len(offsets) else i * FRAME_INTERVAL_SECONDS
        label = f"+{rel:.0f}s"
        cv2.rectangle(tile, (0, 0), (74, 26), (0, 0, 0), -1)
        cv2.putText(tile, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
    return sheet


def encode_for_vlm(frame: np.ndarray, longest_side: int = 0):
    """JPEG-encode a frame, downscaled so the VLM prompt stays inside num_ctx.

    Resolution is the binding constraint, not frame count: 12 frames at native
    2560x1440 is 21,664 prompt tokens (context is 16,128, so the call is
    rejected), while the same 12 at 1280px is 6,514 and costs no more VLM time
    than 2 native frames did.
    """
    longest_side = longest_side or VLM_FRAME_LONGEST_SIDE
    h, w = frame.shape[:2]
    if longest_side and max(h, w) > longest_side:
        scale = longest_side / float(max(h, w))
        frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf if ok else None


def segment_sample_offsets(cumulative_seconds: float,
                           interval: float = 0.0,
                           duration: float = 0.0) -> List[float]:
    """Offsets, in seconds into the file, of every frame sampled for one segment.

    interval=5 over a 60s segment gives 12 frames at +0,5,...,55. interval<=0
    falls back to the original two-frame behaviour (20% and 50% in) so the
    streaming path and old runs stay reproducible.
    """
    interval = interval or FRAME_INTERVAL_SECONDS
    duration = duration or SEGMENT_DURATION
    if interval <= 0 or interval >= duration:
        return [cumulative_seconds + 0.2 * duration,
                cumulative_seconds + 0.5 * duration]
    n = max(1, int(duration // interval))
    return [cumulative_seconds + i * interval for i in range(n)]


def _hms(seconds: float) -> str:
    """Compact h:mm:ss for progress lines."""
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def get_now():
    """Get current time adjusted for IST"""
    return datetime.datetime.now()

# ============================================================================
# MEMORY MONITOR
# ============================================================================

class MemoryMonitor:
    """Monitor and log memory usage"""

    def __init__(self, check_interval=30):
        self.check_interval = check_interval
        self.last_check = time.time()
        self.process = psutil.Process()

    def check(self):
        """Check memory and log if threshold exceeded"""
        now = time.time()
        if now - self.last_check < self.check_interval:
            return

        self.last_check = now

        mem = psutil.virtual_memory()
        mem_percent = mem.percent

        proc_mem = self.process.memory_info().rss / (1024**3)

        gpu_mem_str = "N/A"
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_mem_str = f"{gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f}MB ({gpu.memoryUtil*100:.1f}%)"
        except:
            pass

        logger.info(f"[Memory] RAM: {mem_percent:.1f}% | Process: {proc_mem:.2f}GB | GPU: {gpu_mem_str}")

        if mem_percent > MEMORY_WARNING_THRESHOLD:
            logger.warning(f"⚠️  HIGH MEMORY USAGE: {mem_percent:.1f}%")
            gc.collect()
            logger.info("   Forced garbage collection")

# ============================================================================
# UNIVERSAL SHARED MODEL MANAGER
# ============================================================================

class UniversalModelManager:
    """Singleton to manage ALL models in one place"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.embedding_model = None
        self.embedding_model_lock = threading.Lock()
        logger.info("🔧 UniversalModelManager initialized")

    def get_embedding_model(self):
        """Get or load SentenceTransformer model (thread-safe)"""
        with self.embedding_model_lock:
            if self.embedding_model is None:
                logger.info("🔥 Loading SentenceTransformer ONCE...")
                self.embedding_model = SentenceTransformer(
                    "nomic-ai/nomic-embed-text-v1.5",
                    trust_remote_code=True
                )
                logger.info("✅ SentenceTransformer loaded (SHARED across all cameras)")
            else:
                logger.info("♻️ Reusing shared SentenceTransformer model")
        return self.embedding_model

# ============================================================================
# CAMERA CONFIGURATION LOADER
# ============================================================================

def load_camera_config(csv_path):
    """Live camera config: camera_id,rtmp_url(,location)(,enabled). rtmp_url is
    a live stream URL (http/https/rtsp/rtmp) - no existence check is possible
    or meaningful the way there is for a local file path."""
    try:
        df = pd.read_csv(csv_path)
        required_cols = ['camera_id', 'rtmp_url']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing column: {col}")
                return [], 0

        if 'enabled' in df.columns:
            df = df[df['enabled'].fillna(True).astype(bool)]

        cameras = []
        for _, row in df.iterrows():
            cameras.append({
                'camera_id': str(row['camera_id']),
                'rtmp_url': str(row['rtmp_url']),
                'location': str(row['location']).strip() if 'location' in df.columns
                            and str(row.get('location', '')).strip() else 'Unknown Location',
            })

        return cameras, len(cameras)
    except Exception as e:
        logger.error(f"Error loading camera config: {e}")
        return [], 0

# ============================================================================
# MONGODB FUNCTIONS
# ============================================================================

def initialize_mongodb():
    """Initialize MongoDB for activity logs"""
    try:
        client = MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=30000)
        client.admin.command('ping')
        db = client[MONGO_DATABASE]
        activities_collection = db[MONGO_COLLECTION_ACTIVITIES]

        activities_collection.create_index([
            ("camera_id", 1),
            ("timestamp", -1)
        ])

        logger.info("✓ MongoDB connection successful")
        return client, activities_collection
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection error: {e}")
        raise

# ============================================================================
# AZURE FUNCTIONS
# ============================================================================

def initialize_azure_client():
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        return blob_service_client
    except Exception as e:
        logger.error(f"Failed to initialize Azure: {e}")
        return None


def upload_frame_to_azure(blob_service_client, camera_id: str, segment_id: int,
                          frame_img: np.ndarray, frame_position: str,
                          timestamp: datetime.datetime) -> Optional[str]:
    try:
        if frame_img is None or frame_img.size == 0:
            return None

        time_str = timestamp.strftime("%Y-%m-%d-%H-%M-%S")
        seg_str = f"{segment_id:04d}" if isinstance(segment_id, int) else str(segment_id)
        filename = f"{camera_id}_SEG{seg_str}_{frame_position}_{time_str}.jpg"
        blob_name = f"{AZURE_BLOB_PREFIX}/{filename}"

        success, buf = cv2.imencode(".jpg", frame_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return None

        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=blob_name
        )

        blob_client.upload_blob(
            buf.tobytes(),
            overwrite=True,
            content_settings=ContentSettings(content_type='image/jpeg')
        )

        image_url = f"{STATIC_IMAGE_URL}/{filename}?{AZURE_SAS_TOKEN}"
        return image_url

    except Exception as e:
        logger.error(f"Azure upload failed: {e}")
        return None

def _collapse_repeats(text: str) -> str:
    """Drop verbatim sentence repeats left by a looping decoder.

    repeat_penalty prevents most of these, but a survivor would otherwise be
    embedded and stored as if it were 24 separate observations.
    """
    parts = re.split(r'(?<=[.!?])\s+', text)
    out, seen = [], set()
    for p in parts:
        key = p.strip().lower()
        if len(key) > 15 and key in seen:
            continue
        seen.add(key)
        out.append(p)
    return " ".join(out)


class VideoSegment:
    def __init__(self, segment_id: int, start_time: datetime.datetime,
                 end_time: datetime.datetime, frame_urls: List[str],
                 description: str, cumulative_minutes: float,
                 camera_id: str, location: str = "Unknown Location",
                 source_video: str = "",
                 anchor_confidence: str = "none",
                 video_offset_seconds: float = 0.0,
                 overlay_drift_seconds: Optional[float] = None):
        self.segment_id = segment_id
        self.camera_id = camera_id
        self.location = location
        self.source_video = source_video
        self.start_time = start_time
        self.end_time = end_time
        self.frame_urls = frame_urls
        self.description = description
        self.cumulative_minutes = cumulative_minutes
        # Provenance of the timestamp, so a downstream consumer can tell a
        # surveyed time from a file-mtime guess instead of trusting both equally.
        self.anchor_confidence = anchor_confidence
        self.video_offset_seconds = video_offset_seconds
        self.overlay_drift_seconds = overlay_drift_seconds
        # Set by the caller when the motion gate runs; lets a query ask for
        # "segments where something actually happened".
        self.motion_score = None
        self.motion_gated = False

    def to_dict(self):
        return {
            'segment_id': self.segment_id,
            'camera_id': self.camera_id,
            'location': self.location,
            'source_video': self.source_video,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'cumulative_minutes': self.cumulative_minutes,
            'frame_urls': self.frame_urls,
            'description': self.description,
            'anchor_confidence': self.anchor_confidence,
            'video_offset_seconds': self.video_offset_seconds,
            'overlay_drift_seconds': self.overlay_drift_seconds,
            'motion_score': self.motion_score,
            'motion_gated': self.motion_gated,
        }


def _collapse_repeats(text: str) -> str:
    """Drop verbatim sentence repeats left by a looping decoder.

    repeat_penalty prevents most of these, but a survivor would otherwise be
    embedded and stored as if it were 24 separate observations.
    """
    parts = re.split(r'(?<=[.!?])\s+', text)
    out, seen = [], set()
    for p in parts:
        key = p.strip().lower()
        if len(key) > 15 and key in seen:
            continue
        seen.add(key)
        out.append(p)
    return " ".join(out)


def clean_description_text(text: str) -> str:
    cleaned = _collapse_repeats(text)
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'##\s*', '', cleaned)
    cleaned = re.sub(r'#\s*', '', cleaned)
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)
    # Dates and clock times are deliberately NOT stripped here. Removing them
    # turned "at 21:08:21" into "at" and left descriptions reading "timestamp
    # ( )", destroying the one field a time-based query needs.
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()

    lines = cleaned.split('\n')
    cleaned_lines = [line for line in lines if len(line.strip()) >= 3 or line.strip() == '']
    return '\n'.join(cleaned_lines)

# ============================================================================
# MONGODB DATABASE
# ============================================================================

class MongoVideoDatabase:
    def __init__(self, model_manager, collection):
        self.model_manager = model_manager
        self.embedding_model = model_manager.get_embedding_model()
        self.collection = collection
        logger.info("🗄️ MongoVideoDatabase initialized")

    def add_segment(self, segment: VideoSegment):
        try:
            query_text = f"search_document: {segment.description}"
            with self.model_manager.embedding_model_lock:
                embedding = self.embedding_model.encode(
                    [query_text], convert_to_numpy=True, normalize_embeddings=True
                )

            doc = segment.to_dict()
            doc['embedding'] = embedding[0].tolist()
            doc['timestamp'] = segment.start_time
            
            self.collection.update_one(
                {"camera_id": segment.camera_id, "segment_id": segment.segment_id},
                {"$set": doc},
                upsert=True
            )
            logger.info(f"✅ Segment #{segment.segment_id} stored in MongoDB")
            print(f"✅ Segment #{segment.segment_id} from Camera {segment.camera_id} stored in MongoDB")
        except Exception as e:
            logger.error(f"❌ MongoDB add_segment error: {e}")

    def save(self, camera_id: str):
        pass

# ============================================================================
# VLM DESCRIPTION
# ============================================================================

class _VLMReply:
    """Minimal stand-in for the ollama response object the caller expects."""
    __slots__ = ("content", "thinking")

    def __init__(self, content: str, thinking: str = ""):
        self.content = content
        self.thinking = thinking


def _openrouter_vision(system: str, user: str, image_paths: List[str],
                       temperature: float, max_tokens: Optional[int] = None) -> _VLMReply:
    """One GLM vision call over the sampled frames.

    reasoning.effort=low is not optional here, for the same reason it is not
    optional in alert_dispatch.llm_text: glm-5.3-flash refuses to disable
    reasoning outright, and left alone it spends the whole completion budget
    deliberating and returns content=None. That is precisely the failure the
    qwen3-vl comments below describe, so it is guarded the same way.
    """
    import base64
    import httpx
    from alert_dispatch import _openrouter_key, OPENROUTER_URL

    key = _openrouter_key()
    if not key:
        raise RuntimeError("VLM_BACKEND=openrouter but no OPENROUTER_API_KEY is set")

    parts: List[Dict[str, Any]] = [{"type": "text", "text": user}]
    for path in image_paths:
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    budget = max_tokens if max_tokens is not None else VLM_NUM_PREDICT
    body = {
        "model": OPENROUTER_VLM_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": parts}],
        "max_tokens": budget,
        "temperature": temperature,
        "reasoning": {"effort": "low"},
    }
    logger.info("[AI->] %s %s (%d frames, max_tokens=%d)",
                OPENROUTER_URL, OPENROUTER_VLM_MODEL, len(image_paths), budget)
    started = time.time()
    r = httpx.post(OPENROUTER_URL, timeout=300,
                   headers={"Authorization": f"Bearer {key}",
                            "Content-Type": "application/json"},
                   json=body)
    data = r.json()
    if r.status_code != 200 or "choices" not in data:
        raise RuntimeError(f"OpenRouter HTTP {r.status_code}: {json.dumps(data)[:300]}")

    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    thinking = (msg.get("reasoning") or "") or ""
    usage = data.get("usage") or {}
    logger.info("[AI<-] %s ok in %.1fs | prompt_tok=%s completion_tok=%s "
                "cost=$%s | %d chars", OPENROUTER_VLM_MODEL, time.time() - started,
                usage.get("prompt_tokens"), usage.get("completion_tokens"),
                usage.get("cost"), len(content))
    return _VLMReply(content, thinking)


def _vlm_chat(system: str, user: str, image_paths: List[str],
              temperature: float, max_tokens: Optional[int] = None) -> _VLMReply:
    """Run one vision call on whichever backend VLM_BACKEND selects.

    max_tokens overrides VLM_NUM_PREDICT for calls that need far less than a
    full report - the overlay-clock read wants one line, not 2600 tokens.
    """
    if VLM_BACKEND == "openrouter":
        return _openrouter_vision(system, user, image_paths, temperature, max_tokens)

    response = chat(
        model=VLM_MODEL,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user, 'images': image_paths},
        ],
        options={
            'temperature': temperature,
            'num_predict': max_tokens if max_tokens is not None else VLM_NUM_PREDICT,
            'num_ctx': VLM_NUM_CTX,
            # Without this the non-reasoning model sometimes latches onto a
            # sentence and repeats it verbatim until the cap - one observed
            # segment held "A white car drives past at 32s." 24 times. The
            # penalty makes an already-emitted token progressively less
            # likely, which breaks the loop without altering normal prose.
            'repeat_penalty': VLM_REPEAT_PENALTY,
        },
    )
    return _VLMReply(response.message.content or "",
                     getattr(response.message, 'thinking', "") or "")


def get_segment_description(frames_buffers: List[Any], api_key: str,
                             location: str = "Unknown Location",
                             segment_time: Optional[datetime.datetime] = None,
                             time_trusted: bool = False) -> str:
    """Get AI description"""
    # FIX: temp_files initialised inside the function, before the try block
    temp_files = []

    try:
        image_paths = []
        for i, buffer in enumerate(frames_buffers):
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir='/tmp')
            temp_file.write(buffer.tobytes())
            temp_file.flush()
            temp_file.close()
            temp_files.append(temp_file.name)
            image_paths.append(temp_file.name)

        if not image_paths:
            return "[Error: No frames sampled for VLM]"

        n_frames = len(image_paths)
        span = (n_frames - 1) * FRAME_INTERVAL_SECONDS if n_frames > 1 else 0

        system_instructions = f"""You are a professional police CCTV surveillance analyst specializing in detailed appearance documentation and interaction analysis. This footage is from camera location: {location}. Your report is used by police control-room staff, so any suspicious or alert-worthy activity (reckless/dangerous driving, accidents, fights, snatching/theft, loitering, trespassing, unattended objects, weapons, mob activity) must be called out clearly. Most footage on this camera is routine road traffic (cars/two-wheelers/pedestrians moving normally) - keep that brief and reserve the real detail for anything suspicious or unusual.
Describe ONLY what is strictly visible in the provided frames with emphasis on:
- PERSON APPEARANCE: Physical characteristics, clothing details, accessories, distinctive features
- PERSON ACTIVITIES: Every action, gesture, posture, movement direction, and body language
- OBJECT APPEARANCE: Type, color, size, material, condition, labels, distinctive features
- PERSON-OBJECT INTERACTIONS: How people touch, use, carry, manipulate, or engage with objects
- OBJECT-OBJECT RELATIONSHIPS: Spatial arrangements, containment, support, proximity between objects
- PERSON-PERSON INTERACTIONS: Physical distance, facing direction, gestures toward others, group formations
- VEHICLE APPEARANCE: Type, color, condition, features, cargo, modifications
- Be precise, factual, and objective with granular detail on all appearances and interactions
- Describe lighting (daylight/night/IR) and environmental context
- NEVER use speculative language (possibly, appears to, likely)
- Provide exact spatial context (left/right, foreground/background, distance estimates)
- For appearance details, describe what you see, not what you infer about identity
- Write in plain prose only. NO bullet points. NO Q&A."""

        # The wall-clock time is arithmetic from the video's anchor, which is
        # far more reliable than asking the model to read a burnt-in overlay
        # off a night-time frame. Only assert it when the anchor is trusted.
        # The date and time are already prepended to every stored description by
        # the caller, so the model must not repeat them - restating
        # "13-06-2026 20:59:36" for each vehicle bloats the text, costs output
        # tokens (the dominant cost) and adds nothing.
        if segment_time is not None and time_trusted:
            time_note = (
                f"The recording date and time are already recorded separately and will be "
                f"shown to the reader. Do NOT state the date. Do NOT write full clock "
                f"times. When timing matters, give it as seconds into this clip - "
                f"'15s in', 'by 40s', 'throughout'. Never read the date or time off the "
                f"image overlay.\n\n")
        else:
            time_note = ("The recording date and time are not reliably known. Do not state "
                         "any date or clock time. If timing matters, give it as seconds "
                         "into this clip ('15s in').\n\n")

        if n_frames > 2:
            sequence_note = (
                f"These images are consecutive samples from ONE camera, in chronological "
                f"order, spanning about {span:.0f} seconds of continuous footage. Treat them "
                f"as a TIME SEQUENCE, not as separate scenes. Track each person and vehicle "
                f"across the sequence: where it first appears, which direction it moves, "
                f"whether it stops, and where it leaves. State movement and direction "
                f"explicitly (e.g. 'a white hatchback enters from the left and exits right "
                f"about 20 seconds later'). Describe changes between early and late images. "
                f"Do not count or number the images, and do not comment on how many there "
                f"are - just report what happens over the {span:.0f} seconds.\n\n")
        else:
            sequence_note = ""

        if SEGMENT_STYLE == 'police':
            # The nine-section police report that summarise_video() produces for a
            # whole video, applied to ONE 60-second clip instead. Same headings and
            # the same "None observed." discipline, so a segment reads like the
            # video summary and the two can be compared line for line.
            #
            # Sections 8 and 9 are deliberately rescoped: over 12 hours "pattern"
            # and "quiet periods" mean stretches of minutes, but inside one clip
            # the only axis is seconds, so they are asked for in seconds. Left
            # unscoped the model invents a period it cannot see.
            user_prompt = time_note + sequence_note + (
                f"These frames are ONE continuous {span:.0f}-second clip from CCTV camera "
                f"at {location}. Write a police-style report of this clip using these "
                f"exact sections, in this order.\n\n"
                "1. SCENE - what this camera overlooks, lighting and time of day.\n"
                "2. TRAFFIC AND PEOPLE - what is visible in this clip, with counts and "
                "colours/types.\n"
                "3. VEHICLES IDENTIFIED - report in TWO tiers and never mix them.\n"
                "   CONFIRMED: only a vehicle whose badge you can actually read, or whose "
                "body shape is unmistakable to you. One per line: make and model, colour, "
                "plate if legible, direction of travel, and the second it is seen. Write "
                "'None identified.' when nothing qualifies.\n"
                "   PROBABLE (unconfirmed): a vehicle whose silhouette places it in a "
                "recognisable family but which you cannot confirm. Write it as e.g. "
                "'white compact sedan, Dzire/Aura class (unconfirmed), 15s' or 'dark "
                "compact SUV, Creta/Seltos class (unconfirmed), 40s'. Every line here MUST "
                "carry '(unconfirmed)'. Say what stopped you being sure - distance, motion "
                "blur, rear-only angle, darkness.\n"
                "   A PROBABLE is a lead, never evidence of a model, and must never be "
                "promoted to CONFIRMED. A body type such as 'SUV' or 'hatchback' on its "
                "own is not a model. On night or infrared frames expect almost everything "
                "to be unconfirmed, and much of it not identifiable at all.\n"
                "4. REGISTRATIONS READ - every registration number you can actually read, "
                "one per line with the second and the vehicle it belongs to. State on this "
                "line that these are single automatic reads, not verified against any "
                "registry. Write 'None legible.' if there are none.\n"
                "5. NOTABLE EVENTS - anything suspicious, unusual or alert-worthy, each "
                "with the second it happens: reckless or wrong-way driving, sudden stops, "
                "near-misses, collisions, fighting, snatching, robbery, assault, "
                "unauthorised or restricted-area access, unattended bags or vehicles, "
                "loitering, suspicious contact between people or with objects, a person "
                "lying down, a stopped vehicle in a live lane, fire, smoke or flooding. "
                "Write 'None observed.' if there were none.\n"
                "6. CROWD AND CONGESTION - build-ups, queues and jams, with the second and "
                "any numbers. Write 'None observed.' if there were none.\n"
                "7. TRAFFIC VIOLATIONS - wrong-way driving, helmetless riders, red-light "
                "jumping, three on a motorcycle, driving on the footpath, illegal parking "
                "blocking a crossing, each with the second it happens. Write 'None "
                "observed.' if there were none.\n"
                "8. PATTERN OF ACTIVITY - your read of these 60 seconds: the dominant "
                "traffic, the direction most of it travels, and what changed between the "
                "early and late frames. Not a list.\n"
                "9. QUIET PERIODS - any seconds within this clip where nothing moved. "
                "Write 'None observed.' if the clip is busy throughout.\n\n"
                "Be specific: this is an investigation record, so keep colours, vehicle "
                "types, legible text and the second each thing happens. State ONLY what is "
                "visible in these frames. Never guess a make, model or plate - if it is "
                "not legible say so and move on. Do not invent an incident to fill a "
                "section. Start directly with '1. SCENE', no preamble.")
            system_instructions = (
                f"You are a police CCTV analyst at {location}. Report only what is visible "
                f"in the frames you are given. Be specific about colours, vehicle types and "
                f"legible text. Never guess a make, model or registration - naming one you "
                f"cannot actually read is the worst error you can make.")
        elif FAST_MODE:
            # Fixed labelled lines, one label per analytic category, so coverage
            # is checkable rather than left to the model's discretion. The
            # earlier version asked it to "summarise traffic in ONE sentence",
            # which is why make/model, plates, weapons, incidents, crowding and
            # sign text were absent from every segment.
            user_prompt = (
                f"These images are consecutive samples covering about {span:.0f} seconds "
                f"from one CCTV camera at {location}, in time order.\n\n"
                f"Reply using these labels, one per line, in this order. Give the time of "
                f"anything notable as seconds into the clip ('at 20s').\n\n"
                f"VEHICLES: one entry per distinct vehicle, up to six, most notable first. "
                f"For each: type (car / SUV / hatchback / sedan / van / truck / bus / "
                f"auto-rickshaw / motorcycle / scooter / bicycle / tractor), colour, "
                f"make and model ONLY if you can genuinely read the badge or the body "
                f"shape is unmistakable - otherwise write 'model unidentified', which is "
                f"the right answer on almost every night or infrared frame. A model you "
                f"name must match the body type you gave it. Never guess one, "
                f"number plate ONLY if the characters are actually legible, "
                f"direction of travel, and whether moving, stopped or parked. "
                f"If there are more than six, end with 'plus approx N others'.\n"
                f"PEOPLE: notable individuals - clothing colours, what they carry, what "
                f"they are doing, direction. Give a count if there are many.\n"
                f"CROWD: crowding, queues, congestion, traffic jams or unusual gatherings, "
                f"with rough numbers. Omit this line if traffic and footfall look normal.\n"
                f"OBJECTS: significant objects other than vehicles - bags, carts, barriers, "
                f"animals, debris, stalls, street furniture being used. Omit if none.\n"
                f"TEXT: any legible text - shop and street signs, hoardings, boards, bus "
                f"destination boards, writing on vehicles. Quote it exactly. Omit if none "
                f"is legible.\n"
                f"VIOLATIONS: wrong-way or wrong-side driving, red-light jumping, riding "
                f"without a helmet, more than two on a motorcycle, driving on the footpath "
                f"or in the wrong lane, dangerous overtaking, obvious speeding, illegal "
                f"parking or stopping, blocking a crossing. Write 'none' if there are none.\n"
                f"INCIDENTS: accident or collision or near-miss, fight or altercation, "
                f"theft or snatching or robbery, someone entering a restricted or fenced "
                f"area, an abandoned or unattended bag or vehicle, or any visible weapon "
                f"(knife, firearm, stick, rod). Write 'none' if there are none.\n"
                f"ANOMALY: anything else out of the ordinary for a road scene - a person "
                f"lying down, a stopped vehicle in a live lane, a vehicle reversing into "
                f"traffic, someone loitering or watching parked vehicles, a fire, smoke or "
                f"flooding. Write 'none' if nothing stands out.\n\n"
                f"RULES: report ONLY what is actually visible. Never guess a make, model "
                f"or plate - if it is not legible, say 'plate not legible' and move on. "
                f"Do not invent an incident to fill a line. Omit the optional labels "
                f"entirely when they have nothing. Do not repeat a vehicle across labels. "
                f"No headings beyond these labels, no preamble.")
            system_instructions = (
                f"You are a police CCTV analyst at {location}. Report only what is visible. "
                f"Be specific about colours, vehicle types and legible text. Never guess.")
        else:
            user_prompt = time_note + sequence_note + f"""Report on these surveillance frames from {location}. Use these exact section headings, in this order.

FORMAT: terse labelled facts, not prose. One line per person / object / vehicle as
"attribute: value; attribute: value". Omit any attribute you cannot actually see. Omit a
whole section that has nothing in it - never write "none" or "no people are present". Do
not pad, and do not repeat a detail in two sections.

Sections are ordered by what police act on first. Answer them in order and do not skip
ahead. In sections 5-8 describe at most five entries each, most distinctive first; if
there are more, end with "plus approx N others, similar" rather than listing them.

### 1. ALERTS
Flag each with its time in seconds into the clip: reckless or wrong-way driving, sudden
stops, near-misses, collisions; fighting, snatching, robbery, assault, threatening
behaviour; unauthorised or restricted-area access; unsafe acts; unattended bags or
vehicles; loitering or unusual movement; crowding, mob formation, blocked pathways;
suspicious contact between people or with objects; objects placed oddly; vehicles
stopped or parked where they should not be. If there is genuinely nothing: No alerts.

### 2. CHANGES ACROSS THE CLIP
What changed between the early and late frames: who or what entered and left and from
which direction; paths taken; objects moved, picked up or set down; groups forming or
breaking up; activities starting or ending; posture changes. Times in seconds.

### 3. SCENE
Time of day, lighting (daylight/night/IR), weather, type of area, layout and visible
boundaries. One or two lines.

### 4. VEHICLES
Type (car/truck/van/bus/motorcycle/scooter/auto-rickshaw/bicycle); make and model - write
"model unidentified" unless you can read the badge or the body shape is unmistakable. Naming
a model you cannot actually identify is the worst error you can make here, worse than saying
nothing. A model you do name MUST be consistent with the body type you gave: a hatchback is
not an Innova, an auto-rickshaw has no car model at all. On night or infrared frames the
model is almost never identifiable - say so rather than choosing one; colour(s); size class;
condition; number plate - read it if legible, else
say partially/not legible; distinctive features (roof rack, decals, dents, cargo,
modifications); lights and indicators; movement state (parked/moving/stopped/reversing/
turning/braking); lane or road position; door/boot/bonnet open or shut; visible cargo;
anyone entering, exiting, loading or standing at it; proximity to other vehicles or people.

### 5. PEOPLE
Sex; approx age; build/height; hair (colour/length/style/covered); skin tone; upper
clothing (colour/pattern/sleeves); lower clothing (colour/style); footwear; accessories
(cap/helmet/glasses/watch/bag/jewellery/ID); distinctive marks; posture (standing/sitting/
walking/running/bending/squatting/leaning); what the hands and arms are doing; direction
and speed of movement; where they are facing or looking.

### 6. PERSON-OBJECT
Who; object; action (holding/carrying/opening/placing/picking up/using); how it is held
(in hands/on shoulder/under arm); contact points; manner (careful/forceful/casual).

### 7. PERSON-PERSON
Who and who; distance apart; body orientation (facing/side-by-side/back-to-back); gestures
or pointing; apparent communication; physical contact (handshake/passing an item/guiding);
group formation (clustered/dispersed/in line/scattered); who is in front of or behind whom.

### 8. OBJECTS
Significant objects only. Type; size; colour; material; condition; any brand/label/text;
distinctive features (handles/wheels/locks/markings); spatial relation to other objects
(on/in/under/above/beside, stacked, grouped, mounted, connected by cable or chain);
contents if it is a container.

RULES: start directly with "### 1. ALERTS", no preamble. Only what is visible - never infer
or guess. Use directional terms (left/right/near/far). Give times as seconds into the clip,
never as a date or clock time."""

        response = _vlm_chat(system_instructions, user_prompt, image_paths, 0.1)

        content = response.content
        thinking = response.thinking

        # Log response length for debugging
        logger.info(f"[VLM] Response received. Content len: {len(content)}, Thinking len: {len(thinking)}")

        # NEVER fall back to `thinking`. qwen3-vl is a reasoning model: on a
        # busy scene it can deliberate past the whole token budget and emit no
        # `content` at all. Storing that deliberation is what put "Got it,
        # let's analyze the footage..." into the searchable corpus.
        #
        # Recovery is to RESAMPLE, not to raise num_predict. Measured on one
        # cam13 segment: budget 2200 -> 5,830 chars of thinking and no answer;
        # budget 5000 -> 12,325 chars and still no answer. The model fills
        # whatever ceiling it is given. Re-rolling the same prompt at a higher
        # temperature breaks the loop - the same call that failed at 5,830
        # chars succeeded on the next attempt at 1,724 chars in 8.7s.
        description = content
        for attempt, temp in enumerate(VLM_RETRY_TEMPERATURES, start=1):
            if description.strip():
                break
            logger.warning(
                f"[VLM] Attempt {attempt}: {len(thinking)} chars of reasoning and no "
                f"report - resampling at temperature {temp}")
            retry = _vlm_chat(system_instructions, user_prompt, image_paths, temp)
            description = retry.content
            thinking = retry.thinking

        if not description.strip():
            return ("[Error: VLM produced only reasoning after "
                    f"{len(VLM_RETRY_TEMPERATURES) + 1} attempts]")

        description = description.replace('<think>', '').replace('</think>', '').strip()

        if not description or len(description.strip()) < 5:
            if content: logger.warning(f"[VLM] Raw response too short: {content}")
            return "[Error: Empty response from VLM]"

        # Reasoning that leaked into `content` instead of `thinking`. It reads
        # as first-person deliberation ("let's check each frame", "wait, the
        # user said...") rather than a report, and storing it would put the
        # model's confusion into the searchable corpus as if it were an
        # observation. Cut to the first real section heading when one exists.
        # Each style opens with a different section 1, and this guard has to know
        # which: matching only "1. ENVIRONMENT" would treat a correct police-style
        # report opening "1. SCENE" as unheaded, and reject it the moment the
        # prose happened to contain one of the deliberation phrases below.
        _section_one = r'1\.\s*(?:ENVIRONMENT|SCENE|ALERTS)'
        head = description[:400].lower()
        looks_like_reasoning = (
            not FAST_MODE
            and not re.search(_section_one, head, re.IGNORECASE)
            and any(p in head for p in (
                "let's ", "let me ", "wait,", "okay, ", "got it,", "first, i need",
                "the user ", "hmm,", "maybe the user"))
        )
        if looks_like_reasoning:
            m = re.search(r'(?:^|\n)\s*#{0,3}\s*' + _section_one, description,
                          re.IGNORECASE)
            if m:
                logger.warning("[VLM] Trimmed leaked reasoning preamble "
                               f"({m.start()} chars) before section 1")
                description = description[m.start():].strip()
            else:
                logger.warning("[VLM] Response is reasoning, not a report - rejecting "
                               f"(first 120 chars: {description[:120]!r})")
                return "[Error: VLM returned reasoning instead of a description]"
        return description

    except Exception as e:
        logger.error(f"[VLM] Error: {e}", exc_info=True)
        return f"[Error: {str(e)}]"
    finally:
        time.sleep(0.5)
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass

# ============================================================================
# GPU (NVDEC) VIDEO DECODE
# ============================================================================
# OpenCV here isn't built with CUDA support, so cv2.VideoCapture can't hand
# decode off to the GPU. Instead we shell out to ffmpeg with NVDEC (h264_cuvid)
# and read raw BGR24 frames straight off its stdout pipe.

def probe_video_info(video_path: str):
    """(width, height, fps), parsed BY KEY NAME.

    ffprobe emits csv fields in its own internal order, not the order you asked
    for: requesting width,height,avg_frame_rate,r_frame_rate on this dataset
    returns `1920,1080,250/1,647455000/43200263` - r_frame_rate third. Positional
    unpacking therefore read cam01's nominal 250fps as its average, rejected it as
    implausible, and silently fell back to 25.0. Every camera with a bogus
    r_frame_rate got the same wrong answer.

    avg_frame_rate is the real playback rate; r_frame_rate is only the container's
    nominal time base and reads 250 on these files.
    """
    try:
        raw = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=30).stdout
        stream = (json.loads(raw).get("streams") or [{}])[0]

        def parse_rate(value: Optional[str]) -> float:
            if not value or "/" not in str(value):
                return 0.0
            num, den = str(value).split("/")
            try:
                den = float(den)
                return float(num) / den if den else 0.0
            except ValueError:
                return 0.0

        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))
        if fps <= 0 or fps > 120:
            logger.warning("%s reports an implausible frame rate (avg=%s r=%s); "
                           "assuming 25fps", video_path, stream.get("avg_frame_rate"),
                           stream.get("r_frame_rate"))
            fps = 25.0
        if not width or not height:
            return None, None, None
        return width, height, fps
    except Exception as exc:
        logger.error("ffprobe failed for %s: %s", video_path, exc)
        return None, None, None


def build_decode_cmd(video_path: str, use_gpu: bool) -> List[str]:
    """ffmpeg command piping raw BGR24 frames to stdout, decoding on the GPU
    (NVDEC/CUVID) when available, else falling back to CPU decode.

    The cuvid decoder is chosen from the file's actual codec: this dataset mixes
    H.264 with HEVC (cam27), and a hardcoded h264_cuvid fails outright on the
    HEVC file rather than falling back gracefully.
    """
    if use_gpu:
        decoder = cuvid_decoder_for(probe_video_codec(video_path))
        return [
            "ffmpeg", "-loglevel", "error",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        ] + (["-c:v", decoder] if decoder else []) + [
            "-i", video_path,
            "-vf", "hwdownload,format=nv12",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "-an", "-sn",
            "pipe:1",
        ]
    return [
        "ffmpeg", "-loglevel", "error",
        "-i", video_path,
        "-pix_fmt", "bgr24", "-f", "rawvideo", "-an", "-sn",
        "pipe:1",
    ]


# ── GPU decode capability, probed once ───────────────────────────────────────
# nvidia-smi working does NOT mean ffmpeg can decode on the GPU: in this
# container ffmpeg's cuvid reports CUDA_ERROR_NO_DEVICE even with a healthy
# RTX 3090, because it cannot reach the injected driver. The old code retried
# the GPU on every open and silently fell back, which costs a doomed process
# spawn per attempt - ~26k of them across this dataset. Probe once instead.
_GPU_DECODE_AVAILABLE: Optional[bool] = None
_GPU_DECODE_LOCK = threading.Lock()


def probe_video_codec(video_path: str) -> Optional[str]:
    """Video codec name, so the right cuvid decoder is chosen. cam27 is HEVC
    while the rest are H.264 - hardcoding h264_cuvid fails on it outright."""
    try:
        return subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:
        return None


def cuvid_decoder_for(codec: Optional[str]) -> Optional[str]:
    return {
        "h264": "h264_cuvid", "hevc": "hevc_cuvid", "h265": "hevc_cuvid",
        "mpeg4": "mpeg4_cuvid", "vp9": "vp9_cuvid", "av1": "av1_cuvid",
        "mjpeg": "mjpeg_cuvid", "vc1": "vc1_cuvid",
    }.get((codec or "").lower())


def gpu_decode_available(video_path: str) -> bool:
    """One cheap real decode attempt, cached for the process lifetime."""
    global _GPU_DECODE_AVAILABLE
    if _GPU_DECODE_AVAILABLE is not None:
        return _GPU_DECODE_AVAILABLE
    with _GPU_DECODE_LOCK:
        if _GPU_DECODE_AVAILABLE is not None:
            return _GPU_DECODE_AVAILABLE
        decoder = cuvid_decoder_for(probe_video_codec(video_path))
        if not decoder:
            _GPU_DECODE_AVAILABLE = False
            logger.info("🖥️  GPU decode unavailable (no cuvid decoder for this codec) - using CPU")
            return False
        try:
            result = subprocess.run(
                ["ffmpeg", "-loglevel", "error",
                 "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                 "-c:v", decoder, "-ss", "1", "-i", video_path,
                 "-vf", "hwdownload,format=nv12", "-frames:v", "1",
                 "-f", "null", "-"],
                capture_output=True, timeout=120)
            _GPU_DECODE_AVAILABLE = result.returncode == 0
        except Exception:
            _GPU_DECODE_AVAILABLE = False
        if _GPU_DECODE_AVAILABLE:
            logger.info("🚀 GPU decode (NVDEC) available")
        else:
            logger.info("🖥️  GPU decode (NVDEC) unavailable in this container - "
                        "using CPU decode. Seeking keeps this cheap; no action needed.")
        return _GPU_DECODE_AVAILABLE


def probe_duration(video_path: str) -> Optional[float]:
    """Container duration in seconds, or None if the file is unreadable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out)
    except Exception:
        return None


def extract_frame_at(video_path: str, offset_seconds: float,
                     width: int, height: int, use_gpu: bool = True,
                     timeout: int = 180) -> Optional[np.ndarray]:
    """Decode exactly one frame at offset_seconds.

    -ss BEFORE -i makes ffmpeg keyframe-seek instead of decoding forward from
    zero, which is what turns a 12-hour file into a sub-second lookup. This is
    the primitive that replaces piping every frame: 2 seeks per segment instead
    of ~900 decoded-and-discarded frames. Measured on this dataset: 0.6-0.8s per
    frame on CPU, including seeks 8 hours into a 5 GB file.
    """
    frame_size = width * height * 3
    gpu = use_gpu and gpu_decode_available(video_path)

    def _run(with_gpu: bool) -> bytes:
        if with_gpu:
            decoder = cuvid_decoder_for(probe_video_codec(video_path))
            cmd = [
                "ffmpeg", "-loglevel", "error",
                "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            ] + (["-c:v", decoder] if decoder else []) + [
                "-ss", f"{offset_seconds:.3f}", "-i", video_path,
                "-vf", "hwdownload,format=nv12",
                "-frames:v", "1", "-pix_fmt", "bgr24",
                "-f", "rawvideo", "-an", "-sn", "pipe:1",
            ]
        else:
            cmd = [
                "ffmpeg", "-loglevel", "error",
                "-ss", f"{offset_seconds:.3f}", "-i", video_path,
                "-frames:v", "1", "-pix_fmt", "bgr24",
                "-f", "rawvideo", "-an", "-sn", "pipe:1",
            ]
        return subprocess.run(cmd, capture_output=True, timeout=timeout).stdout

    try:
        raw = _run(gpu)
        if len(raw) < frame_size and gpu:
            raw = _run(False)
        if len(raw) < frame_size:
            return None
        return np.frombuffer(raw[:frame_size], dtype=np.uint8).reshape((height, width, 3)).copy()
    except subprocess.TimeoutExpired:
        logger.warning(f"Frame seek timed out at {offset_seconds:.1f}s in {os.path.basename(video_path)}")
        return None
    except Exception as e:
        logger.warning(f"Frame seek failed at {offset_seconds:.1f}s: {e}")
        return None


def existing_segment_ids(collection, camera_id: str) -> set:
    """Segment IDs already stored for this camera, so a killed run resumes."""
    if collection is None:
        return set()
    try:
        return {
            doc["segment_id"]
            for doc in collection.find({"camera_id": camera_id}, {"segment_id": 1})
            if doc.get("segment_id") is not None
        }
    except Exception as e:
        logger.warning(f"[{camera_id}] Could not read existing segments for resume: {e}")
        return set()


def open_decoder(video_path: str, frame_size: int, camera_id: str):
    """Start the GPU decode pipe; fall back to CPU decode if NVDEC can't
    produce frames for this file (unsupported profile, no free session, etc)."""
    if gpu_decode_available(video_path):
        proc = subprocess.Popen(build_decode_cmd(video_path, use_gpu=True),
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 bufsize=frame_size * 4)
        first_chunk = proc.stdout.read(frame_size)
        if len(first_chunk) == frame_size:
            return proc, first_chunk

        proc.stdout.close()
        proc.terminate()
        try: proc.wait(timeout=3)
        except Exception: proc.kill()
        logger.warning(f"[{camera_id}] GPU (NVDEC) decode produced no frames, falling back to CPU decode")

    proc = subprocess.Popen(build_decode_cmd(video_path, use_gpu=False),
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             bufsize=frame_size * 4)
    first_chunk = proc.stdout.read(frame_size)
    return proc, first_chunk

# ============================================================================
# SEGMENT FINALIZATION
# ============================================================================

def finalize_live_segment(camera_id, location, segment_id,
                          frame1, frame2, segment_start_time, segment_end_time,
                          azure_client, db,
                          stats: Optional[dict] = None,
                          vlm_frames: Optional[List[np.ndarray]] = None,
                          vlm_offsets: Optional[List[float]] = None):
    """Describe the segment via VLM, upload the sampled frames, and save the
    segment to MongoDB - the live counterpart of merge-final-many.py's
    finalize_segment().

    segment_start_time/segment_end_time are wall-clock datetimes captured at
    the instant frame1/frame2 were read off the live pipe - there is no anchor
    registry, no offset arithmetic and no overlay OCR to fall back to or drift-
    check against, because a live frame's capture time IS its timestamp.
    """
    timings = {}
    t0 = time.time()
    frame_urls = []
    for i, (frame, frame_time) in enumerate([(frame1, segment_start_time), (frame2, segment_end_time)], 1):
        url = upload_frame_to_azure(azure_client, camera_id, segment_id, frame, f"frame{i}", frame_time)
        if url: frame_urls.append(url)

    if len(frame_urls) != 2:
        return False

    # A contact sheet of every sampled frame, labelled with its offset into the
    # segment. Without it the description cites moments no stored image shows.
    if CONTACT_SHEET and vlm_frames and len(vlm_frames) > 2:
        try:
            sheet = build_contact_sheet(vlm_frames, vlm_offsets or [], 0.0)
            if sheet is not None:
                sheet_url = upload_frame_to_azure(
                    azure_client, camera_id, segment_id, sheet, "sheet", segment_start_time)
                if sheet_url:
                    frame_urls.append(sheet_url)
        except Exception as e:
            logger.warning(f"[{camera_id}] Segment #{segment_id}: contact sheet failed: {e}")
    timings['azure_upload'] = time.time() - t0

    # Every sampled frame goes to the VLM (downscaled); only frame1/frame2 are
    # uploaded to Azure as evidence thumbnails, so denser sampling does not
    # multiply storage or upload time.
    raw_frames = []
    for frame in (vlm_frames if vlm_frames else [frame1, frame2]):
        buffer = encode_for_vlm(frame)
        if buffer is not None:
            raw_frames.append(buffer)

    t0 = time.time()
    moved, motion_score = (True, 1.0)
    if MOTION_GATE:
        moved, motion_score = frames_have_motion(vlm_frames if vlm_frames else [frame1, frame2])
    timings['motion_check'] = time.time() - t0

    t0 = time.time()
    if not moved:
        # Nothing to describe. Recorded explicitly rather than left absent, so
        # "was anything happening at 3am?" is answerable.
        stamp = segment_start_time.strftime('%d-%m-%Y %H:%M:%S')
        raw_description = (
            f"No activity detected at {location} on {stamp}. The scene is static across the sampled "
            f"frames: no people, vehicles or moving objects. Camera view unchanged.")
        timings['vlm'] = 0.0
        timings['gated'] = 1.0
    else:
        raw_description = get_segment_description(
            raw_frames, OPENROUTER_API_KEY, location,
            segment_time=segment_start_time, time_trusted=True)
        timings['vlm'] = time.time() - t0
        timings['gated'] = 0.0
    cleaned_description = clean_description_text(raw_description)

    # Guarantee the stamp: write it ourselves rather than depending on the
    # model. A live segment's time is always known exactly - it is real
    # wall-clock time - so this line is unconditional, unlike the batch
    # version which has an "anchor unresolved" fallback message.
    stamp_line = (
        f"Recorded {segment_start_time.strftime('%d-%m-%Y')} "
        f"({segment_start_time.strftime('%A')}) "
        f"{segment_start_time.strftime('%H:%M:%S')} to "
        f"{segment_end_time.strftime('%H:%M:%S')} "
        f"at {location}, camera {camera_id} (live).")

    if not cleaned_description.startswith("[Error:"):
        cleaned_description = f"{stamp_line} {cleaned_description}"

    # A VLM failure must NOT be stored. The error string would be embedded and
    # become a searchable "description", poisoning semantic search. Fail the
    # segment instead: the next segment just carries on.
    if raw_description.startswith("[Error:"):
        logger.error(f"[{camera_id}] Segment #{segment_id} VLM FAILED, not storing: "
                     f"{raw_description[:160]}")
        if isinstance(stats, dict):
            stats['timings'] = timings
            stats['vlm_error'] = raw_description[:200]
        return False

    print(f"\n{'='*80}")
    print(f"🤖 [{camera_id}] SEGMENT VLM - #{segment_id} - {segment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(cleaned_description)
    print(f"{'='*80}\n")

    segment = VideoSegment(
        segment_id=segment_id,
        camera_id=camera_id,
        location=location,
        start_time=segment_start_time,
        end_time=segment_end_time,
        frame_urls=frame_urls,
        description=cleaned_description,
        cumulative_minutes=(segment_id * SEGMENT_DURATION) / 60,
        source_video=f"live:{camera_id}",
        anchor_confidence="live",
        video_offset_seconds=0.0,
        overlay_drift_seconds=None,
    )
    segment.motion_score = round(motion_score, 5)
    segment.motion_gated = (not moved)
    t0 = time.time()
    db.add_segment(segment)
    timings['embed_and_store'] = time.time() - t0

    if isinstance(stats, dict):
        stats['timings'] = timings

    # ── real-time alerts ──────────────────────────────────────────────────────
    # Fired here, one minute of footage after the event. Wrapped so that
    # nothing about alerting can break ingestion: a failed alert must never
    # cost a stored segment.
    try:
        from alert_dispatch import raise_alerts_for_segment
        raise_alerts_for_segment(camera_id, location, {
            "_id": getattr(segment, "_id", None) or segment_id,
            "segment_id": segment_id,
            "description": cleaned_description,
            "start_time": segment_start_time.isoformat(),
            "frame_urls": frame_urls,
            "cumulative_minutes": (segment_id * SEGMENT_DURATION) / 60,
        })
    except Exception as e:
        logger.error(f"[{camera_id}] segment #{segment_id} alert failed "
                     f"(segment is stored and unaffected): {e}")

    logger.info(f"[{camera_id}] ✅ Segment #{segment_id} complete")
    return True

# ============================================================================
# PER-CAMERA LIVE WORKER (runs in the thread pool, one per camera, forever)
# ============================================================================

def process_live_camera(camera_config, model_manager, mongo_collection, azure_client,
                        stop_event: threading.Event):
    """Continuously decode one live stream, sample frames every
    FRAME_INTERVAL_SECONDS within each rolling SEGMENT_DURATION-second window,
    and finalize+store a segment each time a window fills - forever, until
    stop_event is set. Reconnects the stream on drop/EOF.

    This is the live counterpart of merge-final-many.py's _process_video_seek:
    same dense multi-frame VLM sampling and segment-storage shape, but frames
    arrive sequentially off a persistent pipe (there is nothing to seek to on
    a live source) and segment_id just counts up forever instead of being
    bounded by a known file duration.
    """
    camera_id = camera_config['camera_id']
    url = camera_config['rtmp_url']
    location = camera_config.get('location', 'Unknown Location')
    threading.current_thread().name = f"{camera_id}-Worker"

    memory_monitor = MemoryMonitor()
    db = MongoVideoDatabase(model_manager=model_manager, collection=mongo_collection)

    already = existing_segment_ids(mongo_collection, camera_id) if RESUME_FROM_MONGO else set()
    segment_id = (max(already) + 1) if already else 0
    if already:
        logger.info(f"[{camera_id}] Resuming - {len(already)} segment(s) already stored, "
                    f"continuing from #{segment_id}")

    # Probe stream dimensions/fps, retrying rather than giving up: a live feed
    # can be briefly unreachable (DVR reboot, network blip) at worker startup
    # just as easily as mid-stream.
    width = height = fps = None
    while not stop_event.is_set():
        width, height, fps = probe_video_info(url)
        if width and height and fps:
            break
        logger.warning(f"[{camera_id}] Could not probe stream, retrying in {RECONNECT_DELAY_S}s...")
        time.sleep(RECONNECT_DELAY_S)
    if stop_event.is_set():
        return {'camera_id': camera_id, 'segments': 0, 'error': 'stopped before probe succeeded'}

    frame_size = width * height * 3
    frames_per_segment = max(1, int(round(SEGMENT_DURATION * fps)))
    # Same dense sampling cadence as the batch script (one frame every
    # FRAME_INTERVAL_SECONDS), expressed as frame INDICES within the segment
    # since a live pipe is read frame-by-frame, not seeked into.
    sample_offsets = segment_sample_offsets(0.0, interval=FRAME_INTERVAL_SECONDS,
                                            duration=SEGMENT_DURATION)
    target_indices = sorted({min(int(round(o * fps)), frames_per_segment - 1)
                             for o in sample_offsets})

    logger.info(f"[{camera_id}] {width}x{height} @ {fps:.2f}fps live (location: {location}) | "
                f"segment={SEGMENT_DURATION}s | {len(target_indices)} sampled frame(s)/segment")

    def open_stream():
        return open_decoder(url, frame_size, camera_id)

    proc, next_chunk = open_stream()
    segments_saved = 0
    frame_index_in_segment = 0
    target_ptr = 0
    vlm_frames: List[np.ndarray] = []
    vlm_capture_times: List[datetime.datetime] = []

    def reset_segment():
        nonlocal frame_index_in_segment, target_ptr, vlm_frames, vlm_capture_times
        frame_index_in_segment = 0
        target_ptr = 0
        vlm_frames = []
        vlm_capture_times = []

    try:
        while not stop_event.is_set():
            memory_monitor.check()
            raw = next_chunk
            next_chunk = None
            eof = raw is None or len(raw) != frame_size

            if eof:
                # A live stream ending is not "done", it is a drop: reconnect
                # and start a fresh segment rather than treating this as EOF.
                try: proc.stdout.close()
                except Exception: pass
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try: proc.kill()
                    except Exception: pass
                logger.warning(f"[{camera_id}] stream ended/dropped, reconnecting "
                               f"in {RECONNECT_DELAY_S}s...")
                reset_segment()
                time.sleep(RECONNECT_DELAY_S)
                if stop_event.is_set():
                    break
                proc, next_chunk = open_stream()
                continue

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            if target_ptr < len(target_indices) and frame_index_in_segment >= target_indices[target_ptr]:
                vlm_frames.append(frame.copy())
                vlm_capture_times.append(get_now())
                target_ptr += 1
            frame_index_in_segment += 1

            if frame_index_in_segment >= frames_per_segment:
                if vlm_frames:
                    mid = len(vlm_frames) // 2
                    frame1, seg_start = vlm_frames[0], vlm_capture_times[0]
                    frame2, seg_end = vlm_frames[mid], vlm_capture_times[mid]
                    try:
                        if finalize_live_segment(
                                camera_id, location, segment_id, frame1, frame2,
                                seg_start, seg_end, azure_client, db,
                                vlm_frames=vlm_frames,
                                vlm_offsets=[i / fps for i in range(len(vlm_frames))]):
                            segments_saved += 1
                    except Exception as e:
                        logger.error(f"[{camera_id}] Segment #{segment_id} error: {e}", exc_info=True)
                else:
                    logger.warning(f"[{camera_id}] Segment #{segment_id}: no frames captured, skipping")

                segment_id += 1
                reset_segment()
                if segment_id % 20 == 0:
                    gc.collect()
                if MAX_SEGMENTS_PER_CAMERA and segment_id >= MAX_SEGMENTS_PER_CAMERA:
                    logger.info(f"[{camera_id}] reached MAX_SEGMENTS_PER_CAMERA="
                                f"{MAX_SEGMENTS_PER_CAMERA}, stopping this camera")
                    break

            if not stop_event.is_set():
                next_chunk = proc.stdout.read(frame_size)

    finally:
        try: proc.stdout.close()
        except Exception: pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try: proc.kill()
            except Exception: pass

    logger.info(f"[{camera_id}] Worker stopped - {segments_saved} segment(s) saved this run")
    return {'camera_id': camera_id, 'segments': segments_saved}

# ============================================================================
# MAIN
# ============================================================================

SUMMARY_COLLECTION = os.getenv('SUMMARY_COLLECTION', 'video_summaries')

# 'ollama' (local, default) or 'openrouter' (GLM). Only the video-level summary
# and the alert pass are affected - the per-segment VLM stays local either way,
# so switching this does not send footage off the box, only the prose about it.
SUMMARY_BACKEND = os.getenv('SUMMARY_BACKEND', 'ollama').lower()
OPENROUTER_MODEL_NAME = os.getenv('OPENROUTER_MODEL', 'z-ai/glm-5.3-flash')

# The video summary is an investigation record, so it is allowed to be long. The
# old 1200 was sized for a local 8B model; GLM has room for far more and the
# nine-section format needs it.
SUMMARY_MAX_TOKENS = int(os.getenv('SUMMARY_MAX_TOKENS',
                                   '6000' if SUMMARY_BACKEND == 'openrouter' else '2500'))


def summarise_video(db, camera_id: str, location: str, source_video: str) -> Optional[str]:
    """Roll a video's per-segment descriptions up into one video-level summary.

    Reads back what was actually stored rather than accumulating in memory, so a
    resumed run summarises the whole video and not just the segments this
    process happened to write.
    """
    try:
        # _id and frame_urls are carried because the alert pass below needs a
        # frame to attach to each alert, and the segment's own id to reference.
        docs = list(db.collection.find(
            {"camera_id": camera_id},
            {"segment_id": 1, "description": 1, "start_time": 1,
             "frame_urls": 1, "_id": 1},
        ).sort("segment_id", 1))
    except Exception as e:
        logger.error(f"[{camera_id}] summary: could not read segments back: {e}")
        return None

    docs = [d for d in docs
            if d.get("description") and not d["description"].startswith("[Error:")]
    if not docs:
        logger.warning(f"[{camera_id}] summary: no usable segment descriptions")
        return None

    # A 12-hour video is ~720 descriptions at ~2.5k chars: far past any context
    # window. Summarise in chunks, then summarise the chunk summaries.
    CHUNK = 25

    def ask(prompt: str, budget: int) -> str:
        # SUMMARY_BACKEND=openrouter sends the summary to GLM (1.3M context, and
        # cheap enough that a long, detailed summary costs fractions of a cent).
        # Default stays ollama so nothing leaves the box unless asked.
        if SUMMARY_BACKEND == "openrouter":
            from alert_dispatch import llm_text
            return llm_text(prompt, max_tokens=budget, temperature=0.2)

        # think=False: a reasoning model spends `budget` on thinking BEFORE the
        # answer, and the answer is what gets stored. Measured on qwen3:8b, two
        # thirds of a 500-token budget went to thinking and the summary was cut
        # off mid-sentence. Set OLLAMA_THINK=true to restore it.
        kwargs = ({} if os.getenv("OLLAMA_THINK", "false").lower() == "true"
                  else {"think": False})
        args = dict(model=SUMMARY_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.2, "num_predict": budget,
                             "num_ctx": VLM_NUM_CTX})
        try:
            r = chat(**args, **kwargs)
        except TypeError:                  # client too old to know `think`
            r = chat(**args)
        return (r.message.content or "").replace("<think>", "").replace("</think>", "").strip()

    def block(subset) -> str:
        return "\n\n".join(
            f"[minute {d['segment_id']}] {d['description'][:1200]}" for d in subset)

    try:
        if len(docs) <= CHUNK:
            body = block(docs)
        else:
            partials = []
            for i in range(0, len(docs), CHUNK):
                subset = docs[i:i + CHUNK]
                logger.info(f"[{camera_id}] summary: condensing minutes "
                            f"{subset[0]['segment_id']}-{subset[-1]['segment_id']} "
                            f"({i // CHUNK + 1}/{(len(docs) + CHUNK - 1) // CHUNK})")
                partials.append(ask(
                    f"These are per-minute CCTV observations from camera {camera_id} "
                    f"({location}), minutes {subset[0]['segment_id']}-{subset[-1]['segment_id']}.\n\n"
                    f"{block(subset)}\n\n"
                    "Condense into a factual paragraph: vehicles and people seen (with "
                    "colours/types), what they did, and anything unusual. Keep the minute "
                    "numbers for anything notable. Omit repetition of the static scenery.\n"
                    "Carry through every make and model the observations name (e.g. "
                    "'Toyota Innova', 'Tata Nexon') together with its colour and plate - "
                    "these are the identifying details and must survive condensing. Never "
                    "add a make or model the observations do not state.",
                    700))
            body = "\n\n".join(f"[part {i+1}] {p}" for i, p in enumerate(partials))

        summary = ask(
            f"The following are observations from CCTV camera {camera_id} at {location}, "
            f"covering {len(docs)} minutes of footage from {source_video}.\n\n{body}\n\n"
            "Write a police-style summary of this video with these sections:\n"
            "1. SCENE - what this camera overlooks, lighting and time of day.\n"
            "2. TRAFFIC AND PEOPLE - what was seen overall, with counts and "
            "colours/types where stated.\n"
            "3. VEHICLES IDENTIFIED - keep the observations' two tiers separate.\n"
            "   CONFIRMED: every vehicle the observations name a make and model for "
            "WITHOUT marking it unconfirmed, one per line: make and model, colour, plate "
            "if one was read, direction of travel, and the minute(s) seen. Write 'None "
            "identified.' if the observations confirm no make or model.\n"
            "   PROBABLE (unconfirmed): every vehicle the observations mark "
            "'(unconfirmed)', carried through with that marking intact and the minute(s) "
            "seen. Never drop the marking and never promote one of these to CONFIRMED - a "
            "class guess repeated across ten minutes is still a guess, not an "
            "identification.\n"
            "4. REGISTRATIONS READ - every registration number the observations quote, "
            "one per line with the minute and the vehicle it belongs to. State on this "
            "line that these are single automatic reads, not verified against any "
            "registry. Write 'None legible.' if there are none.\n"
            "5. NOTABLE EVENTS - anything suspicious, unusual or alert-worthy, each "
            "with its minute number. Write 'None observed.' if there were none.\n"
            "6. CROWD AND CONGESTION - build-ups, queues and jams, with the minutes and "
            "any numbers stated. Write 'None observed.' if there were none.\n"
            "7. TRAFFIC VIOLATIONS - wrong-way driving, helmetless riders, red-light "
            "jumping, illegal parking, each with its minute. Write 'None observed.' if "
            "there were none.\n"
            "8. PATTERN OF ACTIVITY - how the scene changed across the period: when it "
            "was busiest and quietest, what the dominant traffic was, and anything that "
            "recurred. This is the analyst's read of the whole period, not a list.\n"
            "9. QUIET PERIODS - stretches where nothing happened.\n"
            "Be thorough and specific: this is an investigation record, so prefer detail "
            "over brevity and always keep minute numbers, colours, makes, models and "
            "registrations. State only what the observations support. Do not invent "
            "detail. Never name a make or model that the observations do not state - a "
            "body type such as 'SUV' is not a model, and 'model unidentified' means it "
            "stays unidentified.",
            SUMMARY_MAX_TOKENS)
    except Exception as e:
        logger.error(f"[{camera_id}] summary generation failed: {e}")
        return None

    if not summary or len(summary) < 20:
        return None

    # ── alerts ────────────────────────────────────────────────────────────────
    # Deliberately here and nowhere else: the descriptions have just been read
    # back for the summary, so the alert pass reuses them with no extra video
    # decoding and no VLM call. A failure here must not lose the summary that
    # was just produced, so it is caught and reported rather than raised.
    try:
        from alert_dispatch import raise_alerts_for_summary
        alert_stats = raise_alerts_for_summary(camera_id, location, docs)
    except Exception as e:
        logger.error(f"[{camera_id}] alert dispatch failed (summary is unaffected): {e}")
        alert_stats = {"sent": 0, "failed": 0, "skipped": 0}

    try:
        doc = {
            "camera_id": camera_id,
            "location": location,
            "source_video": source_video,
            "summary": summary,
            "alerts_sent": alert_stats.get("sent", 0),
            "alerts_skipped": alert_stats.get("skipped", 0),
            "segments_covered": len(docs),
            "generated_at": get_now(),
            # The model that ACTUALLY wrote it. This recorded SUMMARY_MODEL
            # unconditionally, so a summary written by GLM was stored as having
            # been written by the local model - wrong provenance on a record a
            # police user may later have to account for.
            "model": (OPENROUTER_MODEL_NAME if SUMMARY_BACKEND == "openrouter"
                      else SUMMARY_MODEL),
            "summary_backend": SUMMARY_BACKEND,
            "frame_interval_seconds": FRAME_INTERVAL_SECONDS,
        }
        try:
            with db.model_manager.embedding_model_lock:
                emb = db.embedding_model.encode([f"search_document: {summary}"],
                                                convert_to_numpy=True,
                                                normalize_embeddings=True)
            doc["embedding"] = emb[0].tolist()
        except Exception as e:
            logger.warning(f"[{camera_id}] summary stored without embedding: {e}")

        db.collection.database[SUMMARY_COLLECTION].update_one(
            {"camera_id": camera_id}, {"$set": doc}, upsert=True)
        logger.info(f"[{camera_id}] ✅ video summary stored "
                    f"({len(summary)} chars over {len(docs)} minutes)")
    except Exception as e:
        logger.error(f"[{camera_id}] could not store summary: {e}")

    return summary


def _preflight_vlm() -> bool:
    """Confirm the vision backend can actually answer before processing anything.

    Without this a dead backend produces one failed segment per minute of
    footage, all identical, for hours.
    """
    # OpenRouter backend: Ollama is not involved at all, so checking it would
    # abort a perfectly runnable job. Verify the key and that the model really
    # accepts images instead - a text-only model id here fails on every segment.
    if VLM_BACKEND == "openrouter":
        try:
            import httpx
            from alert_dispatch import _openrouter_key
            if not _openrouter_key():
                logger.error("=" * 78)
                logger.error("VLM_BACKEND=openrouter but OPENROUTER_API_KEY is not set.")
                logger.error("Put it in .env as  OPENROUTER_API_KEY=sk-or-v1-...")
                logger.error("=" * 78)
                return False
            models = httpx.get("https://openrouter.ai/api/v1/models",
                               timeout=30).json()["data"]
            entry = next((m for m in models if m["id"] == OPENROUTER_VLM_MODEL), None)
            if entry is None:
                logger.error(f"OpenRouter does not list '{OPENROUTER_VLM_MODEL}'")
                return False
            modalities = entry.get("architecture", {}).get("input_modalities", [])
            if "image" not in modalities:
                logger.error("=" * 78)
                logger.error(f"'{OPENROUTER_VLM_MODEL}' accepts {modalities} - no image "
                             "input, so it cannot describe frames.")
                logger.error("Set OPENROUTER_VLM_MODEL to a vision model.")
                logger.error("=" * 78)
                return False
            logger.info(f"✅ OpenRouter reachable, '{OPENROUTER_VLM_MODEL}' "
                        f"accepts {modalities}")
            logger.warning("VLM_BACKEND=openrouter - FRAMES are uploaded to a third "
                           "party, not just the prose about them.")
            return True
        except Exception as e:
            logger.error(f"OpenRouter preflight failed: {e}")
            return False

    try:
        import ollama
        names = [m.get('model') or m.get('name') or ''
                 for m in (ollama.list().get('models') or [])]
    except Exception as e:
        logger.error("=" * 78)
        logger.error(f"Ollama is not reachable: {e}")
        logger.error("Start it with:  ollama serve   (or: nohup ollama serve &)")
        logger.error("=" * 78)
        return False

    stem = VLM_MODEL.split(':')[0]
    if not any(n == VLM_MODEL or n.split(':')[0] == stem for n in names):
        logger.error("=" * 78)
        logger.error(f"Ollama is running but '{VLM_MODEL}' is not installed.")
        logger.error(f"Installed: {', '.join(names) or '(none)'}")
        logger.error(f"Pull it with:  ollama pull {VLM_MODEL}")
        logger.error("=" * 78)
        return False

    logger.info(f"✅ Ollama reachable, VLM '{VLM_MODEL}' available")
    return True


def main(config_path: str = LIVE_CAMERAS_CSV_PATH,
         only_cameras: Optional[List[str]] = None,
         max_segments: Optional[int] = None,
         workers: Optional[int] = None):
    logger.info("="*78)
    logger.info("🎥 LIVE CCTV PROSE TIER — one VLM description per 60s segment, real-time")
    logger.info("="*78)

    if not _preflight_vlm():
        return

    cameras, camera_count = load_camera_config(config_path)
    if not cameras or camera_count == 0:
        logger.critical(f"No cameras loaded from {config_path}")
        return

    if only_cameras:
        wanted = set(only_cameras)
        cameras = [c for c in cameras if c["camera_id"] in wanted]
        camera_count = len(cameras)
        if not cameras:
            logger.error(f"None of {sorted(wanted)} found in {config_path}")
            return
        logger.info(f"Restricted to {camera_count} camera(s): "
                    f"{', '.join(c['camera_id'] for c in cameras)}")

    if max_segments:
        global MAX_SEGMENTS_PER_CAMERA
        MAX_SEGMENTS_PER_CAMERA = max_segments
        logger.info(f"Capping at {max_segments} segment(s) per camera (test mode)")

    model_manager = UniversalModelManager()
    model_manager.get_embedding_model()

    mongo_client, activities_collection = initialize_mongodb()
    azure_client = initialize_azure_client()
    if not azure_client:
        logger.error("Azure init failed, aborting")
        mongo_client.close()
        return

    pool = min(workers or MAX_CONCURRENT_VIDEOS, camera_count)
    logger.info(f"Starting {camera_count} live camera(s), {pool} running at a time")

    stop_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=pool, thread_name_prefix="LiveCam")
    futures = {
        executor.submit(process_live_camera, cam, model_manager, activities_collection,
                        azure_client, stop_event): cam
        for cam in cameras
    }
    reported = set()

    def report_failures():
        # Each worker already logs and swallows its own per-segment errors, so
        # this only catches something that killed the worker thread outright -
        # without it a dead camera goes unreported until every other camera
        # also stops, which for a live pipeline may be never.
        for f, cam in futures.items():
            if f.done() and f not in reported:
                reported.add(f)
                exc = f.exception()
                if exc:
                    logger.error(f"[{cam['camera_id']}] worker crashed: {exc}", exc_info=exc)
                else:
                    logger.info(f"[{cam['camera_id']}] worker finished: {f.result()}")

    try:
        while not all(f.done() for f in futures):
            report_failures()
            time.sleep(2)
    except KeyboardInterrupt:
        logger.info("Stopped by user, waiting for camera workers to exit...")
    finally:
        stop_event.set()
        executor.shutdown(wait=True)
        report_failures()
        mongo_client.close()

    logger.info("All camera workers stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Live prose tier: one VLM description per 60-second segment, "
                    "continuously, per live camera stream.")
    parser.add_argument("--config", default=LIVE_CAMERAS_CSV_PATH,
                        help=f"CSV of camera_id,rtmp_url(,location) rows "
                             f"(default {LIVE_CAMERAS_CSV_PATH})")
    parser.add_argument("--cameras", nargs="*", metavar="CAMERA_ID",
                        help="Only these camera_id(s) from the CSV")
    parser.add_argument("--max-segments", type=int, metavar="N",
                        help="Stop each camera after N segments (smoke test, not for production)")
    parser.add_argument("--workers", type=int,
                        help=f"Concurrent live cameras (default {MAX_CONCURRENT_VIDEOS})")
    args = parser.parse_args()

    main(config_path=args.config, only_cameras=args.cameras,
         max_segments=args.max_segments, workers=args.workers)