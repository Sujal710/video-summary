"""
Face search: upload a photo, get back every enrolled identity in MongoDB
(arcis.embeddings-new) ranked by cosine similarity to that photo's face
embedding(s), each with every stored snapshot image, its location, and the
timestamp it was captured at (mongo_enrollment.py's "Sightings" field).

Location/time filtering can happen two ways: face_search_ui.html filters the
full (unfiltered) response client-side, instantly, with no extra round trip;
/api/search also accepts optional location/start_time/end_time form fields
to filter server-side, for callers (curl, scripts) that aren't the UI.

Documents written before Sightings existed only have the older flat
"ImageUrls" list (no location) - those are still read as a fallback, with
the timestamp parsed back out of the filename mongo_enrollment.py wrote it
with ("<PersonName>_<YYYY-MM-DD-HH-MM-SS-mmm>.png") and location "Unknown".

Uses the exact same InsightFace FaceAnalysis model/config (DET_SIZE,
MIN_FACE_SIZE) and the same one-matmul-against-every-embedding matching
approach as mongo_enrollment.py - imported from it directly so both scripts
can never drift apart on matching behavior. The one deliberate difference is
the detection-confidence bar (SEARCH_MIN_DET_SCORE below): enrollment writes
to the DB and stays strict about quality, but search is just a read-only
lookup against a photo the user deliberately chose, so it can afford to be
more lenient.

/api/search also accepts optional `location`, `start_time`, `end_time` form
fields to filter which images are returned per match server-side (the UI
does its own client-side filtering on the full result too, since it's
already there - these exist so a plain curl call can filter without a UI).
start_time/end_time are ISO 8601 ("2026-09-03T15:00:00" or with a "Z"/offset
suffix); a value with no offset is treated as UTC, matching how new
Sightings timestamps are stored.
"""
import datetime
import logging
import os

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import mongo_enrollment as enroll

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FACE_SEARCH_PORT = 8090
SEARCH_MIN_DET_SCORE = 0.30  # more lenient than enroll.MIN_DET_SCORE - see module docstring
SEARCH_MIN_SIMILARITY = 0.30  # only report matches at least this confident

# TLS - vmukti.pem carries the full chain (leaf + intermediate) so clients
# don't need to fetch the intermediate separately; vmukti.key is the
# matching unencrypted private key. Both live one level up from this file.
_CERT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SSL_CERTFILE = os.path.join(_CERT_DIR, "vmukti.pem")
SSL_KEYFILE = os.path.join(_CERT_DIR, "vmukti.key")

app = FastAPI(title="Face Search")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # can't be True together with a wildcard origin
    allow_methods=["*"],
    allow_headers=["*"],
)

_face_app = None


def _ensure_model_loaded():
    """Lazy-loaded on first request (or eagerly in __main__) so importing
    this module for testing doesn't require a GPU to be present."""
    global _face_app
    if _face_app is not None:
        return
    from insightface.app import FaceAnalysis
    logging.info("[FaceSearch] Loading FaceAnalysis model...")
    _face_app = FaceAnalysis(det_size=enroll.DET_SIZE)
    _face_app.prepare(ctx_id=0)
    logging.info("[FaceSearch] Model loaded.")


def _parse_image_timestamp(url, person_name):
    """Recovers the capture time from a snapshot URL written by
    mongo_enrollment.py's _upload_snapshot(): ".../<name>_<timestamp>.png"."""
    filename = url.rsplit('/', 1)[-1]
    stem = filename.rsplit('.', 1)[0]
    prefix = f"{person_name}_"
    ts_str = stem[len(prefix):] if stem.startswith(prefix) else stem
    try:
        return datetime.datetime.strptime(ts_str, "%Y-%m-%d-%H-%M-%S-%f").isoformat()
    except ValueError:
        return None


def _parse_filter_datetime(value, field_name):
    """Parses an optional start_time/end_time query value. A naive value
    (no "Z"/offset) is assumed UTC, matching how new Sightings timestamps
    are stored - so "2026-09-03T15:00:00" lines up with the same instant a
    Sightings entry from that wall-clock UTC time would carry."""
    if not value:
        return None
    try:
        dt = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}: {value!r} (expected ISO 8601)")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def _image_matches_filter(img, location, start_dt, end_dt):
    if location and img.get("location") != location:
        return False
    if start_dt or end_dt:
        if not img.get("timestamp"):
            return False
        ts = datetime.datetime.fromisoformat(img["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        if start_dt and ts < start_dt:
            return False
        if end_dt and ts > end_dt:
            return False
    return True


def _load_person_images():
    """PersonName -> [{url, location, timestamp}], newest first. Prefers the
    "Sightings" field (has real location + timestamp per image); falls back
    to the older flat "ImageUrls" list (location "Unknown", timestamp parsed
    from the filename) for documents written before Sightings existed."""
    client, coll = enroll._connect()
    try:
        images = {}
        for doc in coll.find({}, {"PersonName": 1, "ImageUrls": 1, "Sightings": 1}):
            name = doc.get("PersonName")
            if not name:
                continue

            sightings = doc.get("Sightings")
            if sightings:
                entries = [
                    {
                        "url": s.get("ImageUrl"),
                        "location": s.get("Location") or "Unknown",
                        "timestamp": s["Timestamp"].isoformat() if s.get("Timestamp") else None,
                    }
                    for s in sightings if s.get("ImageUrl")
                ]
            else:
                entries = [
                    {"url": url, "location": "Unknown", "timestamp": _parse_image_timestamp(url, name)}
                    for url in (doc.get("ImageUrls") or [])
                ]

            entries.sort(key=lambda e: e["timestamp"] or "", reverse=True)
            images[name] = entries
        return images
    finally:
        client.close()


@app.get("/")
async def index():
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_search_ui.html")
    if os.path.isfile(ui_path):
        return FileResponse(ui_path)
    # No UI shipped in this deployment - a plain status response instead of a
    # 500/traceback for every GET / (browsers, health checks, etc. hit this).
    return JSONResponse({"status": "ok", "service": "Face Search", "ui": "not installed"})


@app.post("/api/search")
async def search(
    file: UploadFile = File(...),
    location: str = Form(None),
    start_time: str = Form(None),
    end_time: str = Form(None),
):
    _ensure_model_loaded()

    start_dt = _parse_filter_datetime(start_time, "start_time")
    end_dt = _parse_filter_datetime(end_time, "end_time")

    raw = await file.read()
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    faces = _face_app.get(frame)

    logging.info(f"[FaceSearch] Loading embeddings from '{enroll.MONGO_COLLECTION_EMBEDDINGS_NEW}'...")
    known_faces_db = enroll.load_auto_enrolled_faces()
    names, matrix = enroll.build_matrix(known_faces_db)
    person_images = _load_person_images()

    results = []
    for face in faces:
        bbox = face.bbox.astype(int).tolist()
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        det_score = float(getattr(face, 'det_score', 1.0))

        entry = {"bbox": bbox, "det_score": det_score, "matches": [], "skip_reason": None}

        if min(w, h) < enroll.MIN_FACE_SIZE or det_score < SEARCH_MIN_DET_SCORE:
            # Report the detection instead of dropping it silently - without
            # this, a face that just misses the quality bar produces an
            # empty `results` entry with no explanation of why nothing shows.
            entry["skip_reason"] = (f"Face too small/low-confidence to search "
                                     f"(size={min(w, h)}px, det_score={det_score:.2f})")
            results.append(entry)
            continue

        v = np.asarray(face.embedding, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm == 0:
            entry["skip_reason"] = "Zero-norm embedding, could not search."
            results.append(entry)
            continue
        v = v / norm

        if matrix.shape[0]:
            # One matmul against every stored embedding (up to ~10/person)
            # instead of a per-person loop, same as mongo_enrollment.py.
            sims = matrix @ v
            best_per_person = {}
            for name, sim in zip(names, sims):
                sim = float(sim)
                if name not in best_per_person or sim > best_per_person[name]:
                    best_per_person[name] = sim

            for name, sim in sorted(best_per_person.items(), key=lambda kv: kv[1], reverse=True):
                if sim < SEARCH_MIN_SIMILARITY:
                    break  # sorted descending - nothing after this qualifies either
                images = person_images.get(name, [])
                if location or start_dt or end_dt:
                    images = [img for img in images if _image_matches_filter(img, location, start_dt, end_dt)]
                entry["matches"].append({
                    "person_name": name,
                    "similarity": sim,
                    "images": images,
                })
        else:
            entry["skip_reason"] = "No enrolled identities in the database yet."

        results.append(entry)

    return JSONResponse({"faces_found": len(faces), "results": results})


if __name__ == "__main__":
    _ensure_model_loaded()
    if not (os.path.isfile(SSL_CERTFILE) and os.path.isfile(SSL_KEYFILE)):
        raise SystemExit(f"[FaceSearch] Missing TLS cert/key: {SSL_CERTFILE}, {SSL_KEYFILE}")
    logging.info(f"[FaceSearch] Starting HTTPS on port {FACE_SEARCH_PORT}...")
    uvicorn.run(
        app, host="0.0.0.0", port=FACE_SEARCH_PORT, log_level="info",
        ssl_certfile=SSL_CERTFILE, ssl_keyfile=SSL_KEYFILE,
    )
