#!/usr/bin/env python3
"""
anpr-summary.py — full prose summary pipeline with number plates, timed from the
camera's own burnt-in clock.

Same output as merge-final-many.py, plus plates, minus the anchor arithmetic:

  merge-final-many.py   timestamp = video_anchor + offset_into_file
  anpr-summary.py       timestamp = the clock burnt into the frame, per segment

Reading the overlay on every segment is slower than arithmetic and it can fail
outright on a night frame - but it needs no anchor registry, it is correct on a
file that is a concatenation of clips, and it is the only option when a new
video has no entry in video_anchors.json. When OCR fails the segment is stamped
with the processing time and marked `anchor_confidence: "none"` rather than
being given a fabricated one.

The machinery is IMPORTED from merge-final-many.py rather than copied: frame
seeking, motion gating, the VLM description, the Azure upload, the contact
sheet, the embedding, the document shape and the video-level summary are all
the same code. A second copy would drift from the first within a week.

Stages
------
  1. per 60-second segment: sample frames -> read the overlay clock -> VLM
     description -> upload frames -> embed -> store          (resumable)
  2. attach OCR-confirmed plates from `tracks` to those segments
  3. video-level summary                     -> <SUMMARY_COLLECTION>
  4. per-camera plate roster                 -> <ANPR_SUMMARY_COLLECTION>

Plates are never read by the VLM. They come from offline_analytics.py, which
does dedicated plate-region detection, PaddleOCR -> EasyOCR -> tesseract, a
legal-format grammar check and multi-frame voting. Run it on the footage first,
or stage 2 finds nothing and says so.

    OCR reads characters. The VLM describes scenes. Neither does the other's job.

Usage
-----
    python anpr-summary.py --list
    python anpr-summary.py --videos cam02_02_Janpath
    python anpr-summary.py --videos cam02_02_Janpath --max-segments 5
    python anpr-summary.py --plates-only --dry-run        # stage 2+4 only
    python anpr-summary.py --videos NEW_SIGNAL_VIDEO --rewrite-description
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pymongo import ASCENDING, MongoClient, UpdateOne

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("anpr-summary")

HERE = os.path.dirname(os.path.abspath(__file__))

# ── configuration ─────────────────────────────────────────────────────────────

MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING", "")
MONGO_DATABASE = os.getenv("ARCIS_DATABASE", "arcis")
ACTIVITIES_COLLECTION = os.getenv("ACTIVITIES_COLLECTION", "activities1")
TRACKS_COLLECTION = os.getenv("MONGO_COLLECTION_TRACKS", "tracks")
ANPR_SUMMARY_COLLECTION = os.getenv("ANPR_SUMMARY_COLLECTION", "anpr_summaries")
VIDEO_DIR = os.getenv("VIDEO_DIR", os.path.join(HERE, "videos-hackathon"))

# Matches offline_analytics.PLATE_MIN_VOTES: two independent reads must agree
# before a plate is treated as real. Single reads stay in `plate_candidates`
# and are deliberately never used here.
MIN_VOTES = int(os.getenv("PLATE_MIN_VOTES", "2"))
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "llama3.1:latest")


# ── reuse the prose pipeline ──────────────────────────────────────────────────
# merge-final-many.py has a hyphen in its name, so it cannot be imported with a
# normal `import`. Load it by path and take the pieces wholesale.

_MFM = None


def mfm():
    """Load merge-final-many.py once, and cache it."""
    global _MFM
    if _MFM is None:
        path = os.path.join(HERE, "merge-final-many.py")
        if not os.path.exists(path):
            raise SystemExit(f"merge-final-many.py not found next to this script ({path})")
        logger.info("loading the prose pipeline from merge-final-many.py ...")
        spec = importlib.util.spec_from_file_location("mfm", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["mfm"] = module
        spec.loader.exec_module(module)
        _MFM = module
    return _MFM


# ── plate validation ──────────────────────────────────────────────────────────
# Reuse the real grammar. Two copies of a validator drift, and the whole point of
# the check is that "1234567890" must never become a plausible registration.
try:
    from offline_analytics import normalise_plate, plate_is_valid
except Exception as exc:  # pragma: no cover
    logger.warning(f"could not import plate grammar ({exc}); using a local copy")
    _PATTERNS = [
        re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$"),   # GJ01AB1234
        re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"),         # 21BH1234AA
    ]

    def normalise_plate(text: str) -> str:                      # type: ignore[misc]
        return re.sub(r"[^A-Z0-9]", "", (text or "").upper())

    def plate_is_valid(plate: str) -> bool:                     # type: ignore[misc]
        return 6 <= len(plate) <= 11 and any(p.match(plate) for p in _PATTERNS)


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_time(value: Any) -> Optional[datetime]:
    """Segment and track times are ISO strings in these collections."""
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    text = str(value).replace("Z", "").strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text[:26], fmt)
            except ValueError:
                continue
    return None


def overlaps(a_start: datetime, a_end: datetime,
             b_start: datetime, b_end: datetime) -> bool:
    return a_start <= b_end and b_start <= a_end


def connect():
    client = MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=20000)
    client.admin.command("ping")
    return client[MONGO_DATABASE]


def discover_videos(wanted: Optional[List[str]]) -> List[Dict[str, str]]:
    """Video configs in exactly the shape merge-final-many.py expects."""
    m = mfm()
    csv_path = os.path.join(HERE, "videos-hackathon-frame-info.csv")
    cameras: List[Dict[str, str]] = []
    if os.path.exists(csv_path):
        cameras, _ = m.load_video_config_from_csv(csv_path, VIDEO_DIR)
    if not cameras:
        from pathlib import Path
        for p in sorted(Path(VIDEO_DIR).glob("*.mp4")):
            cameras.append({"camera_id": p.stem, "rtmp_url": str(p),
                            "location": re.sub(r"[_-]+", " ", p.stem).strip()})
    if wanted:
        keep = []
        for cam in cameras:
            if any(w.lower() in cam["camera_id"].lower() for w in wanted):
                keep.append(cam)
        missing = [w for w in wanted
                   if not any(w.lower() in c["camera_id"].lower() for c in cameras)]
        for w in missing:
            logger.warning(f"no video matches '{w}'")
        cameras = keep
    return cameras


# ── stage 1: describe segments, timed from the camera frame ───────────────────

def process_video(cam: Dict[str, str], model_manager, collection, azure_client,
                  max_segments: Optional[int], resume: bool) -> Dict[str, Any]:
    """The merge-final-many segment loop, forced onto per-segment overlay OCR.

    anchor=None + anchor_confidence="non_contiguous" is the existing code path
    for a file whose clock cannot be derived arithmetically: finalize_segment
    then reads the burnt-in clock off frame1 and derives frame2's time from the
    sampling span. That is exactly the behaviour wanted here, for every video,
    so no new timestamp logic is introduced - only a different entry into the
    logic that already exists and is already tested.
    """
    m = mfm()
    camera_id = cam["camera_id"]
    video_path = cam["rtmp_url"]
    location = cam.get("location", "Unknown Location")
    source_video = os.path.basename(video_path)

    db = m.MongoVideoDatabase(model_manager=model_manager, collection=collection)
    width, height, fps = m.probe_video_info(video_path)
    if not width or not height:
        logger.error(f"[{camera_id}] could not probe video, skipping")
        return {"camera_id": camera_id, "segments": 0, "error": "probe failed"}

    duration = m.probe_duration(video_path)
    if not duration or duration <= 0:
        logger.error(f"[{camera_id}] could not determine duration, skipping")
        return {"camera_id": camera_id, "segments": 0, "error": "duration unknown"}

    total_segments = max(int(duration // m.SEGMENT_DURATION), 1)
    if max_segments:
        total_segments = min(total_segments, max_segments)

    already = m.existing_segment_ids(collection, camera_id) if resume else set()
    todo = [s for s in range(total_segments) if s not in already]

    logger.info(f"[{camera_id}] {width}x{height} @ {fps:.2f}fps | {duration/3600:.2f}h | "
                f"{total_segments} segment(s) | timestamps: per-segment overlay OCR")
    if already:
        logger.info(f"[{camera_id}] resuming — {len(already)} already stored, skipped")
    logger.info(f"[{camera_id}] {len(todo)} segment(s) to process")

    saved = failed = no_clock = 0
    wall_start = time.time()

    for done, segment_id in enumerate(todo, start=1):
        started = time.time()
        cumulative = segment_id * m.SEGMENT_DURATION
        offsets = m.segment_sample_offsets(cumulative)

        with ThreadPoolExecutor(max_workers=min(m.FRAME_EXTRACT_WORKERS, len(offsets))) as pool:
            frames = list(pool.map(
                lambda o: m.extract_frame_at(video_path, o, width, height), offsets))

        kept = [(o, f) for o, f in zip(offsets, frames) if f is not None]
        if not kept:
            failed += 1
            logger.warning(f"[{camera_id}] segment #{segment_id}: frame seek failed")
            continue

        vlm_frames = [f for _, f in kept]
        offset_1, frame1 = kept[0]
        offset_2, frame2 = kept[len(kept) // 2] if len(kept) > 1 else kept[0]

        stats: Dict[str, Any] = {}
        try:
            ok = m.finalize_segment(
                camera_id, location, source_video, segment_id, cumulative,
                frame1, frame2, azure_client, db,
                anchor=None,                          # ← no arithmetic
                anchor_confidence="non_contiguous",   # ← read the frame's clock
                frame1_offset=offset_1, frame2_offset=offset_2,
                drift_check=False, stats=stats,
                vlm_frames=vlm_frames, vlm_offsets=[o for o, _ in kept])
            if ok:
                saved += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            logger.error(f"[{camera_id}] segment #{segment_id} error: {exc}", exc_info=True)

        t = stats.get("timings", {})
        elapsed = time.time() - wall_start
        mean = elapsed / done
        logger.info(
            f"[{camera_id}] #{segment_id:<4} {done}/{len(todo)} "
            f"({100.0*done/max(len(todo),1):5.1f}%) | seg {time.time()-started:6.2f}s "
            f"| vlm {t.get('vlm',0.0):6.2f} azure {t.get('azure_upload',0.0):5.2f} "
            f"embed {t.get('embed_and_store',0.0):5.2f} | avg {mean:6.2f}s "
            f"| ETA {m._hms((len(todo)-done)*mean)}")

    # How many stored segments ended up without a readable clock: the number to
    # watch, because it is the cost of not using an anchor.
    try:
        no_clock = collection.count_documents(
            {"camera_id": camera_id, "anchor_confidence": "none"})
    except Exception:
        pass

    logger.info(f"[{camera_id}] finished — {saved} saved, {failed} failed, "
                f"{len(already)} skipped")
    if no_clock:
        logger.warning(f"[{camera_id}] {no_clock} stored segment(s) have NO readable "
                       f"overlay clock (anchor_confidence='none'); their times are "
                       f"processing times, not footage times")
    return {"camera_id": camera_id, "location": location, "source_video": source_video,
            "segments": saved, "failed": failed, "skipped": len(already),
            "without_clock": no_clock}


# ── stage 2: confirmed plates -> segments ─────────────────────────────────────

def load_confirmed_plates(db, camera: Optional[str], date: Optional[str],
                          min_votes: int) -> List[Dict[str, Any]]:
    """Tracks whose plate survived voting AND is a legal registration.

    Re-checking the grammar is deliberate belt-and-braces: this may run against
    tracks written by an older build, and a malformed plate reaching a summary
    is exactly the failure the system is designed to avoid.
    """
    query: Dict[str, Any] = {"plate": {"$nin": [None, ""]},
                             "plate_votes": {"$gte": min_votes}}
    if camera:
        query["camera_id"] = camera
    if date:
        query["first_seen"] = {"$regex": f"^{re.escape(date)}"}

    fields = {"camera_id": 1, "location": 1, "plate": 1, "plate_confidence": 1,
              "plate_votes": 1, "vehicle_type": 1, "coco_class": 1, "colour": 1,
              "make_model": 1,
              "first_seen": 1, "last_seen": 1, "track_key": 1, "_id": 0}

    rows, rejected = [], 0
    for t in db[TRACKS_COLLECTION].find(query, fields):
        plate = normalise_plate(t.get("plate"))
        if not plate_is_valid(plate):
            rejected += 1
            continue
        start, end = parse_time(t.get("first_seen")), parse_time(t.get("last_seen"))
        if not start:
            continue
        t["plate"] = plate
        t["_start"], t["_end"] = start, end or start
        rows.append(t)

    if rejected:
        logger.warning(f"{rejected} stored plate(s) failed the grammar check, skipped")
    return rows


def map_plates_to_segments(db, plates: List[Dict[str, Any]], collection: str
                           ) -> Tuple[Dict[Any, List[str]], int]:
    """segment _id -> sorted plates whose sighting falls inside its window."""
    by_camera: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in plates:
        by_camera[p["camera_id"]].append(p)

    mapping: Dict[Any, List[str]] = {}
    scanned = 0

    for camera, cam_plates in by_camera.items():
        days = sorted({p["_start"].strftime("%Y-%m-%d") for p in cam_plates} |
                      {p["_end"].strftime("%Y-%m-%d") for p in cam_plates})
        query = {"camera_id": camera,
                 "$or": [{"start_time": {"$regex": f"^{re.escape(d)}"}} for d in days]}

        for seg in db[collection].find(query, {"start_time": 1, "end_time": 1, "_id": 1}):
            scanned += 1
            s_start = parse_time(seg.get("start_time"))
            s_end = parse_time(seg.get("end_time")) or s_start
            if not s_start:
                continue
            hits = sorted({p["plate"] for p in cam_plates
                           if overlaps(p["_start"], p["_end"], s_start, s_end)})
            if hits:
                mapping[seg["_id"]] = hits

    return mapping, scanned


def enrich_segments(db, mapping: Dict[Any, List[str]], collection: str,
                    rewrite_description: bool, dry_run: bool) -> int:
    """Add a `plates` array, and optionally a PLATES: line in the description.

    The line is APPENDED rather than substituted into "plate not legible". A
    segment can hold several vehicles and nothing in the stored text says which
    plate belongs to which car; writing one into a specific vehicle's clause
    would invent a fact the OCR never established.
    """
    if not mapping:
        return 0
    if dry_run:
        for sid, plates in list(mapping.items())[:10]:
            logger.info(f"  [dry-run] segment {sid} -> {plates}")
        if len(mapping) > 10:
            logger.info(f"  [dry-run] ... and {len(mapping)-10} more")
        return 0

    ops = []
    for sid, plates in mapping.items():
        update: Dict[str, Any] = {"$set": {"plates": plates,
                                           "plates_source": f"{TRACKS_COLLECTION} (voted)"}}
        if rewrite_description:
            doc = db[collection].find_one({"_id": sid}, {"description": 1})
            desc = (doc or {}).get("description") or ""
            if "PLATES:" not in desc:                    # idempotent across re-runs
                update["$set"]["description"] = desc.rstrip() + f" PLATES: {', '.join(plates)}"
        ops.append(UpdateOne({"_id": sid}, update))

    written = 0
    for i in range(0, len(ops), 500):
        written += db[collection].bulk_write(ops[i:i+500], ordered=False).modified_count
    return written


# ── stage 4: per-camera roster ────────────────────────────────────────────────

def _group(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return out


def build_rosters(plates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    per_camera: Dict[str, Dict[str, Any]] = {}
    for camera, rows in _group(plates, "camera_id").items():
        seen: Dict[str, Dict[str, Any]] = {}
        # Make/model is read by the VLM once per TRACK, so one registration seen
        # three times carries three independent opinions. Vote rather than take
        # the first: a single "Innova" from a blurred rear crop is exactly the
        # kind of confident detail an operator would repeat as fact.
        make_votes: Dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            if r.get("make_model"):
                make_votes[r["plate"]][str(r["make_model"]).strip()] += 1
            entry = seen.get(r["plate"])
            if entry is None:
                seen[r["plate"]] = {
                    "plate": r["plate"],
                    "vehicle_type": r.get("vehicle_type") or r.get("coco_class"),
                    "colour": r.get("colour"),
                    "first_seen": r["_start"].isoformat(),
                    "last_seen": r["_end"].isoformat(),
                    "sightings": 1,
                    "max_votes": r.get("plate_votes", 0),
                    "max_confidence": round(float(r.get("plate_confidence") or 0), 3),
                }
            else:
                entry["sightings"] += 1
                entry["first_seen"] = min(entry["first_seen"], r["_start"].isoformat())
                entry["last_seen"] = max(entry["last_seen"], r["_end"].isoformat())
                entry["max_votes"] = max(entry["max_votes"], r.get("plate_votes", 0))
                entry["max_confidence"] = max(entry["max_confidence"],
                                              round(float(r.get("plate_confidence") or 0), 3))
        for plate, entry in seen.items():
            votes = make_votes.get(plate)
            if votes:
                model, agreed = votes.most_common(1)[0]
                entry["make_model"] = model
                entry["make_model_votes"] = agreed
            else:
                entry["make_model"] = None
                entry["make_model_votes"] = 0
        roster = sorted(seen.values(), key=lambda e: e["first_seen"])
        per_camera[camera] = {
            "camera_id": camera,
            "location": rows[0].get("location") or camera,
            "unique_plates": len(roster),
            "total_sightings": sum(e["sightings"] for e in roster),
            "window_start": min(e["first_seen"] for e in roster),
            "window_end": max(e["last_seen"] for e in roster),
            "plates": roster,
        }
    return per_camera


def roster_text(entry: Dict[str, Any], limit: int = 40) -> str:
    def describe(e: Dict[str, Any]) -> str:
        """colour + make/model + type, e.g. 'white Toyota Innova car'.

        An unrecognised model prints as 'model unidentified' rather than being
        left blank, so the model is told the difference between "nobody could
        read it" and "nobody looked" - blank invites it to fill the gap.
        """
        make = e.get("make_model")
        # Worded as prose, not as a tag: the model copies this string through
        # verbatim, so "probable Tata Nexon (unconfirmed)" reads correctly in a
        # sentence where "[UNCONFIRMED]" does not.
        badge = (f"probable {make} (unconfirmed, one view only)"
                 if make and e.get("make_model_votes", 0) < 2 else make)
        return " ".join(x for x in (e.get("colour") or "?",
                                    badge or "model unidentified",
                                    e.get("vehicle_type") or "vehicle") if x)

    lines = [f"{e['plate']}  {describe(e)}"
             f"  first {e['first_seen'][11:19]}  last {e['last_seen'][11:19]}"
             f"  sightings {e['sightings']}" for e in entry["plates"][:limit]]
    if entry["unique_plates"] > limit:
        lines.append(f"... and {entry['unique_plates'] - limit} more")
    return "\n".join(lines)


async def write_narrative(entry: Dict[str, Any]) -> Optional[str]:
    """A short written summary of the plate activity, via llm_backend.

    Given the roster only. It is never asked to read a plate, so it cannot
    invent one.
    """
    try:
        import llm_backend
    except Exception as exc:
        logger.warning(f"narrative skipped, llm_backend unavailable: {exc}")
        return None

    system = ("You are a police ANPR analyst. Summarise registration-number activity "
              "for one camera in 3-5 sentences.\nRULES:\n"
              "- Use ONLY the registrations listed. Never invent or complete a plate.\n"
              "- Quote each registration exactly as written.\n"
              "- Name the make and model where the roster gives one (e.g. 'Toyota Innova', "
              "'Tata Nexon'), written exactly as listed.\n"
              "- Where the roster says 'model unidentified', say the model was not "
              "identified. NEVER guess a make or model from the colour or body type.\n"
              "- A model the roster calls 'probable' or 'unconfirmed' rests on a single "
              "view. You MUST keep that hedge in the sentence that names it. Do not "
              "promote it to established fact.\n"
              "- Mention the busiest period and any vehicle seen more than once.\n"
              "- State plainly that these are automatic reads, not verified against a registry.")
    user = (f"Camera {entry['camera_id']} at {entry['location']}.\n"
            f"Window {entry['window_start']} to {entry['window_end']}.\n"
            f"{entry['unique_plates']} unique registration(s), "
            f"{entry['total_sightings']} sighting(s).\n\n{roster_text(entry)}\n\nWrite the summary.")
    try:
        return (await llm_backend.chat(system, user, ollama_model=SUMMARY_MODEL,
                                       max_tokens=500, temperature=0.3,
                                       label="anpr.narrative")).strip()
    except Exception as exc:
        logger.warning(f"narrative generation failed: {exc}")
        return None


def store_rosters(db, rosters: Dict[str, Dict[str, Any]], dry_run: bool) -> int:
    if dry_run:
        return 0
    db[ANPR_SUMMARY_COLLECTION].create_index([("camera_id", ASCENDING)], unique=True)
    db[ANPR_SUMMARY_COLLECTION].create_index([("plates.plate", ASCENDING)])
    for camera, entry in rosters.items():
        doc = dict(entry)
        doc["generated_at"] = datetime.now().isoformat()
        doc["min_votes"] = MIN_VOTES
        doc["source"] = f"{TRACKS_COLLECTION} (voted + grammar-checked)"
        db[ANPR_SUMMARY_COLLECTION].update_one({"camera_id": camera},
                                               {"$set": doc}, upsert=True)
    return len(rosters)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prose summaries timed from the camera's burnt-in clock, "
                    "with OCR-confirmed number plates attached.")
    ap.add_argument("--videos", nargs="*", help="camera ids / substrings; default all")
    ap.add_argument("--list", action="store_true", help="show what would be processed")
    ap.add_argument("--max-segments", type=int, help="cap segments per video (testing)")
    ap.add_argument("--no-resume", action="store_true", help="re-describe stored segments")
    ap.add_argument("--plates-only", action="store_true",
                    help="skip the video pass; only attach plates and build rosters")
    ap.add_argument("--no-plates", action="store_true", help="skip the plate stages")
    ap.add_argument("--rewrite-description", action="store_true",
                    help="append 'PLATES: ...' to the description so keyword search finds it")
    ap.add_argument("--narrative", action="store_true", help="LLM summary per camera roster")
    ap.add_argument("--min-votes", type=int, default=MIN_VOTES)
    ap.add_argument("--date", help="limit the plate join to one date, YYYY-MM-DD")
    ap.add_argument("--collection", default=ACTIVITIES_COLLECTION)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    logger.info("=" * 78)
    logger.info("ANPR SUMMARY — camera-clock timestamps, prose descriptions, plates")
    logger.info("=" * 78)
    logger.info(f"prose collection   {args.collection}")
    logger.info(f"tracks collection  {TRACKS_COLLECTION}")
    logger.info(f"roster collection  {ANPR_SUMMARY_COLLECTION}")
    logger.info(f"timestamps         burnt-in overlay clock, read per segment")
    logger.info(f"mode               {'DRY RUN' if args.dry_run else 'WRITE'}")

    try:
        raw_db = connect()
    except Exception as exc:
        logger.error(f"MongoDB unreachable: {exc}")
        return 1

    cameras = discover_videos(args.videos) if not args.plates_only else []
    if args.list:
        for cam in cameras:
            logger.info(f"  {cam['camera_id']:34} {cam['location']}")
        logger.info(f"{len(cameras)} video(s)")
        return 0

    results = []
    # ── stage 1 ───────────────────────────────────────────────────────────────
    if not args.plates_only:
        if not cameras:
            logger.error("no videos matched; nothing to describe")
            return 1
        if args.dry_run:
            logger.info("[dry-run] would describe:")
            for cam in cameras:
                logger.info(f"  {cam['camera_id']} ({cam['location']})")
        else:
            m = mfm()
            if not m._preflight_vlm():
                logger.error("the VLM is not reachable; start Ollama and pull the model")
                return 1
            model_manager = m.UniversalModelManager()
            # initialize_mongodb() returns (client, collection), not a collection.
            _mongo_client, collection = m.initialize_mongodb()
            azure_client = m.initialize_azure_client()
            if collection is None or azure_client is None:
                logger.error("MongoDB or Azure could not be initialised")
                return 1
            for cam in cameras:
                results.append(process_video(cam, model_manager, collection, azure_client,
                                             args.max_segments, not args.no_resume))

    # ── stages 2 & 4 ──────────────────────────────────────────────────────────
    if not args.no_plates:
        camera_filter = None
        if args.videos and len(args.videos) == 1 and not args.plates_only and results:
            camera_filter = results[0]["camera_id"]

        plates = load_confirmed_plates(raw_db, camera_filter, args.date, args.min_votes)
        if not plates:
            total = raw_db[TRACKS_COLLECTION].count_documents({})
            logger.warning("")
            logger.warning(f"No confirmed plates in `{TRACKS_COLLECTION}` ({total:,} track(s)).")
            logger.warning("No roster written — an empty one would be worse than none.")
            logger.warning("The plates come from the detection tier, not the VLM. Run:")
            logger.warning("    python offline_analytics.py --slice-seconds 3600 --resume")
        else:
            logger.info(f"confirmed plate sightings: {len(plates)}")
            rosters = build_rosters(plates)
            for camera, entry in rosters.items():
                logger.info(f"  {camera}: {entry['unique_plates']} unique plate(s), "
                            f"{entry['total_sightings']} sighting(s)")
            if args.narrative:
                for entry in rosters.values():
                    text = asyncio.run(write_narrative(entry))
                    if text:
                        entry["narrative"] = text
            mapping, scanned = map_plates_to_segments(raw_db, plates, args.collection)
            logger.info(f"scanned {scanned} segment(s); {len(mapping)} carry a confirmed plate")
            modified = enrich_segments(raw_db, mapping, args.collection,
                                       args.rewrite_description, args.dry_run)
            if not args.dry_run:
                logger.info(f"tagged {modified} segment(s) with `plates`"
                            + (" and a PLATES: line" if args.rewrite_description else ""))
                logger.info(f"wrote {store_rosters(raw_db, rosters, False)} roster document(s)")

    # ── stage 3: video-level summaries ────────────────────────────────────────
    if results and not args.dry_run:
        m = mfm()
        model_manager = m.UniversalModelManager()
        _mongo_client, collection = m.initialize_mongodb()
        db = m.MongoVideoDatabase(model_manager=model_manager, collection=collection)
        for r in results:
            if r.get("segments") or r.get("skipped"):
                logger.info(f"[{r['camera_id']}] building the video-level summary ...")
                m.summarise_video(db, r["camera_id"], r.get("location", ""),
                                  r.get("source_video", ""))

    logger.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
