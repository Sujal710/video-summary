"""
MCP Server for Video Segment Database
Three tools:
  1. get_last_n_hours_summary  — time-range / date / full-DB query
  2. list_available_cameras    — list camera IDs + stats
  3. search_segments_by_activity — hybrid BM25 + semantic search, fused with
     Reciprocal Rank Fusion (RRF)

Embedding model : nomic-embed-text:v1.5  via Ollama (localhost:11434)
FAISS index     : <camera_dir>/text_index.faiss
FAISS metadata  : <camera_dir>/text_metadata.pkl
Segments        : <camera_dir>/segments.pkl
"""

import os
import re
import sys
import json
import math
import logging
import pickle
import requests
from collections import Counter
from datetime import datetime, timedelta, timezone, date as datetime_date, time as datetime_time
from typing import Dict, Any, List, Optional

import numpy as np
import faiss
from pymongo import MongoClient

from fastmcp import FastMCP, Context
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DB_PATH = os.getenv(
    "VIDEO_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_segments_db"),
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")

# How many FAISS neighbours to retrieve before applying date/time filters
FAISS_TOP_K = int(os.getenv("FAISS_TOP_K", "50"))

# Cosine-similarity threshold — results below this score are discarded
FAISS_SCORE_THRESHOLD = float(os.getenv("FAISS_SCORE_THRESHOLD", "0.5"))

# BM25 term-frequency saturation (k1) / length-normalisation (b) parameters
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

# Reciprocal Rank Fusion constant used to combine BM25 + semantic rankings
RRF_K = int(os.getenv("RRF_K", "60"))

# ── MongoDB Configuration ─────────────────────────────────────────────────────
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING", "")
MONGO_DATABASE = 'arcis'
MONGO_COLLECTION_ACTIVITIES = 'activities-hackathon-1'

# ── FastMCP server ─────────────────────────────────────────────────────────────

mcp = FastMCP(name="video-segment-retrieval-agent")

# ── BM25 (lexical) search ────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25Okapi:
    """Minimal BM25 (Okapi) ranker over a fixed, already-tokenized corpus."""

    def __init__(self, corpus: List[List[str]], k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_freqs: List[Counter] = []
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_lens) / self.corpus_size) if self.corpus_size else 0.0

        df: Counter = Counter()
        for doc in corpus:
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)
            for term in freqs:
                df[term] += 1

        self.idf: Dict[str, float] = {
            term: math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        if not self.corpus_size:
            return scores
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


# ── VideoSegment ───────────────────────────────────────────────────────────────

class VideoSegment:
    """One video clip with metadata."""

    def __init__(
        self,
        segment_id: int,
        start_time: datetime,
        end_time: datetime,
        frame_urls: List[str],
        description: str,
        cumulative_minutes: float,
        camera_id: str = None,
        location: str = "Unknown Location",
    ):
        self.segment_id = segment_id
        self.start_time = start_time
        self.end_time = end_time
        self.frame_urls = frame_urls
        self.description = description
        self.cumulative_minutes = cumulative_minutes
        self.camera_id = camera_id or "default"
        self.location = location or "Unknown Location"


# DUMMY CLASS FOR PICKLE LOADING
class DummyVideoSegment:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "VideoSegment" or name == "DummyVideoSegment":
            return VideoSegment
        return super().find_class(module, name)


# ── Per-camera MongoDB index ──────────────────────────────────────────────────

class CameraIndex:
    """
    Holds the segments and embeddings for one camera, fetched from MongoDB.
    """

    def __init__(self, camera_id: str, collection):
        self.camera_id = camera_id
        self.collection = collection
        self.segments: List[VideoSegment] = []
        self.embeddings: List[np.ndarray] = []
        self._load()

    def _load(self):
        try:
            # Fetch all documents for this camera from MongoDB
            docs = list(self.collection.find({"camera_id": self.camera_id}).sort("segment_id", 1))
            self.segments = []
            self.embeddings = []
            
            for doc in docs:
                # Handle start_time/end_time which might be stored as datetime or strings
                start_time = doc.get("start_time")
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                
                end_time = doc.get("end_time")
                if isinstance(end_time, str):
                    end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

                seg = VideoSegment(
                    segment_id=doc.get("segment_id"),
                    start_time=start_time,
                    end_time=end_time,
                    frame_urls=doc.get("frame_urls", []),
                    description=doc.get("description", ""),
                    cumulative_minutes=doc.get("cumulative_minutes", 0),
                    camera_id=self.camera_id,
                    location=doc.get("location", "Unknown Location"),
                )
                self.segments.append(seg)
                
                emb = doc.get("embedding")
                if emb:
                    # Ensure it's a numpy array for similarity calculations
                    arr = np.array(emb, dtype=np.float32)
                    # Normalize for cosine similarity
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                    self.embeddings.append(arr)
                else:
                    arr = np.zeros(768, dtype=np.float32)
                    self.embeddings.append(arr)

                # Also carry the embedding directly on the segment so the
                # hybrid (BM25 + semantic) search can rank a filtered
                # candidate list without re-indexing per camera.
                seg.embedding = arr
            
            logger.info(
                f"[{self.camera_id}] Loaded {len(self.segments)} segments from MongoDB"
            )
        except Exception as e:
            logger.error(f"[{self.camera_id}] Failed to load from MongoDB: {e}")
            self.segments = []
            self.embeddings = []

    @property
    def ready(self) -> bool:
        return len(self.segments) > 0


# ── VideoDatabase ──────────────────────────────────────────────────────────────

class VideoDatabase:
    """Manages segments and retrieval via MongoDB."""

    def __init__(self):
        try:
            self.client = MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=30000)
            self.db_mongo = self.client[MONGO_DATABASE]
            self.collection = self.db_mongo[MONGO_COLLECTION_ACTIVITIES]
            logger.info("✓ MongoDB connection successful in VideoDatabase")
        except Exception as e:
            logger.error(f"MongoDB connection error: {e}")
            raise

        self.segments_by_camera: Dict[str, List[VideoSegment]] = {}
        self.indices_by_camera: Dict[str, CameraIndex] = {}
        self.load_all_cameras()

    # ── loading ───────────────────────────────────────────────────────────────

    def load_all_cameras(self):
        """(Re)load every camera from MongoDB."""
        self.segments_by_camera.clear()
        self.indices_by_camera.clear()

        try:
            # Find all unique camera_ids in the collection
            camera_ids = self.collection.distinct("camera_id")
            
            for cam_id in camera_ids:
                cidx = CameraIndex(cam_id, self.collection)
                if cidx.ready:
                    self.indices_by_camera[cam_id] = cidx
                    self.segments_by_camera[cam_id] = cidx.segments
            
            logger.info(f"Loaded {len(self.segments_by_camera)} camera(s) from MongoDB")
        except Exception as e:
            logger.error(f"Error loading cameras from MongoDB: {e}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def get_all_cameras(self) -> List[str]:
        return list(self.segments_by_camera.keys())

    def _collect_segments(self, camera_id) -> List[VideoSegment]:
        """Return a flat list of segments for the given camera filter."""
        camera_id = self.normalize_camera_id(camera_id)
        if camera_id is None:
            segs = []
            for v in self.segments_by_camera.values():
                segs.extend(v)
            return segs
        if isinstance(camera_id, list):
            segs = []
            for cam in camera_id:
                segs.extend(self.segments_by_camera.get(cam, []))
            return segs
        return list(self.segments_by_camera.get(camera_id, []))

    @staticmethod
    def normalize_camera_id(camera_id: Any) -> Optional[Any]:
        if camera_id is None:
            return None
        if isinstance(camera_id, str):
            if ',' in camera_id:
                return [c.strip() for c in camera_id.split(',') if c.strip()]
            camera_id = camera_id.strip()
            return camera_id if camera_id else None
        return camera_id

    @staticmethod
    def _make_aware(dt: datetime) -> datetime:
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    @staticmethod
    def matches_location(seg_location: Optional[str], location: Any) -> bool:
        """Case-insensitive, punctuation-insensitive substring match, so
        e.g. "ongc" matches "O.N.G.C. Office". `location` may be a single
        string or a list of candidate strings (any-match)."""
        def normalize(s: str) -> str:
            return re.sub(r'[^a-z0-9]', '', s.lower())

        seg_norm = normalize(seg_location or "")
        candidates = location if isinstance(location, list) else [location]
        return any(normalize(str(loc)) in seg_norm for loc in candidates if str(loc).strip())

    def get_all_locations(self) -> List[str]:
        """Distinct, non-empty locations across every loaded segment."""
        locations = {
            seg.location for segs in self.segments_by_camera.values() for seg in segs
            if seg.location and seg.location != "Unknown Location"
        }
        return sorted(locations)

    def get_camera_location(self, camera_id: str) -> str:
        """The most common location tagged on a camera's segments."""
        segs = self.segments_by_camera.get(camera_id, [])
        if not segs:
            return "Unknown Location"
        counts: Dict[str, int] = {}
        for seg in segs:
            counts[seg.location] = counts.get(seg.location, 0) + 1
        return max(counts, key=counts.get)

    # ── absolute time-range query ─────────────────────────────────────────────

    def get_segments_by_absolute_time_range(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
        camera_id=None,
        location=None,
    ) -> List[VideoSegment]:
        segments = self._collect_segments(camera_id)
        if not segments:
            return []

        if start_datetime.tzinfo is None:
            start_datetime = start_datetime.replace(tzinfo=timezone.utc)
        if end_datetime.tzinfo is None:
            end_datetime = end_datetime.replace(tzinfo=timezone.utc)

        filtered = [
            seg for seg in segments
            if seg.start_time and seg.end_time
            and self._make_aware(seg.start_time) <= end_datetime
            and self._make_aware(seg.end_time) >= start_datetime
            and (location is None or self.matches_location(seg.location, location))
        ]
        filtered.sort(key=lambda x: x.start_time)
        return filtered

    # ── hybrid search: BM25 (lexical) + semantic, fused with RRF ───────────────

    def hybrid_search_segments(
        self,
        keywords: List[str],
        camera_id=None,
        location=None,
        filter_date: Optional[datetime_date] = None,
        filter_start_dt: Optional[datetime] = None,
        filter_end_dt: Optional[datetime] = None,
        filter_start_time: Optional[datetime_time] = None,
        filter_end_time: Optional[datetime_time] = None,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = FAISS_TOP_K,
        semantic_score_threshold: float = FAISS_SCORE_THRESHOLD,
        rrf_k: int = RRF_K,
    ) -> Dict[str, Any]:
        """
        Rank segments with BM25 (over `description`) and semantic cosine
        similarity, then fuse both rankings with Reciprocal Rank Fusion.
        There is no substring keyword match — BM25 handles lexical relevance.

        `filter_start_time`/`filter_end_time` are a time-of-day-only filter
        (no date attached), for a query that gives a time window but no date
        at all — e.g. "red truck between 6 and 7" with no day mentioned means
        "that window, on any date", not "ignore the time window". They are
        independent of `filter_date`/`filter_start_dt`/`filter_end_dt`, which
        anchor to a specific date.
        """
        # Collect all segments, then filter down to the date/time/camera/location
        # candidate set that both rankers operate on.
        segments = self._collect_segments(camera_id)
        if not segments:
            return {"results": [], "scored_results": [], "total_checked": 0, "status": "no_segments_in_db"}

        if camera_id is None:
            allowed_cameras = None
        elif isinstance(camera_id, list):
            allowed_cameras = set(camera_id)
        else:
            allowed_cameras = {camera_id}

        candidates: List[VideoSegment] = []
        total = date_filtered = time_filtered = cam_filtered = loc_filtered = 0

        for seg in segments:
            total += 1
            if not seg.start_time:
                time_filtered += 1
                continue
            seg_start = self._make_aware(seg.start_time).astimezone(timezone.utc)
            seg_date = seg_start.date()

            if filter_date and not (filter_start_dt or filter_end_dt):
                if seg_date != filter_date:
                    date_filtered += 1
                    continue

            if filter_start_dt and seg_start < filter_start_dt:
                time_filtered += 1
                continue
            if filter_end_dt and seg_start > filter_end_dt:
                time_filtered += 1
                continue

            if filter_start_time or filter_end_time:
                seg_tod = seg_start.time()
                if filter_start_time and filter_end_time and filter_start_time > filter_end_time:
                    # Window crosses midnight (e.g. 22:00 → 02:00).
                    in_window = seg_tod >= filter_start_time or seg_tod <= filter_end_time
                else:
                    in_window = (
                        (not filter_start_time or seg_tod >= filter_start_time)
                        and (not filter_end_time or seg_tod <= filter_end_time)
                    )
                if not in_window:
                    time_filtered += 1
                    continue

            if allowed_cameras is not None:
                if getattr(seg, 'camera_id', None) not in allowed_cameras:
                    cam_filtered += 1
                    continue

            if location is not None and not self.matches_location(seg.location, location):
                loc_filtered += 1
                continue

            candidates.append(seg)

        base_stats = {
            "total_checked":     total,
            "date_filtered":     date_filtered,
            "time_filtered":     time_filtered,
            "cam_filtered":      cam_filtered,
            "location_filtered": loc_filtered,
        }

        if not candidates:
            return {"results": [], "scored_results": [], "status": "success", **base_stats}

        # ── BM25 ranking ────────────────────────────────────────────────────
        query_tokens: List[str] = []
        for kw in keywords:
            query_tokens.extend(_tokenize(kw))

        bm25_rank_of: Dict[int, int] = {}
        bm25_score_of: Dict[int, float] = {}
        if query_tokens:
            tokenized_docs = [_tokenize(seg.description) for seg in candidates]
            bm25_scores = BM25Okapi(tokenized_docs).get_scores(query_tokens)
            ranked = sorted(
                (i for i, s in enumerate(bm25_scores) if s > 0),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )
            for rank, i in enumerate(ranked, start=1):
                bm25_rank_of[i] = rank
                bm25_score_of[i] = float(bm25_scores[i])

        # ── semantic ranking ───────────────────────────────────────────────
        semantic_rank_of: Dict[int, int] = {}
        semantic_score_of: Dict[int, float] = {}
        if query_vector is not None:
            qv = query_vector.astype(np.float32).flatten()
            norm = np.linalg.norm(qv)
            if norm > 0:
                qv = qv / norm

            sims = []
            for i, seg in enumerate(candidates):
                emb = getattr(seg, 'embedding', None)
                if emb is None:
                    continue
                sims.append((i, float(np.dot(qv, emb))))
            sims.sort(key=lambda x: x[1], reverse=True)

            rank = 0
            for i, score in sims:
                if score < semantic_score_threshold:
                    continue
                rank += 1
                semantic_rank_of[i] = rank
                semantic_score_of[i] = score
                if rank >= top_k:
                    break

        # ── Reciprocal Rank Fusion ──────────────────────────────────────────
        fused_indices = set(bm25_rank_of) | set(semantic_rank_of)
        if not fused_indices:
            return {"results": [], "scored_results": [], "status": "success", **base_stats}

        rrf_scores: Dict[int, float] = {}
        for i in fused_indices:
            score = 0.0
            if i in bm25_rank_of:
                score += 1.0 / (rrf_k + bm25_rank_of[i])
            if i in semantic_rank_of:
                score += 1.0 / (rrf_k + semantic_rank_of[i])
            rrf_scores[i] = score

        ranked_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        results = [candidates[i] for i in ranked_indices]
        scored_results = [
            {
                "segment_id":     candidates[i].segment_id,
                "camera_id":      candidates[i].camera_id,
                "rrf_score":      round(rrf_scores[i], 6),
                "bm25_rank":      bm25_rank_of.get(i),
                "bm25_score":     round(bm25_score_of[i], 4) if i in bm25_score_of else None,
                "semantic_rank":  semantic_rank_of.get(i),
                "semantic_score": round(semantic_score_of[i], 4) if i in semantic_score_of else None,
            }
            for i in ranked_indices
        ]

        return {
            "results": results,
            "scored_results": scored_results,
            "status": "success",
            **base_stats,
        }

    # ── statistics ─────────────────────────────────────────────────────────────

    def get_statistics(self, camera_id=None) -> Dict:
        segments = self._collect_segments(camera_id)
        if not segments:
            return {
                "total_segments": 0,
                "total_frames": 0,
                "total_duration_minutes": 0,
                "cameras": self.get_all_cameras(),
            }
        # Some stored segments have start_time/end_time as None (incomplete
        # rows) — excluded from the duration and time-range calc below, but
        # still counted in total_segments/total_frames.
        timed_segments = [s for s in segments if s.start_time and s.end_time]
        total_duration = sum((s.end_time - s.start_time).total_seconds() for s in timed_segments)
        total_frames = sum(len(s.frame_urls) for s in segments)
        sorted_segs = sorted(timed_segments, key=lambda x: x.start_time)
        return {
            "total_segments": len(segments),
            "total_frames": total_frames,
            "total_duration_minutes": total_duration / 60,
            "first_segment_time": sorted_segs[0].start_time.isoformat() if sorted_segs else None,
            "last_segment_time": sorted_segs[-1].end_time.isoformat() if sorted_segs else None,
            "cameras": (
                self.get_all_cameras()
                if camera_id is None
                else (camera_id if isinstance(camera_id, list) else [camera_id])
            ),
        }


# ── Global DB instance ─────────────────────────────────────────────────────────

db: Optional[VideoDatabase] = None


def initialize_database():
    global db
    logger.info("=" * 60)
    logger.info("INITIALIZING VIDEO DATABASE (MONGODB)")
    logger.info("=" * 60)
    db = VideoDatabase()
    cameras = db.get_all_cameras()
    logger.info(f"Cameras found: {cameras}")

    # Report status per camera
    for cam in cameras:
        cidx = db.indices_by_camera.get(cam)
        if cidx and cidx.ready:
            logger.info(f"  [{cam}] MongoDB Ready — {len(cidx.segments)} segments")
        else:
            logger.warning(f"  [{cam}] Not available in MongoDB")
    logger.info("=" * 60)


# ── Ollama embedding ───────────────────────────────────────────────────────────

def get_query_embedding(text: str) -> Optional[np.ndarray]:
    """
    Call Ollama's /api/embeddings endpoint and return a numpy float32 vector.
    Returns None on failure so the caller can degrade gracefully.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=30,
        )
        print(f"Ollama embedding response status: {resp.status_code}")
        resp.raise_for_status()
        embedding = resp.json().get("embedding")
        if not embedding:
            logger.error("Ollama returned empty embedding")
            return None
        return np.array(embedding, dtype=np.float32)
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot reach Ollama at {OLLAMA_BASE_URL} — is it running?")
        return None
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None


# ── Serialisation helpers ──────────────────────────────────────────────────────

def serialize_segment(seg: VideoSegment, score: Optional[float] = None) -> Dict[str, Any]:
    d = {
        "segment_id": seg.segment_id,
        "camera_id": seg.camera_id,
        "location": seg.location,
        "start_time": seg.start_time.isoformat(),
        "end_time": seg.end_time.isoformat(),
        "start_time_formatted": seg.start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time_formatted": seg.end_time.strftime("%H:%M:%S"),
        "duration_minutes": (seg.end_time - seg.start_time).total_seconds() / 60,
        "cumulative_minutes": seg.cumulative_minutes,
        "frame_urls": seg.frame_urls,
        "description": seg.description,
    }
    if score is not None:
        d["similarity_score"] = round(score, 4)
    return d


def parse_time_string(time_str: str, date_str: Optional[str] = None) -> Optional[datetime]:
    """
    Parse a time string to a UTC-aware datetime.
    Treats all input times as UTC (matching MongoDB/segment storage convention).
    """
    try:
        if "T" in time_str or (len(time_str) > 10 and "-" in time_str and ":" in time_str):
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        if date_str:
            try:
                base_date = datetime.fromisoformat(date_str.split("T")[0])
            except ValueError:
                base_date = datetime.now(timezone.utc)
            base_date = base_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        else:
            base_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        t = time_str.strip().lower()

        if t == "today":
            return base_date
        if t == "yesterday":
            return base_date - timedelta(days=1)

        is_pm = "pm" in t
        is_am = "am" in t
        t = t.replace("am", "").replace("pm", "").strip()

        if ":" in t:
            parts = t.split(":")
            hour, minute = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
            second = int(parts[2]) if len(parts) > 2 else 0
        else:
            hour, minute, second = int(t), 0, 0

        if is_pm and hour != 12:
            hour += 12
        elif is_am and hour == 12:
            hour = 0

        return base_date.replace(hour=hour, minute=minute, second=second)

    except Exception as e:
        logger.error(f"parse_time_string('{time_str}', date='{date_str}'): {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — get_last_n_hours_summary
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool
async def get_last_n_hours_summary(
    ctx: Context,
    camera_id: Optional[Any] = None,
    location: Optional[Any] = None,
    k: int = 999,
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get video segments within a time range.

    Pass combinations of date / start_date / start_time / end_time / end_date to
    narrow results.  If nothing is passed the full database is returned.

    Args:
        camera_id  : Camera ID string, list of IDs, or omit for all cameras.
        location   : Location name (or list of names), matched case-insensitively
                     as a substring against each segment's stored location
                     (e.g. "ongc" matches "O.N.G.C. Office"). Omit for all locations.
        k          : Maximum number of segments to return (default 999).
        date       : YYYY-MM-DD — start date of the range (alias: start_date).
        start_date : YYYY-MM-DD — explicit start date alias for two-date queries.
        start_time : HH:MM:SS  — start of window within the start date.
        end_time   : HH:MM:SS  — end of window within the end date (or start date).
        end_date   : YYYY-MM-DD — end date when range spans multiple days.

    Examples:
        get_last_n_hours_summary(date="2026-04-02")
        get_last_n_hours_summary(date="2026-04-02", start_time="14:00", end_time="16:00")
        get_last_n_hours_summary(start_date="2026-04-05", end_date="2026-04-07")
        get_last_n_hours_summary(start_date="2026-04-05", start_time="22:00", end_date="2026-04-06", end_time="02:00")
        get_last_n_hours_summary(camera_id="ATPL-908610-ARCIS", date="2026-04-02")
        get_last_n_hours_summary(location="O.N.G.C. Office", date="2026-04-02")
        get_last_n_hours_summary()   # returns everything
    """
    # ── start_date is an alias for date ──────────────────────────────────────
    # Prefer start_date if both are supplied (explicit two-date form wins).
    resolved_start_date = start_date or date

    logger.info("=" * 60)
    logger.info("TOOL: get_last_n_hours_summary")
    logger.info(
        f"  camera={camera_id}  location={location}  date={date}  start_date={start_date}  "
        f"start={start_time}  end={end_time}  end_date={end_date}  k={k}"
    )
    logger.info(f"  → resolved_start_date={resolved_start_date}")
    logger.info("=" * 60)

    try:
        db.load_all_cameras()

        from datetime import time as dt_time

        # ── resolve time range ────────────────────────────────────────────────

        if start_time or end_time:
            # Explicit window — default start date to today if still missing
            if not resolved_start_date:
                resolved_start_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            def _parse_t(t_str, fallback_hour):
                try:
                    return dt_time.fromisoformat(t_str)
                except (ValueError, TypeError):
                    return dt_time(fallback_hour, 0, 0)

            s_time = _parse_t(start_time, 0) if start_time else dt_time(0, 0, 0)
            e_time = _parse_t(end_time, 23) if end_time else dt_time(23, 59, 59)

            base = datetime.fromisoformat(resolved_start_date).date()
            start_dt = datetime.combine(base, s_time).replace(tzinfo=timezone.utc)
            end_base = datetime.fromisoformat(end_date).date() if end_date else base
            end_dt = datetime.combine(end_base, e_time).replace(tzinfo=timezone.utc)

        elif resolved_start_date and end_date:
            # Two explicit dates — full day-to-day range
            base = datetime.fromisoformat(resolved_start_date).date()
            start_dt = datetime.combine(base, dt_time(0, 0, 0)).replace(tzinfo=timezone.utc)
            end_base = datetime.fromisoformat(end_date).date()
            end_dt = datetime.combine(end_base, dt_time(23, 59, 59)).replace(tzinfo=timezone.utc)

        elif resolved_start_date:
            # Single date — full day
            base = datetime.fromisoformat(resolved_start_date).date()
            start_dt = datetime.combine(base, dt_time(0, 0, 0)).replace(tzinfo=timezone.utc)
            end_dt = datetime.combine(base, dt_time(23, 59, 59)).replace(tzinfo=timezone.utc)

        else:
            # Entire database
            all_segs = db._collect_segments(camera_id)
            if location is not None:
                all_segs = [s for s in all_segs if db.matches_location(s.location, location)]
            if not all_segs:
                return {"message": "No segments found in database", "count": 0, "segments": [], "status_code": 200}

            make_aware = VideoDatabase._make_aware
            timed_segs = [s for s in all_segs if s.start_time and s.end_time]
            if not timed_segs:
                return {"message": "No timed segments found in database", "count": 0, "segments": [], "status_code": 200}
            start_dt = min(make_aware(s.start_time) for s in timed_segs)
            end_dt = max(make_aware(s.end_time) for s in timed_segs)

        logger.info(f"  Resolved range: {start_dt.isoformat()} → {end_dt.isoformat()}")

        segments = db.get_segments_by_absolute_time_range(start_dt, end_dt, camera_id, location)

        if not segments:
            return {
                "message": f"No segments found for {camera_id or 'any camera'} / {location or 'any location'} in the specified range",
                "camera_id": camera_id,
                "location": location,
                "count": 0,
                "segments": [],
                "time_range": {"start": start_dt.isoformat(), "end": end_dt.isoformat()},
                "status_code": 200,
            }

        segments = segments[-k:]
        serialized = [serialize_segment(seg) for seg in segments]
        logger.info(f"  Returning {len(serialized)} segment(s)")

        return {
            "message": f"Retrieved {len(serialized)} segment(s) from {camera_id or 'all cameras'} / {location or 'all locations'}",
            "camera_id": camera_id,
            "location": location,
            "count": len(serialized),
            "time_range": {"start": start_dt.isoformat(), "end": end_dt.isoformat()},
            "segments": serialized,
            "status_code": 200,
        }

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc(), "status_code": 500}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — list_available_cameras
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool
async def list_available_cameras(ctx: Context, location: Optional[Any] = None) -> Dict[str, Any]:
    """
    List all available camera IDs in the database.

    Args:
        location : Optional location name (or list of names), matched
                    case-insensitively as a substring against each camera's
                    tagged location (e.g. "ongc" matches "O.N.G.C. Office").
                    Omit to list cameras across all locations.

    Returns:
        Dictionary containing list of available cameras with statistics
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Listing Available Cameras (location={location})")
    logger.info(f"{'='*70}\n")

    try:
        # Reload database
        db.load_all_cameras()

        cameras = db.get_all_cameras()

        if not cameras:
            return {
                "message": "No cameras found in database",
                "count": 0,
                "cameras": [],
                "status_code": 200
            }

        # Get statistics (+ location) for each camera, optionally filtered by location
        camera_info = []
        for cam in cameras:
            cam_location = db.get_camera_location(cam)
            if location is not None and not db.matches_location(cam_location, location):
                continue

            stats = db.get_statistics(cam)
            camera_info.append({
                "camera_id": cam,
                "location": cam_location,
                "total_segments": stats.get('total_segments', 0),
                "total_duration_minutes": stats.get('total_duration_minutes', 0),
                "first_recording": stats.get('first_segment_time'),
                "last_recording": stats.get('last_segment_time')
            })

        logger.info(f"[SUCCESS] Found {len(camera_info)} camera(s)\n")

        return {
            "message": f"Found {len(camera_info)} camera(s)",
            "count": len(camera_info),
            "cameras": camera_info,
            "status_code": 200
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Error listing cameras: {str(e)}"
        logger.error(f"[ERROR] {error_msg}")
        logger.error(traceback.format_exc())
        return {
            "error": error_msg,
            "traceback": traceback.format_exc(),
            "status_code": 500
        }

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — search_segments_by_activity
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool
async def search_segments_by_activity(
    ctx: Context,
    query: Any,
    date: Optional[str] = None,         # ← optional: omit to search every date
    camera_id: Optional[Any] = None,
    location: Optional[Any] = None,
    max_results: int = 100,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search for video segments containing specific activities or objects, using
    hybrid search: BM25 (lexical relevance over each segment's description) and
    semantic similarity (embedding cosine similarity) are both computed, then
    fused with Reciprocal Rank Fusion (RRF) into a single relevance ranking.
    There is no plain substring keyword match — BM25 handles term relevance,
    including partial/related term matches and description length normalization.
    Searches all cameras if camera_id is not provided.

    Args:
        query: Keywords to search for - can be a string or list of strings (e.g., 'police' or ['person', 'walking', 'car'])
        date: Optional date to search in YYYY-MM-DD format (e.g., '2026-04-02'). Omit to
              search across every date in the database instead of a single day/range.
        camera_id: Optional Camera ID or list of IDs (e.g., 'ATPL-908610-ARCIS' or ['CAM1', 'CAM2'])
        location: Optional location name or list of names (e.g., 'O.N.G.C. Office'), matched
                  case-/punctuation-insensitively as a substring against each segment's location
        max_results: Maximum number of results to return (default: 10)
        start_time: Optional start time filter (e.g., '14:00')
        end_time: Optional end time filter (e.g., '16:00')
        end_date: Optional end date filter (e.g., '2026-04-03')
    """
    # Normalize query to a list of keywords
    keywords = query
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(',') if k.strip()]
    elif not isinstance(keywords, list):
        keywords = [str(keywords)]
    
    # Parse date and times for filtering. `date` is optional — when omitted,
    # filter_date/filter_start_dt/filter_end_dt all stay None below, and
    # hybrid_search_segments applies no date/time filter at all, searching
    # every segment in the database (every camera too, if camera_id is also
    # omitted).
    filter_date = None
    if date:
        try:
            filter_date = datetime_date.fromisoformat(date)
        except ValueError:
            return {
                "error": f"Invalid date format: '{date}'. Use YYYY-MM-DD (e.g., '2026-04-02')",
                "status_code": 400
            }

    from datetime import time as dt_time
    def _parse_t(t_str, fallback_hour):
        try:
            return dt_time.fromisoformat(t_str)
        except (ValueError, TypeError):
            return dt_time(fallback_hour, 0, 0)

    filter_start_dt = None
    filter_end_dt = None
    filter_start_time = None   # time-of-day only, used when no date at all is given
    filter_end_time = None

    if date and end_date:
        # ── Date-range query: date → end_date ─────────────────────────────────
        # Build an absolute datetime range covering the full span, then clear
        # filter_date so search_segments_by_keywords uses the range branch
        # (filter_start_dt / filter_end_dt) instead of the single-day exact match.
        start_base = datetime.fromisoformat(date).date()
        end_base   = datetime.fromisoformat(end_date).date()

        s_time = _parse_t(start_time, 0) if start_time else dt_time(0, 0, 0)
        e_time = _parse_t(end_time,  23) if end_time  else dt_time(23, 59, 59)

        filter_start_dt = datetime.combine(start_base, s_time).replace(tzinfo=timezone.utc)
        filter_end_dt   = datetime.combine(end_base,   e_time).replace(tzinfo=timezone.utc)
        filter_date     = None          # range filter takes over; single-day check not needed

    elif date:
        # ── Single-date query ─────────────────────────────────────────────────
        if start_time:
            filter_start_dt = datetime.combine(
                filter_date, _parse_t(start_time, 0)
            ).replace(tzinfo=timezone.utc)

        if end_time:
            filter_end_dt = datetime.combine(
                filter_date, _parse_t(end_time, 23)
            ).replace(tzinfo=timezone.utc)
    else:
        # No date at all. A time window without a day ("red truck between 6
        # and 7", no date mentioned) still means something — "that window,
        # any date" — so it's applied as a time-of-day-only filter across
        # every date instead of being silently dropped.
        if start_time:
            filter_start_time = _parse_t(start_time, 0)
        if end_time:
            filter_end_time = _parse_t(end_time, 23)

    logger.info(f"\n{'='*70}")
    logger.info(f"Searching Video Database")
    logger.info(f"{'='*70}")
    logger.info(f"Camera: {camera_id if camera_id else 'ALL'}")
    logger.info(f"Location: {location if location else 'ALL'}")
    logger.info(f"Keywords: {keywords}")
    logger.info(f"Date: {date if date else 'ALL'}")
    if end_date: logger.info(f"End Date: {end_date}")
    if start_time: logger.info(f"Start Time: {start_time}")
    if end_time: logger.info(f"End Time: {end_time}")
    logger.info(f"Max results: {max_results}\n")
    
    try:
        # Reload database
        db.load_all_cameras()

        # Embed the query once up front so BM25 and semantic ranking can run together.
        query_text = query if isinstance(query, str) else " ".join(keywords)
        query_vector = get_query_embedding(query_text)
        if query_vector is None:
            logger.warning("[SEMANTIC] Ollama embedding unavailable — ranking will fall back to BM25 only")

        # Hybrid search: BM25 (lexical) + semantic (embedding) ranking, fused via RRF
        search_res = db.hybrid_search_segments(
            keywords,
            camera_id,
            location=location,
            filter_date=filter_date,
            filter_start_dt=filter_start_dt,
            filter_end_dt=filter_end_dt,
            filter_start_time=filter_start_time,
            filter_end_time=filter_end_time,
            query_vector=query_vector,
            top_k=max(max_results, FAISS_TOP_K),
            semantic_score_threshold=FAISS_SCORE_THRESHOLD,
        )

        segments = search_res["results"]
        scored_results = search_res.get("scored_results", [])
        total_checked = search_res.get("total_checked", 0)
        date_filtered = search_res.get("date_filtered", 0)
        time_filtered = search_res.get("time_filtered", 0)
        cam_filtered = search_res.get("cam_filtered", 0)
        loc_filtered = search_res.get("location_filtered", 0)

        logger.info(
            f"  Filter stats → total={total_checked}  date/time={date_filtered + time_filtered}  "
            f"camera={cam_filtered}  location={loc_filtered}  hybrid_hits={len(segments)}"
        )

        if not segments:
            filter_parts = [f"date {date}"] if date else ["all dates"]
            if end_date:    filter_parts.append(f"to {end_date}")
            if start_time:  filter_parts.append(f"after {start_time}")
            if end_time:    filter_parts.append(f"before {end_time}")
            if camera_id:   filter_parts.append(f"camera {camera_id}")
            if location:    filter_parts.append(f"location {location}")
            debug_info = (
                f" (Checked {total_checked} total → "
                f"{date_filtered + time_filtered} date/time-filtered, "
                f"{cam_filtered} camera-filtered, "
                f"{loc_filtered} location-filtered)"
            )

            if total_checked == 0:
                reason = "No segments found in database."
            elif date_filtered + time_filtered == total_checked:
                reason = "No segments found in the specified date/time range."
            elif cam_filtered > 0 and (date_filtered + time_filtered + cam_filtered) == total_checked:
                reason = f"No segments found for camera '{camera_id}' in the specified date/time range."
            elif loc_filtered > 0 and (date_filtered + time_filtered + cam_filtered + loc_filtered) == total_checked:
                reason = f"No segments found for location '{location}' in the specified date/time range."
            else:
                reason = "Found segments in this time range, but none matched the query via BM25 or semantic search."

            return {
                "message": f"{reason}{debug_info}",
                "search_method": "hybrid_bm25_semantic_rrf",
                "camera_id": camera_id,
                "location": location,
                "keywords": keywords,
                "date": date,
                "count": 0,
                "total_checked":    total_checked,
                "date_filtered":    date_filtered,
                "time_filtered":    time_filtered,
                "cam_filtered":     cam_filtered,
                "location_filtered": loc_filtered,
                "segments": [],
                "status_code": 200,
            }

        # Results already ranked by RRF score (most relevant first); cap to max_results.
        segments = segments[:max_results]
        score_map = {
            (r["camera_id"], r["segment_id"]): r["rrf_score"]
            for r in scored_results
        }
        serialized_segments = [
            serialize_segment(seg, score=score_map.get((seg.camera_id, seg.segment_id)))
            for seg in segments
        ]

        logger.info(f"[SUCCESS] Found {len(segments)} matching segment(s) via hybrid BM25+semantic RRF\n")

        filter_parts = [f"date {date}"] if date else ["all dates"]
        if start_time: filter_parts.append(f"after {start_time}")
        if end_time: filter_parts.append(f"before {end_time}")
        filter_str = " at " + " ".join(filter_parts)

        cam_success = f"camera {camera_id}" if camera_id else "all cameras"
        loc_success = f", location {location}" if location else ""
        return {
            "message": f"Found {len(segments)} segment(s) via hybrid BM25+semantic search (RRF) in {cam_success}{loc_success}{filter_str}",
            "search_method": "hybrid_bm25_semantic_rrf",
            "camera_id": camera_id,
            "location": location,
            "keywords": keywords,
            "date": date,
            "count": len(serialized_segments),
            "segments": serialized_segments,
            "status_code": 200
        }

    except Exception as e:
        import traceback
        error_msg = f"Error searching video database: {str(e)}"
        logger.error(f"[ERROR] {error_msg}")
        logger.error(traceback.format_exc())
        return {
            "error": error_msg,
            "traceback": traceback.format_exc(),
            "status_code": 500
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — search_car_by_plate_number
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool
async def search_car_by_plate_number(
    ctx: Context,
    plate_number: str,
    camera_id: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Search for a vehicle by its license plate number (ANPR detections).

    Always restricted to recognized (OCR-validated) plate readings —
    recognized=True is applied unconditionally, it is never a parameter.

    Args:
        plate_number : Plate number or substring to search for, e.g.
                        "GJ01AB1234" or "GJ01" (matched case-insensitively,
                        substring match — a partial plate returns every
                        recognized plate containing it).
        camera_id     : Optional camera ID (or list of IDs) to restrict the
                         search to. Omit to search across all cameras.

    Returns:
        Matching detections, each with frame_url, plate_number, description,
        camera_id, location, and timestamp — this is the data handed to the
        LLM/VLM summarizer to describe what was found.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Searching for plate: {plate_number!r} (camera_id={camera_id})")
    logger.info(f"{'='*70}\n")

    try:
        query: Dict[str, Any] = {
            "recognized": True,
            "plate_number": {"$regex": re.escape(plate_number), "$options": "i"},
        }
        if camera_id is not None:
            ids = camera_id if isinstance(camera_id, list) else [camera_id]
            query["camera_id"] = {"$in": ids}

        docs = list(db.collection.find(query).sort("timestamp", -1).limit(100))

        if not docs:
            return {
                "message": f"No recognized plates matching '{plate_number}' found",
                "plate_number": plate_number,
                "camera_id": camera_id,
                "count": 0,
                "results": [],
                "status_code": 200,
            }

        results = []
        for d in docs:
            ts = d.get("timestamp")
            results.append({
                "frame_url":    d.get("frame_url"),
                "plate_number": d.get("plate_number"),
                "description":  d.get("description") or d.get("ai_summary") or "",
                "camera_id":    d.get("camera_id"),
                "location":     d.get("location"),
                "timestamp":    ts.isoformat() if hasattr(ts, "isoformat") else ts,
            })

        logger.info(f"[SUCCESS] Found {len(results)} matching detection(s)\n")
        return {
            "message": f"Found {len(results)} detection(s) matching '{plate_number}'",
            "plate_number": plate_number,
            "camera_id": camera_id,
            "count": len(results),
            "results": results,
            "status_code": 200,
        }

    except Exception as e:
        import traceback
        error_msg = f"Error searching plate database: {str(e)}"
        logger.error(f"[ERROR] {error_msg}")
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc(), "status_code": 500}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — list_recognized_plates
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool
async def list_recognized_plates(ctx: Context, camera_id: Optional[Any] = None) -> Dict[str, Any]:
    """
    List every recognized (OCR-validated) license plate detection in the
    database — the ANPR equivalent of list_available_cameras.

    Always restricted to recognized=True — this is applied unconditionally,
    it is never a parameter.

    Args:
        camera_id : Optional camera ID (or list of IDs) to restrict the list
                    to. Omit to list plates across all cameras.

    Returns:
        One entry per recognized plate detection: camera_id, plate_number,
        location, and date.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Listing recognized plates (camera_id={camera_id})")
    logger.info(f"{'='*70}\n")

    try:
        query: Dict[str, Any] = {"recognized": True}
        if camera_id is not None:
            ids = camera_id if isinstance(camera_id, list) else [camera_id]
            query["camera_id"] = {"$in": ids}

        docs = list(db.collection.find(query).sort("timestamp", -1).limit(500))

        plates = []
        for d in docs:
            ts = d.get("timestamp")
            plates.append({
                "camera_id":    d.get("camera_id"),
                "plate_number": d.get("plate_number"),
                "location":     d.get("location"),
                "date":         ts.isoformat() if hasattr(ts, "isoformat") else ts,
            })

        logger.info(f"[SUCCESS] Found {len(plates)} recognized plate(s)\n")
        return {
            "message": f"Found {len(plates)} recognized plate(s)",
            "camera_id": camera_id,
            "count": len(plates),
            "plates": plates,
            "status_code": 200,
        }

    except Exception as e:
        import traceback
        error_msg = f"Error listing recognized plates: {str(e)}"
        logger.error(f"[ERROR] {error_msg}")
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc(), "status_code": 500}


# ── Starlette / SSE wiring ─────────────────────────────────────────────────────

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> Response:
        try:
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as (read_stream, write_stream):
                await mcp_server.run(
                    read_stream, write_stream, mcp_server.create_initialization_options()
                )
            return Response(content="", status_code=200)
        except Exception as e:
            logger.error(f"SSE error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    async def health_check(request: Request) -> JSONResponse:
        cameras = db.get_all_cameras() if db else []
        faiss_status = {}
        if db:
            for cam in cameras:
                cidx = db.indices_by_camera.get(cam)
                faiss_status[cam] = (
                    {"ready": True, "vectors": len(cidx.embeddings)}
                    if cidx and cidx.ready
                    else {"ready": False}
                )
        return JSONResponse({
            "status": "healthy",
            "service": "video-segment-retrieval-mcp",
            "tools": ["get_last_n_hours_summary", "list_available_cameras", "search_segments_by_activity"],
            "cameras": cameras,
            "faiss": faiss_status,
            "embedding_model": OLLAMA_EMBED_MODEL,
            "ollama_url": OLLAMA_BASE_URL,
        })

    return Starlette(
        debug=debug,
        routes=[
            Route("/health", endpoint=health_check, methods=["GET"]),
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCP Video Retrieval Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()

    initialize_database()

    print("\n" + "=" * 60)
    print("MCP VIDEO RETRIEVAL SERVER")
    print("=" * 60)
    print(f"DB path       : {BASE_DB_PATH}")
    print(f"Cameras       : {db.get_all_cameras() if db else 'none'}")
    print(f"Embed model   : {OLLAMA_EMBED_MODEL}")
    print(f"Ollama URL    : {OLLAMA_BASE_URL}")
    print(f"FAISS top-k   : {FAISS_TOP_K}")
    print(f"Score thresh  : {FAISS_SCORE_THRESHOLD}")
    print()
    print("Endpoints:")
    print("  GET  /sse       — MCP SSE connection")
    print("  POST /messages/ — MCP message handler")
    print("  GET  /health    — health check")
    print()
    print("Tools:")
    print("  1. get_last_n_hours_summary")
    print("     date, start_time, end_time, camera_id, k")
    print("  2. list_available_cameras")
    print("     no args — returns all camera IDs + stats")
    print("  3. search_segments_by_activity")
    print("     query, date, camera_id, start_time, end_time, max_results")
    print("     → hybrid BM25 + semantic search, fused via Reciprocal Rank Fusion (RRF)")
    print("=" * 60 + "\n")

    mcp_server = mcp._mcp_server
    starlette_app = create_starlette_app(mcp_server, debug=True)
    uvicorn.run(starlette_app, host=args.host, port=args.port, timeout_keep_alive=60)