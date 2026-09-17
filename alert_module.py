import os
import json
import uuid
import logging
import asyncio
import threading
import pickle
import numpy as np
import httpx
from datetime import datetime, timedelta, timezone, date as datetime_date
from typing import Optional, List, Dict, Any
from pymongo import MongoClient
import zoneinfo as _zi

# Configure logging
logger = logging.getLogger(__name__)

# ── MongoDB Configuration ─────────────────────────────────────────────────────
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

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:latest")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")

# FAISS search parameters
FAISS_TOP_K = int(os.getenv("FAISS_TOP_K", "20"))
FAISS_SCORE_THRESHOLD = float(os.getenv("FAISS_SCORE_THRESHOLD", "0.5"))

# Alert storage
ALERTS_FILE = os.getenv("ALERTS_FILE", "./alerts.json")
ALERT_RULES_FILE = os.getenv("ALERT_RULES_FILE", "./alert_rules.json")

# Timezone
_data_tz_name = os.getenv("DATA_TIMEZONE", "Asia/Kolkata")
DATA_TZ = _zi.ZoneInfo(_data_tz_name)

class VideoSegment:
    """One video clip with metadata"""
    def __init__(self, segment_id, start_time, end_time, frame_urls, 
                 description, cumulative_minutes, camera_id="default"):
        self.segment_id = segment_id
        self.start_time = start_time
        self.end_time = end_time
        self.frame_urls = frame_urls
        self.description = description
        self.cumulative_minutes = cumulative_minutes
        self.camera_id = camera_id

class CameraIndex:
    def __init__(self, camera_id: str, collection):
        self.camera_id = camera_id
        self.collection = collection
        self.segments: List[VideoSegment] = []
        self.embeddings: List[np.ndarray] = []
        self._load()

    def _load(self):
        try:
            docs = list(self.collection.find({"camera_id": self.camera_id}).sort("segment_id", 1))
            self.segments = []
            self.embeddings = []
            for doc in docs:
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
                    camera_id=self.camera_id
                )
                self.segments.append(seg)
                emb = doc.get("embedding")
                if emb:
                    arr = np.array(emb, dtype=np.float32)
                    norm = np.linalg.norm(arr)
                    if norm > 0: arr = arr / norm
                    self.embeddings.append(arr)
                else:
                    self.embeddings.append(np.zeros(768, dtype=np.float32))
        except Exception as e:
            logger.error(f"❌ [{self.camera_id}] Failed to load from MongoDB: {e}")

    @property
    def ready(self) -> bool:
        return len(self.segments) > 0

class MongoDBDatabase:
    def __init__(self):
        self.client = MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=30000)
        self.db_mongo = self.client[MONGO_DATABASE]
        self.collection = self.db_mongo[MONGO_COLLECTION_ACTIVITIES]
        self.segments_by_camera: Dict[str, List[VideoSegment]] = {}
        self.indices_by_camera: Dict[str, CameraIndex] = {}
        self.load_all_cameras()

    def load_all_cameras(self):
        try:
            camera_ids = self.collection.distinct("camera_id")
            self.segments_by_camera.clear()
            self.indices_by_camera.clear()
            for cam_id in camera_ids:
                cidx = CameraIndex(cam_id, self.collection)
                if cidx.ready:
                    self.indices_by_camera[cam_id] = cidx
                    self.segments_by_camera[cam_id] = cidx.segments
        except Exception as e:
            logger.error(f"❌ [Mongo] Error loading cameras: {e}")

    def get_all_cameras(self) -> List[str]:
        return sorted(list(self.segments_by_camera.keys()))

    def get_all_dates(self) -> List[str]:
        try:
            start_times = self.collection.distinct("start_time")
            unique_dates = set()
            for st in start_times:
                if not st: continue
                if isinstance(st, datetime): unique_dates.add(st.strftime("%Y-%m-%d"))
                elif isinstance(st, str):
                    try:
                        dt = datetime.fromisoformat(st.replace("Z", "+00:00"))
                        unique_dates.add(dt.strftime("%Y-%m-%d"))
                    except:
                        if len(st) >= 10: unique_dates.add(st[:10])
            return sorted(list(unique_dates), reverse=True)
        except Exception as e:
            logger.error(f"Error fetching dates: {e}")
            return []

    def get_camera_stats(self, camera_id: str) -> Dict:
        if camera_id not in self.segments_by_camera: return {}
        segments = self.segments_by_camera[camera_id]
        if not segments: return {"total_segments": 0}
        total_duration = sum((s.end_time - s.start_time).total_seconds() for s in segments)
        return {
            "camera_id": camera_id,
            "total_segments": len(segments),
            "total_frames": sum(len(s.frame_urls) for s in segments),
            "total_duration_minutes": total_duration / 60
        }

    def _collect_segments(self, camera_id: Optional[str]) -> List[VideoSegment]:
        # Case-insensitive: the rules file contains "ALL", "All" and "all", and a
        # bare == comparison treats the latter two as a camera literally named
        # "All", which does not exist - so those rules silently never fired.
        if camera_id is None or str(camera_id).strip().upper() == "ALL":
            segs = []
            for v in self.segments_by_camera.values(): segs.extend(v)
            return segs
        return list(self.segments_by_camera.get(camera_id, []))

    @staticmethod
    def _make_aware(dt: datetime) -> datetime:
        return dt if dt.tzinfo else dt.replace(tzinfo=DATA_TZ)

    def get_segments_by_time(self, camera_id: Optional[str], 
                            start_dt: Optional[datetime] = None, 
                            end_dt: Optional[datetime] = None,
                            max_results: int = 100) -> List[VideoSegment]:
        segments = self._collect_segments(camera_id)
        if not segments: return []
        filtered = segments
        if start_dt and end_dt:
            start_dt = self._make_aware(start_dt)
            end_dt = self._make_aware(end_dt)
            filtered = [s for s in segments if self._make_aware(s.start_time) <= end_dt and self._make_aware(s.end_time) >= start_dt]
        filtered.sort(key=lambda x: x.start_time, reverse=True)
        return filtered[:max_results]

    def search_by_keywords(self, query: str, camera_id: Optional[str],
                          filter_start_dt: Optional[datetime] = None,
                          filter_end_dt: Optional[datetime] = None,
                          max_results: int = 20) -> List[VideoSegment]:
        segments = self._collect_segments(camera_id)
        if not segments: return []
        search_term = query.lower().strip()
        if not search_term: return []
        results = []
        for seg in segments:
            seg_start = self._make_aware(seg.start_time)
            if filter_start_dt and seg_start < self._make_aware(filter_start_dt): continue
            if filter_end_dt and seg_start > self._make_aware(filter_end_dt): continue
            if search_term in seg.description.lower(): results.append(seg)
        return sorted(results, key=lambda x: x.start_time, reverse=True)[:max_results]

    def search_by_semantic(self, query_vector: np.ndarray, camera_id: Optional[str],
                          filter_start_dt: Optional[datetime] = None,
                          filter_end_dt: Optional[datetime] = None,
                          top_k: int = 20) -> List[Dict]:
        cameras_to_search = [camera_id] if camera_id else list(self.indices_by_camera.keys())
        qv = query_vector.astype(np.float32).flatten()
        norm = np.linalg.norm(qv)
        if norm > 0: qv = qv / norm
        all_results = []
        for cam in cameras_to_search:
            cidx = self.indices_by_camera.get(cam)
            if not cidx or not cidx.ready: continue
            for i, emb in enumerate(cidx.embeddings):
                score = float(np.dot(qv, emb))
                if score < FAISS_SCORE_THRESHOLD: continue
                seg = cidx.segments[i]
                seg_start = self._make_aware(seg.start_time)
                if filter_start_dt and seg_start < self._make_aware(filter_start_dt): continue
                if filter_end_dt and seg_start > self._make_aware(filter_end_dt): continue
                all_results.append({"segment": seg, "score": score, "camera_id": cam})
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def save_segment(self, segment_data: Dict):
        try:
            self.collection.update_one(
                {"camera_id": segment_data["camera_id"], "segment_id": segment_data["segment_id"]},
                {"$set": segment_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving segment to DB: {e}")
            return False

class AlertManager:
    def __init__(self, rules_file: str, alerts_file: str):
        self.rules_file = rules_file
        self.alerts_file = alerts_file
        self.rules: List[Dict] = []
        self.alerts: List[Dict] = []
        self.load()
    
    def load(self):
        if os.path.exists(self.rules_file):
            try:
                with open(self.rules_file, "r") as f: self.rules = json.load(f)
            except: self.rules = []
        if os.path.exists(self.alerts_file):
            try:
                with open(self.alerts_file, "r") as f: self.alerts = json.load(f)
            except: self.alerts = []
    
    def save_rules(self):
        with open(self.rules_file, "w") as f: json.dump(self.rules, f, indent=2)
    def save_alerts(self):
        with open(self.alerts_file, "w") as f: json.dump(self.alerts, f, indent=2)
    
    def add_rule(self, camera_id: str, prompt: str) -> Dict:
        rule = {"id": str(uuid.uuid4()), "camera_id": camera_id, "prompt": prompt, "status": "active", "created_at": datetime.now(timezone.utc).isoformat()}
        self.rules.append(rule)
        self.save_rules()
        return rule
    
    def delete_rule(self, rule_id: str) -> bool:
        initial_len = len(self.rules)
        self.rules = [r for r in self.rules if r["id"] != rule_id]
        if len(self.rules) < initial_len:
            self.save_rules()
            return True
        return False
    
    def add_alert(self, camera_id: str, matched_prompt: str, description: str, frame_url: str, segment_id: int) -> Dict:
        alert = {"id": str(uuid.uuid4()), "camera_id": camera_id, "matched_prompt": matched_prompt, "description": description, "frame_url": frame_url, "segment_id": segment_id, "timestamp": datetime.now(timezone.utc).isoformat()}
        self.alerts.insert(0, alert)
        self.alerts = self.alerts[:100]
        self.save_alerts()
        return alert

    def delete_alert(self, alert_id: str) -> bool:
        initial_len = len(self.alerts)
        self.alerts = [a for a in self.alerts if a["id"] != alert_id]
        if len(self.alerts) < initial_len:
            self.save_alerts()
            return True
        return False

_EMBEDDER = None
_EMBEDDER_LOCK = threading.Lock()


def _get_embedder():
    """The exact model merge-final-many.py stores document vectors with,
    loaded locally instead of called over Ollama's HTTP endpoint - this
    background watchdog polls every 30s, so it was the steadiest Ollama
    dependency in the whole chat process, not just an occasional call.
    """
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            from sentence_transformers import SentenceTransformer
            logger.info("[Alerts] Loading embedder: nomic-ai/nomic-embed-text-v1.5")
            _EMBEDDER = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    return _EMBEDDER


async def get_embedding(text: str) -> Optional[np.ndarray]:
    try:
        model = await asyncio.to_thread(_get_embedder)
        vec = await asyncio.to_thread(
            model.encode, [text], convert_to_numpy=True, normalize_embeddings=True)
        return vec[0].astype(np.float32)
    except Exception as e:
        logger.error(f"[Alerts] embedding error: {e}")
        return None

async def check_alerts_background(alert_manager: AlertManager, db: MongoDBDatabase):
    logger.info("🕒 [Alerts] Background checker started")
    while True:
        try:
            await asyncio.sleep(30)
            db.load_all_cameras()
            active_rules = [r for r in alert_manager.rules if r["status"] == "active"]
            if not active_rules: continue
            now = datetime.now(timezone.utc)
            start_dt = now - timedelta(minutes=2)
            for rule in active_rules:
                raw_camera = str(rule.get("camera_id", "") or "").strip()
                camera_id = None if raw_camera.upper() == "ALL" else raw_camera
                prompt = rule["prompt"].lower()
                recent_segments = db.search_by_keywords(prompt, camera_id, filter_start_dt=start_dt, filter_end_dt=now, max_results=5)
                if not recent_segments:
                    query_vector = await get_embedding(prompt)
                    if query_vector is not None:
                        semantic_results = db.search_by_semantic(query_vector, camera_id, filter_start_dt=start_dt, filter_end_dt=now, top_k=5)
                        recent_segments = [r["segment"] for r in semantic_results]
                for seg in recent_segments:
                    if not any(a["segment_id"] == seg.segment_id and a["camera_id"] == seg.camera_id for a in alert_manager.alerts[:20]):
                        alert_manager.add_alert(seg.camera_id, rule["prompt"], seg.description, seg.frame_urls[0] if seg.frame_urls else "", seg.segment_id)
                        logger.info(f"🚨 ALERT TRIGGERED: {rule['prompt']} on {seg.camera_id}")
        except Exception as e: logger.error(f"Alert checker error: {e}")
