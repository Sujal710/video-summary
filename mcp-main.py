"""
FastAPI Web Interface for Video Segment Chatbot with SSE
Beautiful UI with real-time streaming responses and image display
IMPROVED: Auto-handles missing camera IDs - FIXED JAVASCRIPT
CACHE: Per-query JSON cache with cosine similarity deduplication
"""

import os
import sys

# 0. CRITICAL: ~/.bashrc exports LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:...,
# which makes the system CUDA 12.9 toolkit's libcublas.so.12 (v12.9.1.4) shadow
# the pip-bundled cuBLAS (v12.8.4.1) our installed torch build (cu128) was
# actually compiled against. That version mismatch breaks EVERY batched GPU
# matmul (bmm/baddbmm/matmul with batch>1, any dtype) used by SAM3 and any
# other GPU-side torch model — confirmed via isolated repro. The dynamic
# linker resolves library search paths once at process startup, so this can't
# be patched by mutating os.environ later (torch must not be imported yet) —
# the process has to relaunch itself with a clean environment first.
if "cuda-12.9" in os.environ.get("LD_LIBRARY_PATH", "") and not os.environ.get("_ARCIS_LD_PATH_FIXED"):
    _env = os.environ.copy()
    _env.pop("LD_LIBRARY_PATH", None)
    _env["_ARCIS_LD_PATH_FIXED"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, _env)

import multiprocessing

# 1. CRITICAL: Environment variables MUST be set before ANY other imports
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtmp_live;1|rtmp_buffer;100"
# This GPU is shared with several long-lived Ollama model processes that
# together hold multiple GB resident — SAM3 (or any torch model here) can hit
# CUDA OOM even with memory nominally free, due to allocator fragmentation.
# expandable_segments lets PyTorch's allocator grow/reuse segments instead of
# needing one contiguous free block, exactly what the OOM error itself
# recommends. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 2. Force 'spawn' method for multiprocessing to isolate memory
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import cv2
cv2.setNumThreads(0) # Disable OpenCV internal threading to prevent segfaults

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import json
import os
import pickle
import hashlib
import time
import tempfile
import uuid
import shutil
from pathlib import Path
from typing import Optional, AsyncGenerator, List, Dict, Any
from datetime import datetime
import uvicorn
import logging
import numpy as np
import requests
import httpx
from supertonic import TTS
import io
import soundfile as sf
from fastapi import File, UploadFile, Form
import db
import alert_module
import re
import cv2
from ollama import chat
from azure.storage.blob import BlobServiceClient, ContentSettings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Azure Configuration ────────────────────────────────────────────────────────
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING", "")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN", "")
AZURE_CONTAINER_NAME = "nvrdatashinobi"
AZURE_BLOB_PREFIX = "live-record/frimages"
STATIC_IMAGE_URL = f"https://nvrdatashinobi.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{AZURE_BLOB_PREFIX}"

def get_azure_client():
    try:
        return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    except Exception as e:
        logger.error(f"Failed to initialize Azure client: {e}")
        return None

class VideoSegmentVLM:
    def __init__(self, segment_id, camera_id, start_time, end_time, frame_urls, description, cumulative_minutes):
        self.segment_id = segment_id
        self.camera_id = camera_id
        self.start_time = start_time
        self.end_time = end_time
        self.frame_urls = frame_urls
        self.description = description
        self.cumulative_minutes = cumulative_minutes

    def to_dict(self):
        return {
            'segment_id': self.segment_id,
            'camera_id': self.camera_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'cumulative_minutes': self.cumulative_minutes,
            'frame_urls': self.frame_urls,
            'description': self.description
        }

def clean_description_text(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'##\s*', '', cleaned)
    cleaned = re.sub(r'#\s*', '', cleaned)
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)
    cleaned = re.sub(r'\d{4}\.\d{1,2}\.\d{1,2}', '', cleaned)
    cleaned = re.sub(r'\d{1,2}:\d{2}:\d{2}', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()

    lines = cleaned.split('\n')
    cleaned_lines = [line for line in lines if len(line.strip()) >= 3 or line.strip() == '']
    return '\n'.join(cleaned_lines)

def upload_frame_to_azure(blob_service_client, camera_id: str, segment_id: int,
                          frame_img: np.ndarray, frame_position: str,
                          timestamp: datetime) -> Optional[str]:
    try:
        if frame_img is None or frame_img.size == 0: return None
        time_str = timestamp.strftime("%Y-%m-%d-%H-%M-%S")
        seg_str = f"{segment_id:04d}"
        filename = f"{camera_id}_SEG{seg_str}_{frame_position}_{time_str}.jpg"
        blob_name = f"{AZURE_BLOB_PREFIX}/{filename}"
        success, buf = cv2.imencode(".jpg", frame_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success: return None
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)
        blob_client.upload_blob(buf.tobytes(), overwrite=True, content_settings=ContentSettings(content_type='image/jpeg'))
        return f"{STATIC_IMAGE_URL}/{filename}?{AZURE_SAS_TOKEN}"
    except Exception as e:
        logger.error(f"Azure upload failed: {e}")
        return None

def get_segment_description(frames_buffers: List[Any]) -> str:
    temp_files = []
    try:
        image_paths = []
        for buffer in frames_buffers:
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir='/tmp')
            temp_file.write(buffer.tobytes())
            temp_file.flush()
            temp_file.close()
            temp_files.append(temp_file.name)
            image_paths.append(temp_file.name)
        
        if not image_paths: return "[Error: No frames sampled]"

        system_instructions = "You are a professional surveillance scene analyst. Describe the scene in detail including people, objects, and interactions. Write in plain prose only."
        user_prompt = "Analyze these surveillance frames and provide a detailed report including environment, people, objects, and activities."

        response = chat(
            model='qwen2.5vl:7b',
            messages=[{'role': 'system', 'content': system_instructions}, {'role': 'user', 'content': user_prompt, 'images': image_paths}],
            options={'temperature': 0.1, 'num_predict': 1000, 'num_ctx': 8000}
        )
        return clean_description_text(response.message.content or "")
    except Exception as e:
        logger.error(f"VLM Error: {e}")
        return f"[Error: {str(e)}]"
    finally:
        for p in temp_files:
            try: os.unlink(p)
            except: pass

# Import the RAG client
from client import IntegratedVideoRAGClient
from graph import build_chat_graph, stream_chat_graph

app = FastAPI(
    title="Video Segment Chatbot",
    description="AI-powered video surveillance chatbot with Ollama and MCP",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Configuration ──────────────────────────────────────────────────────────────

MCP_SERVER_URL = "http://localhost:8088/sse"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")

# Cosine similarity threshold for cache hit (0.0–1.0); 0.92 = very similar queries
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92"))

# Cache TTL (Time To Live) in seconds; 3600 = 1 hour
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

# Where the query cache is persisted — MongoDB collection, same "arcis"
# database (via db.py's db_mongo) everything else in this project uses.
# CACHE_FILE (the old on-disk JSON path) is kept only as a one-time
# migration source: if the collection is empty and this file still exists,
# its entries are imported into MongoDB on first startup, then left alone.
QUERY_CACHE_COLLECTION = os.getenv("QUERY_CACHE_COLLECTION", "query_cache")
CACHE_FILE = os.getenv("QUERY_CACHE_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "query_cache.json"))
HISTORY_FILE = os.getenv("CHAT_HISTORY_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.json"))

# Video output directory — videos are served from here
VIDEO_OUTPUT_DIR = os.getenv("VIDEO_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_videos"))
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# Mount the video directory so the browser can download files via /videos/<filename>
app.mount("/videos", StaticFiles(directory=VIDEO_OUTPUT_DIR), name="videos")

# Global RAG client
rag_client: Optional[IntegratedVideoRAGClient] = None

# Compiled LangGraph for the core chat routing pipeline (contextualize ->
# select_tool -> call_mcp -> generate_summary) — built once at startup, after
# rag_client exists, since its nodes close over that instance.
chat_graph = None

# Global Supertonic TTS instance (loads once at startup)
tts_model: Optional[TTS] = None
tts_voice_style = None

alert_manager: Optional[alert_module.AlertManager] = None
alert_db: Optional[alert_module.MongoDBDatabase] = None
alert_task: Optional[asyncio.Task] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Chat History Store
# ═══════════════════════════════════════════════════════════════════════════════

class ChatStore:
    """Stores chat sessions on disk (for display only)."""
    def __init__(self, history_file: str):
        self.history_file = history_file
        # dict of session_id -> {title, messages: [{role, content, timestamp, image_urls}]}
        self._history: Dict[str, Any] = {}
        import threading
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
            except Exception as e:
                logger.error(f"[History] Load failed: {e}")
                self._history = {}

    def _save(self):
        """Save history to disk in a separate thread to avoid blocking."""
        def save_task():
            with self._lock:
                try:
                    with open(self.history_file, "w", encoding="utf-8") as f:
                        json.dump(self._history, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.error(f"[History] Save failed: {e}")
        
        # Run in a background thread
        import threading
        threading.Thread(target=save_task, daemon=True).start()

    def add_message(self, session_id: str, role: str, content: str, image_urls: List[str] = None):
        if session_id not in self._history:
            # Use the first user message as the title
            title = content[:30] + "..." if len(content) > 30 else content
            self._history[session_id] = {"title": title, "messages": [], "updated_at": time.time()}
        
        self._history[session_id]["messages"].append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "image_urls": image_urls or []
        })
        self._history[session_id]["updated_at"] = time.time()
        self._save()

    def get_sessions(self):
        # Return sorted by most recent
        sessions = []
        for sid, data in self._history.items():
            sessions.append({"id": sid, "title": data["title"], "updated_at": data.get("updated_at", 0)})
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def get_messages(self, session_id: str):
        return self._history.get(session_id, {}).get("messages", [])

    def delete_session(self, session_id: str):
        if session_id in self._history:
            del self._history[session_id]
            self._save()
            return True
        return False

# Global store instance
chat_store: Optional[ChatStore] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Query Cache
# ═══════════════════════════════════════════════════════════════════════════════

class QueryCache:
    """
    Stores (user_query, final_answer) pairs in MongoDB (collection
    QUERY_CACHE_COLLECTION, same "arcis" database db.py's db_mongo already
    connects to — no separate connection is opened here).
    For each incoming query an embedding is computed and cosine-similarity
    is checked against all stored query embeddings.  If similarity >= threshold
    the cached answer is returned without touching the LLM.
    """

    def __init__(self, collection, similarity_threshold: float = 0.92,
                migrate_from_file: Optional[str] = None):
        self.collection = collection
        self.similarity_threshold = similarity_threshold
        self.migrate_from_file = migrate_from_file
        # List of dicts: {id, query, answer, embedding, timestamp}
        self._entries: List[Dict[str, Any]] = []
        # In-memory cache of numpy embeddings for speed
        self._numpy_embeddings: Dict[str, np.ndarray] = {}
        self._last_prune_time = 0
        self._load()

    # ── persistence ────────────────────────────────────────────────────────────

    def _migrate_from_json_file_if_empty(self):
        """One-time import of the old on-disk cache into MongoDB.

        Only runs when the collection has never been populated — once there
        is anything in Mongo, this is skipped every startup after that, so
        it never re-imports or overwrites entries someone deleted since.
        """
        if not self.migrate_from_file or not os.path.exists(self.migrate_from_file):
            return
        try:
            with open(self.migrate_from_file, "r", encoding="utf-8") as f:
                old_entries = json.load(f)
        except Exception as e:
            logger.error(f"[Cache] Could not read old cache file for migration: {e}")
            return
        if not old_entries:
            return
        try:
            self.collection.insert_many(old_entries, ordered=False)
            logger.info(
                f"[Cache] Migrated {len(old_entries)} entries from "
                f"{self.migrate_from_file} into MongoDB collection "
                f"'{self.collection.name}' (one-time)"
            )
        except Exception as e:
            logger.error(f"[Cache] Migration into MongoDB failed: {e}")

    def _load(self):
        try:
            if self.collection.count_documents({}) == 0:
                self._migrate_from_json_file_if_empty()

            self._entries = list(self.collection.find({}, {"_id": 0}))

            # Pre-convert embeddings to numpy arrays
            for entry in self._entries:
                if "embedding" in entry and "id" in entry:
                    self._numpy_embeddings[entry["id"]] = np.array(entry["embedding"], dtype=np.float32)

            logger.info(
                f"[Cache] Loaded {len(self._entries)} cached queries from "
                f"MongoDB collection '{self.collection.name}'"
            )
        except Exception as e:
            logger.error(f"[Cache] Failed to load cache from MongoDB: {e} — starting fresh")
            self._entries = []

    def _persist_entry(self, entry: Dict[str, Any]):
        """Upsert one entry — the MongoDB equivalent of the old whole-file
        rewrite, but scoped to just the document that changed."""
        try:
            self.collection.replace_one({"id": entry["id"]}, entry, upsert=True)
        except Exception as e:
            logger.error(f"[Cache] Failed to save entry to MongoDB: {e}")

    def _delete_entries(self, entry_ids: List[str]):
        if not entry_ids:
            return
        try:
            self.collection.delete_many({"id": {"$in": entry_ids}})
        except Exception as e:
            logger.error(f"[Cache] Failed to delete entries from MongoDB: {e}")

    def _prune(self):
        """Remove entries older than CACHE_TTL seconds, at most once every 60s."""
        now = time.time()
        if now - self._last_prune_time < 60:
            return

        self._last_prune_time = now
        count_before = len(self._entries)
        expired_ids = [
            e["id"] for e in self._entries
            if (now - e.get("timestamp", 0)) >= CACHE_TTL and e.get("id")
        ]
        # Only keep entries where (now - timestamp) < CACHE_TTL
        self._entries = [
            e for e in self._entries
            if (now - e.get("timestamp", 0)) < CACHE_TTL
        ]

        count_after = len(self._entries)
        if count_before != count_after:
            # Sync numpy cache
            valid_ids = {e["id"] for e in self._entries}
            self._numpy_embeddings = {k: v for k, v in self._numpy_embeddings.items() if k in valid_ids}
            logger.info(f"[Cache] Pruned {count_before - count_after} expired entries — {count_after} remain")
            self._delete_entries(expired_ids)

    # ── embedding ──────────────────────────────────────────────────────────────

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Fetch embedding from Ollama and return L2-normalised float32 vector."""
        start_time = time.time()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={
                        "model": OLLAMA_EMBED_MODEL, 
                        "prompt": text,
                        "keep_alive": "5m"
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                vec = resp.json().get("embedding")
                if not vec:
                    return None
                arr = np.array(vec, dtype=np.float32)
                norm = np.linalg.norm(arr)
                emb = arr / norm if norm > 0 else arr
                logger.debug(f"[Cache] Embedding took {time.time() - start_time:.3f}s")
                return emb
        except Exception as e:
            logger.error(f"[Cache] Embedding error (after {time.time() - start_time:.3f}s): {e}")
            return None

    # ── lookup ─────────────────────────────────────────────────────────────────

    async def find_similar(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Return the best cached entry whose query embedding is within
        self.similarity_threshold of the incoming query, or None.
        """
        start_time = time.time()
        # STEP 0: Clean up old entries (throttled)
        self._prune()

        if not self._entries:
            return None

        # STEP 1: Exact match lookup (extremely fast)
        query_clean = query.strip().lower()
        for entry in self._entries:
            if entry.get("query", "").strip().lower() == query_clean:
                logger.info(f"[Cache] EXACT HIT in {time.time() - start_time:.4f}s")
                return entry

        # STEP 2: Similarity search
        query_vec = await self._get_embedding(query)
        if query_vec is None:
            logger.warning("[Cache] Could not embed query — skipping similarity lookup")
            return None

        best_score = -1.0
        best_entry = None

        # Optimization: use pre-computed numpy arrays
        for entry in self._entries:
            eid = entry.get("id")
            if not eid: continue
            
            stored_vec = self._numpy_embeddings.get(eid)
            if stored_vec is None:
                # Fallback if not in memory for some reason
                emb = entry.get("embedding")
                if not emb: continue
                stored_vec = np.array(emb, dtype=np.float32)
                self._numpy_embeddings[eid] = stored_vec
            
            score = float(np.dot(query_vec, stored_vec))
            if score > best_score:
                best_score = score
                best_entry = entry

        duration = time.time() - start_time
        if best_score >= self.similarity_threshold:
            logger.info(f"[Cache] HIT  score={best_score:.4f} in {duration:.4f}s  query='{best_entry['query'][:60]}'")
            return best_entry
        else:
            logger.info(f"[Cache] MISS best_score={best_score:.4f} in {duration:.4f}s")
            return None

    # ── save ───────────────────────────────────────────────────────────────────

    async def add(self, query: str, answer: str, image_urls: List[str] = []) -> bool:
        """
        Embed the query, append a new entry, and persist to MongoDB.
        Returns True on success, False on failure.
        """
        # Always prune before adding to keep it tidy
        self._prune()

        embedding = await self._get_embedding(query)
        if embedding is None:
            logger.warning("[Cache] Could not embed query — entry NOT saved")
            return False

        entry_id = hashlib.sha256(query.encode()).hexdigest()[:16]
        entry = {
            "id": entry_id,
            "query": query,
            "answer": answer,
            "image_urls": image_urls,
            "embedding": embedding.tolist(),
            "timestamp": time.time(),
        }

        # Update in-memory collections
        self._entries = [e for e in self._entries if e.get("id") != entry_id]
        self._entries.append(entry)
        self._numpy_embeddings[entry_id] = embedding

        self._persist_entry(entry)
        return True

    # ── stats ──────────────────────────────────────────────────────────────────

    def count(self) -> int:
        return len(self._entries)

    def all_entries(self) -> List[Dict[str, Any]]:
        return [{"id": e["id"], "query": e["query"], "timestamp": e.get("timestamp", 0)} for e in self._entries]

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by its ID."""
        count_before = len(self._entries)
        self._entries = [e for e in self._entries if e.get("id") != entry_id]
        if len(self._entries) < count_before:
            self._delete_entries([entry_id])
            return True
        return False


# Global cache instance
query_cache: Optional[QueryCache] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Request / Response models
# ═══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class SaveCacheRequest(BaseModel):
    query: str
    answer: str
    image_urls: List[str] = []


class VideoRequest(BaseModel):
    """List of Azure Blob image URLs to stitch into a video clip."""
    image_urls: List[str]
    fps: int = 2          # frames per second in output video (default 2 = slow, good for surveillance stills)
    width: int = 1280     # output resolution width  (height auto-calculated to keep aspect ratio)
    quality: int = 23     # libx264 CRF — lower = better quality, bigger file (18–28 is a good range)

class ProcessVideoRequest(BaseModel):
    rtmp_url: str
    camera_id: str


# ═══════════════════════════════════════════════════════════════════════════════
# Startup / Shutdown
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    global rag_client, chat_graph, query_cache, chat_store, vision_orchestrator, alert_manager, alert_db, alert_task, tts_model, tts_voice_style

    print("=" * 70)
    print("🚀 STARTING VIDEO CHATBOT WEB SERVER (WITH QUERY CACHE)")
    print("=" * 70)

    # Init Supertonic TTS
    try:
        tts_model = TTS(auto_download=True)
        tts_voice_style = tts_model.get_voice_style(voice_name="M1")
        print("✅ Supertonic TTS initialized")
    except Exception as e:
        print(f"❌ Failed to initialize TTS: {e}")

    # Init alert system
    try:
        alert_db = alert_module.MongoDBDatabase()
        alert_manager = alert_module.AlertManager(alert_module.ALERT_RULES_FILE, alert_module.ALERTS_FILE)
        alert_task = asyncio.create_task(alert_module.check_alerts_background(alert_manager, alert_db))
        print("✅ Alert system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize alert system: {e}")

    # Init cache — MongoDB-backed (db.db_mongo, same "arcis" database
    # everything else uses), one-time-migrating any pre-existing on-disk
    # cache file into it.
    query_cache = QueryCache(
        db.db_mongo[QUERY_CACHE_COLLECTION],
        similarity_threshold=CACHE_SIMILARITY_THRESHOLD,
        migrate_from_file=CACHE_FILE,
    )
    print(f"✅ Query cache ready — {query_cache.count()} stored entries (MongoDB: {db.MONGO_DATABASE}.{QUERY_CACHE_COLLECTION})")

    # Init history store
    chat_store = ChatStore(HISTORY_FILE)
    print(f"✅ Chat history ready — {len(chat_store.get_sessions())} sessions ({HISTORY_FILE})")

    # Init RAG client
    try:
        rag_client = IntegratedVideoRAGClient()
        await rag_client.connect_to_mcp_server(MCP_SERVER_URL)
        chat_graph = build_chat_graph(rag_client)
        print("✅ RAG client initialized")
        print("✅ Chat routing graph compiled")
    except Exception as e:
        print(f"❌ Failed to initialize RAG client: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("shutdown")
async def shutdown_event():
    global rag_client
    if rag_client:
        await rag_client.cleanup()
        print("✅ RAG client cleanup complete")


# ═══════════════════════════════════════════════════════════════════════════════
# Cache save endpoint  (called by frontend after user confirms)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/cache/save")
async def save_to_cache(request: SaveCacheRequest):
    """Save a query+answer pair to the cache after user confirmation."""
    if not query_cache:
        raise HTTPException(status_code=500, detail="Cache not initialised")
    success = await query_cache.add(request.query, request.answer, request.image_urls)
    if success:
        return {"status": "saved", "total_entries": query_cache.count()}
    else:
        raise HTTPException(status_code=500, detail="Could not embed query — not saved")


@app.get("/api/cache/list")
async def list_cache():
    """Return all cached queries with IDs and timestamps."""
    if not query_cache:
        raise HTTPException(status_code=500, detail="Cache not initialised")
    return {"count": query_cache.count(), "entries": query_cache.all_entries()}


@app.delete("/api/cache/delete/{entry_id}")
async def delete_from_cache(entry_id: str):
    """Delete a specific entry from the cache."""
    if not query_cache:
        raise HTTPException(status_code=500, detail="Cache not initialised")
    success = query_cache.delete(entry_id)
    if success:
        return {"status": "deleted", "total_entries": query_cache.count()}
    else:
        raise HTTPException(status_code=404, detail="Entry not found")


# ═══════════════════════════════════════════════════════════════════════════════
# TTS Endpoint — Supertonic
# ═══════════════════════════════════════════════════════════════════════════════

class TTSRequest(BaseModel):
    text: str
    speed: float = 1.05

@app.post("/api/tts/speak")
async def tts_speak(request: TTSRequest):
    """
    Synthesize a sentence using Supertonic and return WAV audio bytes.
    Frontend calls this for each sentence as it streams in.
    """
    if not tts_model:
        raise HTTPException(status_code=500, detail="TTS model not initialized")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        # Run in thread so it doesn't block the async event loop
        loop = asyncio.get_event_loop()
        wav, duration = await loop.run_in_executor(
            None,
            lambda: tts_model.synthesize(
                text=text,
                lang="en",
                voice_style=tts_voice_style,
                total_steps=8,
                speed=request.speed,
            )
        )

        # Convert numpy array to WAV bytes in memory
        buf = io.BytesIO()
        sf.write(buf, wav.squeeze(), 44100, format="WAV")
        buf.seek(0)

        logger.info(f"[TTS] Synthesized {duration[0]:.2f}s for: '{text[:50]}'")

        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={"X-Audio-Duration": str(round(float(duration[0]), 3))}
        )

    except Exception as e:
        logger.error(f"[TTS] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# History endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/history/sessions")
async def get_sessions():
    if not chat_store:
        raise HTTPException(status_code=500, detail="History store not initialised")
    return {"sessions": chat_store.get_sessions()}


@app.get("/api/history/load/{session_id}")
async def load_session(session_id: str):
    if not chat_store:
        raise HTTPException(status_code=500, detail="History store not initialised")
    return {"messages": chat_store.get_messages(session_id)}


@app.delete("/api/history/delete/{session_id}")
async def delete_session(session_id: str):
    if not chat_store:
        raise HTTPException(status_code=500, detail="History store not initialised")
    success = chat_store.delete_session(session_id)
    return {"success": success}


# ═══════════════════════════════════════════════════════════════════════════════
# Main chat stream endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat/stream")
async def chat_stream(chat_request: ChatRequest, request: Request):
    """Stream chat response using SSE — with smart camera ID handling and cache lookup."""
    if not rag_client:
        raise HTTPException(status_code=500, detail="RAG client not initialized")
    if not chat_graph:
        raise HTTPException(status_code=500, detail="Chat routing graph not initialized")

    async def generate() -> AsyncGenerator[str, None]:
        try:
            logger.info(f"📥 Query: {chat_request.message} (session: {chat_request.session_id})")

            # Save user message to history
            if chat_store:
                chat_store.add_message(chat_request.session_id, "user", chat_request.message)

            # ── greeting shortcut ──────────────────────────────────────────────
            query_lower = chat_request.message.lower().strip("!?. ")
            greetings = ["hi", "hello", "hey", "namaste", "hola"]
            if query_lower in greetings:
                yield f"data: {json.dumps({'content': 'How can I help you?'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # ── "list cameras" bypass — only for pure listing queries ─────────
            query_lower_full = chat_request.message.lower().strip("!?. ")
            is_list_cameras = query_lower_full in (
                "list cameras", "list all cameras", "cameras", "show cameras", "all cameras", "available cameras", "list available cameras"
            )

            # ── "list recognized plates" bypass — same idea, for ANPR plates ──
            is_list_plates = query_lower_full in (
                "list plates", "list all plates", "list recognized plates",
                "list all recognized plates", "show all plates", "show all number plates",
                "show all number plates detected", "recognized plates", "all recognized plates",
            )
            is_list_bypass = is_list_cameras or is_list_plates

            if not is_list_bypass:
                # ── STEP 1: Check cache ────────────────────────────────────────
                if await request.is_disconnected():
                    logger.info("⏹️ Client disconnected before cache lookup")
                    return

                logger.info("[Cache] Checking cache...")
                start_cache = time.time()
                cached = await query_cache.find_similar(chat_request.message)
                logger.info(f"[Cache] Lookup finished in {time.time() - start_cache:.3f}s")

                if cached:
                    logger.info("[Cache] HIT - Returning cached answer")
                    yield f"data: {json.dumps({'content': '📦 *(Answering from cache — similar question found)'})}"
                    await asyncio.sleep(0.05)

                    answer = cached["answer"]
                    chunk_size = 50
                    for i in range(0, len(answer), chunk_size):
                        if await request.is_disconnected():
                            logger.info("⏹️ Client disconnected during cache stream")
                            return
                        chunk = answer[i : i + chunk_size]
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                        await asyncio.sleep(0.01)

                    # Also yield cached images if any
                    image_urls = cached.get("image_urls", [])
                    if image_urls:
                        for img_url in image_urls:
                            if await request.is_disconnected():
                                return
                            yield f"data: {json.dumps({'image_url': img_url})}\n\n"
                            await asyncio.sleep(0.01)

                    yield "data: [DONE]\n\n"
                    return

            # ── missing camera ID detection ───────────────────────────────────
            needs_camera = any(
                kw in query_lower_full
                for kw in ["hour", "segment", "search", "show", "get", "find", "footage", "video"]
            )
            has_camera_keyword = "camera" in query_lower_full
            has_camera_id = (
                has_camera_keyword
                and any(c.isdigit() or (c.isupper() and c.isalpha()) for c in chat_request.message)
            ) or any(word.isupper() and len(word) > 3 for word in chat_request.message.split())

            if needs_camera and not has_camera_id and "list" not in query_lower_full and "cameras" not in query_lower_full:
                logger.warning("⚠️ Query needs camera ID — auto-listing cameras")

                msg = "⚠️ **Camera ID Required**\n\nI need a camera ID to answer your question. Here are the available cameras:\n\n"
                yield f"data: {json.dumps({'content': msg})}\n\n"
                await asyncio.sleep(0.05)

                try:
                    list_result = await rag_client.process_query("List all cameras")
                    if list_result.get("status") == "success":
                        camera_list = list_result.get("final_answer", "")
                        chunk_size = 100
                        for i in range(0, len(camera_list), chunk_size):
                            if await request.is_disconnected():
                                return
                            yield f"data: {json.dumps({'content': camera_list[i:i+chunk_size]})}\n\n"
                            await asyncio.sleep(0.01)
                        help_msg = f"\n\n💡 **Please ask again with a camera ID:**\nExample: '{chat_request.message} from camera ATPL-908610-ARCIS'"
                        yield f"data: {json.dumps({'content': help_msg})}\n\n"
                    else:
                        yield f"data: {json.dumps({'content': 'Could not retrieve camera list. Please try listing cameras first.'})}\n\n"
                except Exception as e:
                    logger.error(f"Error listing cameras: {e}")
                    yield f"data: {json.dumps({'content': f'Error: {str(e)}'})}\n\n"

                yield "data: [DONE]\n\n"
                return

            # ── STEP 2: Process with LLM ──────────────────────────────────────
            # contextualize -> select_tool -> call_mcp -> generate_summary,
            # all run as one LangGraph (see graph.py). Real LLM text tokens
            # come out via stream_mode="messages"; everything else (headers,
            # image URLs, non-streaming tool 1/4/5 answers) comes out via a
            # plain asyncio.Queue side channel — stream_chat_graph merges
            # both into one ordered sequence of SSE payloads. See graph.py's
            # module docstring for why this two-channel approach is needed.
            try:
                initial_state: Dict[str, Any] = {"query": chat_request.message, "queue": asyncio.Queue()}

                if is_list_cameras:
                    logger.info("📋 'List cameras' detected — bypassing LLM routing")
                    initial_state["tool_name"] = "list_available_cameras"
                    initial_state["params"] = {}
                    initial_state["ctx_query"] = chat_request.message
                elif is_list_plates:
                    logger.info("🚗 'List recognized plates' detected — bypassing LLM routing")
                    initial_state["tool_name"] = "list_recognized_plates"
                    initial_state["params"] = {}
                    initial_state["ctx_query"] = chat_request.message
                else:
                    if await request.is_disconnected():
                        logger.info("⏹️ Client disconnected before chat graph")
                        return

                full_answer = ""
                image_urls: List[str] = []

                async for payload in stream_chat_graph(chat_graph, initial_state):
                    if await request.is_disconnected():
                        logger.info("⏹️ Client disconnected during graph stream")
                        return

                    if payload.get("__done__"):
                        full_answer = payload["full_answer"]
                        image_urls = payload["image_urls"]
                        continue

                    if "content" in payload:
                        yield f"data: {json.dumps({'content': payload['content']})}\n\n"
                    elif "image_url" in payload:
                        yield f"data: {json.dumps({'image_url': payload['image_url']})}\n\n"

                # Save assistant response to history
                if chat_store and full_answer.strip():
                    chat_store.add_message(chat_request.session_id, "assistant", full_answer, image_urls)

                # ── STEP 3: Offer to save to cache ────────────────────────────────
                if not is_list_bypass and full_answer.strip():
                    save_prompt = {
                        "type": "save_prompt",
                        "query": chat_request.message,
                        "answer": full_answer,
                        "image_urls": image_urls,
                    }
                    yield f"data: {json.dumps({'save_prompt': save_prompt})}\n\n"

                yield "data: [DONE]\n\n"
                return

            except asyncio.CancelledError:
                logger.info(f"⏹️ Generation CANCELLED for session: {chat_request.session_id}")
                raise
            except Exception as e:
                logger.error(f"❌ Processing Error: {str(e)}")
                import traceback
                traceback.print_exc()
                error_msg = f"❌ **Error**: {str(e)}\n\n"
                yield f"data: {json.dumps({'content': error_msg})}\n\n"
                yield "data: [DONE]\n\n"
                return

        except asyncio.CancelledError:
            logger.info(f"⏹️ Outer Generation CANCELLED for session: {chat_request.session_id}")
            raise
        except Exception as e:
            logger.error(f"💥 Exception: {str(e)}")
            import traceback
            traceback.print_exc()
            error_msg = f"❌ **Error**: {str(e)}\n\nPlease check the server logs for details."
            yield f"data: {json.dumps({'content': error_msg})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Vision Chat endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat/with_image")
async def chat_with_image(
    image: UploadFile = File(...),
    message: str = Form(...),
    session_id: str = Form("default")
):
    """
    Process an image and a query.
    1. Qwen2.5-VL generates characteristics from the uploaded image (preserving intent).
    2. Search DB for segments based on those characteristics via specialized tool.
    3. Generate a summary using  gemma4:cloud.
    4. Detect the targeted object in retrieved images and show bounding boxes.
    """
    if not vision_orchestrator:
        raise HTTPException(status_code=500, detail="Vision orchestrator not initialized")

    async def generate() -> AsyncGenerator[str, None]:
        try:
            image_bytes = await image.read()
            
            # Save user message to history
            if chat_store:
                chat_store.add_message(session_id, "user", message)

            async for chunk_str in vision_orchestrator.process_query(image_bytes, message, rag_client=rag_client):
                # chunk_str is already a JSON string from vision_orchestrator.process_query
                yield f"data: {chunk_str}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error in vision chat: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════════════════════════
# Video Generation  (/api/generate_video)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/generate_video")
async def generate_video(request: VideoRequest):
    """
    Download a list of Azure Blob image URLs, stitch them into an MP4 using
    OpenCV (cv2), save the file to VIDEO_OUTPUT_DIR, and return a public URL.

    The frontend calls this when the user clicks "Generate Video Clip".
    POST body: { "image_urls": ["https://...", ...], "fps": 2 }
    Response:  { "video_url": "/videos/<uuid>.mp4", "frame_count": N }
    """
    import cv2

    image_urls: List[str] = request.image_urls
    fps: int = max(1, min(request.fps, 30))        # clamp 1–30
    target_width: int = request.width
    crf: int = request.quality

    if not image_urls:
        raise HTTPException(status_code=400, detail="No image URLs provided")

    if len(image_urls) > 500:
        raise HTTPException(status_code=400, detail="Too many frames (max 500)")

    logger.info(f"[Video] Generating video: {len(image_urls)} frames, {fps} fps")

    # ── 1. Download frames into a temp directory ──────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="vmukti_video_")
    try:
        frame_paths: List[str] = []

        headers = {
            "User-Agent": "Mozilla/5.0",
        }

        for idx, url in enumerate(image_urls):
            try:
                resp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda u=url: requests.get(u, headers=headers, timeout=15, stream=True)
                )
                resp.raise_for_status()
                frame_path = os.path.join(tmp_dir, f"frame_{idx:05d}.jpg")
                with open(frame_path, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                frame_paths.append(frame_path)
                logger.debug(f"[Video] Downloaded frame {idx + 1}/{len(image_urls)}")
            except Exception as e:
                logger.warning(f"[Video] Skipping frame {idx} ({url[:60]}…): {e}")

        if not frame_paths:
            raise HTTPException(status_code=502, detail="Could not download any frames from the provided URLs")

        # ── 2. Read first valid frame to get resolution ───────────────────────
        first_img = None
        for fp in frame_paths:
            img = cv2.imread(fp)
            if img is not None:
                first_img = img
                break

        if first_img is None:
            raise HTTPException(status_code=502, detail="Downloaded images could not be decoded by OpenCV")

        orig_h, orig_w = first_img.shape[:2]
        # Maintain aspect ratio
        scale = target_width / orig_w
        out_w = target_width
        out_h = int(orig_h * scale)
        # Ensure even dimensions (required by libx264)
        out_w = out_w if out_w % 2 == 0 else out_w + 1
        out_h = out_h if out_h % 2 == 0 else out_h + 1

        logger.info(f"[Video] Output resolution: {out_w}x{out_h}, fps={fps}")

        # ── 3. Write MP4 ──────────────────────────────────────────────────────
        video_filename = f"{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(VIDEO_OUTPUT_DIR, video_filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (out_w, out_h))

        if not writer.isOpened():
            raise HTTPException(status_code=500, detail="OpenCV VideoWriter failed to open — check codec support")

        good_frames = 0
        for fp in frame_paths:
            img = cv2.imread(fp)
            if img is None:
                continue
            # Resize to target resolution
            img_resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(img_resized)
            good_frames += 1

        writer.release()

        if good_frames == 0:
            if os.path.exists(video_path):
                os.remove(video_path)
            raise HTTPException(status_code=500, detail="No valid frames written — video not created")

        logger.info(f"[Video] ✅ Done — {good_frames} frames written to {video_path}")

        return {
            "status": "ok",
            "video_url": f"/videos/{video_filename}",
            "frame_count": good_frames,
            "resolution": f"{out_w}x{out_h}",
            "fps": fps,
            "message": f"Video clip created from {good_frames} frames."
        }

    finally:
        # Always clean up temp dir
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "rag_client_ready": rag_client is not None,
        "cache_entries": query_cache.count() if query_cache else 0,
        "cache_collection": f"{db.MONGO_DATABASE}.{QUERY_CACHE_COLLECTION}",
        "ollama_configured": True,
    }

def _get_embedding_sync(text: str) -> List[float]:
    """Synchronous wrapper for Ollama embedding."""
    try:
        payload = {"model": OLLAMA_EMBED_MODEL, "prompt": f"search_document: {text}"}
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json().get("embedding", [])
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        return []

@app.post("/api/process_video")
async def process_video_endpoint(req: ProcessVideoRequest):
    """
    Process an external video source and stream results immediately via SSE.
    """
    url = req.rtmp_url
    camera_id = req.camera_id
    azure_client = get_azure_client()
    
    if not azure_client:
        raise HTTPException(status_code=500, detail="Azure Storage not configured")

    logger.info(f"Streaming video processing: {url} for camera {camera_id}")

    def event_stream():
        try:
            for segment in _process_video_generator(url, camera_id, azure_client):
                yield f"data: {json.dumps(segment)}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

def _process_video_generator(url: str, camera_id: str, azure_client: Any):
    logger.info(f"[_process_video_generator] Opening VideoCapture (FFMPEG) for: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        logger.error(f"[_process_video_generator] Failed to open: {url}")
        raise RuntimeError(f"Cannot open video source: {url}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    is_stream = total_frames <= 0
    duration_to_process = 30 if is_stream else (total_frames / fps)
    
    max_segments = 3
    segment_duration = 10
    
    global alert_db

    try:
        for seg_id in range(max_segments):
            start_time = datetime.now()
            sampled_frames = []
            
            frames_to_read = int(segment_duration * fps)
            for i in range(frames_to_read):
                ret, frame = cap.read()
                if not ret: break
                if i == int(frames_to_read * 0.2) or i == int(frames_to_read * 0.5):
                    sampled_frames.append(frame.copy())
            
            if len(sampled_frames) < 2: break
            
            frame_urls = []
            raw_buffers = []
            for i, frame in enumerate(sampled_frames, 1):
                url_azure = upload_frame_to_azure(azure_client, camera_id, seg_id, frame, f"frame{i}", start_time)
                if url_azure: frame_urls.append(url_azure)
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                raw_buffers.append(buf)
                
            description = get_segment_description(raw_buffers)
            
            # Display description in terminal immediately
            print(f"\n{'='*80}")
            print(f"🤖 SEGMENT VLM - #{seg_id} - Camera: {camera_id}")
            print(f"⏰ Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🖼️ Frames: {', '.join(frame_urls)}")
            print(f"{'-'*80}")
            print(description)
            print(f"{'='*80}\n")
            
            embedding = _get_embedding_sync(description)
            
            seg_data = {
                "segment_id": seg_id,
                "camera_id": camera_id,
                "start_time": start_time,
                "end_time": datetime.now(),
                "cumulative_minutes": (seg_id * segment_duration) / 60,
                "frame_urls": frame_urls,
                "description": description,
                "embedding": embedding,
                "timestamp": start_time
            }
            
            if alert_db:
                saved = alert_db.save_segment(seg_data)
                if saved:
                    frontend_item = seg_data.copy()
                    if "embedding" in frontend_item: del frontend_item["embedding"]
                    frontend_item["start_time"] = frontend_item["start_time"].isoformat()
                    frontend_item["end_time"] = frontend_item["end_time"].isoformat()
                    frontend_item["timestamp"] = frontend_item["timestamp"].isoformat()
                    frontend_item["start_time_str"] = start_time.strftime('%Y-%m-%d %H:%M:%S')
                    yield frontend_item
            
            if not is_stream and (seg_id + 1) * segment_duration >= duration_to_process:
                break
    finally:
        cap.release()

@app.get("/api/segments")
async def get_segments(skip: int = 0, limit: int = 5, camera: str = "All",
                       date: str = "All", q: str = "", facets: bool = False):
    """Paginated, filtered segment listing for segments.html.

    Mirrors exactly what the page's refreshData() sends (skip/limit/camera/
    date/q, plus a one-off facets=true) and expects back (segments, total,
    and cameras/dates when facets was requested) — filtering and paging
    happen in MongoDB via db.load_segments_page, not by shipping the whole
    6000+-document collection down on every page turn.
    """
    segments, total = db.load_segments_page(
        skip=skip, limit=limit, camera=camera, date=date, query=q
    )
    result = {"segments": segments, "total": total}
    if facets:
        cameras, dates = db.segment_facets()
        result["cameras"] = cameras
        result["dates"] = dates
    return result

# ── Alert API Routes ─────────────────────────────────────────────────────────

@app.get("/api/alerts")
async def get_alerts():
    if not alert_manager:
        raise HTTPException(status_code=500, detail="Alert manager not initialized")
    return alert_manager.alerts[:100]

@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: str):
    if not alert_manager:
        raise HTTPException(status_code=500, detail="Alert manager not initialized")
    success = alert_manager.delete_alert(alert_id)
    if success: return {"status": "success"}
    raise HTTPException(status_code=404, detail="Alert not found")

@app.get("/api/alert_rules")
async def get_alert_rules():
    if not alert_manager:
        raise HTTPException(status_code=500, detail="Alert manager not initialized")
    return alert_manager.rules

@app.get("/api/cameras")
async def get_cameras():
    if not alert_db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return alert_db.get_all_cameras()

@app.get("/api/dates")
async def get_dates():
    if not alert_db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return alert_db.get_all_dates()

@app.get("/api/alert_segments")
async def get_alert_segments(camera_id: Optional[str] = None, date: Optional[str] = None, keyword: Optional[str] = None):
    if not alert_db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    from datetime import time as dt_time
    if date and date != "ALL":
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
        start_dt = datetime.combine(target_date, dt_time.min).replace(tzinfo=alert_module.DATA_TZ)
        end_dt = datetime.combine(target_date, dt_time.max).replace(tzinfo=alert_module.DATA_TZ)
    else:
        start_dt = end_dt = None

    if keyword:
        segs = alert_db.search_by_keywords(keyword, camera_id, filter_start_dt=start_dt, filter_end_dt=end_dt, max_results=500)
    else:
        segs = alert_db.get_segments_by_time(camera_id, start_dt, end_dt, max_results=500)

    return [{"segment_id": s.segment_id, "camera_id": s.camera_id, "start_time": s.start_time.isoformat(), "end_time": s.end_time.isoformat(), "description": s.description, "frame_urls": s.frame_urls} for s in segs]

@app.get("/alerts", response_class=HTMLResponse)
async def alerts_dashboard():
    """Alert dashboard UI."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "alert_page.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            content = f.read()
            # Fix API path in HTML if needed (our history segments use /api/alert_segments instead of /api/segments)
            content = content.replace('fetch(`/api/segments?${params.toString()}`)', 'fetch(`/api/alert_segments?${params.toString()}`)')
            return HTMLResponse(content=content)
    return HTMLResponse(content="Alert page HTML not found.")

@app.post("/api/segments/delete")
async def delete_segments(request: Request):
    """Delete multiple segments."""
    data = await request.json()
    to_delete = data.get("to_delete", []) # List of [camera_id, segment_id]
    count, errors = db.delete_segments_from_mongo(to_delete)
    return {"count": count, "errors": errors}

@app.get("/segments", response_class=HTMLResponse)
async def segments_ui():
    """Serve the standalone segments management UI."""
    segments_file = os.path.join(os.path.dirname(__file__), "segments.html")
    if os.path.exists(segments_file):
        with open(segments_file, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="Segments UI file not found.")


# ═══════════════════════════════════════════════════════════════════════════════
# Frontend
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = os.path.join(os.path.dirname(__file__), "chat_interface.html")
    if os.path.exists(html_file):
        with open(html_file, "r") as f:
            return HTMLResponse(content=f.read())

    return HTMLResponse(content=r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Segment Chatbot</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            width: 100%;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 1.8em; font-weight: 700; }
        .header-btn {
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            margin-left: 10px;
            font-family: inherit;
            transition: background 0.2s;
        }
        .header-btn:hover { background: rgba(255,255,255,0.35); }
        .chat-area {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
        }
        .message.user { justify-content: flex-end; }
        .message-content {
            max-width: 80%;
            padding: 15px 20px;
            border-radius: 15px;
            line-height: 1.6;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .message.assistant .message-content {
            background: white;
            color: #333;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .message.assistant .message-content h1,
        .message.assistant .message-content h2,
        .message.assistant .message-content h3 { margin-top: 8px; margin-bottom: 4px; }
        .message.assistant .message-content ul,
        .message.assistant .message-content ol { padding-left: 20px; margin: 4px 0; }

        /* ── Cache save banner ── */
        .cache-banner {
            margin-top: 14px;
            padding: 12px 16px;
            background: #f0f4ff;
            border: 1px solid #c7d2fe;
            border-radius: 10px;
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }
        .cache-banner span { flex: 1; font-size: 0.88em; color: #4338ca; min-width: 200px; }
        .cache-yes, .cache-no {
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.85em;
            font-family: inherit;
            cursor: pointer;
            border: none;
            font-weight: 600;
            transition: opacity 0.2s, transform 0.1s;
        }
        .cache-yes { background: #667eea; color: white; }
        .cache-no  { background: #e5e7eb; color: #374151; }
        .cache-yes:hover { opacity: 0.88; }
        .cache-no:hover  { opacity: 0.78; }
        .cache-yes:active, .cache-no:active { transform: scale(0.96); }
        .cache-saved-label {
            font-size: 0.85em;
            color: #059669;
            font-weight: 600;
        }

        .typing-indicator {
            display: none;
            padding: 10px 15px;
            background: white;
            border-radius: 15px;
        }
        .typing-indicator.active { display: block; }
        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #667eea;
            margin: 0 2px;
            animation: typing 1.4s infinite;
        }
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        .input-area {
            padding: 20px 30px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }
        #userInput {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 1em;
            font-family: inherit;
            outline: none;
            transition: border-color 0.2s;
        }
        #userInput:focus { border-color: #667eea; }
        #sendBtn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-family: inherit;
            font-weight: 600;
            transition: opacity 0.2s, transform 0.1s;
        }
        #sendBtn:hover { opacity: 0.9; }
        #sendBtn:active { transform: scale(0.97); }
        #sendBtn:disabled { opacity: 0.5; cursor: not-allowed; }

        /* ── Image gallery ── */
        .image-gallery { margin-top: 16px; padding-top: 12px; border-top: 1px solid #eee; }
        .image-gallery-label { font-size: 0.8em; font-weight: 600; color: #667eea; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px; }
        .image-thumb-wrapper { position: relative; border-radius: 10px; overflow: hidden; aspect-ratio: 16/10; background: #e9ecef; box-shadow: 0 2px 8px rgba(0,0,0,0.08); cursor: pointer; transition: transform 0.2s ease, box-shadow 0.2s ease; }
        .image-thumb-wrapper:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(102,126,234,0.25); }
        .image-thumb-wrapper img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform 0.3s ease; }
        .image-thumb-wrapper:hover img { transform: scale(1.05); }
        .image-thumb-overlay { position: absolute; inset: 0; background: linear-gradient(to top, rgba(0,0,0,0.5) 0%, transparent 50%); opacity: 0; transition: opacity 0.2s ease; display: flex; align-items: flex-end; padding: 8px; }
        .image-thumb-wrapper:hover .image-thumb-overlay { opacity: 1; }
        .image-thumb-overlay span { color: white; font-size: 0.75em; font-weight: 500; }

        /* ── Lightbox ── */
        .lightbox-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 9999; align-items: center; justify-content: center; backdrop-filter: blur(4px); animation: fadeIn 0.2s ease; }
        .lightbox-overlay.active { display: flex; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .lightbox-content { position: relative; max-width: 90vw; max-height: 85vh; }
        .lightbox-content img { max-width: 90vw; max-height: 85vh; border-radius: 12px; box-shadow: 0 8px 40px rgba(0,0,0,0.5); object-fit: contain; }
        .lightbox-close { position: fixed; top: 20px; right: 30px; color: white; font-size: 2.5em; cursor: pointer; background: rgba(255,255,255,0.15); border: none; border-radius: 50%; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; transition: background 0.2s; line-height: 1; }
        .lightbox-close:hover { background: rgba(255,255,255,0.3); }
        .lightbox-nav { position: fixed; top: 50%; transform: translateY(-50%); color: white; font-size: 2.5em; cursor: pointer; background: rgba(255,255,255,0.15); border: none; border-radius: 50%; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; transition: background 0.2s; }
        .lightbox-nav:hover { background: rgba(255,255,255,0.3); }
        .lightbox-prev { left: 20px; }
        .lightbox-next { right: 20px; }
        .lightbox-counter { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%); color: rgba(255,255,255,0.8); font-size: 0.9em; font-weight: 500; background: rgba(0,0,0,0.4); padding: 6px 16px; border-radius: 20px; }

        .templates-area { padding: 10px 30px; background: #f1f3f9; display: flex; gap: 10px; flex-wrap: wrap; border-top: 1px solid #e0e0e0; }
        .template-btn { background: white; border: 1px solid #667eea; color: #667eea; padding: 6px 15px; border-radius: 20px; font-size: 0.85em; cursor: pointer; font-family: inherit; transition: all 0.2s; }
        .template-btn:hover { background: #667eea; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎥 Video Summary Chatbot</h1>
            <div>
                <button class="header-btn" onclick="window.location.href='http://0.0.0.0:8090'">🚨 Alert</button>
                <button class="header-btn" onclick="listCameras()">📹 Cameras</button>
                <button class="header-btn" onclick="viewSegments()">📹 Segments</button>
                <button class="header-btn" onclick="viewCache()">🗃️ Cache</button>
            </div>
        </div>

        <div class="chat-area" id="chatArea">
            <div class="message assistant">
                <div class="message-content">
                    <h2>👋 Welcome!</h2>
                    <p>Ask me about your camera footage. Try:</p>
                    <ul>
                        <li>"Show me last 1 hour" — I'll list cameras for you</li>
                        <li>"List all cameras"</li>
                    </ul>
                </div>
            </div>
            <div class="typing-indicator" id="typingIndicator">
                <span></span><span></span><span></span>
            </div>
        </div>

        <div class="templates-area" id="templatesArea">
            <button class="template-btn" onclick="setPrompt('List all cameras')">📋 List Cameras</button>
            <button class="template-btn" onclick="setPrompt('List all recognized plates')">🚗 List Plates</button>
            <button class="template-btn" onclick="setPrompt('Provide me a summary of last 15 minutes cameraid ANYK-805961-AAAAA')">🕒 Last 15m Summary</button>
            <button class="template-btn" onclick="setPrompt('have you seen any white car in last 15 minutes in cameraid ANYK-804268-AAAAA')">🚗 White Car (15m)</button>
            <button class="template-btn" onclick="setPrompt('have you seen any military person in last 15 minutes in cameraid ANYK-804268-AAAAA')">👮 Military Person (15m)</button>
            <button class="template-btn" onclick="setPrompt('have you seen any SUV car in last 15 minutes in cameraid ANYK-804268-AAAAA')">🚙 SUV Car (15m)</button>
        </div>

        <div class="input-area">
            <input type="text" id="userInput" placeholder="Ask about your camera footage..." onkeypress="if(event.key==='Enter') sendMessage()">
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <!-- Lightbox Modal -->
    <div class="lightbox-overlay" id="lightboxOverlay" onclick="closeLightbox(event)">
        <button class="lightbox-close" onclick="closeLightbox(event)">&times;</button>
        <button class="lightbox-nav lightbox-prev" onclick="navigateLightbox(event, -1)">&#8249;</button>
        <div class="lightbox-content">
            <img id="lightboxImage" src="" alt="Full size frame">
        </div>
        <button class="lightbox-nav lightbox-next" onclick="navigateLightbox(event, 1)">&#8250;</button>
        <div class="lightbox-counter" id="lightboxCounter"></div>
    </div>

    <script>
        var chatArea     = document.getElementById('chatArea');
        var userInput    = document.getElementById('userInput');
        var sendBtn      = document.getElementById('sendBtn');
        var typingIndicator = document.getElementById('typingIndicator');

        var lightboxUrls  = [];
        var lightboxIndex = 0;

        /* ── Prompt helpers ── */
        function setPrompt(text) { userInput.value = text; userInput.focus(); }

        function viewCache() {
            fetch('/api/cache/list').then(r => r.json()).then(data => {
                var msg = '🗃️ **Cache (' + data.count + ' entries)**\n\n';
                if (!data.entries || data.entries.length === 0) {
                    msg += '_No cached queries yet._';
                } else {
                    data.entries.forEach(function(e, i) {
                        msg += (i+1) + '. ' + e.query.substring(0, 80) + (e.query.length > 80 ? '…' : '') + '\n';
                    });
                }
                addMessage(msg, false, null);
            });
        }

        function viewSegments() {
            window.open('/segments', '_blank');
        }

        /* ── Lightbox ── */
        function openLightbox(urls, index) {
            lightboxUrls = urls; lightboxIndex = index || 0;
            updateLightboxImage();
            document.getElementById('lightboxOverlay').classList.add('active');
            document.body.style.overflow = 'hidden';
        }
        function closeLightbox(e) {
            if (e) e.stopPropagation();
            if (e && e.target !== e.currentTarget && !e.target.classList.contains('lightbox-close')) return;
            document.getElementById('lightboxOverlay').classList.remove('active');
            document.body.style.overflow = '';
        }
        function navigateLightbox(e, dir) {
            e.stopPropagation();
            lightboxIndex = (lightboxIndex + dir + lightboxUrls.length) % lightboxUrls.length;
            updateLightboxImage();
        }
        function updateLightboxImage() {
            document.getElementById('lightboxImage').src = lightboxUrls[lightboxIndex];
            document.getElementById('lightboxCounter').textContent = (lightboxIndex+1) + ' / ' + lightboxUrls.length;
        }
        document.addEventListener('keydown', function(e) {
            var ov = document.getElementById('lightboxOverlay');
            if (!ov.classList.contains('active')) return;
            if (e.key === 'Escape') { ov.classList.remove('active'); document.body.style.overflow = ''; }
            if (e.key === 'ArrowLeft')  navigateLightbox(e, -1);
            if (e.key === 'ArrowRight') navigateLightbox(e, 1);
        });

        /* ── Image URL extraction ── */
        function extractImageUrls(text) {
            var regex = /https:\/\/nvrdatashinobi\.blob\.core\.windows\.net[^\s\n<>)}\]]*[^\s\n<>)}\].,;:!?'"]/g;
            var matches = text.match(regex);
            if (!matches) return [];
            var seen = {}, unique = [];
            for (var i = 0; i < matches.length; i++) {
                if (!seen[matches[i]]) { seen[matches[i]] = true; unique.push(matches[i]); }
            }
            return unique;
        }

        /* ── Markdown ── */
        function parseMarkdown(text) {
            text = text.replace(/🖼️[^\n]*https:\/\/nvrdatashinobi[^\n]*/g, '');
            text = text.replace(/https:\/\/nvrdatashinobi\.blob\.core\.windows\.net[^\s\n<>)}\]]*[^\s\n<>)}\].,;:!?'"]/g, '');
            text = text.replace(/🖼️\s*(Images|Frame\s*\d*)\s*:?\s*(\[Available\])?\s*/g, '');
            text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>');
            text = text.replace(/^## (.*$)/gim,  '<h2>$1</h2>');
            text = text.replace(/^# (.*$)/gim,   '<h1>$1</h1>');
            text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            text = text.replace(/^- (.*$)/gim,   '<li>$1</li>');
            text = text.replace(/\n/g, '<br>');
            text = text.replace(/(<br>\s*){3,}/g, '<br><br>');
            return text;
        }

        /* ── Add message ── */
        function addMessage(content, isUser, savePromptData) {
            isUser = isUser || false;
            var messageDiv  = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user' : 'assistant');
            var contentDiv  = document.createElement('div');
            contentDiv.className = 'message-content';

            if (isUser) {
                contentDiv.textContent = content;
            } else {
                var imageUrls = extractImageUrls(content);
                contentDiv.innerHTML = parseMarkdown(content);

                if (imageUrls.length > 0) {
                    var gallery   = document.createElement('div');
                    gallery.className = 'image-gallery';
                    var label = document.createElement('div');
                    label.className = 'image-gallery-label';
                    label.textContent = '📷 Captured Frames (' + imageUrls.length + ')';
                    gallery.appendChild(label);
                    var grid = document.createElement('div');
                    grid.className = 'image-grid';
                    var showCount = Math.min(imageUrls.length, 6);
                    for (var i = 0; i < showCount; i++) grid.appendChild(createThumb(imageUrls, i));
                    gallery.appendChild(grid);

                    if (imageUrls.length > 6) {
                        var extraGrid = document.createElement('div');
                        extraGrid.className = 'image-grid';
                        extraGrid.style.display = 'none';
                        extraGrid.style.marginTop = '10px';
                        for (var j = 6; j < imageUrls.length; j++) extraGrid.appendChild(createThumb(imageUrls, j));
                        gallery.appendChild(extraGrid);
                        var toggleBtn = document.createElement('button');
                        toggleBtn.textContent = 'Show all ' + imageUrls.length + ' images ▼';
                        toggleBtn.style.cssText = 'margin-top:10px;background:none;border:1px solid #667eea;color:#667eea;padding:6px 16px;border-radius:20px;cursor:pointer;font-size:0.85em;font-family:inherit;transition:all 0.2s;';
                        toggleBtn.onmouseenter = function(){ this.style.background='#667eea'; this.style.color='white'; };
                        toggleBtn.onmouseleave = function(){ this.style.background='none'; this.style.color='#667eea'; };
                        toggleBtn.onclick = function() {
                            if (extraGrid.style.display === 'none') {
                                extraGrid.style.display = 'grid';
                                toggleBtn.textContent = 'Show less ▲';
                            } else {
                                extraGrid.style.display = 'none';
                                toggleBtn.textContent = 'Show all ' + imageUrls.length + ' images ▼';
                            }
                        };
                        gallery.appendChild(toggleBtn);
                    }
                    contentDiv.appendChild(gallery);
                }

                /* ── Cache save banner ── */
                if (savePromptData) {
                    var banner = document.createElement('div');
                    banner.className = 'cache-banner';
                    var label2 = document.createElement('span');
                    label2.textContent = '💾 Save this answer to cache so identical questions skip the AI?';
                    banner.appendChild(label2);

                    var yesBtn = document.createElement('button');
                    yesBtn.className = 'cache-yes';
                    yesBtn.textContent = '✅ Yes, save';
                    yesBtn.onclick = function() {
                        fetch('/api/cache/save', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ query: savePromptData.query, answer: savePromptData.answer })
                        }).then(function(r) { return r.json(); }).then(function(data) {
                            banner.innerHTML = '';
                            var saved = document.createElement('span');
                            saved.className = 'cache-saved-label';
                            saved.textContent = '✅ Saved to cache! (' + data.total_entries + ' total entries)';
                            banner.appendChild(saved);
                        }).catch(function() {
                            banner.innerHTML = '<span style="color:#dc2626;font-size:0.85em">❌ Save failed — check server logs.</span>';
                        });
                    };

                    var noBtn = document.createElement('button');
                    noBtn.className = 'cache-no';
                    noBtn.textContent = '✗ No thanks';
                    noBtn.onclick = function() { banner.remove(); };

                    banner.appendChild(yesBtn);
                    banner.appendChild(noBtn);
                    contentDiv.appendChild(banner);
                }
            }

            messageDiv.appendChild(contentDiv);
            chatArea.insertBefore(messageDiv, typingIndicator);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function createThumb(urls, index) {
            var wrapper = document.createElement('div');
            wrapper.className = 'image-thumb-wrapper';
            wrapper.onclick = function() { openLightbox(urls, index); };
            var img = document.createElement('img');
            img.src = urls[index]; img.alt = 'Frame ' + (index+1); img.loading = 'lazy';
            img.onerror = function() {
                this.style.display = 'none';
                wrapper.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#999;font-size:0.8em;">Image unavailable</div>';
            };
            var overlay = document.createElement('div');
            overlay.className = 'image-thumb-overlay';
            var span = document.createElement('span');
            span.textContent = '🔍 Click to enlarge';
            overlay.appendChild(span);
            wrapper.appendChild(img);
            wrapper.appendChild(overlay);
            return wrapper;
        }

        /* ── Send message ── */
        async function sendMessage() {
            var message = userInput.value.trim();
            if (!message) return;

            addMessage(message, true, null);
            userInput.value = '';
            typingIndicator.classList.add('active');
            sendBtn.disabled = true;

            try {
                var response = await fetch('/api/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });

                var reader   = response.body.getReader();
                var decoder  = new TextDecoder();
                var fullResponse = '';
                var pendingSavePrompt = null;
                var buffer = '';

                while (true) {
                    var result = await reader.read();
                    if (result.done) break;

                    buffer += decoder.decode(result.value, { stream: true });

                    // Process all complete SSE events (terminated by \n\n)
                    var parts = buffer.split('\n\n');
                    buffer = parts.pop(); // keep the incomplete tail

                    for (var p = 0; p < parts.length; p++) {
                        var lines = parts[p].split('\n');
                        for (var i = 0; i < lines.length; i++) {
                            var line = lines[i];
                            if (!line.startsWith('data: ')) continue;
                            var data = line.slice(6).trim();
                            if (data === '[DONE]') continue;

                            try {
                                var parsed = JSON.parse(data);
                                if (parsed.content)      fullResponse += parsed.content;
                                // search_segments_by_activity (and a couple of
                                // other tools) send each verified frame's URL
                                // ONLY via this separate event, never merged
                                // into `content` — append it here so
                                // extractImageUrls() in addMessage() below
                                // still finds it; without this, verified
                                // frames were silently never rendered.
                                if (parsed.image_url)    fullResponse += parsed.image_url + '\n';
                                if (parsed.save_prompt)  pendingSavePrompt = parsed.save_prompt;
                            } catch(e) {
                                console.error('Parse error:', e, 'raw:', data);
                            }
                        }
                    }
                }

                // Flush any remaining buffer after stream ends
                if (buffer.trim()) {
                    var lines = buffer.split('\n');
                    for (var i = 0; i < lines.length; i++) {
                        var line = lines[i];
                        if (!line.startsWith('data: ')) continue;
                        var data = line.slice(6).trim();
                        if (data === '[DONE]') continue;
                        try {
                            var parsed = JSON.parse(data);
                            if (parsed.content)     fullResponse += parsed.content;
                            if (parsed.image_url)   fullResponse += parsed.image_url + '\n';
                            if (parsed.save_prompt) pendingSavePrompt = parsed.save_prompt;
                        } catch(e) {}
                    }
                }

                typingIndicator.classList.remove('active');
                addMessage(fullResponse, false, pendingSavePrompt);

            } catch (error) {
                console.error('Error:', error);
                typingIndicator.classList.remove('active');
                addMessage('Sorry, there was an error processing your request.', false, null);
            }

            sendBtn.disabled = false;
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function clearChat() {
            while (chatArea.children.length > 2) chatArea.removeChild(chatArea.firstChild);
        }
        function listCameras() { userInput.value = 'List all cameras'; sendMessage(); }
    </script>
</body>
</html>
    """)


if __name__ == "__main__":
    print("=" * 70)
    print("🌐 VIDEO CHATBOT WEB SERVER (WITH QUERY CACHE)")
    print("=" * 70)
    print("🚀 Server: http://localhost:8085")
    print(f"💾 Cache : MongoDB {db.MONGO_DATABASE}.{QUERY_CACHE_COLLECTION}")
    print(f"🔍 Similarity threshold: {CACHE_SIMILARITY_THRESHOLD}")
    print()
    print("Cache flow:")
    print("  1. Every query (except 'list cameras') → embed → cosine similarity check")
    print("  2. If similar cached query found → return cached answer instantly")
    print("  3. Otherwise → process with LLM → ask user to save to cache")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8085, log_level="info")