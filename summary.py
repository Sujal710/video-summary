"""
Ollama Summarization Service for Video Segments
Generates comprehensive summaries with image references
"""

import logging
import os
import shutil
import json
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple
import asyncio
from datetime import datetime

import cv2
import numpy as np
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv
from langsmith import traceable, get_current_run_tree
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

# Load .env here too — summary.py reads AZURE_* env vars at import time below,
# and client.py imports this module before it calls load_dotenv() itself.
load_dotenv()

logger = logging.getLogger(__name__)

# ── Folder to store images discarded by SAM3 verification ──
DISCARDED_FRAMES_DIR = "discarded_frames"
os.makedirs(DISCARDED_FRAMES_DIR, exist_ok=True)

# ── Azure (for uploading bbox-annotated copies of VLM-verified frames) ──
# Same storage account/container anpr.py and fr-video.py use.
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "nvrdatashinobi")
AZURE_BLOB_PREFIX = os.getenv("AZURE_BLOB_PREFIX", "live-record/frimages")

_azure_blob_service_client: Optional[BlobServiceClient] = None
_warned_no_azure_conn_string = False


def _get_azure_blob_service_client() -> Optional[BlobServiceClient]:
    """Returns None (silently, from the caller's point of view) whenever
    annotation upload can't happen — either the credential is missing or the
    client failed to construct. Both cases are logged here exactly once so a
    "SAM3 found it but no bounding box showed up" report isn't a silent
    fallback to the plain frame — it shows up in the logs instead."""
    global _azure_blob_service_client, _warned_no_azure_conn_string
    if _azure_blob_service_client is None:
        if not AZURE_CONNECTION_STRING:
            if not _warned_no_azure_conn_string:
                logger.warning(
                    "AZURE_CONNECTION_STRING is not set — bbox-annotated frames "
                    "cannot be uploaded, SAM3-verified frames will display as the "
                    "plain (unannotated) original instead."
                )
                _warned_no_azure_conn_string = True
            return None
        try:
            _azure_blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        except Exception as e:
            logger.error(f"Failed to initialize Azure blob client: {e}")
    return _azure_blob_service_client


def _azure_sas_query(connection_string: Optional[str]) -> str:
    """Pull the `SharedAccessSignature=...` segment out of the connection
    string so it can be appended to a plain blob URL for browser access —
    same trick anpr.py/fr-video.py use for their own uploaded frame URLs."""
    if not connection_string:
        return ""
    for part in connection_string.split(";"):
        if part.startswith("SharedAccessSignature="):
            return part[len("SharedAccessSignature="):]
    return ""


_AZURE_SAS_QUERY = _azure_sas_query(AZURE_CONNECTION_STRING)

# ── SAM3 (ultralytics) — open-vocabulary promptable segmentation used to
# verify + locate the searched subject in each candidate frame, replacing the
# earlier qwen2.5vl VLM check. ──
SAM3_MODEL_PATH = os.getenv("SAM3_MODEL_PATH", "/home/vmukti/Downloads/sujal_vmukti/sam3.pt")
SAM3_CONF_THRESHOLD = float(os.getenv("SAM3_CONF_THRESHOLD", "0.5"))

# VERIFY=true  -> ask SAM3 to locate the subject in each candidate frame, show
#                 only what it confirms. Slower (one inference per frame,
#                 serialized on the shared model) but the images are evidence
#                 for the sentence above them.
# VERIFY=false -> no SAM3 call at all: return the frames of the matching
#                 segments straight from the database. Instant, and nothing is
#                 silently discarded, but the images are only "what the search
#                 matched", not "what was checked".
VERIFY = os.getenv('VERIFY', 'true').strip().lower() in ('1', 'true', 'yes', 'on')

# Only applies when VERIFY=false. A broad query can match hundreds of frames,
# which would flood the page and every one of them would have to be
# downloaded by the browser.
MAX_UNVERIFIED_FRAMES = int(os.getenv('MAX_UNVERIFIED_FRAMES', '24'))

# Upper bound on how many candidate frames get an actual SAM3 call per query
# (VERIFY=true only). SAM3 calls are serialized on one shared model instance,
# so this is what actually bounds the wait.
MAX_VERIFY_FRAMES = int(os.getenv('MAX_VERIFY_FRAMES', '30'))

# Frame downloads (ahead of each SAM3 check) used to open one fresh
# httpx.AsyncClient per frame and fire all of them at once via asyncio.gather —
# up to MAX_VERIFY_FRAMES simultaneous fresh HTTPS connections to the same
# Azure blob host. That burst was observed to transiently ConnectTimeout on
# some/all frames even though the host is reachable (verified separately with
# a single request), most likely from exhausting local ephemeral ports/
# conntrack or a brief per-IP throttle. Capping concurrency and reusing one
# pooled client keeps the actual download load steady regardless of how many
# candidate frames a query has.
FRAME_DOWNLOAD_CONCURRENCY = int(os.getenv('FRAME_DOWNLOAD_CONCURRENCY', '6'))

_frame_http_client: Optional["httpx.AsyncClient"] = None


def _get_frame_http_client():
    """Lazily construct one shared, connection-pooled httpx.AsyncClient for
    all frame downloads, instead of a new client (and new TLS handshake) per
    frame."""
    global _frame_http_client
    if _frame_http_client is None:
        import httpx
        _frame_http_client = httpx.AsyncClient(
            timeout=15,
            limits=httpx.Limits(max_connections=FRAME_DOWNLOAD_CONCURRENCY, max_keepalive_connections=FRAME_DOWNLOAD_CONCURRENCY),
        )
    return _frame_http_client

# This GPU is shared with Ollama, which keeps whatever model it last used
# resident for `keep_alive` (5m in this file) after the call finishes — that
# residency is what starves SAM3's own one-time model-to-GPU load of free
# VRAM. When true (default), every SAM3 verification batch force-unloads
# whatever Ollama has loaded right before it needs the GPU, instead of the
# two ever running on it at the same time. Ollama reloads on-demand on its
# next call, so this only costs one extra load on whichever call comes after
# SAM3, not on the SAM3 call itself.
SAM3_UNLOAD_OLLAMA = os.getenv('SAM3_UNLOAD_OLLAMA', 'true').strip().lower() in ('1', 'true', 'yes', 'on')

# ── LlamaIndex context augmentation (retrieval ahead of the summary LLM) ──
# Per query, in-memory only: a VectorStoreIndex is built fresh from that
# call's segment descriptions, retrieved against once, then left to be
# garbage-collected when the function returns. Nothing is written to disk and
# nothing carries over to the next query — each call starts from scratch.
OLLAMA_EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text:v1.5')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
RETRIEVAL_TOP_K = int(os.getenv('RETRIEVAL_TOP_K', '40'))

_embed_model = None


def _get_embed_model():
    """Lazily construct the shared Ollama embedding client once. The client
    itself is just a reusable HTTP wrapper — only the per-query
    VectorStoreIndex built with it (see _retrieve_relevant_segments) is
    ephemeral."""
    global _embed_model
    if _embed_model is None:
        from llama_index.embeddings.ollama import OllamaEmbedding
        _embed_model = OllamaEmbedding(model_name=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    return _embed_model


def rank_frames_for_verification(segments: List[Dict[str, Any]],
                                 keywords: List[str]) -> List[str]:
    """Pick the frames most likely to actually show the subject, best first.

    Scores each segment by its own stored description, which is free, then
    caps the list.
    """
    kws = [str(k).strip().lower() for k in (keywords or []) if str(k).strip()]

    scored = []
    for seg in segments:
        urls = [u for u in (seg.get('frame_urls') or []) if u and u != 'N/A']
        if not urls:
            continue
        desc = (seg.get('description') or '').lower()

        hits = sum(1 for k in kws if k in desc)
        if kws and not hits:
            continue                        # nothing in common; not a candidate

        # Contact sheets are grids of thumbnails, so they read badly as the
        # single frame backing a sentence. Keep them last.
        ranked = sorted(urls, key=lambda u: ('sheet' in u.lower(), u))
        scored.append((hits, ranked))

    # One frame per segment first. Three frames of one moment used to fill the
    # whole cap, which left the images clustered on a couple of segments while
    # the answer text discussed several. Stable sort, so equal scores keep
    # search-result order.
    scored.sort(key=lambda t: -t[0])

    ordered, seen = [], set()
    for depth in range(max((len(r) for _, r in scored), default=0)):
        for _, ranked in scored:
            if depth < len(ranked) and ranked[depth] not in seen:
                seen.add(ranked[depth])
                ordered.append(ranked[depth])
        if len(ordered) >= MAX_VERIFY_FRAMES:
            break

    if len(ordered) > MAX_VERIFY_FRAMES:
        logger.info(f"🎯 verifying top {MAX_VERIFY_FRAMES} of {len(ordered)} "
                    f"candidate frames, one per segment first")
        ordered = ordered[:MAX_VERIFY_FRAMES]
    return ordered


_sam3_predictor = None


def _get_sam3_predictor(force_cpu: bool = False):
    """Lazily load the SAM3 semantic (text-promptable) predictor once.

    GPU requires LD_LIBRARY_PATH to NOT contain the system CUDA 12.9 toolkit path
    (~/.bashrc exports it) — that shadows the pip-bundled cuBLAS our torch build
    (cu128) was compiled against and breaks every batched GPU matmul with
    CUBLAS_STATUS_INVALID_VALUE (confirmed via isolated repro; identical on any
    dtype). mcp-main.py's entrypoint relaunches itself with that var stripped
    before this module (or torch) is ever imported. If summary.py ends up
    imported some other way where that guard didn't run, fall back to CPU
    instead of crashing.

    `force_cpu=True` rebuilds the predictor on CPU even if one already exists
    — used by _run_sam3 when the GPU has no free memory for SAM3's own
    weights (see there): ultralytics only moves the model onto its device on
    the FIRST set_image() call and caches that forever, so if that first
    load OOMs, every later call would otherwise retry and fail identically
    for the rest of the process's life. Rebuilding on CPU here breaks that
    loop — slower per frame, but keeps verification working.
    """
    global _sam3_predictor
    if _sam3_predictor is None or force_cpu:
        import torch
        from ultralytics.models.sam import SAM3SemanticPredictor

        bad_env = "cuda-12.9" in os.environ.get("LD_LIBRARY_PATH", "")
        use_gpu = torch.cuda.is_available() and not bad_env and not force_cpu
        if bad_env:
            logger.warning(
                "LD_LIBRARY_PATH still contains the system CUDA 12.9 toolkit path — "
                "SAM3 would crash on GPU (cuBLAS version mismatch). Falling back to CPU. "
                "Restart via mcp-main.py so its startup guard can strip it."
            )

        # fp16 on GPU only — halves memory footprint (this GPU is shared with
        # several long-lived Ollama models, so headroom is tight) with no
        # measured confidence loss; verified against CPU/fp32 baseline. CPU
        # fallback stays fp32 (some ops aren't implemented for half on CPU).
        overrides = dict(
            conf=SAM3_CONF_THRESHOLD, task="segment", mode="predict",
            model=SAM3_MODEL_PATH, half=use_gpu, device="cuda" if use_gpu else "cpu",
            save=False, verbose=False,
        )
        _sam3_predictor = SAM3SemanticPredictor(overrides=overrides)
        logger.info(f"SAM3 model loaded from {SAM3_MODEL_PATH} on {'GPU' if use_gpu else 'CPU'} (conf>={SAM3_CONF_THRESHOLD})")
    return _sam3_predictor


async def _unload_ollama_models_from_gpu():
    """Force whatever Ollama currently has resident on the GPU to unload,
    right before a SAM3 verification batch needs that VRAM.

    Ollama keeps a model loaded for `keep_alive` after its last use — 5m for
    every call in this codebase — so a chat/embedding call that finished
    seconds ago can still be sitting on the GPU when SAM3 tries its one-time
    model load, which is exactly what starved it into an OOM. This is a
    one-shot "get off the GPU" nudge, not a lock: it queries Ollama's
    `/api/ps` for whatever is currently loaded and asks each to unload via
    `keep_alive: 0`. Ollama reloads on-demand on its own next call, so the
    only cost is one extra load on whichever request comes after SAM3 —
    nothing needs to be explicitly restored here.

    Best-effort: any failure (Ollama unreachable, unexpected response shape)
    is logged and swallowed — this must never be the reason a query fails,
    it only ever tries to make room.
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/ps")
            resp.raise_for_status()
            models = [m.get("name") for m in resp.json().get("models", []) if m.get("name")]
            if not models:
                return
            for name in models:
                try:
                    await client.post(f"{OLLAMA_BASE_URL}/api/generate",
                                      json={"model": name, "keep_alive": 0})
                except Exception as e:
                    logger.warning(f"Could not unload Ollama model {name!r} before SAM3: {e}")
            logger.info(f"🧹 Unloaded Ollama model(s) from GPU before SAM3 verification: {models}")
    except Exception as e:
        logger.warning(f"Could not query/unload Ollama models before SAM3 verification: {e}")


class VideoSegmentSummarizer:
    """Summarize video segments using Ollama LLM"""

    def __init__(self, model: str = "llama3.1:latest"):
        self.model = model
        from ollama import AsyncClient
        self.client = AsyncClient()
        # LangChain-wrapped model, used only by _call_ollama_streaming —
        # LangGraph's stream_mode="messages"/astream_events only see tokens
        # from an actual LangChain chat model invocation (with `config`
        # forwarded into it), not from the raw ollama client above. The
        # non-streaming _call_ollama_async path is unaffected and keeps
        # using self.client directly.
        #
        # A fresh ChatOllama is constructed per call in _call_ollama_streaming
        # (not reused/bound here) because .bind(num_predict=..., temperature=...)
        # is broken in this langchain_ollama version — it passes those straight
        # through to the raw ollama client's chat() as top-level kwargs instead
        # of nesting them under `options`, raising "AsyncClient.chat() got an
        # unexpected keyword argument 'num_predict'" (confirmed empirically).
        # Setting them as ChatOllama constructor fields instead works correctly.
        # SAM3 is one shared model instance with mutable per-call state
        # (set_image caches image features); calls must be fully serialized.
        self.sam3_lock = asyncio.Lock()
        # Caps how many frame downloads run at once (see FRAME_DOWNLOAD_CONCURRENCY).
        self._frame_download_semaphore = asyncio.Semaphore(FRAME_DOWNLOAD_CONCURRENCY)
        logger.info(f"Ollama Summarizer initialized with model: {model} "
                    f"(frame verification: {'SAM3' if VERIFY else 'off'}, "
                    f"verify cap: {MAX_VERIFY_FRAMES}, unverified cap: {MAX_UNVERIFIED_FRAMES})")

    @traceable(name="ollama_chat", run_type="llm", project_name="video-summary")
    async def _call_ollama_async(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Shared Async Ollama chat call"""
        run = get_current_run_tree()
        if run:
            run.metadata.update({
                "component": "summary",
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            })
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_gpu": -1,
                },
                keep_alive="5m"
            )
            return response.message.content or "No content returned from AI."
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return f"Error generating summary: {str(e)}"

    @traceable(name="ollama_chat_streaming", run_type="llm", project_name="video-summary")
    async def _call_ollama_streaming(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 2000,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Same as _call_ollama_async but yields tokens as they arrive.

        Uses self.chat_model (LangChain's ChatOllama) instead of the raw
        ollama client, and forwards `config` into astream() — required for
        LangGraph's stream_mode="messages" to see these tokens when this is
        called from inside a graph node (see graph.py's generate_summary).
        `config` is None when called outside a graph (e.g. no caller passes
        one) — ChatOllama.astream(..., config=None) works the same as
        before, just with nothing to attach a graph tracer to.
        """
        run = get_current_run_tree()
        if run:
            run.metadata.update({"component": "summary", "model": self.model, "max_tokens": max_tokens})
        try:
            from langchain_ollama import ChatOllama
            model = ChatOllama(
                model=self.model, base_url=OLLAMA_BASE_URL,
                num_predict=max_tokens, temperature=0.7, num_gpu=-1, keep_alive="5m",
            )
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            async for chunk in model.astream(messages, config=config):
                token = chunk.content
                if token:
                    yield token
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            yield f"Error: {str(e)}"

    def format_segments_for_summary(self, segments: List[Dict[str, Any]]) -> str:
        """Format segment data for LLM consumption — no camera ID exposed"""
        formatted_parts = []
        for i, seg in enumerate(segments, start=1):
            formatted_parts.append(
                # ── CHANGE 1: Removed camera_id from the formatted text ──
                # Previously: f"**Segment {seg.get('segment_id')}** (Camera: {seg.get('camera_id')})"
                # Now: just a clean numbered entry with no camera reference
                f"**Segment {i}**\n"
                f"Description: {seg.get('description', 'N/A')}\n"
            )
        return "\n---\n\n".join(formatted_parts)

    # ── Context-budget helper ─────────────────────────────────────────────────

    def _budget_segments(
        self,
        segments: List[Dict[str, Any]],
        max_chars: int = 60_000,
    ) -> List[Dict[str, Any]]:
        """
        Return a representative subset of `segments` whose total serialised
        description text stays under `max_chars`.

        Segments are sampled *evenly* across the list rather than head-truncated,
        so the LLM always sees the full time spread instead of only the earliest
        events.  A warning is logged when trimming occurs.
        """
        if not segments:
            return segments

        # Estimate total chars from descriptions only
        total_chars = sum(len(s.get('description', '')) for s in segments)
        if total_chars <= max_chars:
            return segments

        # How many segments fit roughly within the budget?
        avg_chars = total_chars / len(segments)
        target_count = max(1, int(max_chars / avg_chars))

        # Evenly-spaced indices so we sample the whole time range
        step = len(segments) / target_count
        sampled = [segments[int(i * step)] for i in range(target_count)]

        logger.warning(
            f"⚠️  Context budget: {len(segments)} segments → {len(sampled)} sampled "
            f"(~{total_chars:,} chars → ~{max_chars:,} limit). "
            f"Results represent a spread of the full time range."
        )
        return sampled

    # ── Context augmentation (LlamaIndex retrieval) ───────────────────────────

    @traceable(name="llamaindex_retrieval", run_type="retriever", project_name="video-summary")
    def _retrieve_relevant_segments(
        self,
        segments: List[Dict[str, Any]],
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> List[Dict[str, Any]]:
        """Context augmentation step: index this call's segment descriptions
        with LlamaIndex and retrieve just the ones relevant to `query`.

        The VectorStoreIndex built here is local to this call — constructed
        from `segments`, retrieved against once, then dropped when the
        function returns. It is never reused or persisted, so the next query
        builds its own index from scratch.
        """
        run = get_current_run_tree()
        if run:
            run.metadata.update({
                "component": "retrieval",
                "embed_model": OLLAMA_EMBED_MODEL,
                "top_k": top_k,
                "num_segments_in": len(segments),
            })

        if not segments or not query:
            return segments

        try:
            from llama_index.core import VectorStoreIndex, Document
        except ImportError:
            logger.warning("llama-index not installed — skipping retrieval, using all segments")
            return segments

        docs, doc_segments = [], []
        for seg in segments:
            text = (seg.get('description') or '').strip()
            if not text:
                continue
            docs.append(Document(text=text, metadata={'idx': len(doc_segments)}))
            doc_segments.append(seg)

        if not docs:
            return segments

        try:
            index = VectorStoreIndex.from_documents(docs, embed_model=_get_embed_model())
            retriever = index.as_retriever(similarity_top_k=min(top_k, len(docs)))
            nodes = retriever.retrieve(query)
        except Exception as e:
            logger.warning(f"LlamaIndex retrieval failed, using all segments: {e}")
            return segments
        # `index` and `retriever` fall out of scope here — nothing about this
        # query's index survives past this call.

        if not nodes:
            return segments

        picked = [doc_segments[n.node.metadata['idx']] for n in nodes]
        logger.info(f"🔎 LlamaIndex retrieval: {len(segments)} segments → {len(picked)} "
                    f"relevant to query")
        return picked

    def _collect_frame_urls(self, segments: List[Dict[str, Any]]) -> str:
        """Collect frame URLs to ensure UI always has images to display"""
        urls = []
        for seg in segments:
            for url in seg.get('frame_urls', []):
                if url and url != 'N/A':
                    urls.append(url)

        if not urls:
            return ""

        url_lines = "\n".join(list(dict.fromkeys(urls)))
        return f"\n\n---\n📷 Captured Frames:\n{url_lines}\n"

    async def _async_verify_frames(self, urls: List[str], keywords: List[str]) -> List[str]:
        """Verify multiple frames in parallel using SAM3."""
        if SAM3_UNLOAD_OLLAMA:
            await _unload_ollama_models_from_gpu()
        tasks = [self._verify_frame_with_sam3_async(url, keywords) for url in urls]
        try:
            results = await asyncio.gather(*tasks)
            return [display_url for passed, display_url in results if passed]
        except Exception as e:
            logger.error(f"Async verification failed: {e}")
            return urls[:2]

    async def _async_verify_frames_streaming(self, urls: List[str], keywords: List[str]) -> AsyncGenerator[str, None]:
        """Verify frames in parallel using SAM3, yield each passing frame's
        (bbox-annotated, when available) URL immediately."""

        async def verify_and_return(url):
            try:
                passed, display_url = await self._verify_frame_with_sam3_async(url, keywords)
                if passed:
                    return display_url
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Frame verification error for {url}: {e}")
            return None

        if SAM3_UNLOAD_OLLAMA:
            await _unload_ollama_models_from_gpu()

        # Create tasks properly so we can cancel them if client disconnects
        tasks = [asyncio.create_task(verify_and_return(url)) for url in urls]

        try:
            # As e ach task finishes, yield its result
            for coro in asyncio.as_completed(tasks):
                result = await coro
                yield result
        finally:
            # If generator is closed (e.g., client disconnects), cancel remaining tasks
            for t in tasks:
                if not t.done():
                    t.cancel()
            
            # Briefly wait for cleanup
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def _save_discarded_frame(self, tmp_path: str, image_url: str, keywords: List[str], reason: str):
        """
        CHANGE 3: Save a rejected frame to the discarded folder for review.
        Filename encodes: timestamp + last part of URL + keywords used
        Example: 20240601_143022_frame3_red-car.jpg
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Extract filename from URL e.g. "frame3.jpg" from "http://server/frame3.jpg"
            url_filename = image_url.split('/')[-1].split('.')[0]  # "frame3"
            keyword_slug = "-".join(keywords)[:30]                  # "red-car"
            dest_filename = f"{timestamp}_{url_filename}_{keyword_slug}.jpg"
            dest_path = os.path.join(DISCARDED_FRAMES_DIR, dest_filename)
            shutil.copy2(tmp_path, dest_path)
            logger.info(f"🗑️ Discarded frame saved → {dest_path} | reason: {reason}")
        except Exception as e:
            logger.warning(f"Could not save discarded frame: {e}")

    def _upload_annotated_frame(self, img_bgr: np.ndarray, original_url: str) -> Optional[str]:
        """Upload a bbox-annotated copy of a verified frame to Azure as a
        separate blob (`..._bbox.jpg`) — the original frame_url stored in
        Mongo is left untouched, this is purely for display."""
        blob_client_factory = _get_azure_blob_service_client()
        if blob_client_factory is None:
            return None
        try:
            success, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                return None

            original_name = original_url.split('/')[-1].split('?')[0]
            stem, dot, ext = original_name.rpartition('.')
            filename = f"{stem or original_name}_bbox.{ext or 'jpg'}"
            blob_name = f"{AZURE_BLOB_PREFIX}/{filename}"

            blob_client = blob_client_factory.get_blob_client(
                container=AZURE_CONTAINER_NAME, blob=blob_name
            )
            blob_client.upload_blob(
                buf.tobytes(), overwrite=True,
                content_settings=ContentSettings(content_type='image/jpeg')
            )

            base_url = f"https://{blob_client_factory.account_name}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{AZURE_BLOB_PREFIX}"
            url = f"{base_url}/{filename}"
            return f"{url}?{_AZURE_SAS_QUERY}" if _AZURE_SAS_QUERY else url
        except Exception as e:
            logger.warning(f"Annotated frame upload failed, falling back to plain frame: {e}")
            return None

    def _draw_bboxes(self, img_bgr: np.ndarray, detections: List[Tuple[List[float], float]],
                     label: str) -> Optional[np.ndarray]:
        """Draw every detection in `detections` (each an absolute pixel xyxy
        box, already in this frame's own coordinate space — SAM3 returns
        boxes scaled to the original image) onto one copy of the frame.

        A "contact sheet" frame is a grid stitching several source frames
        into one image, so the subject can legitimately appear in more than
        one cell — drawing only the single best-confidence box left every
        other matching cell unannotated in the grid.
        """
        annotated = img_bgr.copy()
        h, w = annotated.shape[:2]
        drawn = 0
        for bbox_px, _conf in detections:
            x1 = int(max(0, min(w, bbox_px[0])))
            y1 = int(max(0, min(h, bbox_px[1])))
            x2 = int(max(0, min(w, bbox_px[2])))
            y2 = int(max(0, min(h, bbox_px[3])))
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label_y = max(y1 - 10, 15)
            cv2.putText(annotated, label[:40], (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            drawn += 1
        return annotated if drawn else None

    @staticmethod
    def _sam3_call_with_oom_retry(fn):
        """Run a zero-arg SAM3 call, retrying once after clearing the CUDA
        cache if it hits CUDA OOM. This GPU is shared with several long-lived
        Ollama model processes — a transient memory spike from one of them
        can OOM us for a moment even though the total we need reliably fits;
        clearing our own (fragmented, partly-unused) cache and retrying once
        recovers most of those instead of wrongly discarding a real match."""
        try:
            return fn()
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            logger.warning(f"SAM3 hit CUDA OOM, clearing cache and retrying once: {e}")
            import torch
            torch.cuda.empty_cache()
            return fn()

    def _run_sam3(self, img_bgr: np.ndarray, prompts: List[str]) -> Tuple[List[Tuple[List[float], float]], Optional[str]]:
        """Blocking SAM3 inference — run via asyncio.to_thread, never called
        directly from async code. Encodes the image once, then tries each
        prompt in turn (cheap: only the text-grounding decode reruns, the
        image features are cached by set_image) until one gets a detection
        above SAM3_CONF_THRESHOLD. Returns (detections, matched_prompt), where
        `detections` is EVERY box above threshold for that prompt, best-first
        — not just the single highest-confidence one, so a subject that
        appears more than once in the frame (e.g. every matching cell of a
        contact-sheet grid) gets every instance back, not only one."""
        predictor = _get_sam3_predictor()
        try:
            self._sam3_call_with_oom_retry(lambda: predictor.set_image(img_bgr))
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            # The model's very first move-to-GPU (inside set_image ->
            # setup_model) OOM'd even after the cache-clear retry — the GPU
            # simply has no free memory for SAM3's own weights right now.
            # ultralytics caches `self.model` only on success, so left alone
            # every future call here would retry this same doomed GPU load
            # and fail identically forever. Fall back to CPU instead.
            logger.error(
                "SAM3 has no free GPU memory to load onto (even after clearing "
                "cache) — falling back to CPU for the rest of this process. "
                "Verification will be slower per frame but will keep working."
            )
            import torch
            torch.cuda.empty_cache()
            predictor = _get_sam3_predictor(force_cpu=True)
            predictor.set_image(img_bgr)
        for prompt in prompts:
            results = self._sam3_call_with_oom_retry(lambda p=prompt: predictor(text=[p]))
            if not results:
                continue
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue
            confs = boxes.conf.tolist()
            xyxy = boxes.xyxy.tolist()
            detections = sorted(zip(xyxy, confs), key=lambda d: -d[1])
            if detections:
                return detections, prompt
        return [], None

    async def _verify_frame_with_sam3_async(self, image_url: str, keywords: List[str]) -> Tuple[bool, str]:
        """Ask SAM3 (open-vocabulary promptable segmentation) whether the
        subject appears in this frame and, if so, where. Returns
        (passed, display_url) — display_url points at a bbox-annotated copy
        when SAM3 located the subject and the upload succeeded, otherwise
        falls back to the original frame URL."""
        import tempfile, httpx
        try:
            # 1. Download image — concurrency-capped (FRAME_DOWNLOAD_CONCURRENCY)
            # and on a shared pooled client, with one retry on a transient
            # connect failure, instead of every candidate frame opening its
            # own connection to Azure blob storage all at once.
            client = _get_frame_http_client()
            async with self._frame_download_semaphore:
                try:
                    resp = await client.get(image_url)
                except (httpx.ConnectTimeout, httpx.ConnectError) as e:
                    logger.warning(f"⚠️ Frame download connect failed, retrying once: {image_url} ({e})")
                    resp = await client.get(image_url)
                if resp.status_code != 200:
                    logger.warning(f"⚠️ Could not download image: {image_url} (status {resp.status_code})")
                    return False, image_url
                content = resp.content

            # 2. Save to temp file (kept for _save_discarded_frame on a miss)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir='/tmp') as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                # Drop blank/whitespace-only entries — an empty element here left a
                # dangling "and" in the old VLM prompt/label and would waste a SAM3 call.
                clean_keywords = [k.strip() for k in keywords if k and k.strip()][:4]
                if not clean_keywords:
                    clean_keywords = ["subject"]

                # Upstream keyword lists are sometimes disjoint attribute fragments
                # (e.g. ["red", "car"] instead of "red car") rather than coherent
                # phrases — SAM3 queries one concept per call, so trying "red" and
                # "car" separately matches ANY red thing or ANY car, not a red car
                # (verified: SAM3 correctly rejects "red car"/"red" alone on a frame
                # with only a white car, but a lone "car" fallback still matched it —
                # so single-word fragments must never be tried standalone, only as
                # part of the joined phrase). Multi-word entries are kept as
                # independent fallbacks since those cases are already-coherent
                # synonym phrases (e.g. ["person drinking water", "water bottle"])
                # that joining into one string would turn nonsensical.
                joined_phrase = ' '.join(clean_keywords)
                multiword_keywords = [k for k in clean_keywords if ' ' in k]
                if len(clean_keywords) > 1 and joined_phrase not in clean_keywords:
                    sam3_prompts = [joined_phrase] + multiword_keywords
                else:
                    sam3_prompts = clean_keywords

                img_arr = np.frombuffer(content, dtype=np.uint8)
                img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    logger.warning(f"⚠️ Could not decode image: {image_url}")
                    return False, image_url

                # SAM3 is a single-concept-per-call detector — serialize access
                # to the shared model instance (set_image mutates its state).
                async with self.sam3_lock:
                    detections, matched_prompt = await asyncio.to_thread(
                        self._run_sam3, img_bgr, sam3_prompts
                    )

                result = bool(detections)
                best_conf = detections[0][1] if detections else 0.0
                logger.info(
                    f"🔍 SAM3 check {sam3_prompts} in {image_url.split('/')[-1]}: "
                    f"{'✅ kept' if result else '❌ discarded'}"
                    f"{f' (matched {matched_prompt!r}, {len(detections)} instance(s), best conf={best_conf:.2f})' if result else ''}"
                )

                display_url = image_url
                if result:
                    annotated = self._draw_bboxes(img_bgr, detections, matched_prompt)
                    if annotated is not None:
                        annotated_url = self._upload_annotated_frame(annotated, image_url)
                        if annotated_url:
                            display_url = annotated_url

                # ── If rejected, save to discarded folder ──
                if not result:
                    self._save_discarded_frame(
                        tmp_path=tmp_path,
                        image_url=image_url,
                        keywords=keywords,
                        reason=f"SAM3: no detection above conf={SAM3_CONF_THRESHOLD} for {sam3_prompts}"
                    )

                return result, display_url

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                f"SAM3 verification error for {image_url}: "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )
            return False, image_url

    async def _collect_verified_frame_urls_async(
            self,
            segments: List[Dict[str, Any]],
            keywords: List[str]) -> str:
        all_urls = []
        for seg in segments:
            for url in seg.get('frame_urls', []):
                if url and url != 'N/A':
                    all_urls.append(url)
        if not all_urls:
            return ""

        to_verify = rank_frames_for_verification(segments, keywords)
        if not to_verify:
            to_verify = list(dict.fromkeys(all_urls))

        # VERIFY=false: return the matching segments' frames as they are — no
        # SAM3 call, nothing silently discarded, capped so a broad query
        # doesn't flood the page.
        if not VERIFY:
            capped = to_verify[:MAX_UNVERIFIED_FRAMES]
            logger.info(f"📸 VERIFY=false — returning {len(capped)} of {len(to_verify)} "
                        f"frame(s), no SAM3 check")
            if len(to_verify) > len(capped):
                logger.info(f"{len(to_verify) - len(capped)} further frame(s) not shown "
                            f"(MAX_UNVERIFIED_FRAMES={MAX_UNVERIFIED_FRAMES})")
            return "\n\n---\n📷 Frames:\n" + "\n".join(capped) + "\n"

        logger.info(f"🔍 Verifying {len(to_verify)} frames for keywords: {keywords}")
        verified_urls = await self._async_verify_frames(to_verify, keywords)
        logger.info(f"✅ {len(verified_urls)}/{len(to_verify)} frames passed SAM3 check")

        if not verified_urls:
            logger.warning("⚠️ No frames passed SAM3 verification — showing none")
            return "\n\n---\n⚠️ No frames visually confirmed for the requested keywords.\n"

        url_lines = "\n".join(verified_urls)
        return (
            f"\n\n---\n"
            f"📷 Verified Frames ({len(verified_urls)} of {len(all_urls)} matched keywords: {', '.join(keywords)}):\n"
            f"{url_lines}\n"
        )

    def _merge_activity_keywords(self, keywords: List[str]) -> List[str]:
        """Post-processing: Merge split person/activity keywords into cohesive phrases."""
        lowered = [k.lower() for k in keywords]
        has_person = any(w in k for k in lowered for w in ["person", "man", "woman", "individual", "people"])
        
        # Objects that imply an action
        activators = {
            "mobile": "using a mobile phone", 
            "phone": "using a phone", 
            "water": "drinking water", 
            "food": "eating food", 
            "laptop": "using a laptop", 
            "computer": "using a computer"
        }
        
        new_keywords = []
        merged_any = False
        for kw in keywords:
            k_low = kw.lower()
            found_act = next((a for a in activators if a in k_low), None)
            
            # If we found an activator (like 'mobile') but the phrase doesn't 
            # already have a verb, and we know there's a person in the set...
            if found_act and has_person and not any(v in k_low for v in ["using", "drinking", "eating", "holding"]):
                merged_phrase = f"person {activators[found_act]}"
                if merged_phrase not in new_keywords:
                    new_keywords.append(merged_phrase)
                merged_any = True
            else:
                new_keywords.append(kw)
        
        if merged_any:
            # Filter out the bare person-tags if we've successfully merged into an action
            final_keywords = []
            for nk in new_keywords:
                nk_low = nk.lower()
                # Drop "person", "man", etc. if an action-phrase was created
                if nk_low in ["person", "man", "woman", "individual", "people"] and any("using" in k or "drinking" in k or "eating" in k for k in new_keywords):
                    continue
                final_keywords.append(nk)
            return final_keywords
        return keywords

    async def _extract_visual_keywords(self, query: str) -> List[str]:
        """PURE LLM keyword generation — generates visual terms from user intent.

        Strips dates, camera IDs, stream names, and bare numbers so only
        visually-detectable subject/action terms reach the VLM verifier.
        """
        import re as _re

        system_prompt = (
            "You are a surveillance image retrieval expert. Your task is to convert a natural language "
            "surveillance query into a short list of VISUAL SEARCH KEYWORDS that a camera can physically detect.\n\n"

            "=== THINKING PROCESS (always follow these steps internally) ===\n"
            "Step 1 — FIND THE VERB: Identify the action/activity word (drinking, using, eating, carrying, talking, sleeping, running…)\n"
            "Step 2 — FORM COMPOUND: The PRIMARY keyword is always SUBJECT + VERB + OBJECT as one phrase. Never break it apart.\n"
            "         e.g. 'person drinking water' stays as 'person drinking water', NOT split into 'person' and 'water'.\n"
            "Step 3 — ADD OBJECT ALONE: Add the object by itself as a secondary keyword (e.g. 'water bottle', 'mobile phone').\n"
            "Step 4 — ADD VISUAL SYNONYMS: What else might a camera see that signals this activity? (hand raised to mouth, bottle in hand…)\n"
            "Step 5 — STRIP NOISE: Remove ALL dates, day-numbers, months, ordinals, camera IDs, stream names, and filler words.\n\n"

            "=== ABSOLUTE RULES ===\n"
            "✗ NEVER output: dates, numbers (1–31), months, ordinals (15th), years, times\n"
            "✗ NEVER output: camera names (stream5, stream6), camera IDs, locations\n"
            "✗ NEVER output: lone filler words (any, seen, use, this, as, have, id)\n"
            "✗ NEVER broaden the object's category. 'bus' must never become 'vehicle'; 'car' must never\n"
            "  become 'vehicle'; 'truck'/'bike'/'motorcycle' must never become 'vehicle' either. A generic\n"
            "  category word matches the WRONG object (e.g. a car when the user asked about a bus) — every\n"
            "  keyword must name the exact same object type the user asked about, only varying phrasing,\n"
            "  color, or state (parked/moving/stationary), never the object type itself.\n"
            "✗ NEVER split a multi-word OBJECT TYPE NAME into overlapping fragments. A compound like\n"
            "  'cement mixer truck' names ONE object type and must appear as a single keyword exactly as\n"
            "  given — do NOT output 'cement mixer' and 'mixer truck' as if they were two separate signals.\n"
            "  'cement mixer' alone matches a handheld/stand mixer or a stationary drum, not the truck, and\n"
            "  'mixer truck' alone is not a phrase a vision model can reliably ground. Same for any compound\n"
            "  object name: 'fire truck', 'garbage truck', 'tow truck', 'auto rickshaw', 'delivery van' —\n"
            "  always keep the whole name as one keyword; add synonyms as WHOLE alternate phrases instead\n"
            "  (e.g. 'concrete mixer truck'), never as a fragment of the original.\n"
            "✓ ALWAYS keep the full compound phrase intact — splitting it loses the user's intent\n"
            "✗ NEVER split a COLOR/SIZE/STATE modifier away from its object into two separate keywords.\n"
            "  'red car' must stay as 'red car' (or 'red sedan') — NEVER output 'red' and 'car' as two\n"
            "  independent entries. 'red' alone matches ANY red thing (a red bag, a red truck, a red shirt)\n"
            "  and 'car' alone matches a car of ANY color — splitting them throws away exactly the\n"
            "  constraint the user gave. Same for 'white truck', 'blue sedan', 'parked car', etc.\n"
            "✓ Read the whole query as one intent before extracting keywords — don't tokenize word-by-word.\n"
            "✓ Output 3–5 keywords maximum\n\n"

            "=== FEW-SHOT EXAMPLES WITH REASONING ===\n\n"

            "Query: \"have you seen any person drinking water on 15th may stream6\"\n"
            "Thinking:\n"
            "  Step 1 → verb = 'drinking', object = 'water'\n"
            "  Step 2 → compound = 'person drinking water'\n"
            "  Step 3 → object alone = 'water bottle'\n"
            "  Step 4 → visual synonyms: 'hand raised to mouth', 'holding bottle'\n"
            "  Step 5 → strip: '15th', 'may', 'stream6' removed\n"
            "Output: [\"person drinking water\", \"water bottle\", \"hand raised to mouth\", \"holding bottle\"]\n\n"

            "Query: \"have you seen any person using mobile on 14th april camera stream5\"\n"
            "Thinking:\n"
            "  Step 1 → verb = 'using', object = 'mobile'\n"
            "  Step 2 → compound = 'person using mobile'\n"
            "  Step 3 → object alone = 'mobile phone'\n"
            "  Step 4 → visual synonyms: 'looking at phone', 'hand holding phone'\n"
            "  Step 5 → strip: '14th', 'april', 'stream5' removed\n"
            "Output: [\"person using mobile\", \"mobile phone\", \"looking at phone\", \"hand holding phone\"]\n\n"

            "Query: \"is any person eating food at their desk on 2nd may\"\n"
            "Thinking:\n"
            "  Step 1 → verb = 'eating', object = 'food'\n"
            "  Step 2 → compound = 'person eating food'\n"
            "  Step 3 → object alone = 'food'\n"
            "  Step 4 → visual synonyms: 'hand to mouth', 'food on desk', 'eating at desk'\n"
            "  Step 5 → strip: '2nd', 'may' removed\n"
            "Output: [\"person eating food\", \"food on desk\", \"hand to mouth\", \"eating at desk\"]\n\n"

            "Query: \"have you seen anyone carrying a bag on 15th\"\n"
            "Thinking:\n"
            "  Step 1 → verb = 'carrying', object = 'bag'\n"
            "  Step 2 → compound = 'person carrying bag'\n"
            "  Step 3 → object alone = 'bag'\n"
            "  Step 4 → visual synonyms: 'holding bag', 'backpack'\n"
            "  Step 5 → strip: '15th' removed\n"
            "Output: [\"person carrying bag\", \"bag\", \"holding bag\", \"backpack\"]\n\n"

            "Query: \"have you seen any white car parked near entrance yesterday stream7\"\n"
            "Thinking:\n"
            "  Step 1 → verb = 'parked', object = 'entrance'\n"
            "  Step 2 → compound = 'white car parked'\n"
            "  Step 3 → object alone = 'white car'\n"
            "  Step 4 → visual synonyms: 'stationary white car', 'car near door' (object type 'car' kept — never 'vehicle')\n"
            "  Step 5 → strip: 'yesterday', 'stream7' removed\n"
            "Output: [\"white car parked\", \"white car\", \"stationary white car\"]\n\n"

            "Query: \"have you seen any yellow bus today\"\n"
            "Thinking:\n"
            "  Step 1 → verb = N/A (no action, just appearance)\n"
            "  Step 2 → compound = 'yellow bus'\n"
            "  Step 3 → object alone = 'bus'\n"
            "  Step 4 → visual synonyms: 'yellow school bus' (object type 'bus' kept — NEVER 'yellow vehicle',\n"
            "           that would also match a yellow car or truck, which is wrong)\n"
            "  Step 5 → strip: 'today' removed\n"
            "Output: [\"yellow bus\", \"bus\", \"yellow school bus\"]\n\n"

            "Query: \"have you seen a red car today\"\n"
            "Thinking:\n"
            "  Step 1 → verb = N/A (no action, just appearance)\n"
            "  Step 2 → compound = 'red car' — color + object kept together as ONE keyword.\n"
            "           NEVER split into 'red' and 'car' (see ABSOLUTE RULES) — 'red' alone matches any\n"
            "           red object and 'car' alone matches a car of any color, losing the user's intent.\n"
            "  Step 3 → object alone = 'car' is only added plain if the user did NOT specify a color;\n"
            "           here a color WAS given, so do not add bare 'car'.\n"
            "  Step 4 → visual synonyms as WHOLE phrases: 'red sedan', 'red automobile'\n"
            "  Step 5 → strip: 'today' removed\n"
            "Output: [\"red car\", \"red sedan\", \"red automobile\"]\n\n"

            "Query: \"have you seen a cement mixer truck anywhere?\"\n"
            "Thinking:\n"
            "  Step 1 → verb = N/A (no action, just an object)\n"
            "  Step 2 → compound = 'cement mixer truck' — a 3-word object type name, kept WHOLE.\n"
            "           NEVER split into 'cement mixer' + 'mixer truck' (see ABSOLUTE RULES) — those are\n"
            "           two fragments of the same name, not two independent signals, and neither one\n"
            "           reliably identifies a mixer TRUCK on its own.\n"
            "  Step 3 → object alone = N/A (already the object)\n"
            "  Step 4 → visual synonyms as WHOLE alternate phrases: 'concrete mixer truck', 'rotating\n"
            "           drum truck' (never a fragment like 'rotating drum' alone)\n"
            "  Step 5 → strip: 'anywhere' removed (not a location filter)\n"
            "Output: [\"cement mixer truck\", \"concrete mixer truck\", \"rotating drum truck\"]\n\n"

            "Query: \"person wearing headset today camera stream5\"\n"
            "Thinking:\n"
            "  Step 1 → verb = 'wearing', object = 'headset'\n"
            "  Step 2 → compound = 'person wearing headset'\n"
            "  Step 3 → object alone = 'headset'\n"
            "  Step 4 → visual synonyms: 'wearing headphones', 'headset on head'\n"
            "  Step 5 → strip: 'today', 'stream5' removed\n"
            "Output: [\"person wearing headset\", \"headset\", \"wearing headphones\"]\n\n"

            "Query: \"two people talking to each other on 10th\"\n"
            "Thinking:\n"
            "  Step 1 → verb = 'talking', count = 'two people'\n"
            "  Step 2 → compound = 'two people talking'\n"
            "  Step 3 → object alone = N/A (interaction, no object)\n"
            "  Step 4 → visual synonyms: 'people facing each other', 'conversation'\n"
            "  Step 5 → strip: '10th' removed\n"
            "Output: [\"two people talking\", \"people facing each other\", \"conversation\"]\n\n"

            "=== OUTPUT FORMAT ===\n"
            "Output ONLY a valid JSON array. NO reasoning text in output. NO explanations. NO extra words.\n"
            "Example output: [\"person drinking water\", \"water bottle\", \"holding bottle\"]"
        )

        user_prompt = (
            f'Query: "{query}"\n\n'
            f'Output ONLY a JSON array of visual keywords (NO dates, NO camera IDs, NO stream names, NO reasoning text):'
        )

        result = await self._call_ollama_async(system_prompt, user_prompt, max_tokens=200, temperature=0.0)
        # ── Parse LLM response ────────────────────────────────────────────────
        try:
            import json
            # Find the LAST JSON array in the response
            # (LLM may output reasoning text before/after the JSON — we want the last one)
            all_matches = list(_re.finditer(r'\[[^\[\]]*\]', result))
            if not all_matches:
                raise ValueError("No JSON array found in LLM response")
            match = all_matches[-1]   # take the LAST match
            keywords = json.loads(match.group())

            # Expanded stop-word set: dates, ordinals, months, camera patterns, filler
            # NOTE: 'using' is intentionally NOT here — "person using X" is a valid phrase
            stop_words = {
                'time', 'when', 'where', 'hour', 'ago', 'today', 'yesterday',
                'january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december',
                'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                'st', 'nd', 'rd', 'th',
                'camera', 'cam', 'stream', 'channel',
                'any', 'have', 'seen', 'this', 'as', 'id',
                'please', 'show', 'find', 'look',
            }

            def _is_valid(kw: str) -> bool:
                kw = kw.strip()
                if not (3 <= len(kw) <= 40):
                    return False
                if _re.fullmatch(r'\d+', kw):           # pure number "15"
                    return False
                if _re.fullmatch(r'\d+(st|nd|rd|th)', kw, _re.I):  # "15th"
                    return False
                if _re.match(r'stream\d*', kw, _re.I):  # "stream5"
                    return False
                tokens = kw.lower().split()
                if all(t in stop_words for t in tokens):
                    return False
                return True

            valid_keywords = [kw.strip() for kw in keywords if isinstance(kw, str) and _is_valid(kw)]
            valid_keywords = self._merge_activity_keywords(valid_keywords)
            logger.info(f"🔍 LLM visual keywords '{query}' → {valid_keywords}")
            return valid_keywords[:5]

        except Exception as e:
            logger.warning(f"LLM keyword parse failed: {e}")

        # ── Emergency fallback — dynamic spaCy NLP extraction ─────────────
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                nlp = spacy.load("en_core_web_sm")

            doc = nlp(query.lower())
            fallback = []
            
            stop_nouns = {"you", "i", "we", "they", "it", "this", "that", "anyone", "someone", "other", "camera", "stream", "id", "stream5", "stream6", "stream7"}
            noise_verbs = {"see", "seen", "have", "has", "is", "are", "was", "were", "use", "using", "look", "find", "show"}
            date_nouns = {"today", "yesterday", "tomorrow", "day", "month", "year", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
            
            for token in doc:
                if token.pos_ == "VERB" and token.lemma_ not in noise_verbs:
                    verb = token
                    
                    # 1. Subject extraction
                    subj = None
                    if verb.dep_ in ("acl", "relcl", "amod") and verb.head.pos_ in ("NOUN", "PRON", "PROPN"):
                        subj = verb.head
                    else:
                        for child in verb.children:
                            if child.dep_ in ("nsubj", "nsubjpass"):
                                subj = child
                                break
                                
                    # 2. Object extraction
                    obj = None
                    for child in verb.children:
                        if child.dep_ in ("dobj", "pobj"):
                            obj = child
                            break
                        if child.dep_ == "prep":
                            for grandchild in child.children:
                                if grandchild.dep_ == "pobj":
                                    obj = grandchild
                                    break
                            if obj: break
                            
                    def get_noun_chunk(noun_token):
                        if not noun_token: return ""
                        modifiers = [c.text for c in noun_token.children if c.dep_ in ("amod", "compound", "nummod") and c.pos_ != "PRON"]
                        if modifiers:
                            return " ".join(modifiers + [noun_token.text])
                        return noun_token.text

                    subj_text = get_noun_chunk(subj)
                    obj_text = get_noun_chunk(obj)
                    
                    if subj and subj.lemma_ in stop_nouns:
                        subj_text = "person" if subj.lemma_ in ("someone", "anyone") else ""
                    if obj and obj.lemma_ in stop_nouns:
                        obj_text = ""
                        
                    parts = []
                    if subj_text: parts.append(subj_text)
                    parts.append(verb.text)
                    if obj_text: parts.append(obj_text)
                    
                    phrase = " ".join(parts).strip()
                    if len(phrase.split()) > 1:  # Only add compounds
                        fallback.append(phrase)
                    if obj_text and obj_text not in stop_nouns:
                        fallback.append(obj_text)
                    if subj_text and subj_text not in stop_nouns:
                        fallback.append(subj_text)
            
            # 3. Handle 'using' manually because 'use/using' is a noise verb (e.g. "use this as camera")
            using_matches = _re.findall(r'\busing\s+([a-z]+)\b', query.lower())
            for match in using_matches:
                if match not in stop_nouns and match not in date_nouns:
                    fallback.append(f"person using {match}")
                    fallback.append(match)

            # 4. Grab standalone nouns that aren't noise or dates
            for chunk in doc.noun_chunks:
                text = chunk.text
                root = chunk.root.lemma_
                if root not in stop_nouns and root not in date_nouns and not chunk.root.is_digit:
                    fallback.append(text)

            if fallback:
                fallback = self._merge_activity_keywords(list(dict.fromkeys(fallback)))
                logger.info(f"🔍 spaCy fallback keywords for '{query}': {fallback}")
                return fallback[:5]

        except Exception as e:
            logger.warning(f"spaCy fallback failed, using regex: {e}")

        # ── Emergency regex fallback — smarter compound extraction ─────────────
        compound_pattern = _re.compile(
            r'(person|people|man|woman|someone|anyone)\s+'
            r'(drinking|eating|carrying|holding|wearing|using|watching|reading|writing|'
            r'sleeping|sitting|standing|lying|running|walking|talking|looking|typing|'
            r'operating|handling|lifting|pushing|pulling|opening|closing|checking|working)'
            r'(?:\s+(\w+))?',  # optional object
            _re.I
        )
        fallback = []
        for m in compound_pattern.finditer(query):
            subject = m.group(1).lower()
            verb    = m.group(2).lower()
            obj     = m.group(3).lower() if m.group(3) else None
            if obj and obj not in {'on', 'in', 'at', 'with', 'the', 'a', 'an', 'to', 'for', 'of'}:
                fallback.append(f"{subject} {verb} {obj}")
                fallback.append(obj)
            else:
                fallback.append(f"{subject} {verb}")
        
        if not fallback:
            bare = _re.findall(
                r'\b(person|people|man|woman|crowd|car|vehicle|truck|van|bus|bike|motorcycle|'
                r'mobile|phone|laptop|water|bottle|cup|food|bag|helmet|headset|headphones|'
                r'backpack|umbrella|jacket|shirt|chair|table|desk)\b',
                query.lower()
            )
            fallback = list(dict.fromkeys(bare))
            
        fallback = self._merge_activity_keywords(list(dict.fromkeys(fallback)))
        logger.info(f"🔍 Fallback keywords for '{query}': {fallback}")
        return fallback[:5]




    # ─────────────────────────────────────────────
    # CHANGE 1 applied to ALL 4 summary methods:
    # - Removed camera_id from system_prompt and user_prompt
    # - Removed camera_id from the response header
    # ─────────────────────────────────────────────

    # ─────────────────────────────────────────────
    # Streaming Summary Methods
    # ─────────────────────────────────────────────

    @traceable(name="generate_summary", run_type="chain", project_name="video-summary")
    async def summarize_time_range_streaming(
            self, query: str, contextualized_query: str,
            segments: List[Dict[str, Any]], camera_id: str, hours: float,
            queue: Optional[asyncio.Queue] = None,
            config: Optional[RunnableConfig] = None) -> AsyncGenerator[str, None]:
        """`queue`/`config` are set only when called from graph.py's
        generate_summary node. Static text (headers, no-data messages, frame
        URLs) is mirrored onto `queue` as {"content"/"image_url": ...} for
        the SSE side channel, since LangGraph's stream_mode="messages" only
        carries the LLM's own tokens (see graph.py). The actual LLM tokens
        below are NOT also pushed to `queue` — with `config` forwarded into
        _call_ollama_streaming, they already reach the client via
        stream_mode="messages", so pushing them here too would double-send
        them. Callers that don't pass queue/config (e.g. vision_orchestrator)
        are unaffected — this generator still yields everything exactly as
        before."""
        def _emit(chunk: str):
            if queue is None or chunk == "__KEEPALIVE__":
                return
            if chunk.startswith("http"):
                queue.put_nowait({"image_url": chunk.strip()})
            else:
                queue.put_nowait({"content": chunk})

        run = get_current_run_tree()
        if run:
            run.metadata.update({
                "component": "summary", "tool": "get_last_n_hours_summary",
                "camera_id": camera_id, "num_segments": len(segments), "model": self.model,
            })

        if not segments:
            time_msg = f"in the last {hours} hours" if hours > 0 else "in the specified time range"
            msg = f"# ⚠️ No Data Found\n\nNo footage found {time_msg}."
            _emit(msg)
            yield msg
            return

        header = "# 📹 Video Summary\n\n"
        _emit(header)
        yield header

        segments = self._budget_segments(self._retrieve_relevant_segments(segments, query))
        segments_text = self.format_segments_for_summary(segments)
        system_prompt = (
            "You are a video surveillance analyst answering a specific question about a time window of footage.\n\n"
            "RULES:\n"
            "- Lead with a direct answer to the user's question in the first 1-2 sentences.\n"
            "- Then provide supporting chronological context if needed.\n"
            "- Do NOT mention camera IDs or technical identifiers in your response.\n"
            "- Use transitional language (initially, later, following this) instead of timestamps.\n"
            "- If nothing relevant to the question occurred, state that clearly."
        )
        user_prompt = (
            f"QUESTION TO ANSWER: {query}\n\n"
            f"Time window: last {hours} hours\n\n"
            f"Segment descriptions:\n{segments_text}\n\n"
            f"Answer the question directly: {query}"
        )

        async for token in self._call_ollama_streaming(system_prompt, user_prompt, config=config):
            yield token

        urls = []
        for seg in segments:
            for url in seg.get('frame_urls', []):
                if url and url != 'N/A':
                    urls.append(url)

        unique_urls = list(dict.fromkeys(urls))
        if unique_urls:
            frames_header = "\n\n---\n📷 Captured Frames:\n"
            _emit(frames_header)
            yield frames_header
            for url in unique_urls:
                _emit(url)
                yield f"{url}\n"

    @traceable(name="generate_summary", run_type="chain", project_name="video-summary")
    async def summarize_segments_streaming(
            self, query: str, contextualized_query: str,
            segments: List[Dict[str, Any]], camera_id: str) -> AsyncGenerator[str, None]:
        run = get_current_run_tree()
        if run:
            run.metadata.update({
                "component": "summary", "tool": "summarize_segments",
                "camera_id": camera_id, "num_segments": len(segments), "model": self.model,
            })

        if not segments:
            yield "# ⚠️ No segments found."
            return

        yield "# 📹 Video Segments Summary\n\n"

        segments = self._budget_segments(self._retrieve_relevant_segments(segments, query))
        segments_text = self.format_segments_for_summary(segments)
        system_prompt = (
            "You are a video surveillance analyst. You answer questions using footage segment descriptions.\n\n"
            "RULES:\n"
            "- Your response must directly address the user's question — not summarize everything.\n"
            "- Start with a one-sentence direct answer, then elaborate with relevant details.\n"
            "- Omit segment details that don't relate to the question.\n"
            "- Do NOT mention camera IDs or technical identifiers in your response.\n"
            "- Be specific about appearance, count, actions, and movement when relevant.\n"
            "- No conversational filler. Formal report style."
        )
        user_prompt = (
            f"QUESTION TO ANSWER: {query}\n\n"
            f"Segment descriptions:\n{segments_text}\n\n"
            f"Answer the question directly: {query}"
        )

        async for token in self._call_ollama_streaming(system_prompt, user_prompt):
            yield token

        urls = []
        for seg in segments:
            for url in seg.get('frame_urls', []):
                if url and url != 'N/A':
                    urls.append(url)
        
        unique_urls = list(dict.fromkeys(urls))
        if unique_urls:
            yield "\n\n---\n📷 Captured Frames:\n"
            for url in unique_urls:
                yield f"{url}\n"

    @traceable(name="generate_summary", run_type="chain", project_name="video-summary")
    async def summarize_search_results_streaming(
            self, query: str, contextualized_query: str,
            segments: List[Dict[str, Any]], camera_id: str,
            keywords: List[str],
            queue: Optional[asyncio.Queue] = None,
            config: Optional[RunnableConfig] = None) -> AsyncGenerator[str, None]:
        """See summarize_time_range_streaming's docstring for what queue/config
        are for — same pattern here."""
        def _emit(chunk: str):
            if queue is None or chunk == "__KEEPALIVE__":
                return
            if chunk.startswith("http"):
                queue.put_nowait({"image_url": chunk.strip()})
            else:
                queue.put_nowait({"content": chunk})

        run = get_current_run_tree()
        if run:
            run.metadata.update({
                "component": "summary", "tool": "search_segments_by_activity",
                "camera_id": camera_id, "keywords": keywords,
                "num_segments": len(segments), "model": self.model,
            })

        if not segments:
            msg = f"# ⚠️ No matches found for keywords: {', '.join(keywords)}"
            _emit(msg)
            yield msg
            return

        header = "# 🔍 Search Results\n\n"
        _emit(header)
        yield header

        segments = self._budget_segments(self._retrieve_relevant_segments(segments, query))
        segments_text = self.format_segments_for_summary(segments)
        system_prompt = (
            "You are a video surveillance analyst. You answer specific questions about recorded footage.\n\n"
            "RULES:\n"
            "- Your ENTIRE response must be structured as a direct answer to the user's question.\n"
            "- Open your response by restating what the user asked in one sentence, then answer it immediately.\n"
            "- Use segment descriptions only as evidence to support your answer — do not narrate them.\n"
            "- If the segments do not contain enough information to answer the question, say so explicitly.\n"
            "- Do NOT mention camera IDs or technical identifiers in your response.\n"
            "- Be specific: clothing colors, counts, directions of movement, vehicle types matter.\n"
            "- No filler phrases. No segment-by-segment walkthrough. One cohesive answer."
        )
        user_prompt = (
            f"QUESTION TO ANSWER: {query}\n\n"
            f"Search keywords used: {', '.join(keywords)}\n"
            f"Evidence from matching segments (use these to build your answer):\n"
            f"{segments_text}\n\n"
            f"Now answer the question: {query}"
        )

        async for token in self._call_ollama_streaming(system_prompt, user_prompt, config=config):
            yield token

        # ── Verification Step ──
        # Provided keywords/characteristics are normally already coherent phrases —
        # get_image_characteristics() explicitly forbids splitting e.g. "person" and
        # "mobile" into separate strings. But the tool-selection LLM occasionally
        # hands a plain-text query's keywords back as disjoint attribute fragments
        # (e.g. "red car" → ["red", "car"]); querying the verifier with a lone "car"
        # or "red" matches ANY car / ANY red thing, not what was actually asked. When
        # that fragment shape shows up, re-derive keywords from the user's own query
        # with the same LLM used for the summary, instead of trusting the fragments.
        keywords_look_fragmented = (
            len(keywords) > 1 and all(' ' not in k.strip() for k in keywords if k and k.strip())
        )
        if keywords and len(keywords) > 0 and not keywords_look_fragmented:
            # Ensure passed keywords are also merged for robustness
            keywords = self._merge_activity_keywords(keywords)
            logger.info(f"🎨 Using provided visual characteristics for verification: {keywords}")
            keywords_for_images = keywords
        else:
            if keywords_look_fragmented:
                logger.info(f"🎨 Provided keywords look like disjoint fragments {keywords} — re-deriving from query via LLM")
            else:
                logger.info(f"🎨 Extracting visual keywords for verification from query: {query}")
            try:
                keywords_for_images = await self._extract_visual_keywords(query)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Keyword extraction failed: {e}")
                keywords_for_images = keywords

        if not keywords_for_images:
            keywords_for_images = keywords
        
        all_urls = []
        for seg in segments:
            for url in seg.get('frame_urls', []):
                if url and url != 'N/A':
                    all_urls.append(url)
        all_urls = list(dict.fromkeys(all_urls))

        if all_urls:
            candidates = rank_frames_for_verification(segments, keywords_for_images) or all_urls

            # VERIFY=false: no SAM3 call — stream the matching frames straight
            # from the database, capped so a broad query doesn't flood the page.
            if not VERIFY:
                capped = candidates[:MAX_UNVERIFIED_FRAMES]
                logger.info(f"📸 VERIFY=false — streaming {len(capped)} of {len(candidates)} "
                            f"frame(s), no SAM3 check")
                frames_header = "\n\n---\n📷 Frames:\n"
                _emit(frames_header)
                yield frames_header
                for url in capped:
                    _emit(url)
                    yield url + "\n"
            else:
                logger.info(f"📸 Starting SAM3 verification for {len(candidates)} frames...")
                frames_header = "\n\n---\n📷 Verified Frames:\n"
                _emit(frames_header)
                yield frames_header

                try:
                    async for verified_url in self._async_verify_frames_streaming(candidates, keywords_for_images):
                        if verified_url:
                            _emit(verified_url)
                            yield verified_url + "\n"
                        else:
                            # Yield special marker so backend can check for disconnection
                            yield "__KEEPALIVE__"
                except asyncio.CancelledError:
                    logger.info("⏹️ SAM3 verification generator cancelled")
                    raise
                except Exception as e:
                    logger.error(f"Error in verification stream: {e}")

    async def summarize_time_range(
            self, query: str, contextualized_query: str,
            segments: List[Dict[str, Any]], camera_id: str, hours: float) -> str:

        if not segments:
            time_msg = f"in the last {hours} hours" if hours > 0 else "in the specified time range"
            # camera_id still used internally for the no-data message but not shown in normal flow
            return f"# ⚠️ No Data Found\n\nNo footage found {time_msg}."

        segments = self._budget_segments(self._retrieve_relevant_segments(segments, query))
        segments_text = self.format_segments_for_summary(segments)
        system_prompt = (
            # ── Removed: "Camera: {camera_id}" instruction ──
            "You are a video surveillance analyst answering a specific question about a time window of footage.\n\n"
            "RULES:\n"
            "- Lead with a direct answer to the user's question in the first 1-2 sentences.\n"
            "- Then provide supporting chronological context if needed.\n"
            "- Do NOT mention camera IDs or technical identifiers in your response.\n"
            "- Use transitional language (initially, later, following this) instead of timestamps.\n"
            "- If nothing relevant to the question occurred, state that clearly."
        )
        user_prompt = (
            f"QUESTION TO ANSWER: {query}\n\n"
            # ── Removed: f"Camera: {camera_id} | " ──
            f"Time window: last {hours} hours\n\n"
            f"Segment descriptions:\n{segments_text}\n\n"
            f"Answer the question directly: {query}"
        )

        summary = await self._call_ollama_async(system_prompt, user_prompt, max_tokens=2000)
        image_urls = self._collect_frame_urls(segments)
        # ── Removed camera_id from header ──
        header = f"# 📹 Video Summary\n\n"
        return header + (summary or "") + image_urls

    async def summarize_segments(
            self, query: str, contextualized_query: str,
            segments: List[Dict[str, Any]], camera_id: str) -> str:

        if not segments:
            return f"# ⚠️ No segments found."

        segments = self._budget_segments(self._retrieve_relevant_segments(segments, query))
        segments_text = self.format_segments_for_summary(segments)
        system_prompt = (
            "You are a video surveillance analyst. You answer questions using footage segment descriptions.\n\n"
            "RULES:\n"
            "- Your response must directly address the user's question — not summarize everything.\n"
            "- Start with a one-sentence direct answer, then elaborate with relevant details.\n"
            "- Omit segment details that don't relate to the question.\n"
            "- Do NOT mention camera IDs or technical identifiers in your response.\n"
            "- Be specific about appearance, count, actions, and movement when relevant.\n"
            "- No conversational filler. Formal report style."
        )
        user_prompt = (
            f"QUESTION TO ANSWER: {query}\n\n"
            # ── Removed: f"Camera: {camera_id}\n\n" ──
            f"Segment descriptions:\n{segments_text}\n\n"
            f"Answer the question directly: {query}"
        )

        summary = await self._call_ollama_async(system_prompt, user_prompt, max_tokens=2500)
        image_urls = self._collect_frame_urls(segments)
        # ── Removed camera_id from header ──
        header = f"# 📹 Video Segments Summary\n\n"
        return header + (summary or "") + image_urls

    async def summarize_search_results(
            self, query: str, contextualized_query: str,
            segments: List[Dict[str, Any]], camera_id: str,
            keywords: List[str]) -> str:

        if not segments:
            return f"# ⚠️ No matches found for keywords: {', '.join(keywords)}"

        segments = self._budget_segments(self._retrieve_relevant_segments(segments, query))
        segments_text = self.format_segments_for_summary(segments)
        system_prompt = (
            "You are a video surveillance analyst. You answer specific questions about recorded footage.\n\n"
            "RULES:\n"
            "- Your ENTIRE response must be structured as a direct answer to the user's question.\n"
            "- Open your response by restating what the user asked in one sentence, then answer it immediately.\n"
            "- Use segment descriptions only as evidence to support your answer — do not narrate them.\n"
            "- If the segments do not contain enough information to answer the question, say so explicitly.\n"
            "- Do NOT mention camera IDs or technical identifiers in your response.\n"
            "- Be specific: clothing colors, counts, directions of movement, vehicle types matter.\n"
            "- No filler phrases. No segment-by-segment walkthrough. One cohesive answer."
        )
        user_prompt = (
            f"QUESTION TO ANSWER: {query}\n\n"
            f"Search keywords used: {', '.join(keywords)}\n"
            # ── Removed: f"Camera: {camera_id}\n\n" ──
            f"Evidence from matching segments (use these to build your answer):\n"
            f"{segments_text}\n\n"
            f"Now answer the question: {query}"
        )

        summary = await self._call_ollama_async(system_prompt, user_prompt, max_tokens=2000)

        keywords_for_images = await self._extract_visual_keywords(query) or keywords
        image_urls = await self._collect_verified_frame_urls_async(segments, keywords_for_images)

        # ── Removed camera_id from header ──
        header = f"# 🔍 Search Results\n\n"
        return header + (summary or "") + image_urls

    async def summarize_plate_search(
            self, query: str, contextualized_query: str,
            results: List[Dict[str, Any]], plate_number: str) -> str:
        """Summarize search_car_by_plate_number's results: each detection
        already carries frame_url, plate_number, description (the segment's
        original AI/VLM-generated scene description), camera_id, location and
        timestamp — that's the evidence handed to the LLM here, same pattern
        as summarize_search_results()."""

        if not results:
            return f"# ⚠️ No recognized plate found matching '{plate_number}'."

        evidence_parts = []
        for i, r in enumerate(results, start=1):
            evidence_parts.append(
                f"**Detection {i}**\n"
                f"Plate: {r.get('plate_number', 'N/A')}\n"
                f"Camera: {r.get('camera_id', 'N/A')}\n"
                f"Location: {r.get('location', 'N/A')}\n"
                f"Time: {r.get('timestamp', 'N/A')}\n"
                f"Scene description: {r.get('description', 'N/A')}\n"
            )
        evidence_text = "\n---\n\n".join(evidence_parts)

        system_prompt = (
            "You are a police ANPR (Automatic Number Plate Recognition) analyst. You answer "
            "questions about a specific vehicle's detected plate-reading history.\n\n"
            "RULES:\n"
            "- Your ENTIRE response must directly answer the user's question about this plate.\n"
            "- State clearly which camera(s), location(s), and time(s) the plate was seen at.\n"
            "- Use the scene descriptions as supporting evidence — summarize relevant details "
            "(vehicle appearance, activity, other objects/people) only where useful, don't repeat everything verbatim.\n"
            "- If detections span multiple cameras/locations, order them chronologically.\n"
            "- No filler phrases. One cohesive answer."
        )
        user_prompt = (
            f"QUESTION TO ANSWER: {query}\n\n"
            f"Plate number searched: {plate_number}\n\n"
            f"Detections found (use these to build your answer):\n"
            f"{evidence_text}\n\n"
            f"Now answer the question: {query}"
        )

        summary = await self._call_ollama_async(system_prompt, user_prompt, max_tokens=1500)

        urls = [r.get('frame_url', '') for r in results if r.get('frame_url')]
        image_urls = ""
        if urls:
            url_lines = "\n".join(dict.fromkeys(urls))
            image_urls = f"\n\n---\n📷 Captured Frames:\n{url_lines}\n"

        header = f"# 🚗 Plate Search — {plate_number}\n\n"
        return header + (summary or "") + image_urls

    async def summarize_person_activities(
            self, query: str, contextualized_query: str,
            person_name: str, activities: List[Dict[str, Any]],
            hourly_summary: List[Any], time_range: Dict[str, str],
            cameras: List[str]) -> str:

        if not activities:
            return f"# ⚠️ No activities found for {person_name}."

        activities_text = "\n".join([
            # ── Removed camera_id from activity lines ──
            # Previously: f"- Camera {act.get('camera_id')}: {act.get('activity')}"
            # Now: just the activity itself
            f"- {act.get('activity', 'Activity')}"
            for act in activities
        ])
        system_prompt = (
            f"You are a video surveillance analyst tracking the activities of {person_name}.\n\n"
            "RULES:\n"
            "- Answer the user's specific question about this person first, directly.\n"
            "- Then provide supporting detail: where they went, what they did, in sequence.\n"
            "- Do NOT mention camera IDs or technical identifiers in your response.\n"
            "- Use 'first', 'then', 'later' for ordering. No timestamps.\n"
            "- If the footage doesn't answer the question, say so clearly."
        )
        user_prompt = (
            f"QUESTION TO ANSWER: {query}\n\n"
            f"Person: {person_name}\n\n"
            f"Detected activities:\n{activities_text}\n\n"
            f"Answer the question directly: {query}"
        )

        summary = await self._call_ollama_async(system_prompt, user_prompt, max_tokens=3000)

        urls = [
            act.get('frame_url', '')
            for act in activities
            if act.get('frame_url') and act.get('frame_url') != 'N/A'
        ]
        image_urls = ""
        if urls:
            url_lines = "\n".join(list(dict.fromkeys(urls)))
            image_urls = f"\n\n---\n📷 Captured Frames:\n{url_lines}\n"

        header = f"# 👤 Activity Report - {person_name}\n\n"
        return header + (summary or "") + image_urls