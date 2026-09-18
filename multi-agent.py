"""
Multi-agent concierge API that fans a single user question out to two
already-running A2A agents and combines their answers:

  - AGENT1_URL (default http://localhost:8091)
    -> Video Surveillance Chat Agent, see New-Code/a2a_server.py
  - AGENT2_URL (default http://localhost:8092)
    -> Arcis Election Surveillance Chat Agent, see ElectionArcis .../a2a_server.py

Both agents speak the same hand-rolled A2A 0.2/0.3 JSON-RPC wire format
(message/send, message/stream, tasks/get, tasks/cancel; TextPart-only parts;
image/video/file URLs appended as plain lines inside the answer text rather
than as separate structured parts) -- this client is written directly against
that shape instead of a generic SDK, since that's what both agents actually
speak.

Run with:
    uvicorn agent:app --reload
Frontend (static SPA) is served at:
    GET /
"""

import asyncio
import json
import os
import re
import uuid
from typing import AsyncGenerator, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

AGENTS = {
    "agent1": os.environ.get("AGENT1_URL", "http://localhost:8091").rstrip("/"),
    "agent2": os.environ.get("AGENT2_URL", "http://localhost:8092").rstrip("/"),
}
_HTTP_TIMEOUT = httpx.Timeout(60.0, connect=5.0)

# Query router -- a small local model picks which agent(s) a question is
# actually for for, based on each agent's own card (name/description/skills),
# instead of fanning every question out to both agents unconditionally.
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "llama3.1:latest")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
_ROUTER_TIMEOUT = httpx.Timeout(15.0, connect=3.0)

# Fixed test identity forwarded to agent2 (the Arcis Election Surveillance
# Chat Agent) only -- that agent's get_stream_status/get_district_camera_stats
# tools require a logged-in user's email to scope results (see its db.py).
# AGENT2_PASSWORD isn't sent anywhere yet: a2a_server.py's message metadata
# only reads "email"/"auth_token", never a raw password -- it's kept here so
# a future login step (POST arcis_backend_R-D's /api/auth/login) can obtain a
# real auth_token for write actions without another round of wiring.
AGENT2_EMAIL = os.environ.get("AGENT2_EMAIL", "krushnal@vmukti.com")
AGENT2_PASSWORD = os.environ.get("AGENT2_PASSWORD", "Krushnal@123")

app = FastAPI(title="Multi-Agent Concierge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agent cards fetched once at startup (name/description used for UI labels).
# Kept as a mutable cache rather than re-fetched per request so a request
# doesn't pay for two extra round trips; refreshed lazily if an agent was
# down at startup and a later call succeeds.
_agent_cards: dict[str, Optional[dict]] = {key: None for key in AGENTS}


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # A2A contextId; frontend persists this per conversation


class ChatResponse(BaseModel):
    reply: str
    session_id: str


# ── A2A JSON-RPC client helpers ───────────────────────────────────────────────

async def _fetch_agent_card(client: httpx.AsyncClient, key: str) -> Optional[dict]:
    try:
        resp = await client.get(f"{AGENTS[key]}/.well-known/agent.json")
        resp.raise_for_status()
        card = resp.json()
        _agent_cards[key] = card
        return card
    except Exception:
        return None


def _agent_label(key: str) -> str:
    card = _agent_cards.get(key)
    return card["name"] if card else key


def _extract_task_text(task: dict) -> str:
    """Pull the agent's answer text out of a Task object (see the reference
    a2a_server.py's Task/Artifact shape): prefer the last artifact's text,
    fall back to the last agent message in history."""
    for artifact in reversed(task.get("artifacts") or []):
        parts = artifact.get("parts") or []
        texts = [p.get("text", "") for p in parts if p.get("kind") == "text"]
        joined = "".join(texts).strip()
        if joined:
            return joined
    for message in reversed(task.get("history") or []):
        if message.get("role") == "agent":
            parts = message.get("parts") or []
            texts = [p.get("text", "") for p in parts if p.get("kind") == "text"]
            joined = "".join(texts).strip()
            if joined:
                return joined
    return ""


async def _call_agent_send(client: httpx.AsyncClient, key: str, message: str, context_id: str) -> str:
    """message/send against one A2A agent. Never raises -- unreachable/erroring
    agents surface as a bracketed note in the combined reply instead of
    failing the whole request."""
    message_payload = {
        "role": "user",
        "parts": [{"kind": "text", "text": message}],
        "messageId": str(uuid.uuid4()),
        "contextId": context_id,
    }
    if key == "agent2":
        message_payload["metadata"] = {"email": AGENT2_EMAIL}
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {"message": message_payload},
    }
    label = _agent_label(key)
    try:
        resp = await client.post(AGENTS[key], json=payload)
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            return f"[{label} error: {body['error'].get('message', 'unknown error')}]"
        text = _extract_task_text(body.get("result") or {})
        return text or f"[{label} returned an empty answer.]"
    except Exception as exc:
        return f"[{label} unavailable ({type(exc).__name__}). Skipping this agent for now.]"


def _agent_card_summary(key: str) -> str:
    card = _agent_cards.get(key)
    if not card:
        return f'"{key}": offline, capabilities unknown'
    skills = card.get("skills") or []
    tags = sorted({t for s in skills for t in s.get("tags", [])})
    examples = [ex for s in skills for ex in (s.get("examples") or [])][:4]
    lines = [f'"{key}": {card["name"]} — {card["description"]}']
    if tags:
        lines.append(f"  tags: {', '.join(tags)}")
    if examples:
        lines.append(f"  example questions it answers: {'; '.join(examples)}")
    return "\n".join(lines)


async def classify_agents(client: httpx.AsyncClient, message: str) -> list[str]:
    """Ask a small local LLM which configured agent(s) should handle this
    query, based on each agent's own card. Routing only narrows the fan-out
    -- any failure to reach the router or parse its answer falls back to
    calling every agent, so a query is never silently dropped."""
    all_keys = list(AGENTS.keys())
    if len(all_keys) <= 1:
        return all_keys

    catalog = "\n".join(_agent_card_summary(key) for key in all_keys)
    prompt = (
        "You route a user's question to the right assistant(s) below. Reply "
        "with ONLY a JSON array of the agent keys that should answer, e.g. "
        '["agent2"].\n\n'
        "Default to picking exactly ONE key: whichever agent's description/"
        "tags/example questions most closely match this question's intent "
        "and phrasing. Only return more than one key if the question "
        "explicitly asks to compare, combine, or check across systems (e.g. "
        '"check both", "compare X and Y") -- a topic merely appearing on '
        "both agents' cards (like \"cameras\" or \"vehicles\") is NOT enough "
        "by itself to pick both; pick the single closer match instead. "
        "Never invent a key that isn't listed.\n\n"
        f"{catalog}\n\n"
        f"User question: {message}\n\nJSON array:"
    )
    try:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": ROUTER_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=_ROUTER_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            return all_keys
        picked = [k for k in json.loads(match.group(0)) if k in AGENTS]
        return picked or all_keys
    except Exception:
        return all_keys


async def run_multi_agent(message: str, session_id: str) -> str:
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        for key, card in _agent_cards.items():
            if card is None:
                await _fetch_agent_card(client, key)

        routed_keys = await classify_agents(client, message)
        results = await asyncio.gather(
            *(_call_agent_send(client, key, message, session_id) for key in routed_keys)
        )

    sections = [
        f"### {_agent_label(key)}\n{text}"
        for key, text in zip(routed_keys, results)
    ]
    return "\n\n".join(sections)


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        await asyncio.gather(*(_fetch_agent_card(client, key) for key in AGENTS))


@app.get("/health")
async def health():
    """Liveness probe + per-agent reachability, used by the frontend's status dot."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
        cards = await asyncio.gather(*(_fetch_agent_card(client, key) for key in AGENTS))
    return {
        "status": "ok",
        "agents": [
            {
                "key": key,
                "url": AGENTS[key],
                "online": card is not None,
                "name": card["name"] if card else None,
                "description": card["description"] if card else None,
            }
            for key, card in zip(AGENTS.keys(), cards)
        ],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a question and get back one combined answer from both agents."""
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="Message must not be empty.")
    session_id = req.session_id or str(uuid.uuid4())
    try:
        reply = await run_multi_agent(req.message, session_id)
        return ChatResponse(reply=reply, session_id=session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Same as /chat but streams each agent's answer as soon as it's ready
    (rather than waiting for all of them), as JSON-encoded SSE events:
      {"routed": [{"key": "agent2", "agent": "<name>"}, ...]}   -- sent first
      {"agent": "<name>", "key": "agent2", "text": "...", "final": true}
      ... one per routed agent ...
      {"done": true, "session_id": "..."}
    The frontend seeds one card per entry in "routed" (not one per configured
    agent) and detects image/video URLs inside "text" to show them inline.
    """
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="Message must not be empty.")
    session_id = req.session_id or str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            for key, card in _agent_cards.items():
                if card is None:
                    await _fetch_agent_card(client, key)

            routed_keys = await classify_agents(client, req.message)
            yield f"data: {json.dumps({'routed': [{'key': k, 'agent': _agent_label(k)} for k in routed_keys]})}\n\n"

            async def run_one(key: str):
                text = await _call_agent_send(client, key, req.message, session_id)
                return key, text

            for coro in asyncio.as_completed([run_one(key) for key in routed_keys]):
                key, text = await coro
                event = {
                    "agent": _agent_label(key),
                    "key": key,
                    "text": text,
                    "final": True,
                }
                yield f"data: {json.dumps(event)}\n\n"

        yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Frontend (static SPA) ─────────────────────────────────────────────────────
# Mounted last so it doesn't shadow the API routes above; open http://localhost:8000/
_frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8003)))
