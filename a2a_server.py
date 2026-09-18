"""
A2A (Agent2Agent protocol) wrapper around this project's chatbot.ask().

Speaks the widely-deployed 0.2/0.3-era A2A wire format -- JSON-RPC 2.0 methods
named "message/send"/"message/stream"/"tasks/get"/"tasks/cancel", lowercase
TaskState strings ("submitted"/"working"/...), and Part objects discriminated
by a "kind" field ("text"). Modeled directly on this same repo family's own
a2a_server.py reference (a video-surveillance A2A agent wrapping a LangGraph
chat_graph) -- kept wire-compatible with it rather than the newer protobuf-derived
`a2a-sdk` schema (ALL_CAPS enums, bundled DB/migrations layer), which is not
what real A2A clients in the wild currently speak.

Unlike that reference, chatbot.ask() is synchronous (no async generator, no
token-level streaming) and is itself a full request/response call into a
LangGraph graph with real per-session state -- a paused write-action
confirmation (see chatbot.py's _tools_node/interrupt()) persists in that
graph's checkpointer across turns, keyed by session_id. This wrapper reuses
the A2A contextId as that session_id, so a caller keeps sending the same
contextId across a conversation to stay on the same session -- including
resuming a pending confirmation with a follow-up "yes"/"no" message.

This is the SAME process family as app.py's REST API (imports chatbot.py
directly, no extra service) -- run it as a second process alongside app.py,
on its own port, since both import chatbot.py's module-level graph/singletons.

Run with:
    python a2a_server.py
Agent card is served at:
    GET /.well-known/agent.json
JSON-RPC endpoint (message/send, message/stream, tasks/get, tasks/cancel):
    POST /
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from chatbot import ask, reset_session

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

A2A_HOST = os.getenv("A2A_HOST", "0.0.0.0")
A2A_PORT = int(os.getenv("A2A_PORT", "8092"))
A2A_PUBLIC_URL = os.getenv("A2A_PUBLIC_URL", f"http://localhost:{A2A_PORT}/")

# Exact strings chatbot.ask() returns on a hard failure (see chatbot.py's
# GraphRecursionError/Exception handlers) -- matched by prefix below so those
# two known failure paths map to TaskState "failed" instead of the default
# "input-required" used for an actual pending write-action confirmation. Both
# cases share the same "qa_id is None" signal from ask(), so this is the only
# way to tell them apart without changing chatbot.py's return shape.
_FAILURE_PREFIXES = (
    "I wasn't able to finish answering that",
    "Sorry, I'm having trouble reaching the assistant service",
)


# ═══════════════════════════════════════════════════════════════════════════
# Wire types (A2A 0.2/0.3 JSON-RPC shape)
# ═══════════════════════════════════════════════════════════════════════════

class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str


Part = TextPart  # only text parts are produced/consumed here -- see module docstring


class Message(BaseModel):
    role: Literal["user", "agent"]
    parts: List[Part]
    messageId: str
    taskId: Optional[str] = None
    contextId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskStatus(BaseModel):
    state: Literal[
        "submitted", "working", "input-required",
        "completed", "canceled", "failed",
    ]
    message: Optional[Message] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Artifact(BaseModel):
    artifactId: str
    name: Optional[str] = None
    parts: List[Part]


class Task(BaseModel):
    id: str
    contextId: str
    status: TaskStatus
    artifacts: List[Artifact] = Field(default_factory=list)
    history: List[Message] = Field(default_factory=list)
    kind: Literal["task"] = "task"
    metadata: Optional[Dict[str, Any]] = None


class TaskStatusUpdateEvent(BaseModel):
    taskId: str
    contextId: str
    status: TaskStatus
    final: bool
    kind: Literal["status-update"] = "status-update"


class TaskArtifactUpdateEvent(BaseModel):
    taskId: str
    contextId: str
    artifact: Artifact
    append: bool = False
    lastChunk: bool = True
    kind: Literal["artifact-update"] = "artifact-update"


# ═══════════════════════════════════════════════════════════════════════════
# In-memory task store -- no DB, matches "lightweight wrapper" scope. A task
# lives for the life of this process; nothing here needs to survive a
# restart. The actual conversation state that DOES need to survive a paused
# confirmation across turns lives in chatbot.py's own LangGraph checkpointer,
# keyed by contextId (see _handle_message below) -- this dict is purely A2A
# bookkeeping (tasks/get, tasks/cancel).
# ═══════════════════════════════════════════════════════════════════════════

_tasks: Dict[str, Task] = {}


def _extract_query(message: Dict[str, Any]) -> str:
    parts = message.get("parts") or []
    texts = [p.get("text", "") for p in parts if p.get("kind") == "text"]
    return " ".join(t for t in texts if t).strip()


def _jsonrpc_result(req_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _append_media_links(answer: str, images: List[str], videos: List[str], files: List[str]) -> str:
    """Same convention app.py's REST /chat response uses for the frontend:
    images/videos/files are separate payloads from the pipeline, never part
    of the answer text itself. Part here is TextPart-only (see module
    docstring), so the only way to surface them in a Task's single "answer"
    artifact is to append them as their own lines instead of silently
    dropping them."""
    extra_lines = [*images, *videos, *files]
    if not extra_lines:
        return answer
    return answer.rstrip("\n") + "\n" + "\n".join(extra_lines)


# ═══════════════════════════════════════════════════════════════════════════
# Agent Card
# ═══════════════════════════════════════════════════════════════════════════

AGENT_CARD: Dict[str, Any] = {
    "name": "Arcis Election Surveillance Chat Agent",
    "description": (
        "In-app assistant for Arcis, an election-monitoring video "
        "surveillance platform. Answers questions about cameras, live "
        "streams, AI detection alerts, GPS/vehicle tracking, incidents, "
        "inventory, helpdesk activity, and subscriptions -- organized by "
        "district and assembly constituency -- and can perform a small set "
        "of confirmed write actions (log an incident, log helpdesk "
        "activity, update AI detection settings)."
    ),
    "url": A2A_PUBLIC_URL,
    "version": "1.0.0",
    "capabilities": {"streaming": True, "pushNotifications": False},
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "skills": [
        {
            "id": "arcis_election_surveillance_chat",
            "name": "Arcis election surveillance chat",
            "description": (
                "Natural-language Q&A over camera/stream status, AI "
                "detection alerts, GPS/vehicle tracking, incidents, "
                "district/assembly camera coverage, and how-to guidance for "
                "the Arcis web app; can also create an incident, log "
                "helpdesk activity, or update AI detection settings after "
                "an explicit confirm step."
            ),
            "tags": ["election", "surveillance", "cameras", "incidents", "gps"],
            "examples": [
                "How many cameras are connected right now?",
                "Show all districts where cameras are installed",
                "Give me the names of all cameras in Rajkot",
                "Log a helpdesk visit for camera ATPL-908610-ARCIS",
            ],
            "inputModes": ["text"],
            "outputModes": ["text"],
        }
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Arcis Election Surveillance A2A Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/.well-known/agent.json")
async def agent_card():
    return JSONResponse(AGENT_CARD)


def _classify_state(qa_id: Optional[str], answer: str) -> str:
    if qa_id is not None:
        return "completed"
    if answer.startswith(_FAILURE_PREFIXES):
        return "failed"
    # qa_id is None and it's not a known failure string -> a paused write
    # action awaiting "yes"/"no" (see chatbot.py's interrupt()/_tools_node).
    return "input-required"


def _make_task(
    task_id: str, context_id: str, state: str,
    user_message: Message, agent_text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Task:
    history = [user_message]
    artifacts: List[Artifact] = []
    if agent_text is not None:
        agent_message = Message(
            role="agent",
            parts=[TextPart(text=agent_text)],
            messageId=str(uuid.uuid4()),
            taskId=task_id,
            contextId=context_id,
        )
        history.append(agent_message)
        artifacts.append(Artifact(artifactId=str(uuid.uuid4()), name="answer", parts=[TextPart(text=agent_text)]))
    return Task(
        id=task_id,
        contextId=context_id,
        status=TaskStatus(state=state),
        artifacts=artifacts,
        history=history,
        metadata=metadata,
    )


async def _run_ask(context_id: str, query: str, message: Dict[str, Any]) -> tuple[str, List[str], List[str], List[str], Optional[str]]:
    """chatbot.ask() is synchronous and takes an internal lock (see
    chatbot.py) -- runs it off the event loop thread so one slow/blocked
    session can't stall every other request this process is serving."""
    metadata = message.get("metadata") or {}
    auth_token = metadata.get("auth_token")
    email = metadata.get("email")
    confirm = metadata.get("confirm")  # explicit override; otherwise ask() parses "yes"/"no" from the text itself
    return await asyncio.to_thread(
        ask, context_id, query, auth_token=auth_token, confirm=confirm, email=email
    )


async def _handle_message_send(params: Dict[str, Any]) -> Task:
    message = params.get("message") or {}
    query = _extract_query(message)
    task_id = message.get("taskId") or str(uuid.uuid4())
    # contextId doubles as chatbot.py's session_id/thread_id -- a caller must
    # reuse the same contextId across turns to stay on one conversation
    # (including resuming a paused confirmation).
    context_id = message.get("contextId") or str(uuid.uuid4())

    user_message = Message(
        role="user",
        parts=[TextPart(text=query)],
        messageId=message.get("messageId", str(uuid.uuid4())),
        taskId=task_id,
        contextId=context_id,
    )

    if not query:
        task = _make_task(task_id, context_id, "failed", user_message,
                           "No text found in the message -- nothing to ask.")
        _tasks[task_id] = task
        return task

    try:
        answer, images, videos, files, qa_id = await _run_ask(context_id, query, message)
        full_answer = _append_media_links(answer, images, videos, files)
        state = _classify_state(qa_id, answer)
        task = _make_task(task_id, context_id, state, user_message, full_answer,
                           metadata={"qa_id": qa_id} if qa_id else None)
    except Exception as e:
        logger.error(f"message/send failed: {e}", exc_info=True)
        task = _make_task(task_id, context_id, "failed", user_message, f"Error: {e}")

    _tasks[task_id] = task
    return task


async def _stream_message_events(params: Dict[str, Any], req_id: Any):
    """Async generator of raw SSE lines for message/stream. chatbot.ask()
    has no token-level streaming (it's one blocking call into a LangGraph
    graph, see module docstring) -- this still yields the standard
    working -> artifact-update -> status-update(final) sequence, just as a
    single artifact chunk instead of many, so streaming-only A2A clients get
    a correctly-shaped event sequence rather than nothing."""
    message = params.get("message") or {}
    query = _extract_query(message)
    task_id = message.get("taskId") or str(uuid.uuid4())
    context_id = message.get("contextId") or str(uuid.uuid4())

    user_message = Message(
        role="user",
        parts=[TextPart(text=query)],
        messageId=message.get("messageId", str(uuid.uuid4())),
        taskId=task_id,
        contextId=context_id,
    )

    def _sse(event: BaseModel) -> str:
        return f"data: {json.dumps(_jsonrpc_result(req_id, event.model_dump()))}\n\n"

    if not query:
        task = _make_task(task_id, context_id, "failed", user_message,
                           "No text found in the message -- nothing to ask.")
        _tasks[task_id] = task
        yield _sse(TaskStatusUpdateEvent(taskId=task_id, contextId=context_id, status=task.status, final=True))
        return

    _tasks[task_id] = _make_task(task_id, context_id, "submitted", user_message)
    yield _sse(TaskStatusUpdateEvent(
        taskId=task_id, contextId=context_id,
        status=TaskStatus(state="working"), final=False,
    ))

    try:
        answer, images, videos, files, qa_id = await _run_ask(context_id, query, message)
        full_answer = _append_media_links(answer, images, videos, files)

        artifact = Artifact(artifactId=task_id, name="answer", parts=[TextPart(text=full_answer)])
        yield _sse(TaskArtifactUpdateEvent(
            taskId=task_id, contextId=context_id, artifact=artifact,
            append=False, lastChunk=True,
        ))

        state = _classify_state(qa_id, answer)
        task = _make_task(task_id, context_id, state, user_message, full_answer,
                           metadata={"qa_id": qa_id} if qa_id else None)
        _tasks[task_id] = task
        yield _sse(TaskStatusUpdateEvent(taskId=task_id, contextId=context_id, status=task.status, final=True))

    except asyncio.CancelledError:
        # The client closing the SSE connection lands here (StreamingResponse
        # cancels the generator on disconnect) -- this is the actual way a
        # message/stream task stops early. tasks/cancel (below) is a separate
        # JSON-RPC call on a NEW request and has no handle onto this
        # generator to cancel it directly; it only updates the stored Task's
        # bookkeeping for a subsequent tasks/get.
        task = _tasks.get(task_id)
        if task:
            task.status = TaskStatus(state="canceled")
        raise
    except Exception as e:
        logger.error(f"message/stream failed: {e}", exc_info=True)
        task = _make_task(task_id, context_id, "failed", user_message, f"Error: {e}")
        _tasks[task_id] = task
        yield _sse(TaskStatusUpdateEvent(taskId=task_id, contextId=context_id, status=task.status, final=True))


@app.post("/")
async def jsonrpc_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_jsonrpc_error(None, -32700, "Parse error"), status_code=400)

    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params") or {}

    if method == "message/send":
        task = await _handle_message_send(params)
        return JSONResponse(_jsonrpc_result(req_id, task.model_dump()))

    if method == "message/stream":
        async def event_stream():
            async for line in _stream_message_events(params, req_id):
                yield line

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    if method == "tasks/get":
        task_id = params.get("id")
        task = _tasks.get(task_id)
        if task is None:
            return JSONResponse(_jsonrpc_error(req_id, -32001, f"Task not found: {task_id}"), status_code=404)
        return JSONResponse(_jsonrpc_result(req_id, task.model_dump()))

    if method == "tasks/cancel":
        task_id = params.get("id")
        task = _tasks.get(task_id)
        if task is None:
            return JSONResponse(_jsonrpc_error(req_id, -32001, f"Task not found: {task_id}"), status_code=404)
        if task.status.state in ("submitted", "working", "input-required"):
            # Unlike a plain in-flight LLM call, "input-required" here means
            # chatbot.py's graph is genuinely paused on a write-action
            # confirmation (see interrupt() in chatbot.py's _tools_node) --
            # cancel actually clears that paused session server-side too, so
            # a later message/send with this same contextId starts a fresh
            # turn instead of silently resuming the old confirmation.
            reset_session(task.contextId)
            task.status = TaskStatus(state="canceled")
        return JSONResponse(_jsonrpc_result(req_id, task.model_dump()))

    return JSONResponse(_jsonrpc_error(req_id, -32601, f"Method not found: {method}"), status_code=404)


if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("ARCIS ELECTION SURVEILLANCE A2A AGENT")
    print("=" * 70)
    print(f"Agent card : http://localhost:{A2A_PORT}/.well-known/agent.json")
    print(f"JSON-RPC   : http://localhost:{A2A_PORT}/  (message/send, message/stream, tasks/get, tasks/cancel)")
    print("Wraps      : chatbot.py's ask()/reset_session() -- same Mongo/Ollama/OpenRouter config as app.py")
    print("=" * 70)

    uvicorn.run(app, host=A2A_HOST, port=A2A_PORT, log_level="info")
