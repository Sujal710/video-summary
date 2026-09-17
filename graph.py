"""
LangGraph version of the chat routing + answer-generation pipeline:

    contextualize -> select_tool -> call_mcp -> generate_summary

This is the same sequence that used to live inline in mcp-main.py's
/api/chat/stream STEP 2 block — moved here as an explicit graph so the flow
shows up as a node-and-edge diagram in LangSmith/LangGraph Studio instead of
only a nested call-stack trace. Node bodies are unchanged logic (tool
selection, parameter normalization, per-tool answer generation) — only where
the code lives changed.

Streaming: generate_summary's actual text tokens come from
VideoSegmentSummarizer's ChatOllama-backed streaming methods
(summarize_time_range_streaming / summarize_search_results_streaming), which
now accept a `config` argument. When this node forwards its own `config` into
them, LangGraph's stream_mode="messages" sees those tokens automatically —
mcp-main.py consumes the compiled graph with astream(stream_mode=["messages"])
to get them live, one token at a time.

That covers LLM-generated text only. Everything else this node needs to send
to the client (static headers, "no data" messages, frame image URLs, and the
non-streaming tool 1/4/5 answers) is NOT an LLM token, so "messages" mode
can't carry it — LangGraph's "custom" stream_mode (get_stream_writer()/the
injected `writer` parameter) would normally be the way to send arbitrary data
like this, but it does not work from async nodes on Python < 3.11 (confirmed
empirically: even LangGraph's own documented get_stream_writer() example
raises "Called get_config outside of a runnable context" in this
environment, which runs Python 3.10). So this node instead takes a plain
asyncio.Queue passed in via state["queue"] (constructed by mcp-main.py before
invoking the graph) and pushes {"content": ...}/{"image_url": ...} onto it
directly — a manual side channel that sidesteps LangGraph's broken one
entirely. mcp-main.py drains stream_mode="messages" and this queue
concurrently and merges both into the SSE response.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List
from typing_extensions import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)


class ChatState(TypedDict, total=False):
    query: str                     # raw user message
    ctx_query: str                 # contextualized query
    tool_name: str
    params: Dict[str, Any]
    tool_result: Dict[str, Any]
    # Typed as Any, not asyncio.Queue: pydantic can't build a JSON schema for
    # a raw asyncio.Queue (breaks the /schemas endpoint), and Studio's own
    # "run" UI can't construct one when you type in test input either — it
    # just omits the field entirely, so generate_summary must tolerate this
    # being absent (see there) rather than assume mcp-main.py always
    # supplies one.
    queue: Any                      # side channel for non-LLM-token SSE content
    full_answer: str
    image_urls: List[str]


def build_chat_graph(rag_client):
    """Compile the chat graph, closing over the already-initialized
    IntegratedVideoRAGClient instance (the same one mcp-main.py builds once
    at startup) so nodes reuse its contextual_agent/query_generator/
    summarizer/MCP session rather than constructing their own."""

    async def contextualize(state: ChatState) -> Dict[str, Any]:
        # "list cameras" / "list plates" bypass queries already carry a
        # ctx_query set by the caller — nothing to resolve.
        if state.get("ctx_query"):
            return {}
        ctx_query = await rag_client.contextual_agent.process_query(state["query"])
        return {"ctx_query": ctx_query}

    async def select_tool(state: ChatState) -> Dict[str, Any]:
        # Bypass queries already carry a tool_name — skip LLM tool selection.
        if state.get("tool_name"):
            return {}

        llm_response = await rag_client.query_generator.generate_tool_call(state["ctx_query"])
        if not llm_response or "tool_call" not in llm_response:
            raise ValueError("Could not determine correct action for this query.")

        tool_call = llm_response["tool_call"]
        tool_name = tool_call["name"]
        params = tool_call.get("parameters", {}) or {}

        # search_segments_by_activity has no "start_date" param — only
        # get_last_n_hours_summary does — but the LLM sometimes reaches for
        # it anyway to mean "start of range" for tool 3 too. Remap it into
        # "date" (tool 3's actual range-start param) before the strip below
        # and before the "ensure date present" fallback further down.
        if tool_name == "search_segments_by_activity" and "start_date" in params:
            params["date"] = params.pop("start_date")

        # Strip any params the tool-selection LLM hallucinated that the
        # target tool doesn't actually accept — the MCP server rejects the
        # whole call with a validation error otherwise.
        allowed_params = rag_client.VALID_PARAMS.get(tool_name)
        if allowed_params is not None:
            bad_params = [k for k in params if k not in allowed_params]
            if bad_params:
                logger.warning(f"⚠️ Removing unexpected params for {tool_name}: {bad_params}")
                params = {k: v for k, v in params.items() if k in allowed_params}

        # Normalize parameters for tool 3 (keywords)
        if tool_name == "search_segments_by_activity":
            q = params.get("query")
            if isinstance(q, str) and ',' in q:
                params["query"] = [x.strip() for x in q.split(',') if x.strip()]
            elif not isinstance(q, list):
                params["query"] = [str(q)] if q else []

            if not params.get("query"):
                raise ValueError(
                    "The AI couldn't determine what to search for — "
                    "please rephrase your question with a specific "
                    "object, person, or activity."
                )
            # `date` is intentionally left as whatever the LLM decided (or
            # omitted) — a keyword search with no date mentioned means
            # "search every date", not "default to today".

        # Normalize date for tool 2
        if tool_name == "get_last_n_hours_summary" and "date" not in params:
            from datetime import datetime, timezone
            params["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        logger.info(f"🛠️ Tool identified: {tool_name} | Params: {params}")
        return {"tool_name": tool_name, "params": params}

    async def call_mcp(state: ChatState) -> Dict[str, Any]:
        mcp_result = await rag_client.call_tool_with_reconnect(state["tool_name"], state["params"])
        if not mcp_result or not mcp_result.content:
            raise ValueError("Empty response from MCP server")

        raw_text = mcp_result.content[0].text
        try:
            tool_result = json.loads(raw_text)
        except json.JSONDecodeError:
            if "validation error" in raw_text.lower():
                raise ValueError(f"Server rejected tool call: {raw_text.strip()}")
            raise ValueError(f"Invalid JSON from MCP server: {raw_text[:200]!r}")

        return {"tool_result": tool_result}

    async def generate_summary(state: ChatState, config: RunnableConfig) -> Dict[str, Any]:
        """Dispatches to the right answer-generation path for tool_name. Real
        LLM text reaches the client via stream_mode="messages" (see module
        docstring); everything else is pushed onto state["queue"] directly.

        state.get (not state[...]) for queue: mcp-main.py always supplies
        one, but Studio's own "run" UI can't construct an asyncio.Queue from
        typed-in test input, so it's simply absent when invoked that way —
        _emit below no-ops in that case rather than crashing, same pattern
        summary.py's streaming methods already use for the same reason."""
        queue = state.get("queue")

        def _emit(payload: Dict[str, Any]):
            if queue is not None:
                queue.put_nowait(payload)

        tool_name = state["tool_name"]
        tool_result = state["tool_result"]
        params = state["params"]
        query = state["query"]
        ctx_query = state["ctx_query"]

        full_answer = ""
        image_urls: List[str] = []

        # ── Tools 1/4/5: non-streaming summary, manually chunked for the SSE wire ──
        if tool_name in ("list_available_cameras", "list_recognized_plates", "search_car_by_plate_number"):
            answer = await rag_client._generate_summary(tool_name, tool_result, query, ctx_query, params)
            full_answer = answer
            chunk_size = 50
            for i in range(0, len(answer), chunk_size):
                _emit({"content": answer[i:i + chunk_size]})

            if tool_name == "search_car_by_plate_number":
                for r in tool_result.get("results", []):
                    img_url = r.get("frame_url")
                    if img_url:
                        image_urls.append(img_url)
                        _emit({"image_url": img_url})

        # ── Tool 2: get_last_n_hours_summary (real streaming) ──────────────
        elif tool_name == "get_last_n_hours_summary":
            segments = tool_result.get("segments", [])
            camera_id = tool_result.get("camera_id", params.get("camera_id", "all cameras"))

            hours = 0.0
            tr = tool_result.get("time_range", {})
            if tr.get("start") and tr.get("end"):
                try:
                    from datetime import datetime
                    s = datetime.fromisoformat(tr["start"].replace("Z", "+00:00"))
                    e = datetime.fromisoformat(tr["end"].replace("Z", "+00:00"))
                    hours = (e - s).total_seconds() / 3600
                except Exception:
                    pass

            async for chunk in rag_client.summarizer.summarize_time_range_streaming(
                query, ctx_query, segments, camera_id, hours, queue=queue, config=config
            ):
                if chunk == "__KEEPALIVE__":
                    continue
                if chunk.startswith("http"):
                    image_urls.append(chunk.strip())
                else:
                    full_answer += chunk

        # ── Tool 3: search_segments_by_activity (real streaming with images) ──
        elif tool_name == "search_segments_by_activity":
            segments = tool_result.get("segments", [])
            camera_id = tool_result.get("camera_id", params.get("camera_id", "all cameras"))
            keywords = params.get("query", [])
            if isinstance(keywords, str):
                keywords = [keywords]

            async for chunk in rag_client.summarizer.summarize_search_results_streaming(
                query, ctx_query, segments, camera_id, keywords, queue=queue, config=config
            ):
                if chunk == "__KEEPALIVE__":
                    continue
                if chunk.startswith("http"):
                    image_urls.append(chunk.strip())
                else:
                    full_answer += chunk

        else:
            # Fallback for unknown tools
            answer = json.dumps(tool_result, indent=2)
            full_answer = answer
            _emit({"content": answer})

        return {"full_answer": full_answer, "image_urls": image_urls}

    graph = StateGraph(ChatState)
    graph.add_node("contextualize", contextualize)
    graph.add_node("select_tool", select_tool)
    graph.add_node("call_mcp", call_mcp)
    graph.add_node("generate_summary", generate_summary)

    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "select_tool")
    graph.add_edge("select_tool", "call_mcp")
    graph.add_edge("call_mcp", "generate_summary")
    graph.add_edge("generate_summary", END)

    return graph.compile()


async def stream_chat_graph(chat_graph, initial_state: ChatState):
    """Run the compiled graph once, yielding SSE-shaped payloads —
    {"content": ...} / {"image_url": ...} — merged from two sources that
    have to be drained concurrently:

      1. LangGraph's stream_mode="messages": real LLM tokens, forwarded
         automatically once generate_summary passes `config` into the
         ChatOllama call underneath it.
      2. state["queue"]: everything else generate_summary needs to send
         (headers, "no data" messages, frame image URLs, non-streaming
         tool 1/4/5 answers) — pushed there directly since LangGraph's own
         "custom" stream mode doesn't work on this Python version (see
         module docstring).

    `initial_state["queue"]` must be a fresh asyncio.Queue() the caller
    creates per request — this function owns driving the graph via
    astream(), consumes that queue as a side effect of that, and pushes a
    sentinel onto it once the graph finishes so queue-draining stops too.

    Yields normal SSE payload dicts throughout, then one final
    {"__done__": True, "full_answer": ..., "image_urls": [...]} dict —
    reconstructed from everything actually streamed, not from the node's
    return value — for the caller to save to chat history / offer to cache.
    """
    queue: asyncio.Queue = initial_state["queue"]
    out_q: asyncio.Queue = asyncio.Queue()
    _DONE = object()
    full_answer_parts: List[str] = []
    image_urls: List[str] = []

    async def pump_messages():
        async for _mode, chunk in chat_graph.astream(initial_state, stream_mode=["messages"]):
            msg, _meta = chunk
            content = getattr(msg, "content", None)
            if content:
                full_answer_parts.append(content)
                out_q.put_nowait({"content": content})
        # The graph (and therefore generate_summary) is fully done now —
        # let the queue drainer stop once it's drained anything already
        # enqueued, instead of waiting forever for the next item.
        queue.put_nowait(None)
        out_q.put_nowait(_DONE)

    async def pump_queue():
        while True:
            item = await queue.get()
            if item is None:
                break
            if "image_url" in item:
                image_urls.append(item["image_url"])
            elif "content" in item:
                full_answer_parts.append(item["content"])
            out_q.put_nowait(item)
        out_q.put_nowait(_DONE)

    t1 = asyncio.create_task(pump_messages())
    t2 = asyncio.create_task(pump_queue())

    try:
        done_count = 0
        while done_count < 2:
            item = await out_q.get()
            if item is _DONE:
                done_count += 1
                continue
            yield item
    finally:
        # Reached on normal completion (both pumps already finished, so
        # these are no-ops) or on early exit (caller's `async for` broke,
        # e.g. client disconnected) — cancel both pumps, which propagates
        # into and cancels the still-running graph execution too.
        for t in (t1, t2):
            if not t.done():
                t.cancel()
        await asyncio.gather(t1, t2, return_exceptions=True)

    yield {"__done__": True, "full_answer": "".join(full_answer_parts), "image_urls": image_urls}


_studio_graph = None
_studio_graph_lock = None


async def make_graph(config: Dict[str, Any] = None):
    """Factory entrypoint for `langgraph dev` / LangGraph Studio (see
    langgraph.json's "graphs" entry). mcp-main.py doesn't use this — it
    builds rag_client itself at startup and calls build_chat_graph directly.
    This exists only so the CLI/Studio has a real, connected rag_client to
    run the graph against, since build_chat_graph alone needs one passed in.

    Requires the same services mcp-main.py needs to already be running:
      - Ollama (localhost:11434)
      - The MCP video-retrieval server, server1.py (localhost:8088) —
        `python server1.py`

    Cached at module level: langgraph dev's in-memory runtime calls this
    factory fresh on EVERY API request (every /schemas, /graph, /subgraphs
    call, confirmed by log inspection) — not once at startup like
    mcp-main.py does. Reconnecting a new MCP SSE session per call, and
    leaving each prior one dangling, crashed with
    "RuntimeError: Attempted to exit cancel scope in a different task than
    it was entered in" once enough had piled up (anyio's task groups require
    the same task to enter and exit a scope; the eventually-garbage-collected
    old connections were being torn down from whatever task happened to run
    next). Building rag_client/the graph once and reusing it — same pattern
    mcp-main.py already uses for its own single long-lived instance — avoids
    opening a new connection per request entirely.
    """
    global _studio_graph, _studio_graph_lock
    if _studio_graph_lock is None:
        _studio_graph_lock = asyncio.Lock()

    async with _studio_graph_lock:
        if _studio_graph is None:
            from client import IntegratedVideoRAGClient

            rag_client = IntegratedVideoRAGClient()
            await rag_client.connect_to_mcp_server("http://localhost:8088/sse")
            _studio_graph = build_chat_graph(rag_client)

    return _studio_graph
