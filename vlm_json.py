"""
Robust JSON extraction from a vision-language model reply.

Every VLM call in this pipeline asks for JSON and gets it *most* of the time.
The failure modes are boringly varied — a markdown fence, a sentence of preamble,
reasoning text leaking into the content, a truncated closing brace, a trailing
comma — and each one silently discarded a real result before this existed. In a
verification path that matters twice over: an unparseable reply is indistinguish-
able from "not an incident", so parser fragility quietly becomes a recall bug.

    from vlm_json import extract_json, response_text
    data = extract_json(response_text(response))
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>|<thinking>.*?</thinking>", re.S | re.I)
_FENCE = re.compile(r"```(?:json|JSON)?\s*|\s*```")


def response_text(response: Any) -> str:
    """The assistant's content, whichever shape the ollama client returns.

    The client has moved between plain dicts and pydantic models across versions;
    attribute access works on the model, subscripting on the dict, and neither
    works on both. Reasoning models also put chain-of-thought in a separate
    `thinking` field, which must be ignored rather than parsed.
    """
    message = None
    for accessor in (
        lambda: response.message,
        lambda: response["message"],
        lambda: response.get("message"),
    ):
        try:
            message = accessor()
            if message is not None:
                break
        except (AttributeError, KeyError, TypeError):
            continue
    if message is None:
        return str(response or "")

    for accessor in (
        lambda: message.content,
        lambda: message["content"],
        lambda: message.get("content"),
    ):
        try:
            content = accessor()
            if content is not None:
                return str(content)
        except (AttributeError, KeyError, TypeError):
            continue
    return str(message)


def _balanced_object(text: str) -> Optional[str]:
    """Outermost balanced {...}, respecting strings and escapes.

    A greedy regex breaks on a brace inside a quoted description, which is
    exactly what a model writes when it quotes a sign or a plate.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    # Unclosed: the reply was truncated mid-object. Close it and let the
    # field-level repair below salvage what arrived.
    if depth > 0:
        return text[start:] + ('"' if in_string else "") + "}" * depth
    return None


def _repair(candidate: str) -> Optional[Dict[str, Any]]:
    """Try increasingly aggressive repairs, applied CUMULATIVELY.

    Independent alternatives are not enough: a python-dict-style reply needs both
    the quote swap and the True/False/None swap at once, and applying either
    alone still fails to parse.
    """
    attempts = [candidate]
    # trailing commas
    attempts.append(re.sub(r",\s*([}\]])", r"\1", attempts[-1]))
    # python literals
    attempts.append(re.sub(r"\bTrue\b", "true",
                    re.sub(r"\bFalse\b", "false",
                    re.sub(r"\bNone\b", "null", attempts[-1]))))
    # single-quoted keys and values (last: it is the most likely to corrupt an
    # apostrophe inside a legitimate double-quoted string)
    attempts.append(re.sub(r"'([^']*)'", r'"\1"', attempts[-1]))

    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _field_scrape(text: str) -> Dict[str, Any]:
    """Last resort: pull known fields out of whatever arrived.

    Preferred over returning nothing, because for the incident verifier "nothing"
    is silently read as a rejection.
    """
    out: Dict[str, Any] = {}
    boolean = re.search(r'"?confirmed"?\s*[:=]\s*(true|false|yes|no)', text, re.I)
    if boolean:
        out["confirmed"] = boolean.group(1).lower() in ("true", "yes")
    number = re.search(r'"?confidence"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text, re.I)
    if number:
        try:
            out["confidence"] = float(number.group(1))
        except ValueError:
            pass
    for key in ("observed", "reason", "vehicle_type", "colour", "make_model", "plate_text",
                "carrying", "upper_colour", "lower_colour"):
        match = re.search(rf'"?{key}"?\s*[:=]\s*"([^"]*)"', text, re.I)
        if match:
            out[key] = match.group(1)
    array = re.search(r'"?actions"?\s*[:=]\s*\[([^\]]*)\]', text, re.I)
    if array:
        out["actions"] = [
            item.strip().strip('"\'') for item in array.group(1).split(",") if item.strip()
        ]
    return out


def extract_json(text: str, expect_keys: Optional[list] = None) -> Dict[str, Any]:
    """Best-effort dict from a model reply. Returns {} only if nothing usable."""
    if not text:
        return {}
    cleaned = _THINK_BLOCK.sub(" ", text)
    cleaned = _FENCE.sub(" ", cleaned).strip()

    parsed = _repair(cleaned)
    if parsed is not None:
        return parsed

    candidate = _balanced_object(cleaned)
    if candidate:
        parsed = _repair(candidate)
        if parsed is not None:
            return parsed

    scraped = _field_scrape(cleaned)
    if scraped:
        logger.debug("VLM reply needed field-level scraping: %r", cleaned[:160])
        if expect_keys and not any(key in scraped for key in expect_keys):
            return {}
        return scraped

    logger.debug("VLM reply unparseable: %r", cleaned[:200])
    return {}
