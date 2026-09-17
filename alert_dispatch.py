#!/usr/bin/env python3
"""
alert_dispatch.py - raise VMS alerts from the prose the VLM already wrote.

Alerts are derived at SUMMARY TIME, not during the per-segment scan: the summary
step already reads every stored description back out of MongoDB, so one extra
pass over that same text costs no video decoding and no VLM calls. Nothing here
runs during ingestion.

    segments in MongoDB ──► summarise_video() ──► GLM reads the descriptions
                                                    │
                                                    ├─► detailed summary (stored)
                                                    └─► alerts ──► POST /api/Analytics/analytics

What it can and cannot claim
    CAN   - crowd anomalies, unattended objects, suspicious activity, and
            vehicle-attribute alerts ("red bus", "white Toyota Innova"), because
            the VLM prose already describes those.
    CANNOT- registry-grade ANPR, vehicle tracking or cross-camera identity.
            Plates spotted in prose are SINGLE-READ: no multi-frame voting and no
            grammar check, which is exactly what offline_analytics.py exists to
            provide. They are emitted as `anpr_unconfirmed` and must never be
            treated as a confirmed registration.

Safety
    Posting is OFF by default. Set ALERT_POST_ENABLED=true to actually send.
    Until then every alert is logged as it would have been posted, so a run can
    be inspected before anything reaches a live control room.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ALERT_API_URL = os.getenv(
    "ALERT_API_URL", "https://vmsai2026.vmukti.com:8082/api/Analytics/analytics")

# Off by default. Nothing is sent until this is explicitly turned on.
ALERT_POST_ENABLED = os.getenv("ALERT_POST_ENABLED", "false").lower() == "true"

# The endpoint presents a certificate its chain does not validate here; set
# ALERT_API_VERIFY_TLS=true once a proper chain is in place.
ALERT_API_VERIFY_TLS = os.getenv("ALERT_API_VERIFY_TLS", "false").lower() == "true"
ALERT_API_TIMEOUT = float(os.getenv("ALERT_API_TIMEOUT", "30"))

# A ceiling per camera. A model that decides every minute is suspicious would
# otherwise post 720 alerts from one video and get the feed muted on day one.
MAX_ALERTS_PER_CAMERA = int(os.getenv("MAX_ALERTS_PER_CAMERA", "60"))

# ── the analytic id per alert type ────────────────────────────────────────────
# 41 is the only value supplied ("Box Detection"). Every type therefore defaults
# to it, which is ALMOST CERTAINLY WRONG for a real deployment: the receiving
# system uses an_id to route and classify. Fill these in from the VMS analytic
# registry before enabling posting, or every alert lands in one bucket.
DEFAULT_AN_ID = int(os.getenv("ALERT_DEFAULT_AN_ID", "41"))
AN_ID_BY_TYPE: Dict[str, int] = {
    "crowd_anomaly":       DEFAULT_AN_ID,
    "unattended_object":   DEFAULT_AN_ID,
    "suspicious_activity": DEFAULT_AN_ID,
    "vehicle_of_interest": DEFAULT_AN_ID,
    "anpr_unconfirmed":    DEFAULT_AN_ID,
    "traffic_violation":   DEFAULT_AN_ID,
}
# Optional JSON override: {"crowd_anomaly": 44, "unattended_object": 45, ...}
_an_id_json = os.getenv("ALERT_AN_ID_MAP")
if _an_id_json:
    try:
        AN_ID_BY_TYPE.update({k: int(v) for k, v in json.loads(_an_id_json).items()})
    except Exception as exc:
        logger.warning("ALERT_AN_ID_MAP ignored, not valid JSON: %s", exc)

# ── camera_id -> the VMS device id ────────────────────────────────────────────
# Ours look like "cam13_stream-13"; the VMS expects "VSPL-149178-ARCIS". There is
# no rule connecting the two, so it has to be a lookup. An unmapped camera is
# SKIPPED rather than posted under a made-up id - a real alert filed against the
# wrong camera sends officers to the wrong junction.
CAMERA_DID_MAP_FILE = os.getenv("CAMERA_DID_MAP", "camera_did_map.json")


def _load_camera_did_map() -> Dict[str, str]:
    path = CAMERA_DID_MAP_FILE
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return {str(k): str(v) for k, v in json.load(fh).items()
                    if not str(k).startswith("_")}
    except FileNotFoundError:
        logger.warning("no %s - every camera is unmapped, alerts will be skipped",
                       CAMERA_DID_MAP_FILE)
        return {}
    except Exception as exc:
        logger.error("could not read %s: %s", CAMERA_DID_MAP_FILE, exc)
        return {}


CAMERA_DID = _load_camera_did_map()

# ══════════════════════════════════════════════════════════════════════════════
# GLM via OpenRouter
# ══════════════════════════════════════════════════════════════════════════════

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.3-flash")


def _openrouter_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        try:                                   # reuse the key llm_backend holds
            import llm_backend
            key = getattr(llm_backend, "OPENROUTER_API_KEY", "") or ""
        except Exception:
            pass
    return key


def llm_text(prompt: str, max_tokens: int = 4000, temperature: float = 0.2,
             system: Optional[str] = None) -> str:
    """One synchronous GLM call. Returns "" on any failure - never raises.

    reasoning.effort=low is not optional. glm-5.3-flash REFUSES to disable
    reasoning ("Reasoning is mandatory for this endpoint"), and left alone it
    spends the whole completion budget thinking: measured at max_tokens=300, 298
    of 300 tokens were reasoning and `content` came back None. At effort=low the
    same call used 3 reasoning tokens - and returned a better answer, keeping
    "a white Innova" where the default run had flattened it to "a white car".
    """
    key = _openrouter_key()
    if not key:
        logger.error("no OPENROUTER_API_KEY - cannot reach %s", OPENROUTER_MODEL)
        return ""

    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning": {"effort": "low"},
    }
    # Every outbound AI call announces itself. Without this the only trace was
    # httpx's own logger, which merge-final-many does not enable - so a run gave
    # no way to tell whether it had reached OpenRouter at all.
    logger.info("[AI->] %s %s (max_tokens=%d, prompt=%d chars)",
                OPENROUTER_URL, OPENROUTER_MODEL, max_tokens, len(prompt))
    started = time.time()
    try:
        r = httpx.post(OPENROUTER_URL, timeout=180,
                       headers={"Authorization": f"Bearer {key}",
                                "Content-Type": "application/json"},
                       json=body)
        data = r.json()
    except Exception as exc:
        logger.error("[AI!!] OpenRouter call failed after %.1fs: %s",
                     time.time() - started, exc)
        return ""
    if r.status_code != 200 or "choices" not in data:
        logger.error("[AI!!] OpenRouter HTTP %s: %s",
                     r.status_code, json.dumps(data)[:300])
        return ""
    content = (data["choices"][0]["message"].get("content") or "").strip()
    usage = data.get("usage") or {}
    finish = (data["choices"][0].get("finish_reason") or "")
    if not content:
        # Empty content is indistinguishable from "nothing to report" downstream,
        # so it must not pass quietly: it usually means reasoning ate the budget.
        logger.warning("[AI!!] %s returned NO content (finish=%s, completion_tok=%s) "
                       "- raise max_tokens; treating as no result",
                       OPENROUTER_MODEL, finish, usage.get("completion_tokens"))
    logger.info("[AI<-] %s ok in %.1fs | prompt_tok=%s completion_tok=%s "
                "cost=$%s | %d chars returned",
                OPENROUTER_MODEL, time.time() - started,
                usage.get("prompt_tokens"), usage.get("completion_tokens"),
                usage.get("cost"), len(content))
    return content


# ══════════════════════════════════════════════════════════════════════════════
# Alert extraction
# ══════════════════════════════════════════════════════════════════════════════

ALERT_TYPES = ("crowd_anomaly", "unattended_object", "suspicious_activity",
               "vehicle_of_interest", "anpr_unconfirmed", "traffic_violation")

_EXTRACT_SYSTEM = (
    "You are a police control-room analyst reading automatically generated CCTV "
    "observations. You raise an alert ONLY for something an operator would act "
    "on. Routine traffic and ordinary pedestrians are NOT alerts. You never "
    "invent detail that the observations do not state."
)

_EXTRACT_PROMPT = """These are per-minute CCTV observations from camera {camera} at {location}.

{block}

Return ONLY a JSON object, no prose, in exactly this shape:
{{"alerts": [{{"segment_id": <the minute number>, "type": "<one of {types}>", "msg": "<short operator-facing label, at most 90 characters>"}}]}}

Raise an alert only for:
- crowd_anomaly       unusual gathering, congestion, queue or crowd build-up, with a number if stated
- unattended_object   a bag, package or vehicle left unattended or abandoned
- suspicious_activity loitering, someone watching parked vehicles, forced entry, a fight, a theft, a weapon, an accident, a person lying down
- vehicle_of_interest a vehicle worth flagging by its description - state colour, type and the make/model if the observations give one, e.g. "Red Bus Detected", "White Toyota Innova Detected"
- anpr_unconfirmed    a registration number that the observations state is legible; quote it exactly, e.g. "Plate Read (unconfirmed) GJ01AB1234"
- traffic_violation   wrong-way driving, red-light jumping, riding without a helmet, three on a motorcycle, driving on the footpath, illegal parking blocking a crossing

Rules:
- Use ONLY what the observations state. Never guess a plate, a make or a model.
- One alert per distinct thing. Do not repeat the same vehicle or the same crowd across consecutive minutes; report it once, at the minute it first appears.
- segment_id MUST be one of the minute numbers shown above.
- If nothing in this block warrants an alert, return {{"alerts": []}}.
- msg is what an operator sees in a list. Make it specific and readable.
"""


def _parse_alert_json(raw: str) -> List[Dict[str, Any]]:
    if not raw:
        return []
    text = raw.strip()
    if text.startswith("```"):                      # ```json ... ``` fences
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)    # first object in loose prose
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except Exception:
            return []
    alerts = data.get("alerts") if isinstance(data, dict) else data
    return [a for a in (alerts or []) if isinstance(a, dict)]


def extract_alerts(camera_id: str, location: str,
                   docs: List[Dict[str, Any]], chunk: int = 25) -> List[Dict[str, Any]]:
    """Read the stored descriptions and return the alerts worth raising.

    Chunked because a 12-hour camera is ~720 descriptions: well inside GLM's
    1.3M context on paper, but a single call over all of it reliably returns a
    thin, generic list. Smaller blocks keep the model specific.
    """
    by_id = {str(d.get("segment_id")): d for d in docs}
    found: List[Dict[str, Any]] = []
    seen: set = set()

    for start in range(0, len(docs), chunk):
        subset = docs[start:start + chunk]
        block = "\n\n".join(
            f"[minute {d.get('segment_id')}] {str(d.get('description') or '')[:1500]}"
            for d in subset)
        raw = llm_text(
            _EXTRACT_PROMPT.format(camera=camera_id, location=location,
                                   block=block, types="|".join(ALERT_TYPES)),
            max_tokens=1500, temperature=0.1, system=_EXTRACT_SYSTEM)

        for alert in _parse_alert_json(raw):
            atype = str(alert.get("type") or "").strip().lower()
            msg = str(alert.get("msg") or "").strip()
            seg = str(alert.get("segment_id"))
            if atype not in ALERT_TYPES or not msg:
                continue
            if seg not in by_id:                    # a minute it invented
                continue
            key = (atype, msg.lower())              # same thing, twice: keep one
            if key in seen:
                continue
            seen.add(key)
            found.append({"type": atype, "msg": msg[:90], "segment": by_id[seg]})
            if len(found) >= MAX_ALERTS_PER_CAMERA:
                logger.warning("[%s] alert cap %d reached, stopping extraction",
                               camera_id, MAX_ALERTS_PER_CAMERA)
                return found
    return found


# ══════════════════════════════════════════════════════════════════════════════
# Posting
# ══════════════════════════════════════════════════════════════════════════════

# Every alert type currently maps to the same an_id (41), because the VMS
# analytic registry is not published anywhere reachable - the endpoint has no
# list/types/swagger route. That would make a collision, a crowd and a plate read
# indistinguishable in the receiving system. `msg` is ours to shape, so the type
# is carried there as a readable prefix: an operator can still triage even while
# an_id is uniform, and nothing has to change when the real ids arrive.
TYPE_LABEL = {
    "crowd_anomaly":       "Crowd Anomaly",
    "unattended_object":   "Unattended Object",
    "suspicious_activity": "Suspicious Activity",
    "vehicle_of_interest": "Vehicle of Interest",
    "anpr_unconfirmed":    "ANPR Unconfirmed",
    "traffic_violation":   "Traffic Violation",
}
ALERT_MSG_PREFIX = os.getenv("ALERT_MSG_PREFIX", "true").lower() == "true"


def _msg(alert_type: str, msg: str) -> str:
    label = TYPE_LABEL.get(alert_type)
    if not ALERT_MSG_PREFIX or not label or msg.lower().startswith(label.lower()):
        return msg[:120]
    return f"{label}: {msg}"[:120]


# The brief said 'vidurl keep N/A in every case', but the endpoint rejects that:
#   {"success":false,"message":"vidurl must be an http(s) URL","recievedVidurl":"N/A"}
# So a literal N/A cannot be sent. ALERT_VIDURL decides what goes instead:
#   ""        (default) reuse the frame URL - always a valid http(s) URL
#   <url>     a fixed placeholder of your choosing
#   "N/A"     send it anyway, and let the endpoint reject it
ALERT_VIDURL = os.getenv("ALERT_VIDURL", "")


def _vidurl(image_url: str) -> str:
    if not ALERT_VIDURL:
        return image_url
    if ALERT_VIDURL.upper() == "N/A":
        logger.warning("ALERT_VIDURL=N/A - the endpoint rejects non-URL values, "
                       "this alert will be refused with HTTP 400")
    return ALERT_VIDURL


def _stamp(dt: datetime) -> str:
    """YYYY-mm-DD-HH-MM-SS-mmm, which is what the endpoint demands.

    ISO 8601 is refused outright:
      HTTP 535 {"message":"Server does not read this Date and Time",
                "suggestion":"Send the date in format YYYY-mm-DD-HH-MM-SS-000"}
    The sample payload in the brief used ISO, so this had to come from the API.
    It is also the convention already used in the stored frame filenames
    (…_frame1_2026-06-13-21-36-57.jpg), so the two now agree.
    """
    return dt.strftime("%Y-%m-%d-%H-%M-%S-") + f"{dt.microsecond // 1000:03d}"


def _sendtime(segment: Dict[str, Any]) -> str:
    """The segment's own wall-clock time as 2026-08-24-11-30-00-000.

    The alert time is when the thing HAPPENED, not when the summary ran - a
    control room searching by time would otherwise find every alert stamped with
    the moment the batch job finished.
    """
    raw = str(segment.get("start_time") or "")
    for parse in (lambda s: datetime.fromisoformat(s),
                  lambda s: datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")):
        try:
            dt = parse(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return _stamp(dt.astimezone(timezone.utc))
        except Exception:
            continue
    return _stamp(datetime.now(timezone.utc))


def build_payload(alert: Dict[str, Any], camera_id: str) -> Optional[Dict[str, Any]]:
    """The exact shape the endpoint expects, or None if it cannot be built."""
    segment = alert["segment"]
    cameradid = CAMERA_DID.get(camera_id)
    if not cameradid:
        logger.warning("[%s] no VMS device id in %s - alert skipped: %s",
                       camera_id, CAMERA_DID_MAP_FILE, alert["msg"])
        return None

    frames = [u for u in (segment.get("frame_urls") or [])
              if isinstance(u, str) and u.startswith("http")]
    if not frames:
        logger.warning("[%s] segment %s has no frame url - alert skipped: %s",
                       camera_id, segment.get("segment_id"), alert["msg"])
        return None

    return {
        "_id": str(segment.get("_id") or ""),
        "cameradid": cameradid,
        "sendtime": _sendtime(segment),
        "msg": _msg(alert["type"], alert["msg"]),
        "imgurl": frames[0],
        "vidurl": _vidurl(frames[0]),
        "an_id": AN_ID_BY_TYPE.get(alert["type"], DEFAULT_AN_ID),
        "ImgCount": 1,
    }


def post_alerts(alerts: List[Dict[str, Any]], camera_id: str) -> Dict[str, int]:
    """POST each alert. Returns {sent, failed, skipped}."""
    stats = {"sent": 0, "failed": 0, "skipped": 0}
    if not alerts:
        return stats

    payloads = []
    for alert in alerts:
        payload = build_payload(alert, camera_id)
        if payload is None:
            stats["skipped"] += 1
        else:
            payloads.append(payload)

    if not ALERT_POST_ENABLED:
        logger.warning("[%s] ALERT_POST_ENABLED is false - %d alert(s) NOT sent:",
                       camera_id, len(payloads))
        for p in payloads:
            logger.warning("[ALERT-DRY] would POST an_id=%s %s | %s",
                           p["an_id"], p["sendtime"], p["msg"])
        stats["skipped"] += len(payloads)
        return stats

    with httpx.Client(timeout=ALERT_API_TIMEOUT, verify=ALERT_API_VERIFY_TLS) as client:
        for payload in payloads:
            try:
                logger.info("[ALERT->] POST %s | an_id=%s cam=%s | %s",
                            ALERT_API_URL, payload["an_id"],
                            payload["cameradid"], payload["msg"])
                # FLAT, not {"data": {...}}. The wrapper was inferred from the
                # sample payload; the endpoint answers it with
                # {"success":false,"message":"All fields are required"} because it
                # sees none of the fields. Sent flat, validation proceeds.
                r = client.post(ALERT_API_URL, json=payload,
                                headers={"Content-Type": "application/json"})
                if 200 <= r.status_code < 300:
                    stats["sent"] += 1
                    logger.info("[ALERT<-] accepted HTTP %s: %s",
                                r.status_code, (r.text or "")[:160])
                    logger.info("[%s] alert sent: %s", camera_id, payload["msg"])
                else:
                    stats["failed"] += 1
                    logger.error("[ALERT!!] [%s] rejected HTTP %s: %s | %s",
                                 camera_id, r.status_code, r.text[:200], payload["msg"])
            except Exception as exc:
                stats["failed"] += 1
                logger.error("[ALERT!!] [%s] POST failed: %s | %s",
                             camera_id, exc, payload["msg"])
    return stats


def raise_alerts_for_summary(camera_id: str, location: str,
                             docs: List[Dict[str, Any]]) -> Dict[str, int]:
    """The single entry point called from the summary step."""
    if os.getenv("ENABLE_SUMMARY_ALERTS", "true").lower() != "true":
        logger.info("[%s] summary alerts disabled (ENABLE_SUMMARY_ALERTS)", camera_id)
        return {"sent": 0, "failed": 0, "skipped": 0}
    # In realtime mode every segment has already been assessed as it was stored.
    # Running the batch pass too would post each alert a second time.
    if ALERT_MODE == "realtime":
        logger.info("[%s] summary alert pass skipped - ALERT_MODE=realtime, "
                    "alerts were raised per segment", camera_id)
        return {"sent": 0, "failed": 0, "skipped": 0}
    try:
        alerts = extract_alerts(camera_id, location, docs)
    except Exception as exc:
        logger.error("[%s] alert extraction failed: %s", camera_id, exc)
        return {"sent": 0, "failed": 0, "skipped": 0}

    logger.info("[%s] %d alert(s) derived from %d segment description(s)",
                camera_id, len(alerts), len(docs))
    stats = post_alerts(alerts, camera_id)
    logger.info("[%s] alerts: %d sent, %d failed, %d skipped",
                camera_id, stats["sent"], stats["failed"], stats["skipped"])
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# REAL-TIME: one segment at a time, as it is stored
# ══════════════════════════════════════════════════════════════════════════════
#
# ALERT_MODE=realtime posts as each segment lands, instead of waiting for the
# video to finish. On a 12-hour file that is the difference between an alert
# arriving 60 seconds after the event and arriving 3 hours later.
#
# The cost of doing this naively is one GLM call per segment - 718 calls for one
# camera. Almost all of them would be wasted: a quiet minute's description says
# "1. ALERTS No alerts" and there is nothing to extract. So a free text gate runs
# first and only alert-worthy descriptions reach the model. On the cam14 sample
# that skipped roughly three quarters of segments.

ALERT_MODE = os.getenv("ALERT_MODE", "realtime").lower()   # realtime | summary | both

# Phrases that mean "nothing here". A description whose alert-bearing sections
# are all negative never reaches the model.
_NEGATIVE = ("no alerts", "none observed", "none.", "no violations",
             "nothing unusual", "no incidents", "none detected")

# Words that suggest something worth a second look. Deliberately broad - the
# model is the precision stage, this is only the recall stage.
_TRIGGERS = (
    "accident", "collision", "crash", "near-miss", "near miss",
    "fight", "altercation", "assault", "weapon", "knife", "gun", "firearm",
    "theft", "snatch", "robbery", "stolen",
    "abandoned", "unattended", "left behind",
    "loiter", "suspicious", "lying down", "trespass", "restricted",
    "crowd", "queue", "congestion", "jam", "gathering", "protest",
    "wrong way", "wrong-way", "wrong side", "red light", "red-light",
    "redlight", "jumping the signal", "without helmet",
    "no helmet", "helmetless", "three riders", "three people", "four people",
    "footpath", "illegal parking", "blocking", "overloaded", "speeding",
    "fire", "smoke", "flood",
)

# One alert per (type, subject) per this many minutes of footage. Without it a
# crowd that persists for twenty minutes posts twenty identical alerts.
ALERT_COOLDOWN_MINUTES = int(os.getenv("ALERT_COOLDOWN_MINUTES", "10"))

# Budget for the per-segment call. 400 was too tight: reasoning is mandatory on
# glm-5.3-flash and even at effort=low it occasionally spends the whole ceiling,
# returning 0 chars - which looks exactly like "no alert here" and silently drops
# a real one. Observed on cam13 segment 9: completion_tok=400, content empty, on
# a segment that had produced an alert at a larger budget.
ALERT_SEGMENT_MAX_TOKENS = int(os.getenv("ALERT_SEGMENT_MAX_TOKENS", "1200"))
_recent: Dict[str, Dict[str, float]] = {}


# Two things make a naive substring scan useless on a structured description,
# and both were measured on real cam16 output rather than guessed:
#
# 1. HEADINGS ARE FIXED TEXT. SEGMENT_STYLE=police heads a section "6. CROWD AND
#    CONGESTION", so "crowd" and "congestion" are present in every segment ever
#    written, including one whose every section reads "None observed."
# 2. THE MODEL IS ASKED TO STATE NEGATIVES, in exactly this vocabulary: "No
#    collisions, fires, unattended items or altercations observed." A denial is
#    the strongest possible evidence that nothing happened, and it was passing
#    the gate.
#
# Together those put 100% of segments through a filter designed to pass ~3%.
# Note also that clean_description_text() collapses ALL whitespace before
# storage, so the stored description is ONE line: nothing here may be anchored
# to line starts, which is why the section names are matched explicitly.
_SECTION_NAMES = (
    "ALERTS", "CHANGES ACROSS THE CLIP", "SCENE", "ENVIRONMENT",
    "TRAFFIC AND PEOPLE", "VEHICLES IDENTIFIED", "VEHICLES",
    "REGISTRATIONS READ", "NOTABLE EVENTS", "CROWD AND CONGESTION",
    "TRAFFIC VIOLATIONS", "PATTERN OF ACTIVITY", "QUIET PERIODS", "PEOPLE",
    "PERSON-OBJECT", "PERSON-PERSON", "OBJECTS",
)
_ANY_HEADING = (r"#{0,3}\s*\d{1,2}\.\s*(?:"
                + "|".join(re.escape(n) for n in _SECTION_NAMES) + r")\b")
_HEADING_RE = re.compile(_ANY_HEADING, re.I)

# A denial SPAN, not the whole clause: "no contact observed, but movement is
# fast ... possible near-miss" must keep the half after the comma. Removing the
# clause would delete the near-miss along with the denial.
_DENIAL_RE = re.compile(
    r"\bno(?:ne|thing)?\b[^.;]{0,150}?"
    r"\b(?:observed|detected|present|reported|builds|legible|identified|noted)\b",
    re.I)

# Sections whose body, when it is not a denial, is itself the finding.
_REPORT_SECTIONS = ("ALERTS", "NOTABLE EVENTS", "TRAFFIC VIOLATIONS")


def _section_body(text: str, name: str) -> Optional[str]:
    """The text under one numbered heading, up to the next known heading."""
    m = re.search(rf"#{{0,3}}\s*\d{{1,2}}\.\s*{re.escape(name)}\b(.*?)"
                  rf"(?={_ANY_HEADING}|\Z)", text, re.S | re.I)
    return m.group(1).strip(" -–—•:\t") if m else None


def _positive_prose(text: str) -> str:
    """The description with headings and denial spans removed."""
    return _DENIAL_RE.sub(" ", _HEADING_RE.sub(" ", text)).lower()


def _worth_asking(description: str) -> bool:
    raw = description or ""
    if not raw.strip():
        return False

    # A findings section that states something, in any words, counts. This is
    # the case a keyword list cannot cover: "a person is slumped over the
    # handlebars" is alert-worthy and contains no trigger word.
    for name in _REPORT_SECTIONS:
        body = _section_body(raw, name)
        if body and len(_DENIAL_RE.sub(" ", body).strip(" -–—•:.\t")) > 15:
            return True

    # Otherwise fall back to the keyword sweep, over positive prose only. This
    # is what still catches the unstructured and FAST_MODE descriptions, which
    # have no sections to read.
    return any(t in _positive_prose(raw) for t in _TRIGGERS)


def _subject(msg: str) -> str:
    """A crude key for 'the same thing again' - first three meaningful words."""
    words = re.findall(r"[a-z]{3,}", msg.lower())
    return " ".join(words[:3])


def _on_cooldown(camera_id: str, atype: str, msg: str, minute: float) -> bool:
    key = f"{atype}|{_subject(msg)}"
    seen = _recent.setdefault(camera_id, {})
    last = seen.get(key)
    if last is not None and (minute - last) < ALERT_COOLDOWN_MINUTES:
        return True
    seen[key] = minute
    return False


_SEGMENT_PROMPT = """This is one minute of automatically generated CCTV observation from camera {camera} at {location}.

{description}

Return ONLY a JSON object, no prose:
{{"alerts": [{{"type": "<one of {types}>", "msg": "<short operator-facing label, at most 90 characters>"}}]}}

Raise an alert ONLY for something an operator would act on right now:
- crowd_anomaly       unusual gathering, congestion, queue or crowd build-up
- unattended_object   a bag, package or vehicle left unattended or abandoned
- suspicious_activity loitering, watching parked vehicles, a fight, a theft, a weapon, an accident, a person lying down
- vehicle_of_interest a vehicle worth flagging by description - colour, type, and make/model only if the observation states one, e.g. "Red Bus Detected"
- anpr_unconfirmed    a registration the observation quotes as legible; quote it exactly
- traffic_violation   wrong-way driving, red-light jumping, no helmet, three on a motorcycle, footpath riding, illegal parking

Rules:
- Routine traffic and ordinary pedestrians are NOT alerts. Most minutes produce none.
- Use ONLY what the observation states. Never guess a plate, make or model.
- If nothing warrants an alert, return {{"alerts": []}}.
"""


def raise_alerts_for_segment(camera_id: str, location: str,
                             segment_doc: Dict[str, Any]) -> Dict[str, int]:
    """Called immediately after a segment is stored. Never raises.

    Must not break ingestion: a failure here is logged and the scan continues.
    A segment that produces no alert costs nothing - the gate is a substring
    test, not a model call.
    """
    stats = {"sent": 0, "failed": 0, "skipped": 0}
    if ALERT_MODE not in ("realtime", "both"):
        return stats

    description = str(segment_doc.get("description") or "")
    if description.startswith("[Error:") or not _worth_asking(description):
        return stats

    seg_id = segment_doc.get("segment_id")
    try:
        minute = float(segment_doc.get("cumulative_minutes")
                       or (float(seg_id) if seg_id is not None else 0.0))
    except (TypeError, ValueError):
        minute = 0.0

    logger.info("[%s] segment %s looks alert-worthy - asking %s",
                camera_id, seg_id, OPENROUTER_MODEL)
    raw = llm_text(
        _SEGMENT_PROMPT.format(camera=camera_id, location=location,
                               description=description[:4000],
                               types="|".join(ALERT_TYPES)),
        max_tokens=ALERT_SEGMENT_MAX_TOKENS, temperature=0.1, system=_EXTRACT_SYSTEM)

    alerts = []
    for alert in _parse_alert_json(raw):
        atype = str(alert.get("type") or "").strip().lower()
        msg = str(alert.get("msg") or "").strip()
        if atype not in ALERT_TYPES or not msg:
            continue
        if _on_cooldown(camera_id, atype, msg, minute):
            logger.info("[%s] segment %s alert suppressed (cooldown): %s",
                        camera_id, seg_id, msg)
            continue
        alerts.append({"type": atype, "msg": msg[:90], "segment": segment_doc})

    if not alerts:
        return stats
    logger.info("[%s] segment %s -> %d alert(s)", camera_id, seg_id, len(alerts))
    return post_alerts(alerts, camera_id)
