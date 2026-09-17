"""
Integrated Video Segment RAG Client with Ollama Summarization

Three tools (matching server exactly):
  1. list_available_cameras       — no params
  2. get_last_n_hours_summary     — date/time range query
  3. search_segments_by_activity  — keyword search (query required, date OPTIONAL —
                                     omit entirely when the user doesn't mention one,
                                     so the search runs across every date)

Routing logic:
  - "list cameras / stats"        → list_available_cameras
  - query contains "summary" OR asks for a general overview of a time period
                                  → get_last_n_hours_summary
  - everything else (objects, people types, vehicles, activities, events)
                                  → search_segments_by_activity
"""

import asyncio
import json
import logging
import re
import os
from typing import Optional, Dict, Any

import anyio
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from ollama import AsyncClient
from dotenv import load_dotenv
from langsmith import traceable, get_current_run_tree

from summary import VideoSegmentSummarizer
from context import VideoContextualAgent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class VideoQueryGenerator:
    """Uses Ollama to map a user query to one of three MCP tool calls."""

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        logger.info(f"✅ Query Generator initialized: {self.model}")

    # ── system prompt ─────────────────────────────────────────────────────────

    def get_system_prompt(self) -> str:
        from datetime import datetime, timedelta
        now = datetime.now()
        today     = now.strftime("%Y-%m-%d")
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")

        return f"""You are a routing assistant for a video surveillance system.
Your ONLY job is to pick the correct tool and extract its parameters from the user query.

TODAY  = {today}
YESTERDAY = {yesterday}

════════════════════════════════════════════════
AVAILABLE TOOLS  (EXACTLY FIVE — no others)
════════════════════════════════════════════════
CRITICAL: The "name" field MUST be EXACTLY one of these five strings, verbatim:
  "list_available_cameras" | "get_last_n_hours_summary" | "search_segments_by_activity"
  | "search_car_by_plate_number" | "list_recognized_plates"
NEVER invent a different tool name (e.g. there is no "list_cameras_by_location" —
a location filter is just a `location` PARAMETER on list_available_cameras or
get_last_n_hours_summary, never a different tool name).

TOOL 1 ─ list_available_cameras
  When : User wants to know which cameras exist, or asks for camera list/info.
  Params:
    location    (string, optional — omit if not mentioned; see LOCATION RULES)
  Examples:
    "list all cameras"
    "which cameras are available?"
    "what cameras do we have?"
    "which cameras are at the ONGC office?"
    "list cameras near Janpath"

TOOL 2 ─ get_last_n_hours_summary
  When : User explicitly asks for a SUMMARY, OVERVIEW, or REPORT of footage.
         Also use this when user asks about RECORDING STATUS or FOOTAGE EXISTENCE
         for a time range — e.g. 'is any video recorded', 'any footage before X',
         'was anything recorded', 'is there recording'.
         Do NOT use this for specific object/person/activity searches.
  Params:
    date        (YYYY-MM-DD, optional — use when only ONE date is provided)
    start_date  (YYYY-MM-DD, optional — use as the START date when TWO dates are provided)
    end_date    (YYYY-MM-DD, optional — use for the END date of a multi-day range)
    start_time  (HH:MM:SS, optional)
    end_time    (HH:MM:SS, optional)
    camera_id   (string, optional — omit if not mentioned)
    location    (string, optional — omit if not mentioned; see LOCATION RULES; combine freely with camera_id)
    k           (int, optional — max segments)
  Date rules for this tool:
    • Single date  → use `date` only
    • Two dates    → use `start_date` + `end_date` (do NOT use `date` for two-date queries)
    • Cross-midnight range → use `start_date` (or `date`) + `end_date`
  Examples:
    "give me a summary of the last 2 hours"
    "summarize what happened on 2026-04-05 from 14:00 to 16:00"
    "overview of camera CAM-01 on April 5th"
    "what happened yesterday on camera ATPL-908610-ARCIS?"
    "summary from 2026-04-05 to 2026-04-07"
    "give me an overview between April 3rd and April 6th"
    "summary of the ONGC office on 2026-04-05"
    "what happened near Janpath between 14:00 and 16:00 today?"

TOOL 3 ─ search_segments_by_activity
  When : ANY query that is NOT a camera list and NOT explicitly asking for a
         summary/overview.  This covers:
           • specific objects  (white car, truck, bag, box)
           • people types      (police, military, person in uniform, worker)
           • animals
           • events / incidents (fight, fire, crowd)
           • presence checks   ("any car?", "was there a person?")
           • activity descriptions ("someone running", "vehicle parked")
         If in doubt → USE THIS TOOL.
  Params:
    query       (string OR list of strings, REQUIRED — keywords to match)
    date        (YYYY-MM-DD, OPTIONAL — include ONLY if the user's query actually
                 names a date/day (an explicit date, "today", "yesterday", a weekday
                 name, etc). If the query says nothing about when, OMIT this param
                 entirely — do NOT default it to today. Omitting it searches every
                 date in the database instead of restricting to just one day.)
    start_time  (HH:MM:SS, optional)
    end_time    (HH:MM:SS, optional)
    end_date    (YYYY-MM-DD, optional — use for cross-date ranges)
    camera_id   (string, optional — omit if not mentioned)
    location    (string, optional — omit if not mentioned; see LOCATION RULES; combine freely with camera_id)
    max_results (int, optional)
  Examples:
    "any white car at the ONGC office today?"
    "was there a person near Janpath yesterday?"

TOOL 4 ─ search_car_by_plate_number
  When : User gives a specific LICENSE PLATE NUMBER (or part of one) and wants
         to find that vehicle — e.g. "find car GJ01AB1234", "search plate
         MH12XY1234", "has plate DL8CAB1234 been seen?". A plate number looks
         like STATE-CODE + digits + letters + digits (e.g. GJ01AB1234,
         MH-12-XY-1234) — NOT a camera ID (XXXX-NNNNNN-XXXX format).
         Do NOT use this tool for generic vehicle searches with no plate
         number given ("any white car?" → use tool 3 instead).
  Params:
    plate_number (string, REQUIRED — the plate number/substring, copy as
                  given; strip spaces/dashes only if the user wrote it with
                  separators, e.g. "GJ 01 AB 1234" → "GJ01AB1234")
    camera_id    (string, optional — omit if not mentioned)
  Examples:
    "find car with plate GJ01AB1234"
    "search for number plate MH12XY1234"
    "has GJ01 been seen on any camera?"

TOOL 5 ─ list_recognized_plates
  When : User wants a LIST of all recognized/detected license plates —
         not a search for one specific plate. E.g. "list all recognized
         plates", "show all number plates detected", "list plates on
         camera cam_pakwan".
  Params:
    camera_id (string, optional — omit if not mentioned)
  Examples:
    "list all recognized plates"
    "show me every detected number plate"
    "list plates for camera cam_pakwan"

════════════════════════════════════════════════
DATE / TIME RULES
════════════════════════════════════════════════
- Tool 2 (get_last_n_hours_summary): ALWAYS include a date. If no date mentioned → use today: {today}
- Tool 3 (search_segments_by_activity): include `date` ONLY when the user's query
  actually names a date/day. If no date is mentioned, OMIT `date` entirely — do
  NOT default it to today. A bare keyword search ("any bus?", "show me any
  transport vehicle") means "ever", not "today", so it must search every date.
- "today" → {today}
- "yesterday" → {yesterday}
- Times use 24-hour HH:MM:SS.
- Ambiguous times in a surveillance context lean PM:
    "6 to 7"   → 18:00:00 to 19:00:00
    "2 to 4"   → 14:00:00 to 16:00:00
- Copy timestamps verbatim when they appear in the query
  (e.g. "2026-04-05 16:17:49" → date="2026-04-05", start_time="16:17:49")
- If the time range spans across midnight, provide both `date` and `end_date`.

════════════════════════════════════════════════
CAMERA ID RULES
════════════════════════════════════════════════
- Camera IDs look like: XXXX-123456-XXXX  (letters-digits-letters)
- If the user mentions one → include camera_id exactly.
- If NOT mentioned → OMIT camera_id entirely (do not guess or default).

════════════════════════════════════════════════
LOCATION RULES  (tools 1, 2, and 3)
════════════════════════════════════════════════
- A LOCATION is a place/site name — e.g. "ONGC office", "Janpath", "Chiman bhai
  Bridge", "CN Vidhyalaya" — NOT a camera ID (which is the XXXX-123456-XXXX format).
- If the user names a place (not a formatted camera ID) → include it as `location`,
  copied close to how they said it (matching is case- and punctuation-insensitive
  server-side, so "ongc" / "ONGC" / "O.N.G.C." all work — don't over-format it).
- location and camera_id are independent — combine both if the user gives a
  specific camera AND a place name that could disambiguate/confirm it.
- For tool 3, a place name is a LOCATION filter, never a search keyword — do NOT
  put it in `query` (e.g. "car at Janpath" → query=["car",...], location="Janpath",
  not query=["car","janpath"]).
- If NO place name is mentioned → OMIT location entirely (do not guess or default).

════════════════════════════════════════════════
KEYWORD RULES  (tool 3 only) — CRITICAL
════════════════════════════════════════════════
- query must be a JSON array of lowercase strings.
- Expand ONLY to SPECIFIC related synonyms.
- DO NOT use generic terms when specific ones are mentioned.

SPECIFIC EXPANSION RULES:

1. CAR-RELATED (keep specific, no generic "vehicle"):
   "car"        → ["car","sedan","automobile","hatchback","coupe","suv"]
   "white car"  → ["white car","white sedan","white automobile","white hatchback"]
   "red car"    → ["red car","red sedan","red automobile"]
   
2. TRUCK-RELATED (keep specific, no generic "vehicle"):
   "truck"      → ["truck","lorry","pickup","pickup truck","semi","trailer","big rig"]
   "white truck"→ ["white truck","white lorry","white pickup"]
   
3. VEHICLE (only if user specifically says "vehicle"):
   "vehicle"    → ["vehicle","car","truck","van","automobile","transport"]
   
4. PEOPLE:
   "police"     → ["police","cop","officer","law enforcement","policeman"]
   "military"   → ["military","soldier","army","armed forces","uniform"]
   "person"     → ["person","man","woman","individual","pedestrian","people"]
   "worker"     → ["worker","construction worker","laborer","employee"]
   
5. OBJECTS:
   "bag"        → ["bag","backpack","handbag","purse","luggage","suitcase"]
   "box"        → ["box","package","parcel","crate","container"]
   
6. ANIMALS:
   "dog"        → ["dog","canine","puppy"]
   "cat"        → ["cat","feline","kitten"]
   
7. LOCATION/FEATURES:
   "gate"       → ["gate","entrance","doorway","barrier"]
   "parking"    → ["parking","parking lot","parking area","parked"]

CRITICAL RULES:
- If user says "car" → DO NOT add "truck" or "vehicle"
- If user says "truck" → DO NOT add "car" or "vehicle"
- If user says "vehicle" → THEN add both car and truck terms
- NEVER split a "modifier + object" phrase into two separate array entries. The
  user's intent is the WHOLE phrase, not its parts in isolation — e.g. for
  "red car", output "red car" (and "red sedan"/"red automobile") as single
  entries; NEVER output "red" and "car" as two separate keywords. "red" alone
  would match anything red (a red bag, a red truck, a red shirt) and "car"
  alone would match any car of any color — together they lose exactly the
  intent the user stated. This applies to every color/size/state modifier
  ("white truck", "blue sedan", "parked car", "big rig"), not just red.
- Keep color modifiers with the object (e.g., "white car" stays together) —
  read the query as a whole to decide what belongs together, don't tokenize
  it word-by-word.
- Do NOT add unrelated words.

════════════════════════════════════════════════
RESPONSE FORMAT  — STRICT JSON, NOTHING ELSE
════════════════════════════════════════════════
{{
  "tool_call": {{
    "name": "<tool_name>",
    "parameters": {{ ... }}
  }},
  "reasoning": "<one sentence>"
}}

════════════════════════════════════════════════
FEW-SHOT EXAMPLES
════════════════════════════════════════════════

Q: "list all cameras"
{{
  "tool_call": {{"name": "list_available_cameras", "parameters": {{}}}},
  "reasoning": "User wants to see available cameras."
}}

Q: "which cameras are at the ONGC office?"
{{
  "tool_call": {{"name": "list_available_cameras", "parameters": {{"location": "ONGC office"}}}},
  "reasoning": "Camera list scoped to a named place, not a camera ID."
}}

Q: "any white car today?"
{{
  "tool_call": {{
    "name": "search_segments_by_activity",
    "parameters": {{
      "query": ["white car","white sedan","white automobile","white hatchback"],
      "date": "{today}"
    }}
  }},
  "reasoning": "Specific CAR search — using car-specific keywords only, no truck or vehicle."
}}

Q: "show footage from 2026-04-05 22:00 to 2026-04-06 02:00"
{{
  "tool_call": {{
    "name": "search_segments_by_activity",
    "parameters": {{
      "query": ["activity"],
      "date": "2026-04-05",
      "start_time": "22:00:00",
      "end_date": "2026-04-06",
      "end_time": "02:00:00"
    }}
  }},
  "reasoning": "Cross-midnight range needs separate end_date."
}}

Q: "show me a truck from yesterday"
{{
  "tool_call": {{
    "name": "search_segments_by_activity",
    "parameters": {{
      "query": ["truck","lorry","pickup","pickup truck","semi"],
      "date": "{yesterday}"
    }}
  }},
  "reasoning": "Specific TRUCK search — using truck-specific keywords only, no car or vehicle."
}}

Q: "any vehicle in parking lot?"
{{
  "tool_call": {{
    "name": "search_segments_by_activity",
    "parameters": {{
      "query": ["vehicle","car","truck","sedan","automobile","parking","parking lot"]
    }}
  }},
  "reasoning": "User said VEHICLE (generic) — now we include both car AND truck terms. No date mentioned, so `date` is omitted (searches every date)."
}}

Q: "is there a red truck near the gate today between 6 and 7?"
{{
  "tool_call": {{
    "name": "search_segments_by_activity",
    "parameters": {{
      "query": ["red truck","red lorry","red pickup","gate","entrance"],
      "date": "{today}",
      "start_time": "18:00:00",
      "end_time": "19:00:00"
    }}
  }},
  "reasoning": "Specific red TRUCK + location — truck keywords only, plus gate-related terms. 'today' is mentioned, so date is included (start_time/end_time need a `date` to anchor to — if no day is mentioned at all, omit date AND leave start_time/end_time out too, since they can't be applied without one)."
}}

Q: "show activities of white car from 2026-04-02 14:00:00 to 2026-04-02 16:11:51"
{{
  "tool_call": {{
    "name": "search_segments_by_activity",
    "parameters": {{
      "query": ["white car","white sedan","white automobile"],
      "date": "2026-04-02",
      "start_time": "14:00:00",
      "end_time": "16:11:51"
    }}
  }},
  "reasoning": "Specific white CAR search — car keywords only, no truck terms."
}}

Q: "have you seen any police in camera ANYK-804268-AAAAA?"
{{
  "tool_call": {{
    "name": "search_segments_by_activity",
    "parameters": {{
      "query": ["police","cop","officer","law enforcement"],
      "camera_id": "ANYK-804268-AAAAA"
    }}
  }},
  "reasoning": "Police search with specific police-related keywords. No date mentioned, so `date` is omitted (searches every date on this camera)."
}}

Q: "any person carrying a bag on camera ATPL-908610-ARCIS on 2026-04-03?"
{{
  "tool_call": {{
    "name": "search_segments_by_activity",
    "parameters": {{
      "query": ["person","carrying","bag","backpack","handbag"],
      "date": "2026-04-03",
      "camera_id": "ATPL-908610-ARCIS"
    }}
  }},
  "reasoning": "Person + bag search with relevant keywords."
}}

Q: "was there a blue sedan today?"
{{
  "tool_call": {{
    "name": "search_segments_by_activity",
    "parameters": {{
      "query": ["blue sedan","blue car","blue automobile"],
      "date": "{today}"
    }}
  }},
  "reasoning": "Specific blue SEDAN — car-type keywords only."
}}

Q: "show me any transport vehicle"
{{
  "tool_call": {{
    "name": "search_segments_by_activity",
    "parameters": {{
      "query": ["vehicle","transport","car","truck","van","automobile"]
    }}
  }},
  "reasoning": "Generic VEHICLE/transport — includes all vehicle types. No date mentioned, so `date` is omitted (searches every date)."
}}

Q: "give me a summary from 2026-04-05 14:00:00 to 16:00:00 for camera ATPL-908610-ARCIS"
{{
  "tool_call": {{
    "name": "get_last_n_hours_summary",
    "parameters": {{
      "date": "2026-04-05",
      "start_time": "14:00:00",
      "end_time": "16:00:00",
      "camera_id": "ATPL-908610-ARCIS"
    }}
  }},
  "reasoning": "User explicitly asked for a summary with date and time range."
}}

Q: "overview of all footage on 2026-04-05"
{{
  "tool_call": {{
    "name": "get_last_n_hours_summary",
    "parameters": {{
      "date": "2026-04-05"
    }}
  }},
  "reasoning": "'Overview' signals a summary request — full day, all cameras."
}}

Q: "summary from 2026-04-03 to 2026-04-06"
{{
  "tool_call": {{
    "name": "get_last_n_hours_summary",
    "parameters": {{
      "start_date": "2026-04-03",
      "end_date": "2026-04-06"
    }}
  }},
  "reasoning": "Two dates provided — use start_date + end_date for multi-day summary."
}}

Q: "give me an overview between April 3rd and April 5th on camera ATPL-908610-ARCIS"
{{
  "tool_call": {{
    "name": "get_last_n_hours_summary",
    "parameters": {{
      "start_date": "2026-04-03",
      "end_date": "2026-04-05",
      "camera_id": "ATPL-908610-ARCIS"
    }}
  }},
  "reasoning": "Two-date range summary with specific camera."
}}

Q: "summarize footage from 2026-04-05 22:00 to 2026-04-06 02:00"
{{
  "tool_call": {{
    "name": "get_last_n_hours_summary",
    "parameters": {{
      "start_date": "2026-04-05",
      "start_time": "22:00:00",
      "end_date": "2026-04-06",
      "end_time": "02:00:00"
    }}
  }},
  "reasoning": "Cross-midnight summary — start_date + end_date with times."
}}

Q: \"have you seen any auto-rickshaw from 2026-04-10 to 2026-04-15 on camera ANYK-807722-AAAAA?\"
{{
  \"tool_call\": {{
    \"name\": \"search_segments_by_activity\",
    \"parameters\": {{
      \"query\": [\"auto-rickshaw\",\"auto rickshaw\",\"rickshaw\",\"tuk-tuk\",\"three-wheeler\"],
      \"date\": \"2026-04-10\",
      \"end_date\": \"2026-04-15\",
      \"camera_id\": \"ANYK-807722-AAAAA\"
    }}
  }},
  \"reasoning\": \"Date range search + specific camera — always keep camera_id when user mentions it.\"
}}

Q: \"any truck on camera ATPL-908610-ARCIS from 2026-04-02 to 2026-04-05 between 14:00 and 18:00?\"
{{
  \"tool_call\": {{
    \"name\": \"search_segments_by_activity\",
    \"parameters\": {{
      \"query\": [\"truck\",\"lorry\",\"pickup\",\"pickup truck\",\"semi\"],
      \"date\": \"2026-04-02\",
      \"end_date\": \"2026-04-05\",
      \"start_time\": \"14:00:00\",
      \"end_time\": \"18:00:00\",
      \"camera_id\": \"ATPL-908610-ARCIS\"
    }}
  }},
  \"reasoning\": \"Truck search with date range, time window, and specific camera.\"
}}

Q: "summary of the ONGC office on 2026-04-05"
{{
  "tool_call": {{
    "name": "get_last_n_hours_summary",
    "parameters": {{
      "date": "2026-04-05",
      "location": "ONGC office"
    }}
  }},
  "reasoning": "Named place (not a camera ID) → location param, single-date summary."
}}

Q: "what happened near Janpath between 14:00 and 16:00 today?"
{{
  "tool_call": {{
    "name": "get_last_n_hours_summary",
    "parameters": {{
      "date": "{today}",
      "start_time": "14:00:00",
      "end_time": "16:00:00",
      "location": "Janpath"
    }}
  }},
  "reasoning": "Place name signals location filter; time window taken verbatim."
}}

Q: "any white car at the ONGC office today?"
{{
  "tool_call": {{
    "name": "search_segments_by_activity",
    "parameters": {{
      "query": ["white car","white sedan","white automobile","white hatchback"],
      "date": "{today}",
      "location": "ONGC office"
    }}
  }},
  "reasoning": "Specific CAR search scoped to a named place — location filter, not a query keyword."
}}

Q: "find car with plate GJ01AB1234"
{{
  "tool_call": {{
    "name": "search_car_by_plate_number",
    "parameters": {{"plate_number": "GJ01AB1234"}}
  }},
  "reasoning": "A specific plate number was given — plate search tool, not activity search."
}}

Q: "has plate MH12XY1234 been seen on camera cam_pakwan?"
{{
  "tool_call": {{
    "name": "search_car_by_plate_number",
    "parameters": {{"plate_number": "MH12XY1234", "camera_id": "cam_pakwan"}}
  }},
  "reasoning": "Specific plate number scoped to a named camera."
}}

Q: "list all recognized plates"
{{
  "tool_call": {{"name": "list_recognized_plates", "parameters": {{}}}},
  "reasoning": "User wants every recognized plate listed, not a search for one specific plate."
}}

Q: "show all number plates detected on cam_pakwan"
{{
  "tool_call": {{
    "name": "list_recognized_plates",
    "parameters": {{"camera_id": "cam_pakwan"}}
  }},
  "reasoning": "Listing plates scoped to one camera."
}}

CRITICAL CAMERA RULE: If the user mentions a camera ID (format XXXX-NNNNNN-XXXX), you MUST include
camera_id in the parameters — even when a date range (end_date) is also present. NEVER drop camera_id.

CRITICAL LOCATION RULE: If the user mentions a place/site name (not a camera ID) for tool 1, 2, or 3,
you MUST include it as location. NEVER drop location. Never put a place name into camera_id or into
tool 3's query keywords.

REMINDER: For get_last_n_hours_summary with TWO dates always use start_date + end_date.
For a SINGLE date use just date.

REMINDER: For search_segments_by_activity with cross-date ranges use date + end_date.

RESPOND WITH ONLY THE JSON OBJECT. NO MARKDOWN. NO EXPLANATION OUTSIDE THE JSON."""

    # ── parameter fix ─────────────────────────────────────────────────────────

    def _fix_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise parameter names (strip spaces, fix known typos)."""
        fixes = {
            "person_名字": "person_name",
            "start_时间": "start_time",
            "end_时间": "end_time",
            "person_ name": "person_name",
            "start_ time": "start_time",
            "end_ time": "end_time",
        }
        cleaned = {}
        for k, v in params.items():
            norm = k.strip().replace(" ", "_")
            corrected = fixes.get(k, fixes.get(norm, norm))
            if corrected != k:
                logger.info(f"🔧 Fixed param: '{k}' → '{corrected}'")
            cleaned[corrected] = v
        return cleaned

    # ── JSON parser ───────────────────────────────────────────────────────────

    def _parse_response(self, text: str) -> Dict[str, Any]:
        text = re.sub(r'```json|```', '', text).strip()
        start = text.find('{')
        if start == -1:
            raise ValueError("No JSON object found in LLM response")

        brace, in_str, esc = 0, False, False
        clean_text = ""
        
        for i in range(start, len(text)):
            c = text[i]
            clean_text += c
            if esc:
                esc = False
                continue
            if c == '\\':
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if not in_str:
                if c == '{':
                    brace += 1
                elif c == '}':
                    brace -= 1
                    if brace == 0:
                        # We found a complete object
                        break
        
        # Robustness: if we finished the loop but brace > 0 or in_str is True,
        # the JSON is truncated. Attempt to fix it.
        if in_str:
            clean_text += '"'
        while brace > 0:
            clean_text += '}'
            brace -= 1

        try:
            parsed = json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON even after cleanup: {clean_text}")
            raise ValueError(f"Invalid JSON in LLM response: {e}")

        # Normalise wrapper
        if "tool_call" not in parsed:
            if "name" in parsed and "parameters" in parsed:
                parsed = {"tool_call": {"name": parsed["name"], "parameters": parsed["parameters"]}}
            else:
                # If it's just the tool name and params at root
                keys = parsed.keys()
                if "name" in keys or "parameters" in keys:
                     parsed = {"tool_call": {
                         "name": parsed.get("name", "unknown"),
                         "parameters": parsed.get("parameters", {})
                     }}
                else:
                    raise ValueError("Response missing 'tool_call' key")

        # Fix params
        tc = parsed.get("tool_call", {})
        if "parameters" in tc:
            tc["parameters"] = self._fix_parameters(tc["parameters"])

        return parsed

    # ── main entry ────────────────────────────────────────────────────────────

    @traceable(name="select_tool", run_type="chain", project_name="video-summary")
    async def generate_tool_call(self, query: str) -> Optional[Dict[str, Any]]:
        logger.info("=" * 60)
        logger.info("STEP 1 — LLM tool selection")
        logger.info(f"Query: {query}")
        logger.info("=" * 60)

        run = get_current_run_tree()
        if run:
            run.metadata.update({"component": "client", "model": self.model})

        try:
            client = AsyncClient()
            response = await client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user",   "content": f"Query: {query}"},
                ],
                options={
                    'temperature':    0.0,
                    'num_predict':    1024,
                    'top_k':          40,
                    'top_p':          0.9,
                    'repeat_penalty': 1.1,
                    'num_gpu':        -1,
                    'num_thread':     8,
                    'seed':           42,
                },
                keep_alive="5m"
            )

            raw = response.message.content.strip()
            logger.info(f"LLM raw: {raw}")

            parsed = self._parse_response(raw)
            tc = parsed.get("tool_call", {})
            logger.info(f"✅ Tool: {tc.get('name')}")
            logger.info(f"   Params: {json.dumps(tc.get('parameters', {}), indent=2)}")
            logger.info(f"   Reason: {parsed.get('reasoning', '')}")
            return parsed

        except Exception as e:
            logger.error(f"❌ generate_tool_call error: {e}")
            import traceback; traceback.print_exc()
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATED RAG CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class IntegratedVideoRAGClient:

    # Valid parameters for each of the three tools
    VALID_PARAMS = {
        "list_available_cameras":    ["location"],
        "get_last_n_hours_summary":  ["camera_id", "location", "k", "date", "start_date", "start_time", "end_time", "end_date"],
        "search_segments_by_activity": [
            "query", "date", "end_date", "camera_id", "location", "max_results", "start_time", "end_time"
        ],
        "search_car_by_plate_number": ["plate_number", "camera_id"],
        "list_recognized_plates":     ["camera_id"],
    }

    def __init__(self, model: str = "llama3.2:3b"):
        self.model            = model
        self.query_generator  = VideoQueryGenerator(model)
        self.summarizer       = VideoSegmentSummarizer(model=" gemma4:cloud")
        self.contextual_agent = VideoContextualAgent(model=model)
        self.session: Optional[ClientSession] = None
        self._mcp_server_url: Optional[str] = None
        # Guards reconnect_to_mcp_server so two requests that both see the
        # dead session at once don't race to tear down/rebuild it together.
        self._reconnect_lock = asyncio.Lock()
        logger.info(f"✅ IntegratedVideoRAGClient ready ({model})")

    # ── MCP connection ────────────────────────────────────────────────────────

    async def connect_to_mcp_server(self, server_url: str):
        self._mcp_server_url = server_url
        logger.info(f"Connecting to MCP server: {server_url}")

        # A prior connection's context managers, if any (reconnect path) —
        # best-effort close; the old transport is already broken so tearing
        # it down cleanly can itself raise, but that must never block getting
        # a fresh connection up.
        if hasattr(self, '_session_ctx'):
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing stale MCP session (ignored): {e}")
        if hasattr(self, '_streams_ctx'):
            try:
                await self._streams_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing stale MCP streams (ignored): {e}")

        self._streams_ctx = sse_client(url=server_url, timeout=600.0)
        streams = await self._streams_ctx.__aenter__()
        self._session_ctx = ClientSession(*streams)
        self.session = await self._session_ctx.__aenter__()
        await self.session.initialize()

        tools = (await self.session.list_tools()).tools
        print("\n" + "=" * 60)
        print("✅ VIDEO RAG CLIENT CONNECTED")
        print("=" * 60)
        print(f"Server  : {server_url}")
        print(f"Tools   : {[t.name for t in tools]}")
        print(f"Model   : {self.model}")
        print("=" * 60 + "\n")

    async def call_tool_with_reconnect(self, tool_name: str, parameters: dict):
        """session.call_tool, transparently reconnecting once if the shared
        MCP session has died.

        mcp-main.py opens ONE session at process startup and reuses it for
        every request over the app's whole lifetime (see startup_event). If
        that underlying SSE connection ever drops — server1.py restarting,
        a network blip, an idle timeout — every call on it raises
        ClosedResourceError forever, with nothing to notice or recover:
        the entire chatbot goes down until someone manually restarts
        mcp-main.py. Reconnecting once here means a transient drop heals
        itself on the very next query instead.
        """
        try:
            return await self.session.call_tool(tool_name, parameters)
        except (anyio.ClosedResourceError, anyio.BrokenResourceError) as e:
            async with self._reconnect_lock:
                logger.warning(
                    f"MCP session dead ({type(e).__name__}) — reconnecting to "
                    f"{self._mcp_server_url} and retrying once"
                )
                await self.connect_to_mcp_server(self._mcp_server_url)
            return await self.session.call_tool(tool_name, parameters)

    async def cleanup(self):
        if hasattr(self, '_session_ctx'):
            await self._session_ctx.__aexit__(None, None, None)
        if hasattr(self, '_streams_ctx'):
            await self._streams_ctx.__aexit__(None, None, None)

    # ── main query processor ──────────────────────────────────────────────────

    @traceable(name="video_rag_query", run_type="chain", project_name="video-summary")
    async def process_query(self, query: str) -> Dict[str, Any]:
        run = get_current_run_tree()
        if run:
            run.metadata.update({"component": "root", "model": self.model, "original_query": query})

        try:
            # ── Step 0: contextualise ─────────────────────────────────────────
            logger.info("=" * 60)
            logger.info("STEP 0 — Contextualise query")
            logger.info("=" * 60)

            query_lower = query.lower()
            is_meta = (
                ('list' in query_lower and 'camera' in query_lower)
                or ('list' in query_lower and 'plate' in query_lower)
            )

            if is_meta:
                ctx_query = query
                logger.info("Meta-query — skipping contextualiser")
            else:
                ctx_query = await self.contextual_agent.process_query(query)

            logger.info(f"Original     : {query}")
            logger.info(f"Contextualised: {ctx_query}")

            # ── Step 1: LLM selects tool ──────────────────────────────────────
            result = await self.query_generator.generate_tool_call(ctx_query)

            if not result or not result.get("tool_call"):
                return {"error": "LLM could not select a tool", "query": query, "status": "failed"}

            tool_name  = result["tool_call"]["name"]
            parameters = result["tool_call"].get("parameters", {})

            # Guard: only allow the three server tools
            if tool_name not in self.VALID_PARAMS:
                logger.warning(f"⚠️ LLM returned unknown tool '{tool_name}' — defaulting to search")
                tool_name  = "search_segments_by_activity"
                parameters = {}

            # Strip any extra params the LLM hallucinated
            allowed = self.VALID_PARAMS[tool_name]
            bad     = [k for k in parameters if k not in allowed]
            if bad:
                logger.warning(f"⚠️ Removing unexpected params: {bad}")
            parameters = {k: v for k, v in parameters.items() if k in allowed}

            # Normalise 'query' to a list for search tool
            if tool_name == "search_segments_by_activity":
                q = parameters.get("query")
                if isinstance(q, str):
                    parameters["query"] = [x.strip() for x in q.split(',') if x.strip()]
                elif not isinstance(q, list):
                    # Covers q being None (LLM omitted "query" entirely) or
                    # any other scalar. `parameters.get("query", [])` used to
                    # default to a list here, so this branch never ran and
                    # "query" was left OUT of parameters altogether — the MCP
                    # server then rejected the call with an opaque pydantic
                    # "missing required argument" error.
                    parameters["query"] = [str(q)] if q else []

                if not parameters.get("query"):
                    raise ValueError(
                        "The AI couldn't determine what to search for — "
                        "please rephrase your question with a specific "
                        "object, person, or activity."
                    )

                # `date` is intentionally left as whatever the LLM decided (or
                # omitted) — a keyword search with no date mentioned means
                # "search every date", not "default to today". Do NOT
                # auto-inject a date here.

            # Ensure date for summary tool too
            if tool_name == "get_last_n_hours_summary":
                has_any_date = (
                    "date" in parameters
                    or "start_date" in parameters
                    or "end_date" in parameters
                )
                if not has_any_date:
                    from datetime import datetime, timezone
                    parameters["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    logger.info(f"🔧 Auto-added date for summary: {parameters['date']}")

            # ── Step 2: call MCP tool ─────────────────────────────────────────
            logger.info("=" * 60)
            logger.info("STEP 2 — Execute MCP tool")
            logger.info(f"Tool  : {tool_name}")
            logger.info(f"Params: {json.dumps(parameters, indent=2)}")
            logger.info("=" * 60)

            try:
                mcp_result = await self.call_tool_with_reconnect(tool_name, parameters)
            except Exception as e:
                logger.error(f"❌ MCP call failed: {e}")
                return {"error": f"MCP tool error: {e}", "status": "failed"}

            if not mcp_result or not mcp_result.content:
                return {"error": "Empty response from MCP server", "status": "failed"}

            raw_text = mcp_result.content[0].text

            try:
                tool_result = json.loads(raw_text)
            except json.JSONDecodeError:
                if "validation error" in raw_text.lower():
                    logger.error(f"Server validation error: {raw_text.strip()}")
                    return {"error": f"Server validation failed: {raw_text.strip()}", "status": "failed"}
                logger.error(f"JSON parse error. Raw: {repr(raw_text[:200])}")
                return {"error": "Invalid JSON from MCP server", "status": "failed"}

            logger.info("✅ MCP responded successfully")

            # ── Step 3: generate summary ──────────────────────────────────────
            logger.info("=" * 60)
            logger.info("STEP 3 — Generate summary")
            logger.info("=" * 60)

            final_answer = await self._generate_summary(
                tool_name, tool_result, query, ctx_query, parameters
            )

            self._display_final_answer(query, final_answer)

            return {
                "status":               "success",
                "query":                query,
                "contextualized_query": ctx_query,
                "tool_used":            tool_name,
                "final_answer":         final_answer,
            }

        except Exception as e:
            logger.error(f"process_query error: {e}")
            import traceback; traceback.print_exc()
            return {"error": str(e), "query": query, "status": "failed"}

    # ── vision query processor ────────────────────────────────────────────────

    async def process_image_query(self, characteristics: list, user_query: str, ctx_query: str) -> dict:
        """
        Special entry point for multimodal vision pipeline.
        Uses characteristics as keywords and ctx_query for filters.
        """
        logger.info("=" * 60)
        logger.info("MULTIMODAL STEP — MCP Search via characteristics")
        logger.info(f"Keywords: {characteristics}")
        logger.info(f"Context : {ctx_query}")
        logger.info("=" * 60)

        # 1. Get tool call from contextualized query to extract filters (dates/camera)
        result = await self.query_generator.generate_tool_call(ctx_query)
        
        tool_name = "search_segments_by_activity"
        if not result or not result.get("tool_call"):
            logger.warning("LLM could not select a tool for vision query — defaulting to search_segments_by_activity with characteristics")
            from datetime import datetime
            parameters = {
                "query": characteristics, 
                "date": datetime.now().strftime("%Y-%m-%d")
            }
        else:
            parameters = result["tool_call"].get("parameters", {})
            # Override/Inject the visual characteristics as the query keywords
            parameters["query"] = characteristics
            # Ensure it uses the correct tool
            tool_name = "search_segments_by_activity"

        # Guard: validate params against allowed list for search_segments_by_activity
        allowed = self.VALID_PARAMS["search_segments_by_activity"]
        parameters = {k: v for k, v in parameters.items() if k in allowed}
        
        # Ensure date is always present (required by server for this tool)
        if "date" not in parameters:
            from datetime import datetime, timezone
            parameters["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            
        logger.info(f"Executing vision-search: {tool_name} with {json.dumps(parameters, indent=2)}")
        
        try:
            mcp_result = await self.call_tool_with_reconnect(tool_name, parameters)
            if not mcp_result or not mcp_result.content:
                return {"segments": [], "status": "no_results"}
            
            tool_result = json.loads(mcp_result.content[0].text)
            return tool_result
        except Exception as e:
            logger.error(f"process_image_query MCP error: {e}")
            return {"segments": [], "error": str(e), "status": "failed"}

    # ── summary dispatcher ────────────────────────────────────────────────────

    @traceable(name="generate_summary", run_type="chain", project_name="video-summary")
    async def _generate_summary(
        self,
        tool_name:   str,
        tool_result: Dict[str, Any],
        orig_query:  str,
        ctx_query:   str,
        parameters:  Dict[str, Any],
    ) -> str:
        run = get_current_run_tree()
        if run:
            run.metadata.update({
                "component": "summary",
                "tool_name": tool_name,
                "camera_id": parameters.get("camera_id"),
                "location": parameters.get("location"),
                "model": self.summarizer.model,
            })

        try:
            # ── list_available_cameras ────────────────────────────────────────
            if tool_name == "list_available_cameras":
                cameras = tool_result.get("cameras", [])
                if not cameras:
                    return "No cameras found in the database."

                lines = []
                for cam in cameras:
                    lines.append(
                        f"📹 **{cam['camera_id']}**\n"
                        f"   • Location: {cam.get('location', 'Unknown Location')}\n"
                        f"   • Range  : {cam.get('first_recording','N/A')} → {cam.get('last_recording','N/A')}\n"
                        f"   • Segments: {cam.get('total_segments', 0)}\n"
                        f"   • Duration: {cam.get('total_duration_minutes', 0):.1f} min"
                    )
                return "# 📹 Available Cameras\n\n" + "\n\n".join(lines)

            # ── get_last_n_hours_summary ──────────────────────────────────────
            elif tool_name == "get_last_n_hours_summary":
                segments  = tool_result.get("segments", [])
                camera_id = tool_result.get("camera_id", parameters.get("camera_id", "all cameras"))
                logger.info(f"Total segments: {len(segments)}")
                for i, seg in enumerate(segments[:2]):   # print first 2 only
                   logger.info(f"Segment {i} keys: {list(seg.keys())}")
                   logger.info(f"Segment {i} frame data: { {k:v for k,v in seg.items() if 'frame' in k.lower() or 'url' in k.lower() or 'image' in k.lower()} }")

                # Derive a human-readable hours label from the time range if available
                hours = 0.0
                tr = tool_result.get("time_range", {})
                if tr.get("start") and tr.get("end"):
                    try:
                        from datetime import datetime
                        s = datetime.fromisoformat(tr["start"].replace("Z", "+00:00"))
                        e = datetime.fromisoformat(tr["end"].replace("Z", "+00:00"))
                        hours = (e - s).total_seconds() / 3600
                    except Exception:
                        hours = 0.0

                logger.info(f"Summarising {len(segments)} segments for time-range summary")
                return await self.summarizer.summarize_time_range(
                    orig_query, ctx_query, segments, camera_id, hours
                )

            # ── search_segments_by_activity ───────────────────────────────────
            elif tool_name == "search_segments_by_activity":
                segments  = tool_result.get("segments", [])
                camera_id = tool_result.get("camera_id", parameters.get("camera_id", "all cameras"))
                keywords  = parameters.get("query", [])
                if isinstance(keywords, str):
                    keywords = [keywords]

                logger.info(f"Summarising {len(segments)} search results")
                return await self.summarizer.summarize_search_results(
                    orig_query, ctx_query, segments, camera_id, keywords
                )

            # ── search_car_by_plate_number ────────────────────────────────────
            elif tool_name == "search_car_by_plate_number":
                results      = tool_result.get("results", [])
                plate_number = tool_result.get("plate_number", parameters.get("plate_number", ""))

                logger.info(f"Summarising {len(results)} plate-search result(s) for '{plate_number}'")
                return await self.summarizer.summarize_plate_search(
                    orig_query, ctx_query, results, plate_number
                )

            # ── list_recognized_plates ────────────────────────────────────────
            elif tool_name == "list_recognized_plates":
                plates = tool_result.get("plates", [])
                if not plates:
                    return "No recognized plates found in the database."

                lines = []
                for p in plates:
                    lines.append(
                        f"🚗 **{p.get('plate_number', 'Unknown')}**\n"
                        f"   • Camera  : {p.get('camera_id', 'Unknown')}\n"
                        f"   • Location: {p.get('location', 'Unknown Location')}\n"
                        f"   • Date    : {p.get('date', 'N/A')}"
                    )
                return "# 🚗 Recognized Plates\n\n" + "\n\n".join(lines)

            else:
                return json.dumps(tool_result, indent=2)

        except Exception as e:
            logger.error(f"_generate_summary error: {e}")
            return f"Summary generation failed: {e}"

    # ── display helper ────────────────────────────────────────────────────────

    def _display_final_answer(self, query: str, answer: str):
        print(f"\n{'='*60}")
        print("📝 FINAL ANSWER")
        print(f"{'='*60}")
        print(f"Q: {query}")
        print(f"{'='*60}\n")
        print(answer)
        print(f"\n{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8088/sse")

    client = IntegratedVideoRAGClient()

    try:
        await client.connect_to_mcp_server(MCP_SERVER_URL)

        print("\n" + "=" * 60)
        print("🚀 VIDEO SURVEILLANCE CHATBOT — READY")
        print("=" * 60)
        print("Examples:")
        print("  📋  'list all cameras'")
        print("  📊  'give me a summary of 2026-04-05 from 14:00 to 16:00'")
        print("  🚗  'any white car today?'")
        print("  👮  'have you seen any police on camera ATPL-908610-ARCIS?'")
        print("  🎒  'person carrying a bag between 6 and 7'")
        print("  🔍  'is there a truck near the gate on 2026-04-03?'")
        print("\nType 'quit' to exit.")
        print("=" * 60 + "\n")

        while True:
            try:
                query = input("💬 Your question: ").strip()
                if not query:
                    continue
                if query.lower() in ('quit', 'exit', 'q'):
                    print("\n👋 Goodbye!")
                    break
                await client.process_query(query)
                print("\n" + "-" * 60 + "\n")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                break

    finally:
        await client.cleanup()
        print("✅ Disconnected.")


if __name__ == "__main__":
    asyncio.run(main())