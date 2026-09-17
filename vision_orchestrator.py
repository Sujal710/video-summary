import asyncio
import os
import json
import logging
import base64
import tempfile
import httpx
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
from pymongo import MongoClient
import numpy as np
from sentence_transformers import SentenceTransformer
from ollama import AsyncClient
from summary import VideoSegmentSummarizer
from context import VideoContextualAgent

# Configure logging
logger = logging.getLogger(__name__)

class VisionOrchestrator:
    def __init__(self, ollama_model: str = "qwen3-vl:latest", embed_model: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.ollama_model = ollama_model
        self.client = AsyncClient()
        # Default was "minimax-m3:cloud", a cloud model that was never pulled -
        # it resolved to a 404 the moment anyone uploaded an image. Falls back to
        # the same local model the router uses.
        self.summarizer = VideoSegmentSummarizer(
            model=os.getenv("SUMMARY_MODEL", os.getenv("ROUTER_MODEL", "qwen3:8b")))
        self.context_agent = VideoContextualAgent()
        
        # Initialize Embedding Model for semantic fallback
        logger.info(f"Loading embedding model: {embed_model}")
        self.embed_model = SentenceTransformer(embed_model, trust_remote_code=True)
        
    def _extract_json(self, text: str) -> Optional[Any]:
        """Robustly extract JSON from text even if wrapped in markdown or contains extra prose."""
        try:
            # Clean markdown blocks
            text = re.sub(r'```json|```', '', text).strip()
            
            # Find the first occurrence of { or [
            start_curly = text.find('{')
            start_bracket = text.find('[')
            
            if start_curly == -1 and start_bracket == -1:
                return None
                
            start = start_curly if (start_bracket == -1 or (start_curly != -1 and start_curly < start_bracket)) else start_bracket
            
            brace_count = 0
            in_string = False
            escape = False
            
            for i in range(start, len(text)):
                char = text[i]
                if escape:
                    escape = False
                    continue
                if char == '\\':
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char in '{[':
                        brace_count += 1
                    elif char in '}]':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[start:i+1]
                            return json.loads(json_str)
            return None
        except Exception as e:
            logger.error(f"JSON extraction failed: {e}")
            return None

    async def get_image_characteristics(self, image_bytes: bytes, user_query: str) -> List[str]:
        """Generate a visual fingerprint of the PRIMARY ENTITY the user is asking about.

        The characteristics describe the WHOLE entity (e.g. the whole person) so that
        a downstream VLM can locate that entity as a single unit, not its individual
        accessories or clothing items.
        """
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            prompt = (
                f"USER QUERY: '{user_query}'\n\n"
                "Your job is to extract a VISUAL FINGERPRINT of the PRIMARY ENTITY the user is asking about.\n"
                "Rules:\n"
                "1. First, decide WHAT the user is asking about from their query (e.g. the PERSON in the image).\n"
                "2. Describe ONLY that entity as a WHOLE — do NOT split it into individual parts like 'watch', 'laptop', 'table'.\n"
                "3. MANDATORY ACTIVITY PHRASE: If the person is doing something (using a phone, drinking, eating, typing), you MUST provide a single phrase describing the action: 'person using a mobile phone', 'person drinking water', 'person eating food'.\n"
                "4. NEGATIVE CONSTRAINT: Never return 'person' and 'mobile' as two separate strings. That is a failure. Always combine them into 'person using mobile'.\n"
                "5. For a PERSON: describe their overall appearance as a single subject — body build, posture, clothing style/color (e.g. 'person wearing a blue plaid shirt').\n"
                "6. Aim for 5-8 descriptive phrases that together uniquely identify this entity.\n"
                "7. Output ONLY a valid JSON array of strings — no markdown, no explanation.\n\n"
                "Example output for a person using a phone:\n"
                '["person using a mobile phone", "blue plaid shirt", "seated in a black office chair", "short dark hair", "wearing glasses"]'
            )

            response = await self.client.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt, 'images': [tmp_path]}],
                options={'temperature': 0.1}
            )

            content = response.message.content.strip()
            keywords = self._extract_json(content)

            if isinstance(keywords, list):
                # ── Post-processing: Merge split person/activity keywords ──────
                # If 'person' and 'mobile' exist separately, merge them
                lowered = [k.lower() for k in keywords]
                has_person = any(w in k for k in lowered for w in ["person", "man", "woman", "individual"])
                
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
                        if nk_low in ["person", "man", "woman", "individual"] and any("using" in k or "drinking" in k or "eating" in k for k in new_keywords):
                            continue
                        final_keywords.append(nk)
                    keywords = final_keywords

                logger.info(f"Entity-level characteristics extracted: {keywords}")
                return keywords
            return []
        except Exception as e:
            logger.error(f"Error in intent extraction: {e}")
            return []
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def detect_object_with_bbox(
        self,
        image_url: str,
        characteristics: List[str],
        user_query: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Locate the PRIMARY ENTITY that matches the user's query and return a single
        bounding box around that WHOLE entity (e.g. the full body of the person).

        IMPORTANT: The bounding box must enclose the entity as a unit — NOT individual
        accessories (watch, shirt, laptop) or background objects.

        Qwen2.5-VL returns bounding boxes as [x_min, y_min, x_max, y_max] normalised 0-1000.
        """
        tmp_path = None
        try:
            # ── 1. Fetch image ────────────────────────────────────────────────
            async with httpx.AsyncClient() as http:
                resp = await http.get(image_url, timeout=15)
                if resp.status_code != 200:
                    logger.warning(
                        f"Cannot fetch image for detection (HTTP {resp.status_code}): {image_url}"
                    )
                    return None
                image_content = resp.content

            # Write to temp file — Ollama reads file paths directly
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp.write(image_content)
                tmp_path = tmp.name

            char_str = ", ".join(characteristics)
            # Determine the entity label from query context (default: "person")
            entity_hint = "person" if any(
                w in user_query.lower() for w in ["person", "man", "woman", "he", "she", "who", "doing"]
            ) else "object"

            # ── 2. Try prompts in order of specificity ────────────────────────
            # Each prompt stresses that the bbox must cover the WHOLE entity, not parts.
            prompts = [
                # Attempt 1: entity-aware, strict
                (
                    f"USER QUESTION: '{user_query}'\n"
                    f"Visual description of the target {entity_hint}: {char_str}\n\n"
                    f"Task: Draw ONE bounding box that tightly encloses the ENTIRE {entity_hint} described above. "
                    "The box must cover the whole body / whole object — NOT just a clothing item, accessory, or background element.\n"
                    "Return ONLY valid JSON (no markdown, no explanation):\n"
                    '{"detected_objects": [{"label": "<entity label>", "bbox": [x1, y1, x2, y2]}]}\n'
                    "Coordinates are integers 0-1000 (normalised: 0=left/top edge, 1000=right/bottom edge)."
                ),
                # Attempt 2: simpler re-phrase
                (
                    f"Locate the {entity_hint} who matches this description: {char_str}. "
                    f"Return a SINGLE bounding box around the COMPLETE {entity_hint} (full body, not just a part). "
                    "JSON format only — key 'detected_objects', each entry has 'label' and 'bbox' [x1,y1,x2,y2] 0-1000."
                ),
                # Attempt 3: minimal fallback
                (
                    f"Find the {entity_hint} in the image. "
                    f"The {entity_hint} can be identified by: {characteristics[0] if characteristics else entity_hint}. "
                    f"Return ONE bounding box around the WHOLE {entity_hint}. "
                    'JSON: {"detected_objects": [{"label": "...", "bbox": [x1,y1,x2,y2]}]}'
                ),
            ]

            for attempt, prompt in enumerate(prompts):
                try:
                    response = await self.client.chat(
                        model=self.ollama_model,
                        messages=[{
                            'role': 'user',
                            'content': prompt,
                            'images': [tmp_path],
                        }],
                        options={'temperature': 0.0, 'num_predict': 300}
                    )

                    content = response.message.content.strip()
                    logger.debug(f"Detection attempt {attempt+1} raw: {content[:300]}")

                    result = self._extract_json(content)

                    # Accept dict with detected_objects key
                    if isinstance(result, dict) and 'detected_objects' in result:
                        objs = result['detected_objects']
                        if objs:
                            # Post-process: pick the largest bbox (most likely to be the whole entity)
                            result['detected_objects'] = self._pick_largest_bbox(objs)
                            logger.info(f"Entity Detection (attempt {attempt+1}): {result}")
                            return result

                    # Accept a bare list of bbox dicts
                    if isinstance(result, list) and result:
                        best = self._pick_largest_bbox(result)
                        wrapped = {"detected_objects": best}
                        logger.info(f"Entity Detection bare list (attempt {attempt+1}): {wrapped}")
                        return wrapped

                except Exception as inner_e:
                    logger.warning(f"Detection attempt {attempt+1} failed: {inner_e}")
                    continue

            logger.warning("All detection attempts exhausted — no bbox found.")
            return None

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── Helper ────────────────────────────────────────────────────────────────
    def _pick_largest_bbox(self, objs: List[Dict]) -> List[Dict]:
        """Return a list containing only the object whose bounding box has the largest area.

        This heuristic ensures that when a VLM returns multiple small detections
        (e.g. 'watch', 'shirt') the one with the biggest bbox — almost always the
        whole person — is chosen.
        """
        if not objs:
            return objs

        def area(o: Dict) -> float:
            bbox = o.get("bbox", [])
            if len(bbox) != 4:
                return 0.0
            x1, y1, x2, y2 = bbox
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        largest = max(objs, key=area)
        return [largest]

    # ── Frame verification (mirrors summary.py) ───────────────────────────────
    DISCARDED_FRAMES_DIR = "discarded_frames"

    async def _verify_frame_presence(self, image_url: str, characteristics: List[str]) -> bool:
        """
        Download the surveillance frame and ask the VLM (YES/NO) whether the
        queried subject is visible — same approach as summary.py _verify_frame_with_vlm_async.
        Rejected frames are copied to discarded_frames/ for review.
        """
        import shutil
        from datetime import datetime as _dt
        import io
        os.makedirs(self.DISCARDED_FRAMES_DIR, exist_ok=True)
        tmp_path: Optional[str] = None
        try:
            # 1. Download frame
            async with httpx.AsyncClient() as dl:
                resp = await dl.get(image_url, timeout=15)
                if resp.status_code != 200:
                    logger.warning(f"[Verify] Could not download {image_url} (HTTP {resp.status_code})")
                    return False
                content = resp.content

            # 2. Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir='/tmp') as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # 3. Build subject description from characteristics list
            subject_desc = ', '.join(characteristics) if characteristics else 'the queried subject'
            prompt = (
                f"You are a surveillance image analyst. Your ONLY job is to determine whether a "
                f"specific subject is visible in this camera frame.\n\n"
                f"Subject to find: {subject_desc}\n\n"
                f"Instructions:\n"
                f"- Answer YES if the subject (or someone matching these characteristics) is present, "
                f"even if partially visible or at an angle.\n"
                f"- Answer YES if you can reasonably identify them based on clothing, colour, posture, or context.\n"
                f"- Answer NO only if the subject is definitively absent from the frame.\n"
                f"- Surveillance footage may be low quality — do not reject just because it is not crisp.\n"
                f"- Reply with ONLY one word: YES or NO."
            )

            response = await self.client.chat(
                model='qwen3-vl:latest',
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [tmp_path]
                }],
                options={'temperature': 0.0, 'num_predict': 5, 'num_gpu': -1},
                keep_alive="5m"
            )

            answer = (response.message.content or "").strip().upper()
            passed = 'YES' in answer
            logger.info(
                f"[Verify] {'✅ PRESENT' if passed else '❌ ABSENT'} — "
                f"{image_url.split('/')[-1]} | VLM: {answer}"
            )

            # 4. Save rejected frame to discarded folder
            if not passed:
                try:
                    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
                    slug = "-".join(characteristics)[:30]
                    fname = f"{ts}_{image_url.split('/')[-1].split('.')[0]}_{slug}.jpg"
                    dest = os.path.join(self.DISCARDED_FRAMES_DIR, fname)
                    shutil.copy2(tmp_path, dest)
                    logger.info(f"🗑️  Discarded frame → {dest}")
                except Exception as save_err:
                    logger.warning(f"Could not save discarded frame: {save_err}")

            return passed

        except Exception as e:
            logger.error(f"[Verify] Error: {e}")
            return False
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def process_query(self, image_bytes: bytes, user_query: str, rag_client: Any) -> AsyncGenerator[str, None]:
        """
        Multimodal Pipeline:
        1. Extract characteristics (Intent-Preserving)
        2. Resolve query context (Time/Camera)
        3. Search via MCP (Keywords + Semantic Fallback)
        4. Summarize (Minimax Streaming)
        5. Targeted Bounding Box Detection
        """
        yield json.dumps({"status": "Extracting visual characteristics (preserving intent)..."}) + "\n"
        characteristics = await self.get_image_characteristics(image_bytes, user_query)
        if not characteristics:
            yield json.dumps({"error": "Failed to analyze image characteristics."}) + "\n"
            return

        yield json.dumps({"status": "Resolving temporal and camera references..."}) + "\n"
        contextualized_query = await self.context_agent.process_query(user_query)

        yield json.dumps({"status": "Searching surveillance database via specialized tool..."}) + "\n"
        
        # Use the process_image_query method in rag_client which calls search_by_image_features
        search_res = await rag_client.process_image_query(characteristics, user_query, contextualized_query)
        docs = search_res.get("segments", [])
        camera_id = search_res.get("camera_id", "Multiple")

        if not docs:
            yield json.dumps({"status": "No matching footage found.", "final_answer": "I found no footage matching those characteristics in the specified timeframe."}) + "\n"
            return

        yield json.dumps({"status": "Generating intelligence report..."}) + "\n"
        # Use summarize_search_results_streaming to ensure the summary focuses on the visual characteristics
        summary_gen = self.summarizer.summarize_search_results_streaming(
            user_query, contextualized_query, docs, camera_id, characteristics
        )
        async for token in summary_gen:
            if token.startswith("#"): continue
            yield json.dumps({"token": token}) + "\n"

        yield json.dumps({"status": "Locating targeted object in retrieved frames..."}) + "\n"

        detection_result = None
        target_image_url = None

        # Collect all candidate URLs from the top-3 docs (skip N/A and blanks)
        candidate_urls: List[str] = []
        for doc in docs[:3]:
            for url in doc.get("frame_urls", []):
                if url and url != "N/A" and url not in candidate_urls:
                    candidate_urls.append(url)

        # Pre-validate URLs with a HEAD request before sending to the VLM
        valid_urls: List[str] = []
        async with httpx.AsyncClient() as http_check:
            for url in candidate_urls[:6]:   # cap at 6 to avoid long waits
                try:
                    head = await http_check.head(url, timeout=5)
                    if head.status_code < 400:
                        valid_urls.append(url)
                except Exception:
                    pass   # unreachable URL — skip silently

        logger.info(f"Detection: {len(valid_urls)} reachable candidate URLs from {len(candidate_urls)} total.")

        # Collect all detections
        detections_found = []

        for url in valid_urls:
            # ── Step A: Direct detection (trusting search results) ───────────
            logger.info(f"Targeting frame for detection: {url.split('/')[-1]}")
            
            # Subject confirmed via search → detect bounding box
            detection_result = await self.detect_object_with_bbox(url, characteristics, user_query)
            if not (detection_result and detection_result.get('detected_objects')):
                logger.info(f"No subject detected in frame: {url.split('/')[-1]}")
                continue

            # ── Step B: Subject detected → Draw bounding box ──────────────────
            objs = detection_result.get("detected_objects", [])
            bbox = objs[0].get("bbox", [])
            annotated_b64 = None
            
            try:
                from PIL import Image, ImageDraw
                import io

                async with httpx.AsyncClient() as dl:
                    resp = await dl.get(url, timeout=15)
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")

                W, H = img.size
                if len(bbox) == 4:
                    v1, v2, v3, v4 = bbox
                    # Heuristic: Determine if coordinates are 0-1, 0-1000, or raw pixels.
                    max_v = max(v1, v2, v3, v4)
                    
                    if max_v <= 1.0:
                        v1, v2, v3, v4 = v1 * W, v2 * H, v3 * W, v4 * H
                    elif max_v > 1005:
                        pass 
                    else:
                        v1 = (v1 * W) / 1000.0
                        v2 = (v2 * H) / 1000.0
                        v3 = (v3 * W) / 1000.0
                        v4 = (v4 * H) / 1000.0

                    x_coords = [v1, v3]
                    y_coords = [v2, v4]
                    
                    x1 = max(0, min(W, int(min(x_coords))))
                    x2 = max(0, min(W, int(max(x_coords))))
                    y1 = max(0, min(H, int(min(y_coords))))
                    y2 = max(0, min(H, int(max(y_coords))))

                    if x2 <= x1: x2 = min(W, x1 + 10)
                    if y2 <= y1: y2 = min(H, y1 + 10)

                    draw = ImageDraw.Draw(img)
                    lw = max(3, int(min(W, H) * 0.004))
                    for t in range(lw):
                        rect_x1, rect_y1 = x1 + t, y1 + t
                        rect_x2, rect_y2 = x2 - t, y2 - t
                        if rect_x2 >= rect_x1 and rect_y2 >= rect_y1:
                            draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], outline=(0, 100, 255))

                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90)
                annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
                logger.info(f"✅ Bounding box drawn on frame: {url.split('/')[-1]} | bbox={bbox}")
                
                # Yield intermediate result
                yield json.dumps({
                    "status": "Found subject in frame",
                    "detected_image_url": url,
                    "annotated_image": annotated_b64
                }) + "\n"
                
                detections_found.append({
                    "url": url,
                    "detection": detection_result,
                    "annotated_image": annotated_b64
                })

            except Exception as draw_err:
                logger.warning(f"Could not draw bbox on frame {url}: {draw_err}")

        # ── Final Status ─────────────────────────────────────────────────────
        if detections_found:
            final_detection = detections_found[0] # For backward compatibility
            yield json.dumps({
                "status": "Analysis Complete",
                "detection": final_detection["detection"],
                "detected_image_url": final_detection["url"],
                "annotated_image": final_detection["annotated_image"],
                "all_detections": detections_found
            }) + "\n"
        else:
            yield json.dumps({
                "status": "Analysis Complete",
                "final_answer": "No bounding boxes could be generated for the verified frames."
            }) + "\n"
