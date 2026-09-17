

from contextual import video_contextual_query
from ollama import AsyncClient
from langsmith import traceable


import re

# Any of these appearing in the query means it has SOME date/time reference
# that video_contextual_query() would need to resolve. If none of them match,
# there is nothing for the contextual LLM to do — a bare keyword search like
# "have you seen a red car" mentions no date/time at all, so calling the LLM
# only risks it hallucinating a date/time that was never asked for.
_TEMPORAL_PATTERN = re.compile(
    r'\b('
    r'today|yesterday|tomorrow|tonight'
    r'|monday|tuesday|wednesday|thursday|friday|saturday|sunday'
    r'|january|february|march|april|may|june|july|august|september|october|november|december'
    r'|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec'
    r'|\d{1,2}(st|nd|rd|th)'                                   # "4th", "15th"
    r'|\d{1,2}\s*(am|pm)\b|\b(am|pm)\b'                        # "9am", "9 am", "pm"
    r'|\d{1,2}:\d{2}(:\d{2})?'                                 # "14:00", "14:00:00"
    r'|\bhour|\bhours|\bminute|\bminutes|\bmin\b|\bmins\b'
    r'|\bbefore\b|\bafter\b|\bsince\b|\buntil\b|\btill\b'
    r'|\blast\b|\bpast\b|\bbetween\b'
    r'|\d{1,2}\s*(to|-)\s*\d{1,2}\b'                           # "10 to 12", "9-5"
    r'|\d{4}-\d{2}-\d{2}'                                      # "2026-04-05"
    r')\b',
    re.IGNORECASE,
)


def _has_temporal_reference(query: str) -> bool:
    return bool(_TEMPORAL_PATTERN.search(query))


class VideoContextualAgent:
    def __init__(self, model: str = "llama3.1:latest"):
        """
        Initialize the Video Contextual Agent

        Args:
            model: Model name to use (default: qwen:7b)
        """
        self.model = model

    @traceable(name="contextual_agent", run_type="chain", project_name="video-summary")
    async def process_query(self, user_query: str) -> str:
        """
        Process user query with contextual understanding for video segments

        Args:
            user_query: Raw user query

        Returns:
            Contextualized query with resolved time and camera references
        """
        from langsmith import get_current_run_tree
        run = get_current_run_tree()
        if run:
            run.metadata.update({
                "component": "contextual",
                "model": self.model,
                "original_query": user_query,
            })

        # ✅ NEW: Skip if already has absolute datetime
        if re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', user_query):
            print(f"[INFO] Already resolved, skipping: {user_query}")
            return user_query

        # ✅ NEW: Skip entirely if the query has no date/time reference at all
        # (e.g. "have you seen a red car") — there is nothing for the
        # contextual LLM to resolve, so bypass it rather than risk it
        # inventing a date/time the user never mentioned.
        if not _has_temporal_reference(user_query):
            print(f"[INFO] No date/time reference, bypassing contextual LLM: {user_query}")
            return user_query

        try:
            # Generate system and user prompts
            system_prompt, user_prompt = video_contextual_query(user_query)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call Ollama API with GPU optimizations
            print(f"[INFO] Processing query with Ollama (model: {self.model})")
            client = AsyncClient()
            response = await client.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': 0.0,
                    'num_predict': 150,
                    'top_k': 10,
                    'top_p': 0.7,
                    'num_gpu': -1,
                    'num_thread': 8,
                },
                keep_alive="5m" # Keep model in memory for 5 mins
            )
            
            # Get the response and clean it
            contextualized = response.message.content.strip()
            
            # Remove common prefixes that the model might add
            prefixes_to_remove = [
                "Output:",
                "output:",
                "Your response:",
                "Response:",
                "Here's the query:",
                "Query:",
                "Rewritten query:",
                "Contextualized query:",
            ]
            
            for prefix in prefixes_to_remove:
                if contextualized.startswith(prefix):
                    contextualized = contextualized[len(prefix):].strip()
            
            print(f"[INFO] Original: {user_query}")
            print(f"[INFO] Contextualized: {contextualized}")
            
            return contextualized
            
        except Exception as e:
            print(f"[ERROR] Error in contextual query processing: {e}")
            # In case of error, return original query
            return user_query


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize agent with Ollama
    agent = VideoContextualAgent()
    
    print("="*70)
    print("VIDEO CONTEXTUAL AGENT - TESTING")
    print("="*70)
    
    # Test cases
    test_queries = [
        "Show me camera ATPL-908610-ARCIS",  # Should add "last 1 hour"
        "Get footage from 10 to 12",  # Should convert to "from 10:00:00 to 12:00:00"
        "What about 2pm to 4pm?",  # Should convert to "from 14:00:00 to 16:00:00"
        "Show last 30 minutes",  # Should convert to "last 0.5 hours"
        "Get segments from 10:05 to 12:05",  # Should convert to "from 10:05:00 to 12:05:00"
        "What about this camera?",  # Follow-up - should reference previous camera
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: {query}")
        print(f"{'='*70}")
        
        contextualized = asyncio.run(agent.process_query(query))
        print(f"Result: {contextualized}\n")
    
    print("="*70)
    print("TESTING COMPLETE")
    print("="*70)