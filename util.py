# util.py - retry wrapper for OpenAI API calls
import time
from openai import RateLimitError, APIConnectionError, APITimeoutError

def with_retry(fn, max_attempts=8):
    for attempt in range(max_attempts):
        try:
            return fn()
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            wait = min(5 * (attempt + 1), 30)
            print(f"  {type(e).__name__}, waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("retries exhausted")
