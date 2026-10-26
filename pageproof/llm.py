"""Small LLM helpers: one JSON-returning call with retries on transient errors."""

import json
import re
import time
from functools import lru_cache

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from pageproof import config

TRANSIENT = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


@lru_cache(maxsize=1)
def client() -> OpenAI:
    return OpenAI(api_key=config.llm_api_key(), base_url=config.llm_base_url(),
                  timeout=180, max_retries=0)


def parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def complete_json(system: str, user: str, model: str, max_tokens: int = 6000,
                  attempts: int = 4) -> tuple[dict | None, dict]:
    """(parsed JSON or None, usage dict). Reasoning models spend hidden tokens, so max_tokens is generous."""
    for attempt in range(attempts):
        try:
            resp = client().chat.completions.create(
                model=model, temperature=0, max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            u = resp.usage
            usage = {"model": model, "prompt_tokens": u.prompt_tokens if u else 0,
                     "completion_tokens": u.completion_tokens if u else 0}
            return parse_json(resp.choices[0].message.content or ""), usage
        except TRANSIENT:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt * 2)
    return None, {}
