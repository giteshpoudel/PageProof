"""Answer a question from retrieved passages, citing pages, with a supporting quote.

The model sees only the retrieved OCR passages, each labelled with its page id,
and returns JSON: the answer, the page ids it relied on, and a short quote
copied from one passage. The quote is what gets highlighted on the page image.
"""

import json
import re
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel, Field

from pageproof import config
from pageproof.qa.retrieve import SearchResult, search

NOT_FOUND = "Not found in the retrieved pages."

SYSTEM = f"""You answer questions about books using only the numbered passages given.
The passages come from OCR of scanned pages and may contain recognition errors; read through
obvious misspellings, but never add facts the passages don't contain.

Reply with a JSON object:
{{"answer": "<one or two sentences>",
  "citations": ["<page id of each passage you used>"],
  "quote": "<5 to 25 words copied exactly from the passage that best supports the answer>",
  "quote_page": "<page id of the passage the quote comes from>"}}

If the passages don't contain the answer, reply
{{"answer": "{NOT_FOUND}", "citations": [], "quote": "", "quote_page": ""}}"""


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    quote: str = ""
    quote_page: str = ""
    passages: list[SearchResult] = Field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def found(self) -> bool:
        return self.answer.strip() != NOT_FOUND and bool(self.citations)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=config.llm_api_key(), base_url=config.llm_base_url(), timeout=120)


def format_passages(results: list[SearchResult]) -> str:
    return "\n\n".join(
        f"[{i}] page id: {r.chunk.page_id} ({r.chunk.title}, page {r.chunk.page_no})\n{r.chunk.text}"
        for i, r in enumerate(results, 1)
    )


def parse_reply(text: str) -> dict:
    """The JSON object in the reply (tolerates a code fence or text around it)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"answer": NOT_FOUND, "citations": []}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"answer": NOT_FOUND, "citations": []}


def answer_from(question: str, results: list[SearchResult], client=None, model: str | None = None) -> Answer:
    client = client or _client()
    model = model or config.llm_model()
    # deepseek-flash reasons before answering and those tokens count against max_tokens;
    # EDGAR saw empty answers when the limit was too low.
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=4000,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Passages:\n\n{format_passages(results)}\n\nQuestion: {question}"}],
    )
    data = parse_reply(resp.choices[0].message.content or "")
    valid = {r.chunk.page_id for r in results}
    citations = [c for c in data.get("citations", []) if c in valid]  # drop invented page ids
    quote_page = data.get("quote_page", "") if data.get("quote_page", "") in valid else ""
    usage = resp.usage
    return Answer(question=question, answer=str(data.get("answer", NOT_FOUND)).strip(),
                  citations=list(dict.fromkeys(citations)), quote=str(data.get("quote", "")).strip(),
                  quote_page=quote_page or (citations[0] if citations else ""), passages=results,
                  model=model, prompt_tokens=usage.prompt_tokens if usage else 0,
                  completion_tokens=usage.completion_tokens if usage else 0)


def ask(question: str, source: str = "tesseract", level: int = 0, k: int = 5,
        mode: str = "hybrid", client=None) -> Answer:
    return answer_from(question, search(question, source, level, k, mode), client=client)
