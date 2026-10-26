"""Grade an answer against a reference: an LLM judge plus a mechanical exact-match check.

The judge (JUDGE_MODEL, deepseek-v4-pro) is a different model from the one being
evaluated. Before its verdicts are trusted at scale they are compared with hand
labels on ~60 answers (Cohen's kappa; see evals/judge_validation.py).
"""

import re

from pageproof import config
from pageproof.llm import complete_json
from pageproof.ocr.metrics import normalize

VERDICTS = ("correct", "partial", "incorrect")

JUDGE_SYSTEM = """You grade answers to questions about books against a reference answer. Be strict and literal.

- "correct": the answer states the reference answer's fact (wording may differ; names and numbers must match,
  allowing obvious spelling variants of names). Extra true detail is fine.
- "partial": the answer has part of the reference (e.g. one of two items) or is right but vague.
- "incorrect": wrong, contradicts the reference, names a different person/number/thing, or gives no answer.
- "refused": true if the answer says the information isn't available instead of answering.

Return JSON only: {"verdict": "correct|partial|incorrect", "refused": true/false, "reason": "<one short sentence>"}"""


def judge(question: str, reference: str, evidence: str, answer: str) -> dict:
    user = (f"Question: {question}\nReference answer: {reference}\nEvidence from the book: {evidence}\n\n"
            f"Answer to grade: {answer}")
    data, usage = complete_json(JUDGE_SYSTEM, user, model=config.judge_model())
    verdict = (data or {}).get("verdict")
    if verdict not in VERDICTS:
        verdict = "incorrect"
    return {"verdict": verdict, "refused": bool((data or {}).get("refused", False)),
            "reason": str((data or {}).get("reason", "")), "usage": usage}


def exact_match(reference: str, answer: str) -> bool:
    """The reference's words appear in order in the answer (after normalization)."""
    ref = re.sub(r"[^\w\s]", " ", normalize(reference).lower()).split()
    ans = " ".join(re.sub(r"[^\w\s]", " ", normalize(answer).lower()).split())
    return bool(ref) and " ".join(ref) in ans
