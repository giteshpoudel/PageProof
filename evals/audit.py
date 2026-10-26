"""Audit the question set with two independent checkers, so only a few questions need a person.

    uv run python evals/audit.py run       # both checkers on every question -> evals/audit.jsonl
    uv run python evals/audit.py sample    # flagged questions + a random 20 that passed -> evals/review_sample.jsonl
    uv run python evals/audit.py estimate  # after hand review: error rate among passed questions, with a 95% bound

The checkers use a different prompt from the drafter: they look for reasons to reject.
One runs on deepseek-v4-pro (the drafting model), one on deepseek-flash (a different model),
and a question is flagged if either objects. The random sample of passed questions measures
how many bad ones slip through both.
"""

import json
import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import typer

from pageproof import config
from pageproof.llm import complete_json

app = typer.Typer(no_args_is_help=True)
EVALS = Path(__file__).resolve().parent
QUESTIONS = EVALS / "questions.jsonl"
AUDIT = EVALS / "audit.jsonl"
SAMPLE = EVALS / "review_sample.jsonl"
REVIEWS = EVALS / "question_reviews.jsonl"
SEED = 20261026
CHECKS = ("answerable", "single_answer", "unambiguous", "no_outside_knowledge", "answer_matches")

AUDIT_SYSTEM = """You audit an evaluation question for a question-answering system over book pages.
Look for reasons to REJECT it. You get the page, the question, the reference answer and an evidence quote.

Check each, true or false:
- answerable: a reader of this page alone can answer the question.
- single_answer: there is one correct answer; another reading of the page wouldn't give a different, equally valid one.
- unambiguous: the question names enough context (book, scene, speaker) to point at this passage, not similar ones elsewhere in the book.
- no_outside_knowledge: answering needs nothing beyond the page (no knowledge of other chapters or history).
- answer_matches: the reference answer is correct and complete for the question as asked.

Return JSON only:
{"answerable": true, "single_answer": true, "unambiguous": true, "no_outside_knowledge": true,
 "answer_matches": true, "problem": "<one short sentence if any check is false, else empty>"}"""


@app.callback()
def main() -> None:
    """Audit the question set."""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def page_texts() -> dict[str, str]:
    rows = duckdb.sql(f"select page_id, blocks_json from '{config.DATA}/corpus/pages.parquet'").fetchall()
    return {r[0]: "\n\n".join(b["text"] for b in json.loads(r[1])) for r in rows}


def _check(q: dict, page: str, model: str) -> dict:
    user = (f"Book: {q['title']}\n\nPage:\n{page}\n\nQuestion: {q['question']}\n"
            f"Reference answer: {q['answer']}\nEvidence quote: {q['evidence']}")
    data, _ = complete_json(AUDIT_SYSTEM, user, model=model)
    data = data or {}
    failed = [c for c in CHECKS if data.get(c) is not True]  # a missing or unparsable check counts as failed
    return {"model": model, "failed": failed, "problem": str(data.get("problem", ""))}


def _audit_one(args) -> dict:
    q, page = args
    checks = [_check(q, page, config.judge_model()), _check(q, page, config.llm_model())]
    return {"qid": q["qid"], "flagged": any(c["failed"] for c in checks), "checks": checks}


@app.command()
def run(workers: int = 8) -> None:
    pages = page_texts()
    qs = read_jsonl(QUESTIONS)
    done = {a["qid"] for a in read_jsonl(AUDIT)}
    todo = [(q, pages[q["source_page"]]) for q in qs if q["qid"] not in done]
    with ThreadPoolExecutor(workers) as pool:
        new = list(pool.map(_audit_one, todo))
    rows = read_jsonl(AUDIT) + new
    write_jsonl(AUDIT, rows)
    flagged = [a for a in rows if a["flagged"]]
    by_model = Counter(c["model"] for a in rows for c in a["checks"] if c["failed"])
    print(f"{len(flagged)}/{len(rows)} flagged; objections by model {dict(by_model)}")
    print("failed checks:", dict(Counter(f for a in rows for c in a["checks"] for f in c["failed"])))


@app.command()
def sample(n_passed: int = 20) -> None:
    audit = {a["qid"]: a for a in read_jsonl(AUDIT)}
    qs = read_jsonl(QUESTIONS)
    flagged = [q["qid"] for q in qs if audit[q["qid"]]["flagged"]]
    passed = [q["qid"] for q in qs if not audit[q["qid"]]["flagged"]]
    rng = random.Random(SEED)
    picked = rng.sample(passed, min(n_passed, len(passed)))
    rows = [{"qid": qid, "why": "flagged"} for qid in flagged] + [{"qid": qid, "why": "random"} for qid in picked]
    write_jsonl(SAMPLE, rows)
    print(f"review set: {len(flagged)} flagged + {len(picked)} random passed = {len(rows)}")


def upper_bound(k: int, n: int, conf: float = 0.95) -> float:
    """Largest p with P(X <= k | n, p) >= 1 - conf: the exact one-sided Clopper-Pearson upper bound."""
    if n == 0 or k >= n:
        return 1.0
    cdf = lambda p: sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))  # noqa: E731
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) > 1 - conf else (lo, mid)
    return lo


@app.command()
def estimate() -> None:
    """Share of passed questions a person rejected, with a one-sided 95% upper bound (Clopper-Pearson)."""
    reviews = {r["qid"]: r for r in read_jsonl(REVIEWS)}
    rnd = [s["qid"] for s in read_jsonl(SAMPLE) if s["why"] == "random" and s["qid"] in reviews]
    bad = sum(reviews[q]["review"] == "rejected" for q in rnd)
    n = len(rnd)
    upper = upper_bound(bad, n)
    flg = [s["qid"] for s in read_jsonl(SAMPLE) if s["why"] == "flagged" and s["qid"] in reviews]
    print(f"random passed: {bad}/{n} rejected; 95% upper bound on bad share {upper:.1%}")
    print(f"flagged: {sum(reviews[q]['review'] == 'rejected' for q in flg)}/{len(flg)} rejected")
    Path("results/p4_audit.json").write_text(json.dumps({
        "random_passed_reviewed": n, "random_passed_rejected": bad, "upper95_bad_share": upper,
        "flagged_reviewed": len(flg), "flagged_rejected": sum(reviews[q]["review"] == "rejected" for q in flg)}, indent=2) + "\n")


if __name__ == "__main__":
    app()
