"""Check the LLM judge against hand labels before trusting it at scale.

    uv run python evals/judge_validation.py answers   # run the pipeline on dev questions in 3 conditions, judge each
    uv run python evals/judge_validation.py sample    # pick 60 answers for hand labelling -> judge_sample.jsonl
    uv run python evals/judge_validation.py kappa     # agreement of judge vs hand labels (judge_labels.jsonl)

Conditions are clean L0, tesseract L3 and tesseract L5, so the sample has right, partly
right and wrong answers. Non-correct verdicts are oversampled: a judge that says "correct"
to everything would still agree often on an easy sample.
Hand labels are made without seeing the judge's verdict.
"""

import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from pageproof.evals.judge import VERDICTS, exact_match, judge
from pageproof.qa.answer import ask

app = typer.Typer(no_args_is_help=True)
EVALS = Path(__file__).resolve().parent
QUESTIONS = EVALS / "questions.jsonl"
ANSWERS = EVALS / "judge_answers.jsonl"
SAMPLE = EVALS / "judge_sample.jsonl"
LABELS = EVALS / "judge_labels.jsonl"
CONDITIONS = [("clean", 0), ("tesseract", 3), ("tesseract", 5)]
SEED = 20261026


@app.callback()
def main() -> None:
    """Validate the judge against hand labels."""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _answer_one(args) -> dict:
    q, source, level = args
    a = ask(q["question"], source, level)
    j = judge(q["question"], q["answer"], q["evidence"], a.answer)
    return {"aid": f"{q['qid']}|{source}|L{level}", "qid": q["qid"], "source": source, "level": level,
            "question": q["question"], "reference": q["answer"], "evidence": q["evidence"],
            "answer": a.answer, "citations": a.citations, "gold_pages": q["gold_pages"],
            "retrieved": [p.chunk.page_id for p in a.passages],
            "exact_match": exact_match(q["answer"], a.answer),
            "judge": {k: j[k] for k in ("verdict", "refused", "reason")}}


@app.command()
def answers(workers: int = 6) -> None:
    dev = [q for q in read_jsonl(QUESTIONS) if q["split"] == "dev"]
    done = {r["aid"] for r in read_jsonl(ANSWERS)}
    todo = [(q, s, lv) for q in dev for s, lv in CONDITIONS if f"{q['qid']}|{s}|L{lv}" not in done]
    print(f"answering {len(todo)} (question, condition) pairs")
    with ThreadPoolExecutor(workers) as pool:
        new = list(pool.map(_answer_one, todo))
    rows = read_jsonl(ANSWERS) + new
    write_jsonl(ANSWERS, rows)
    for s, lv in CONDITIONS:
        sel = [r for r in rows if (r["source"], r["level"]) == (s, lv)]
        print(f"{s} L{lv}: {dict(Counter(r['judge']['verdict'] for r in sel))}")


@app.command()
def sample(n: int = 60) -> None:
    """Half the sample from non-correct verdicts (as far as they go), spread over conditions."""
    rows = read_jsonl(ANSWERS)
    rng = random.Random(SEED)
    rng.shuffle(rows)
    hard = [r for r in rows if r["judge"]["verdict"] != "correct"]
    easy = [r for r in rows if r["judge"]["verdict"] == "correct"]
    picked = hard[: n // 2]
    picked += easy[: n - len(picked)]
    rng.shuffle(picked)
    write_jsonl(SAMPLE, picked)
    print(f"{len(picked)} answers: {dict(Counter(r['judge']['verdict'] for r in picked))}; "
          f"by condition {dict(Counter(f'{r['source']} L{r['level']}' for r in picked))}")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n**2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


@app.command()
def kappa(labels_file: Path = LABELS, out: Path = Path("results/p4_judge_kappa.json")) -> None:
    """Labels come from a JSONL of {"aid": ..., "label": "correct|partial|incorrect"}: hand labels by default,
    or a second judge's (judge_labels_claude.jsonl -> results/p4_judge_vs_claude.json)."""
    labels = {r["aid"]: r["label"] for r in read_jsonl(labels_file) if r.get("label") in VERDICTS}
    rows = [r for r in read_jsonl(SAMPLE) if r["aid"] in labels]
    human = [labels[r["aid"]] for r in rows]
    judged = [r["judge"]["verdict"] for r in rows]
    binary = lambda xs: ["correct" if x == "correct" else "not" for x in xs]  # noqa: E731
    k3, k2 = cohen_kappa(human, judged), cohen_kappa(binary(human), binary(judged))
    agree = sum(h == j for h, j in zip(human, judged))
    print(f"{len(rows)} labelled; agreement {agree}/{len(rows)}; kappa 3-class {k3:.3f}, binary {k2:.3f}")
    confusion = Counter(zip(human, judged))
    print("labels \\ judge " + "  ".join(f"{v:>9s}" for v in VERDICTS))
    for h in VERDICTS:
        print(f"{h:>13s}  " + "  ".join(f"{confusion[(h, j)]:9d}" for j in VERDICTS))
    out.write_text(json.dumps({"n": len(rows), "agreement": agree, "kappa_3class": k3, "kappa_binary": k2,
                               "confusion": {f"{h}|{j}": c for (h, j), c in confusion.items()}}, indent=2) + "\n")
    disagree = [(r["aid"], labels[r["aid"]], r["judge"]["verdict"], r["judge"]["reason"]) for r in rows
                if labels[r["aid"]] != r["judge"]["verdict"]]
    for d in disagree:
        print("  disagree:", d)


if __name__ == "__main__":
    app()
