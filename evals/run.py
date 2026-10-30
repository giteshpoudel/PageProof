"""Final eval runs on the test split: every OCR source x level, three conditions, three runs.

    uv run python evals/run.py answers            # resumable; appends to evals/runs/answers.jsonl
    uv run python evals/run.py summary            # accuracy per cell with bootstrap CIs -> results/p4_test_*.csv

Conditions (see OCR_PROJECT_PLAN.md, experimental design):
- full: retrieve 5 passages from that index, then answer (recall@5 comes from the same retrieval)
- oracle: the gold page(s) OCR text from that index, no retrieval (isolates reading)
- closed: no context at all (the memorization floor; run once per run, not per index)

Runs are the outer loop, so after run 1 every cell already has one complete pass.
Each answer is judged by JUDGE_MODEL. Rows are written as they finish, so a crash loses nothing.
"""

import json
import random
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from pageproof import config
from pageproof.evals.judge import exact_match, judge
from pageproof.index.chunking import Chunk
from pageproof.index.store import indexed_sources, page_texts
from pageproof.llm import complete_json
from pageproof.qa.answer import answer_from
from pageproof.qa.retrieve import SearchResult, search

app = typer.Typer(no_args_is_help=True)
EVALS = Path(__file__).resolve().parent
QUESTIONS = EVALS / "questions.jsonl"
OUT = EVALS / "runs" / "answers.jsonl"
# clean text plus every OCR source that has been indexed (Tesseract; Unlimited-OCR after the GPU run)
CELLS = [("clean", 0)] + [(s, lv) for s in indexed_sources() for lv in range(6)]
RUNS = 3
SEED = 20261026
CLOSED_SYSTEM = """Answer the question about a book from memory, in one sentence.
If you don't know, say "I don't know". Return JSON only: {"answer": "..."}"""

_lock = threading.Lock()
_pages: dict[tuple[str, int], dict[str, Chunk]] = {}


@app.callback()
def main() -> None:
    """Final eval runs."""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def key(run: int, mode: str, source: str, level: int, qid: str) -> str:
    return f"r{run}|{mode}|{source}|L{level}|{qid}"


def gold_passages(q: dict, source: str, level: int) -> list[SearchResult]:
    """Whole gold pages as passages, shaped like retrieval results."""
    if (source, level) not in _pages:
        with _lock:
            if (source, level) not in _pages:
                _pages[(source, level)] = {
                    page_id: Chunk(chunk_id=f"{source}-L{level}:{page_id}:page", index=f"{source}-L{level}",
                                   source=source, level=level, page_id=page_id, image_id=image_id,
                                   book_id=book_id, title=title, page_no=page_no, position=0, text=text or "")
                    for image_id, page_id, book_id, title, page_no, text in page_texts(source, level)}
    pages = _pages[(source, level)]
    return [SearchResult(chunk=pages[p], score=1.0) for p in q["gold_pages"] if p in pages]


def _one(task) -> dict | None:
    run, mode, source, level, q = task
    try:
        t0 = time.time()
        if mode == "closed":
            data, usage = complete_json(CLOSED_SYSTEM, q["question"], model=config.llm_model(), max_tokens=12000)
            answer, citations, retrieved = str((data or {}).get("answer", "")), [], []
            quote, quote_page = "", ""
            tokens = (usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        else:
            results = (search(q["question"], source, level, k=5) if mode == "full"
                       else gold_passages(q, source, level))
            a = answer_from(q["question"], results)
            answer, citations, quote, quote_page = a.answer, a.citations, a.quote, a.quote_page
            retrieved = [r.chunk.page_id for r in results]
            tokens = (a.prompt_tokens, a.completion_tokens)
        j = judge(q["question"], q["answer"], q["evidence"], answer)
        gold = set(q["gold_pages"])
        row = {"key": key(run, mode, source, level, q["qid"]), "run": run, "mode": mode, "source": source,
               "level": level, "qid": q["qid"], "book_id": q["book_id"],
               "memorized": q["closed_book"]["verdict"] == "correct", "answer": answer, "citations": citations,
               "quote": quote, "quote_page": quote_page,
               "retrieved": retrieved, "gold_retrieved": bool(gold & set(retrieved)) if mode == "full" else None,
               "gold_cited": bool(gold & set(citations)), "exact_match": exact_match(q["answer"], answer),
               "verdict": j["verdict"], "refused": j["refused"], "reason": j["reason"],
               "answer_tokens": tokens, "judge_tokens": (j["usage"].get("prompt_tokens", 0),
                                                         j["usage"].get("completion_tokens", 0)),
               "seconds": round(time.time() - t0, 1)}
        with _lock, OUT.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row
    except Exception:  # noqa: BLE001 - one failed call shouldn't stop a multi-hour run; resume retries it
        with _lock:
            print(f"failed {key(run, mode, source, level, q['qid'])}:\n{traceback.format_exc(limit=2)}", flush=True)
        return None


@app.command()
def answers(split: str = "test", runs: int = RUNS, workers: int = 10) -> None:
    OUT.parent.mkdir(exist_ok=True)
    qs = [q for q in read_jsonl(QUESTIONS) if q["split"] == split]
    done = {r["key"] for r in read_jsonl(OUT)}
    print("cells: clean + " + ", ".join(sorted({s for s, _ in CELLS} - {"clean"})), flush=True)
    for run in range(1, runs + 1):
        tasks = [(run, "closed", "none", 0, q) for q in qs]
        tasks += [(run, mode, s, lv, q) for mode in ("full", "oracle") for s, lv in CELLS for q in qs]
        tasks = [t for t in tasks if key(t[0], t[1], t[2], t[3], t[4]["qid"]) not in done]
        print(f"run {run}: {len(tasks)} answers to do", flush=True)
        t0, n_ok = time.time(), 0
        with ThreadPoolExecutor(workers) as pool:
            for i, row in enumerate(pool.map(_one, tasks), 1):
                n_ok += row is not None
                if i % 100 == 0:
                    print(f"  run {run}: {i}/{len(tasks)} ({n_ok} ok) {time.time() - t0:.0f}s", flush=True)
        print(f"run {run} finished: {n_ok}/{len(tasks)} ok in {time.time() - t0:.0f}s", flush=True)


def bootstrap(per_q: dict[str, float], n: int = 2000) -> tuple[float, float, float]:
    """Mean over questions with a 95% percentile bootstrap CI (questions resampled)."""
    vals = list(per_q.values())
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(SEED)
    means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n))
    return sum(vals) / len(vals), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def with_memorized(rows: list[dict]) -> list[dict]:
    """Set `memorized` from these runs' own closed-book answers (correct in at least half of them).
    The flag stored at answer time came from question selection, where 38 of 376 closed-book
    answers were cut off by a 3,000-token limit; it's only the fallback."""
    closed = defaultdict(list)
    for r in rows:
        if r["mode"] == "closed":
            closed[r["qid"]].append(r["verdict"] == "correct")
    return [r | {"memorized": sum(closed[r["qid"]]) / len(closed[r["qid"]]) >= 0.5} if closed[r["qid"]] else r
            for r in rows]


@app.command()
def summary(runs: str = "") -> None:
    """runs: comma-separated run numbers to include (default all), e.g. --runs 1 while later runs are incomplete."""
    rows = read_jsonl(OUT)
    if runs:
        rows = [r for r in rows if str(r["run"]) in runs.split(",")]
    rows = with_memorized(rows)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["mode"], r["source"], r["level"])].append(r)
    lines = ["mode,source,level,answers,runs,questions,accuracy,ci_low,ci_high,partial_share,refused_share,"
             "recall_at_5,gold_cited,accuracy_not_memorized,accuracy_memorized"]
    for (mode, source, level), rs in sorted(cells.items()):
        by_q = defaultdict(list)
        for r in rs:
            by_q[r["qid"]].append(r)
        mean_of = lambda f, sel=None: {q: sum(f(r) for r in v) / len(v)  # noqa: E731
                                       for q, v in by_q.items() if sel is None or sel(v[0])}
        acc, lo, hi = bootstrap(mean_of(lambda r: r["verdict"] == "correct"))
        avg = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
        recall = avg(mean_of(lambda r: bool(r["gold_retrieved"]))) if mode == "full" else float("nan")
        lines.append(",".join(str(x) for x in [
            mode, source, level, len(rs), len({r["run"] for r in rs}), len(by_q),
            f"{acc:.3f}", f"{lo:.3f}", f"{hi:.3f}",
            f"{avg(mean_of(lambda r: r['verdict'] == 'partial')):.3f}",
            f"{avg(mean_of(lambda r: r['refused'])):.3f}", f"{recall:.3f}",
            f"{avg(mean_of(lambda r: r['gold_cited'])):.3f}",
            f"{avg(mean_of(lambda r: r['verdict'] == 'correct', lambda r: not r['memorized'])):.3f}",
            f"{avg(mean_of(lambda r: r['verdict'] == 'correct', lambda r: r['memorized'])):.3f}"]))
    out = Path("results/p4_test_cells.csv")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    tok = [sum(r["answer_tokens"]) + sum(r["judge_tokens"]) for r in rows]
    print(f"{len(rows)} answers; {sum(tok):,} tokens total (answer + judge)")


if __name__ == "__main__":
    app()
