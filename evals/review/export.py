"""Turn questions.jsonl and judge_sample.jsonl into documents for the review page.

    uv run python evals/review/export.py questions OUT_DIR   # one JSON file per question, with its page text
    uv run python evals/review/export.py labels OUT_DIR      # one JSON file per sampled answer, judge verdict left out

The page (evals/review/review.html) stores hand calls in its own database; they are read
back into evals/question_reviews.jsonl and evals/judge_labels.jsonl.
"""

import json
import sys
from pathlib import Path

import duckdb

from pageproof import config

EVALS = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def page_info() -> dict[str, tuple[int, str]]:
    rows = duckdb.sql(f"select page_id, page_no, blocks_json from '{config.DATA}/corpus/pages.parquet'").fetchall()
    return {r[0]: (r[1], "\n\n".join(b["text"] for b in json.loads(r[2]))) for r in rows}


def questions() -> list[dict]:
    info = page_info()
    docs = []
    for q in read_jsonl(EVALS / "questions.jsonl"):
        page_no, text = info[q["source_page"]]
        docs.append({"path": f"questions/{q['qid']}", "data": {
            "qid": q["qid"], "split": q["split"], "book_id": q["book_id"], "title": q["title"],
            "source_page": q["source_page"], "page_no": page_no, "page_text": text,
            "question": q["question"], "answer": q["answer"], "evidence": q["evidence"], "kind": q["kind"],
            "gold_pages": q["gold_pages"], "closed_book_verdict": q["closed_book"]["verdict"],
            "review": "pending", "edited_question": "", "edited_answer": "", "note": ""}})
    return docs


def labels() -> list[dict]:
    titles = {q["qid"]: q["title"] for q in read_jsonl(EVALS / "questions.jsonl")}
    docs = []
    for i, r in enumerate(read_jsonl(EVALS / "judge_sample.jsonl"), 1):
        docs.append({"path": f"labels/a{i:03d}", "data": {
            "aid": r["aid"], "short_id": f"a{i:03d}", "title": titles.get(r["qid"], ""),
            "question": r["question"], "reference": r["reference"], "evidence": r["evidence"],
            "answer": r["answer"], "citations": r["citations"], "gold_pages": r["gold_pages"], "label": ""}})
    return docs


if __name__ == "__main__":
    kind, out = sys.argv[1], Path(sys.argv[2])
    docs = questions() if kind == "questions" else labels()
    for d in docs:
        path = out / f"{d['path']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d["data"], ensure_ascii=False))
    print(f"{len(docs)} docs -> {out}")
