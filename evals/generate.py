"""Build the eval question set (evals/questions.jsonl).

    uv run python evals/generate.py draft        # one candidate question per clean page -> candidates.jsonl
    uv run python evals/generate.py verify       # mechanical checks: exact quote, answer in quote, distinctive
    uv run python evals/generate.py gold         # every page that supports the answer -> gold page set
    uv run python evals/generate.py closedbook   # can the evaluated model answer with no pages at all?
    uv run python evals/generate.py assemble     # choose ~200, split dev/test -> questions.jsonl

Drafting and checking use JUDGE_MODEL (deepseek-v4-pro), a different, stronger model than
the one being evaluated (deepseek-flash). Nothing it writes is trusted as is: quotes must be
exact substrings of the page, and the final set is reviewed by hand.
"""

import json
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import typer

from pageproof import config
from pageproof.evals.judge import judge
from pageproof.llm import complete_json
from pageproof.ocr.metrics import normalize

app = typer.Typer(no_args_is_help=True)
EVALS = Path(__file__).resolve().parent
CANDIDATES = EVALS / "candidates.jsonl"
QUESTIONS = EVALS / "questions.jsonl"
CORPUS = config.DATA / "corpus"
SEED = 20261026


@app.callback()
def main() -> None:
    """Build the eval question set."""


def pages() -> list[dict]:
    rows = duckdb.sql(f"""select page_id, book_id, title, page_no, text, blocks_json
                          from '{CORPUS}/pages.parquet' order by page_id""").fetchall()
    return [{"page_id": r[0], "book_id": r[1], "title": r[2], "page_no": r[3], "text": r[4],
             "paragraphs": "\n\n".join(b["text"] for b in json.loads(r[5]))} for r in rows]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def norm(text: str) -> str:
    return normalize(text).lower()


# --- draft ---------------------------------------------------------------------------------

DRAFT_SYSTEM = """You write evaluation questions for a question-answering system over scanned book pages.
You get ONE page. Write ONE question that this page alone answers.

Rules:
- Ask about a specific detail: a name, number, amount, object, place, or what someone said or did.
  Prefer minor details that a reader who knows the book well would NOT remember without the page.
  Avoid the book's famous plot points and themes.
- Make the question self-contained: name the book (or its author) and enough context to find the passage.
  Never mention a page number: readers of other editions don't have them.
- The answer must be short: 1 to 12 words, stated on the page.
- The question must not contain its answer.
- Copy an evidence quote of 8 to 40 words, character for character, from the page; it must contain the answer.

Return JSON only:
{"question": "...", "answer": "...", "evidence": "...", "kind": "name|number|object|place|action|quote"}
If the page has no suitable detail (e.g. a chapter heading only), return {"skip": true}."""


PAGE_REF = [  # page mentions the drafter sometimes adds despite the prompt
    (re.compile(r"^On pages?\s+\d+\s+of\s+", re.I), "In "),
    (re.compile(r"^According to pages?\s+\d+\s+of\s+", re.I), "According to "),
    (re.compile(r",\s*according to pages?\s+\d+,", re.I), ","),
    (re.compile(r",?\s*\(pages?\s+\d+\)", re.I), ""),
    (re.compile(r"\s*,?\s*according to pages?\s+\d+\s+of\s+", re.I), ", according to "),
    (re.compile(r",?\s+on pages?\s+\d+(?=[,?])", re.I), ""),
    (re.compile(r",\s*pages?\s+\d+,", re.I), ","),
    (re.compile(r"\s+pages?\s+\d+,", re.I), ","),
    (re.compile(r",?\s+pages?\s+\d+\b", re.I), ""),
]


def strip_page_refs(question: str) -> str:
    for pattern, repl in PAGE_REF:
        question = pattern.sub(repl, question)
    return re.sub(r"\s+([,?])", r"\1", question).strip()


def _draft_one(page: dict) -> dict | None:
    data, usage = complete_json(DRAFT_SYSTEM, f"Book: {page['title']}, page {page['page_no']}\n\n{page['paragraphs']}",
                                model=config.judge_model())
    if not data or data.get("skip") or not data.get("question"):
        return None
    return {"qid": f"q-{page['page_id']}", "source_page": page["page_id"], "book_id": page["book_id"],
            "title": page["title"], "question": strip_page_refs(data["question"]), "answer": str(data.get("answer", "")).strip(),
            "evidence": str(data.get("evidence", "")).strip(), "kind": data.get("kind", ""), "draft_usage": usage}


@app.command()
def draft(workers: int = 8) -> None:
    done = {c["source_page"] for c in read_jsonl(CANDIDATES)}
    todo = [p for p in pages() if p["page_id"] not in done]
    print(f"drafting {len(todo)} pages with {config.judge_model()}")
    with ThreadPoolExecutor(workers) as pool:
        new = [c for c in pool.map(_draft_one, todo) if c]
    write_jsonl(CANDIDATES, read_jsonl(CANDIDATES) + new)
    print(f"{len(new)} new candidates, {len(read_jsonl(CANDIDATES))} total")


# --- verify --------------------------------------------------------------------------------

STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "by", "with", "from", "at", "as",
        "is", "are", "was", "were", "be", "his", "her", "he", "she", "it", "that", "which", "what",
        "who", "how", "does", "did", "do", "had", "has", "they", "their", "this", "him", "them"}
WORD = re.compile(r"[a-z0-9]+")


def content_words(text: str) -> set[str]:
    return {w for w in WORD.findall(norm(text)) if w not in STOP}


def check(c: dict, by_id: dict[str, dict]) -> list[str]:
    """Problems with a candidate; empty means it passes."""
    page = by_id[c["source_page"]]
    problems = []
    ev, ans = norm(c["evidence"]), content_words(c["answer"])
    if not ev or ev not in norm(page["paragraphs"]):
        problems.append("evidence is not an exact quote from the page")
    if not ans:
        problems.append("empty answer")
    elif len(ans & content_words(c["evidence"])) / len(ans) < 0.5:
        problems.append("answer is not in the evidence quote")
    if len(c["answer"].split()) > 12:
        problems.append("answer longer than 12 words")
    if re.search(r"\bpages?\s+\d+", c["question"], re.I):
        problems.append("question names a page number")
    if ans and ans <= content_words(c["question"]):
        problems.append("question contains the answer")
    hits = sum(ev in norm(p["paragraphs"]) for p in by_id.values()) if ev else 0
    if hits > 2:
        problems.append(f"evidence appears on {hits} pages (not distinctive)")
    return problems


@app.command()
def verify() -> None:
    by_id = {p["page_id"]: p for p in pages()}
    rows = read_jsonl(CANDIDATES)
    for c in rows:
        c["problems"] = check(c, by_id)
    write_jsonl(CANDIDATES, rows)
    ok = [c for c in rows if not c["problems"]]
    print(f"{len(ok)}/{len(rows)} pass mechanical checks")
    from collections import Counter

    print(Counter(p.split(" (")[0] for c in rows for p in c["problems"]).most_common())


# --- gold page sets ------------------------------------------------------------------------

GOLD_SYSTEM = """You check which book pages support a reference answer to a question.
A page supports it if a reader of that page ALONE could give the reference answer.
Return JSON only: {"supporting": [<passage numbers>]}"""


def keyword_candidates(text: str, all_pages: list[dict], n: int = 8) -> list[dict]:
    """Pages ranked by rare-word overlap (a simple IDF), independent of the system under test."""
    terms = content_words(text)
    docs = [content_words(p["paragraphs"]) for p in all_pages]
    df = {t: sum(t in d for d in docs) for t in terms}
    idf = {t: math.log((1 + len(docs)) / (1 + df[t])) for t in terms if df[t]}
    ranked = sorted(range(len(all_pages)), key=lambda i: sum(w for t, w in idf.items() if t in docs[i]), reverse=True)
    return [all_pages[i] for i in ranked[:n]]


def _gold_one(args) -> dict:
    c, all_pages = args
    cands = keyword_candidates(f"{c['question']} {c['answer']} {c['evidence']}", all_pages)
    if c["source_page"] not in {p["page_id"] for p in cands}:
        cands.append(next(p for p in all_pages if p["page_id"] == c["source_page"]))
    passages = "\n\n".join(f"[{i}] {p['title']}, page {p['page_no']}\n{p['paragraphs']}" for i, p in enumerate(cands, 1))
    data, usage = complete_json(GOLD_SYSTEM, f"Question: {c['question']}\nReference answer: {c['answer']}\n\n{passages}",
                                model=config.judge_model())
    nums = [n for n in (data or {}).get("supporting", []) if isinstance(n, int) and 1 <= n <= len(cands)]
    found = [cands[n - 1]["page_id"] for n in nums]
    c["gold_pages"] = sorted(set(found) | {c["source_page"]})
    c["gold_source_confirmed"] = c["source_page"] in found
    c["gold_usage"] = usage
    return c


@app.command()
def gold(workers: int = 8) -> None:
    all_pages = pages()
    rows = read_jsonl(CANDIDATES)
    todo = [c for c in rows if not c.get("problems") and "gold_pages" not in c]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(_gold_one, [(c, all_pages) for c in todo]))
    write_jsonl(CANDIDATES, rows)
    done = [c for c in rows if "gold_pages" in c]
    print(f"gold sets for {len(done)}: {sum(len(c['gold_pages']) > 1 for c in done)} have more than one page; "
          f"{sum(not c['gold_source_confirmed'] for c in done)} where the checker didn't confirm the source page")


# --- closed book ---------------------------------------------------------------------------

CLOSED_SYSTEM = """Answer the question about a book from memory, in one sentence.
If you don't know, say "I don't know". Return JSON only: {"answer": "..."}"""


def _closed_one(c: dict) -> dict:
    data, _ = complete_json(CLOSED_SYSTEM, c["question"], model=config.llm_model(), max_tokens=3000)
    answer = str((data or {}).get("answer", ""))
    j = judge(c["question"], c["answer"], c["evidence"], answer)
    c["closed_book"] = {"answer": answer, "verdict": j["verdict"]}
    return c


@app.command()
def closedbook(workers: int = 8) -> None:
    rows = read_jsonl(CANDIDATES)
    todo = [c for c in rows if "gold_pages" in c and "closed_book" not in c]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(_closed_one, todo))
    write_jsonl(CANDIDATES, rows)
    done = [c for c in rows if "closed_book" in c]
    print(f"closed book correct: {sum(c['closed_book']['verdict'] == 'correct' for c in done)}/{len(done)}")


# --- assemble ------------------------------------------------------------------------------


@app.command()
def assemble(total: int = 200, dev: int = 50) -> None:
    """Pick `total` questions spread evenly over books, preferring ones the model can't answer
    from memory; split dev/test evenly by book."""
    rows = [c for c in read_jsonl(CANDIDATES) if "closed_book" in c and c["gold_source_confirmed"]]
    rng = random.Random(SEED)
    by_book: dict[int, list[dict]] = {}
    for c in rows:
        by_book.setdefault(c["book_id"], []).append(c)
    books = sorted(by_book)
    share = lambda n, i: n // len(books) + (i < n % len(books))  # noqa: E731  (spread the remainder)
    chosen = []
    for i, book_id in enumerate(books):
        cs = by_book[book_id]
        rng.shuffle(cs)
        cs.sort(key=lambda c: c["closed_book"]["verdict"] == "correct")  # memorized last (stable)
        picked = cs[:share(total, i)]
        rng.shuffle(picked)
        n_dev = share(dev, i)
        for i, c in enumerate(picked):
            c["split"] = "dev" if i < n_dev else "test"
        chosen += picked
    keep = ("qid", "split", "book_id", "title", "source_page", "gold_pages", "question", "answer",
            "evidence", "kind", "closed_book")
    out = [{k: c[k] for k in keep} | {"review": "pending"} for c in sorted(chosen, key=lambda c: c["qid"])]
    write_jsonl(QUESTIONS, out)
    print(f"{len(out)} questions ({sum(c['split'] == 'dev' for c in out)} dev, {sum(c['split'] == 'test' for c in out)} test); "
          f"closed-book correct: {sum(c['closed_book']['verdict'] == 'correct' for c in out)}")


if __name__ == "__main__":
    app()
