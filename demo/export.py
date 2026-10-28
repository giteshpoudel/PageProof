"""Build the demo's data bundle: everything the app needs, with no corpus, index or Tesseract.

    uv run python demo/export.py                      # -> demo/bundle/
    uv run python demo/export.py --space build/space  # also lay out a static Hugging Face Space

For each test question's source page at every damage level: the page image (JPEG), what Tesseract
read and its CER, and the highlight box for the evidence quote, found ahead of time. Answers are the
saved eval runs (the lowest-numbered run per question and condition).
"""

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import typer
from PIL import Image

from pageproof.config import DATA
from pageproof.qa.locate import locate

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
LEVELS = range(6)
JPEG_QUALITY = 82
app = typer.Typer()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_box(args) -> tuple[str, int, list[int] | None, str]:
    """Box of the quote on the level image; if not found, the box from the clean (L0) image scaled across."""
    page_id, level, path, clean_path, quote = args
    image = Image.open(path).convert("L")
    hit = locate(image, quote)
    if hit:
        return page_id, level, list(hit.box), "found"
    if level:
        clean = Image.open(clean_path).convert("L")
        hit = locate(clean, quote)
        if hit:
            sx, sy = image.width / clean.width, image.height / clean.height
            x0, y0, x1, y1 = hit.box
            return page_id, level, [int(x0 * sx), int(y0 * sy) - 8, int(x1 * sx), int(y1 * sy) + 8], "from_clean"
    return page_id, level, None, "none"


@app.command()
def main(out: Path = DEMO / "bundle", space: Path | None = None, workers: int = 8) -> None:
    qs = [q for q in read_jsonl(ROOT / "evals" / "questions.jsonl") if q["split"] == "test"]
    by_page = {q["source_page"]: q for q in qs}
    if out.exists():
        shutil.rmtree(out)
    (out / "img").mkdir(parents=True)

    labels = {p: f"{t}, page {n}" for p, t, n in duckdb.sql(
        f"select page_id, title, page_no from '{DATA}/corpus/pages.parquet'").fetchall()}
    imgs = {(p, lv): path for p, lv, path in duckdb.sql(
        f"select page_id, level, path from '{DATA}/corpus/images.parquet' where \"set\" = 'level'").fetchall()}
    ocr = {(p, lv): (text, cer) for p, lv, text, cer in duckdb.sql(f"""
        select i.page_id, i.level, r.plain, r.cer from '{DATA}/ocr/tesseract.parquet' r
        join '{DATA}/corpus/images.parquet' i using (image_id) where i."set" = 'level'""").fetchall()}

    tasks = [(p, lv, str(ROOT / imgs[(p, lv)]), str(ROOT / imgs[(p, 0)]), q["evidence"])
             for p, q in by_page.items() for lv in LEVELS]
    with ProcessPoolExecutor(workers) as pool:
        boxes = {(p, lv): (box, how) for p, lv, box, how in pool.map(find_box, tasks, chunksize=8)}

    pages = {}
    for p in by_page:
        levels = {}
        for lv in LEVELS:
            name = f"{p}_L{lv}.jpg"
            src = ROOT / imgs[(p, lv)]
            if src.suffix.lower() in (".jpg", ".jpeg"):  # damaged levels are JPEGs already; re-encoding grows them
                shutil.copy(src, out / "img" / name)
            else:
                Image.open(src).convert("RGB").save(out / "img" / name, quality=JPEG_QUALITY)
            text, cer = ocr[(p, lv)]
            box, how = boxes[(p, lv)]
            levels[lv] = {"image": f"img/{name}", "ocr_text": text, "cer": cer, "box": box, "how": how}
        pages[p] = levels

    answers = {}
    keep = ("qid", "mode", "source", "level", "run", "answer", "citations", "quote", "quote_page",
            "verdict", "gold_retrieved")
    test_ids = {q["qid"] for q in qs}
    for r in sorted(read_jsonl(ROOT / "evals" / "runs" / "answers.jsonl"), key=lambda r: -r["run"]):
        if r["qid"] in test_ids:
            answers[f"{r['qid']}|{r['mode']}|{r['source']}|{r['level']}"] = {k: r.get(k) for k in keep}

    questions = [{k: q[k] for k in ("qid", "title", "question", "answer", "evidence", "source_page", "gold_pages")}
                 for q in qs]
    (out / "data.json").write_text(json.dumps(
        {"questions": questions, "answers": answers, "pages": pages, "labels": labels}, ensure_ascii=False))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    hows = [h for _, h in boxes.values()]
    print(f"{len(questions)} questions, {len(answers)} answers, {len(pages) * len(LEVELS)} images, "
          f"{size / 1e6:.0f} MB -> {out}")
    print(f"highlights: {hows.count('found')} found, {hows.count('from_clean')} from the clean page, "
          f"{hows.count('none')} not located")

    if space:
        if space.exists():
            shutil.rmtree(space)
        shutil.copytree(DEMO / "space", space)
        shutil.copy(DEMO / "static" / "index.html", space / "index.html")
        shutil.copytree(out, space / "bundle")
        print(f"Space -> {space}")


if __name__ == "__main__":
    app()
