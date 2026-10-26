"""OCR runs over the corpus: one Parquet file of outputs per source.

Each row: image_id, page_id, source, raw text, plain text, CER, WER, seconds.
Runs resume: images already in the output file are skipped.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from pageproof.ocr.metrics import cer, wer

OCR_DIR = Path("data/ocr")
CORPUS = Path("data/corpus")


def pending_images(source: str, out: Path, corpus: Path = CORPUS) -> list[tuple]:
    """(image_id, page_id, path, ground-truth text) for images not yet OCR'd by this source."""
    done = set()
    if out.exists():
        done = set(pq.read_table(out, columns=["image_id"]).column("image_id").to_pylist())
    rows = duckdb.sql(f"""
        select i.image_id, i.page_id, i.path, p.text
        from '{corpus}/images.parquet' i join '{corpus}/pages.parquet' p using (page_id)
        order by i.image_id
    """).fetchall()
    return [r for r in rows if r[0] not in done]


def append_rows(out: Path, rows: list[dict]) -> None:
    """Append by rewriting: the files are small (a few MB), and this keeps one file per source."""
    table = pa.Table.from_pylist(rows)
    if out.exists():
        table = pa.concat_tables([pq.read_table(out), table], promote_options="default")
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)


def _tesseract_one(item: tuple) -> dict:
    from PIL import Image

    from pageproof.ocr.tesseract import ocr_text

    image_id, page_id, path, text = item
    t = time.perf_counter()
    hyp = ocr_text(Image.open(path))
    secs = time.perf_counter() - t
    return {"image_id": image_id, "page_id": page_id, "source": "tesseract",
            "raw": hyp, "plain": hyp, "cer": cer(text, hyp), "wer": wer(text, hyp),
            "seconds": secs}


def run_tesseract(workers: int | None = None, batch: int = 200, log=print) -> Path:
    out = OCR_DIR / "tesseract.parquet"
    todo = pending_images("tesseract", out)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    log(f"tesseract: {len(todo)} images to do, {workers} workers")
    # One Tesseract thread per process; parallelism comes from the pool.
    os.environ["OMP_THREAD_LIMIT"] = "1"
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i in range(0, len(todo), batch):
            rows = list(pool.map(_tesseract_one, todo[i:i + batch], chunksize=4))
            append_rows(out, rows)
            done = i + len(rows)
            rate = done / (time.perf_counter() - start)
            log(f"  {done}/{len(todo)}  {rate:.1f} images/s")
    return out
