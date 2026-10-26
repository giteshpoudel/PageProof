"""Build the synthetic corpus: page images at every degradation level, with ground truth.

Output (under data/corpus/):
  pdf/pg<id>.pdf      rendered books
  img/pg<id>/...      page images
  pages.parquet       one row per base page: exact text and text blocks with boxes
  images.parquet      one row per image: which page, level or leave-one-out set, seed, path
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pymupdf
from PIL import Image

from pageproof.corpus.books import BOOKS, Book
from pageproof.corpus.degrade import EFFECTS, N_LEVELS, level_severity, save_degraded
from pageproof.corpus.render import BASE_DPI, clean_html, fetch_html, render_pdf

PAGES_PER_BOOK = 65
LOO_PAGES_PER_BOOK = 10  # leave-one-out set: first 10 of each book's base pages, at level 5
SEED = 20261026


def page_id(book_id: int, page_no: int) -> str:
    return f"pg{book_id}-p{page_no:04d}"


def choose_pages(book: Book, page_count: int, n: int) -> list[int]:
    """A contiguous run of n body pages, at a seeded offset between 10% and 85% of the book.

    Contiguous so that questions can span neighbouring pages; away from the ends
    to skip front matter, tables of contents and back matter.
    """
    rng = np.random.default_rng(SEED + book.id)
    lo, hi = int(page_count * 0.10), int(page_count * 0.85) - n
    start = int(rng.integers(lo, max(lo + 1, hi)))
    return list(range(start + 1, start + n + 1))  # 1-based page numbers


def _seed(book_id: int, page_no: int) -> int:
    return SEED + book_id * 10_000 + page_no


def build(root: Path = Path("data/corpus"), pages_per_book: int = PAGES_PER_BOOK,
          loo_pages: int = LOO_PAGES_PER_BOOK, books=BOOKS, log=print) -> tuple[Path, Path]:
    pdf_dir, img_dir = root / "pdf", root / "img"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    page_rows, image_rows = [], []

    for book in books:
        pdf = pdf_dir / f"pg{book.id}.pdf"
        if not pdf.exists():
            render_pdf(clean_html(fetch_html(book.id, Path("data/html"))), pdf)
        doc = pymupdf.open(pdf)
        chosen = choose_pages(book, doc.page_count, pages_per_book)
        out = img_dir / f"pg{book.id}"
        out.mkdir(parents=True, exist_ok=True)

        for k, page_no in enumerate(chosen):
            page = doc[page_no - 1]
            pid = page_id(book.id, page_no)
            blocks = [
                {"bbox": [round(v, 2) for v in b[:4]], "text": b[4]}
                for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()
            ]
            page_rows.append({
                "page_id": pid, "book_id": book.id, "title": book.title, "layout": book.layout,
                "page_no": page_no, "page_count": doc.page_count,
                "width_pt": page.rect.width, "height_pt": page.rect.height,
                "text": page.get_text("text"), "blocks_json": json.dumps(blocks, ensure_ascii=False),
            })

            pix = page.get_pixmap(dpi=BASE_DPI, colorspace=pymupdf.csGRAY)
            clean = Image.frombytes("L", (pix.width, pix.height), pix.samples)
            seed = _seed(book.id, page_no)
            variants = [("level", level, EFFECTS) for level in range(N_LEVELS)]
            if k < loo_pages:
                variants += [("loo", N_LEVELS - 1, tuple(e for e in EFFECTS if e != drop))
                             for drop in EFFECTS]

            for kind, level, effects in variants:
                dropped = sorted(set(EFFECTS) - set(effects))
                name = f"p{page_no:04d}_L{level}" + (f"_no-{dropped[0]}" if kind == "loo" else "")
                path = save_degraded(clean, level_severity(level), seed, out / name, effects)
                with Image.open(path) as im:
                    w, h = im.size
                image_rows.append({
                    "image_id": f"{pid}-{path.stem.split('_', 1)[1]}", "page_id": pid,
                    "book_id": book.id, "set": kind, "level": level,
                    "severity": level_severity(level), "effects": ",".join(effects),
                    "dropped_effect": dropped[0] if kind == "loo" else None,
                    "seed": seed, "path": str(path), "width_px": w, "height_px": h,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
        log(f"pg{book.id} {book.title}: pages {chosen[0]}-{chosen[-1]} of {doc.page_count}")
        doc.close()

    pages_path, images_path = root / "pages.parquet", root / "images.parquet"
    pq.write_table(pa.Table.from_pylist(page_rows), pages_path)
    pq.write_table(pa.Table.from_pylist(image_rows), images_path)
    log(f"{len(page_rows)} base pages, {len(image_rows)} images -> {pages_path}, {images_path}")
    return pages_path, images_path
