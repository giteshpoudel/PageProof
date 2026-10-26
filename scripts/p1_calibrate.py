"""Calibrate degradation strength: Tesseract CER per level and per single effect.

Target: level 0 near 0% CER, a steady rise, and level 5 hard but not blank
(Tesseract roughly 20-50% CER), so the curve has room on both ends.

    uv run python scripts/p1_calibrate.py
"""

import csv
import statistics
from pathlib import Path

import pymupdf
from PIL import Image

from pageproof.corpus.books import BOOKS
from pageproof.corpus.degrade import EFFECTS, N_LEVELS, degrade, level_severity
from pageproof.ocr.metrics import cer
from pageproof.ocr.tesseract import ocr_text

PDF_DIR = Path("data/corpus/pdf")


def sample_pages():
    for book in BOOKS:
        doc = pymupdf.open(PDF_DIR / f"pg{book.id}.pdf")
        for frac in (0.35, 0.65):
            i = int(doc.page_count * frac)
            pix = doc[i].get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
            img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
            yield book.id, i + 1, img, doc[i].get_text("text")


def main() -> None:
    rows = []
    for book_id, page_no, img, gt in sample_pages():
        seed = book_id * 10_000 + page_no
        for level in range(N_LEVELS):
            hyp = ocr_text(degrade(img, level_severity(level), seed))
            rows.append((book_id, page_no, "combined", level, cer(gt, hyp)))
        for effect in EFFECTS:
            for name, effects in ((f"only-{effect}", (effect,)),
                                  (f"all-but-{effect}", tuple(e for e in EFFECTS if e != effect))):
                hyp = ocr_text(degrade(img, 1.0, seed, effects=effects))
                rows.append((book_id, page_no, name, 5, cer(gt, hyp)))
        print(f"pg{book_id} p{page_no}: " + " ".join(
            f"{r[4]:.3f}" for r in rows if r[:2] == (book_id, page_no) and r[2] == "combined"))

    Path("results").mkdir(exist_ok=True)
    with open("results/p1_calibration.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["book_id", "page", "effects", "level", "tesseract_cer"])
        w.writerows([(*r[:4], f"{r[4]:.5f}") for r in rows])

    print("\ncombined, median Tesseract CER by level:")
    for level in range(N_LEVELS):
        v = [r[4] for r in rows if r[2] == "combined" and r[3] == level]
        print(f"  level {level}: median {statistics.median(v):.3f}  min {min(v):.3f}  max {max(v):.3f}")
    print("level 5, one effect alone / all effects but one:")
    for effect in EFFECTS:
        only = statistics.median(r[4] for r in rows if r[2] == f"only-{effect}")
        but = statistics.median(r[4] for r in rows if r[2] == f"all-but-{effect}")
        print(f"  {effect:7s}: alone {only:.3f}   all but it {but:.3f}")


if __name__ == "__main__":
    main()
