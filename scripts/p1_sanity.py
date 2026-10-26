"""P1 sanity check: Tesseract on every level-0 image should be near 0% CER.

A high CER on a clean page means the ground truth and the image disagree
(a rendering or alignment bug), not an OCR problem.

    uv run python scripts/p1_sanity.py
"""

import csv
import statistics

import duckdb
from PIL import Image

from pageproof.ocr.metrics import cer
from pageproof.ocr.tesseract import ocr_text


def main() -> None:
    rows = duckdb.sql("""
        select i.page_id, p.title, i.path, p.text
        from 'data/corpus/images.parquet' i join 'data/corpus/pages.parquet' p using (page_id)
        where i.set = 'level' and i.level = 0 order by i.page_id
    """).fetchall()
    scores = []
    for page_id, title, path, text in rows:
        scores.append((page_id, title, cer(text, ocr_text(Image.open(path)))))
    with open("results/p1_sanity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page_id", "title", "tesseract_cer_level0"])
        w.writerows([(a, b, f"{c:.5f}") for a, b, c in scores])
    for title in sorted({s[1] for s in scores}):
        v = [s[2] for s in scores if s[1] == title]
        print(f"{title:36s} median {statistics.median(v):.4f}  max {max(v):.4f}")
    v = [s[2] for s in scores]
    print(f"all {len(v)} pages: median {statistics.median(v):.4f}  mean {statistics.mean(v):.4f}  max {max(v):.4f}")
    print("worst:", sorted(scores, key=lambda s: -s[2])[:5])


if __name__ == "__main__":
    main()
