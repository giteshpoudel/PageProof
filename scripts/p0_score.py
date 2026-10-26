"""Rescore saved P0 OCR outputs without rerunning the models.

Updates the cer/wer columns of results/p0_<label>.csv in place (timings are kept).

    uv run python scripts/p0_score.py --label mac
"""

import argparse
import csv
import json
from pathlib import Path

from pageproof.ocr.metrics import cer, wer
from pageproof.ocr.unlimited import to_plain_text

SOURCES = {"tesseract": ("tesseract", ".txt", str), "unlimited-ocr": ("unlimited", ".md", to_plain_text)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    gt = {}
    with open("data/p0/pg1342/pages.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            gt[p["page_no"]] = p["text"]

    path = Path("results") / f"p0_{args.label}.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        prefix, ext, clean = SOURCES[r["source"]]
        out = Path("data/p0/ocr") / f"{prefix}-{args.label}" / f"page_{int(r['page']):04d}{ext}"
        hyp = clean(out.read_text(encoding="utf-8"))
        ref = gt[int(r["page"])]
        r["cer"], r["wer"] = f"{cer(ref, hyp):.5f}", f"{wer(ref, hyp):.5f}"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for source in SOURCES:
        sel = [r for r in rows if r["source"] == source]
        if sel:
            n = len(sel)
            print(f"{source:14s} mean CER {sum(float(r['cer']) for r in sel) / n:.4f}  "
                  f"mean WER {sum(float(r['wer']) for r in sel) / n:.4f}  "
                  f"{sum(float(r['seconds']) for r in sel) / n:.1f}s/page  (n={n})")


if __name__ == "__main__":
    main()
