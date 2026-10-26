"""P0 feasibility: OCR 10 rendered pages with Tesseract and Unlimited-OCR.

Writes raw outputs to data/p0/ocr/<source>/ and per-page metrics to
results/p0_<label>.csv. Run the same script on CUDA with --label cuda, then
compare the two Unlimited-OCR outputs with scripts/p0_parity.py.

    uv run python scripts/p0_feasibility.py --label mac
"""

import argparse
import csv
import platform
import time
from pathlib import Path

from pageproof.corpus.render import render_book
from pageproof.ocr.metrics import cer, wer

BOOK_ID, FIRST_PAGE, N_PAGES = 1342, 22, 10  # Pride and Prejudice, body text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="machine label, e.g. mac or cuda")
    ap.add_argument("--model-dir", default="models/unlimited-ocr")
    ap.add_argument("--skip-tesseract", action="store_true")
    ap.add_argument("--n-pages", type=int, default=N_PAGES)
    args = ap.parse_args()

    data = Path("data/p0")
    pages = render_book(BOOK_ID, data, FIRST_PAGE, args.n_pages)
    rows = []

    if not args.skip_tesseract:
        from pageproof.ocr.tesseract import ocr_page as tess

        out = data / "ocr" / f"tesseract-{args.label}"
        out.mkdir(parents=True, exist_ok=True)
        for p in pages:
            t = time.perf_counter()
            hyp = tess(p.image)
            secs = time.perf_counter() - t
            (out / f"page_{p.page_no:04d}.txt").write_text(hyp, encoding="utf-8")
            rows.append(("tesseract", p.page_no, cer(p.text, hyp), wer(p.text, hyp), secs))
            print(f"tesseract  p{p.page_no}  CER {rows[-1][2]:.4f}  {secs:.1f}s")

    from pageproof.ocr.unlimited import UnlimitedOCR, to_plain_text

    t = time.perf_counter()
    model = UnlimitedOCR(args.model_dir)
    print(f"Unlimited-OCR loaded on {model.device} in {time.perf_counter() - t:.1f}s")
    out = data / "ocr" / f"unlimited-{args.label}"
    out.mkdir(parents=True, exist_ok=True)
    for p in pages:
        t = time.perf_counter()
        raw = model.ocr_page(p.image)
        secs = time.perf_counter() - t
        (out / f"page_{p.page_no:04d}.md").write_text(raw, encoding="utf-8")
        hyp = to_plain_text(raw)
        rows.append(("unlimited-ocr", p.page_no, cer(p.text, hyp), wer(p.text, hyp), secs))
        print(f"unlimited  p{p.page_no}  CER {rows[-1][2]:.4f}  {secs:.1f}s  ({len(raw)} chars)")

    results = Path("results")
    results.mkdir(exist_ok=True)
    path = results / f"p0_{args.label}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "page", "cer", "wer", "seconds", "machine"])
        machine = f"{platform.machine()} {platform.processor()} ({model.device})"
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.5f}", f"{r[3]:.5f}", f"{r[4]:.2f}", machine])

    for source in sorted({r[0] for r in rows}):
        sel = [r for r in rows if r[0] == source]
        n = len(sel)
        print(f"{source:14s} mean CER {sum(r[2] for r in sel) / n:.4f}  "
              f"mean WER {sum(r[3] for r in sel) / n:.4f}  "
              f"{sum(r[4] for r in sel) / n:.1f}s/page  (n={n})")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
