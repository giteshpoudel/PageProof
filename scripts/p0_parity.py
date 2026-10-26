"""Mac vs CUDA parity: CER between the two machines' Unlimited-OCR outputs.

    uv run python scripts/p0_parity.py --a unlimited-mac --b unlimited-cuda
"""

import argparse
from pathlib import Path

from pageproof.ocr.metrics import cer
from pageproof.ocr.unlimited import to_plain_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="unlimited-mac")
    ap.add_argument("--b", default="unlimited-cuda")
    ap.add_argument("--root", default="data/p0/ocr")
    args = ap.parse_args()

    a_dir, b_dir = Path(args.root) / args.a, Path(args.root) / args.b
    scores = []
    for a_file in sorted(a_dir.glob("page_*.md")):
        b_file = b_dir / a_file.name
        if not b_file.exists():
            continue
        a = to_plain_text(a_file.read_text(encoding="utf-8"))
        b = to_plain_text(b_file.read_text(encoding="utf-8"))
        scores.append(cer(b, a))  # CUDA output as the reference
        print(f"{a_file.stem}  CER {scores[-1]:.4f}  identical={a == b}")
    if scores:
        print(f"mean parity CER {sum(scores) / len(scores):.4f} over {len(scores)} pages")
    else:
        print("no matching pages found")


if __name__ == "__main__":
    main()
