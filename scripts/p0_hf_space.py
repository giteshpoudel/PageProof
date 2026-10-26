"""P0 quick CUDA check: OCR the P0 pages through the public Hugging Face Space.

The Space (baidu/Unlimited-OCR, ZeroGPU A10G) runs the original CUDA code with
the same settings as our adapter: bf16, gundam mode, max_length 8192,
no_repeat_ngram_size 35, window 128. It does not pin a model revision, and it
returns text with layout tags already stripped.

    uv run --with gradio_client python scripts/p0_hf_space.py
"""

import time
from pathlib import Path

from gradio_client import Client, handle_file

PAGES = Path("data/p0/pg1342")
OUT = Path("data/p0/ocr/unlimited-hfspace")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = Client("baidu/Unlimited-OCR", verbose=False)
    for image in sorted(PAGES.glob("page_*.png")):
        out = OUT / f"{image.stem}.md"
        if out.exists():
            continue
        t = time.perf_counter()
        result = client.predict(handle_file(str(image)), "gundam", "document parsing.",
                                api_name="/run_ocr")
        text = result["text"] if isinstance(result, dict) else str(result)
        out.write_text(text, encoding="utf-8")
        print(f"{image.stem}: {len(text)} chars, {time.perf_counter() - t:.1f}s round trip")


if __name__ == "__main__":
    main()
