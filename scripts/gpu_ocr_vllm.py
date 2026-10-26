"""Unlimited-OCR over the whole corpus through a vLLM server (run on the rented GPU).

Start the server first (see docs/gpu_runbook.md), then:

    uv run --extra gpu python scripts/gpu_ocr_vllm.py --gpu-price 0.34

Sends every corpus image concurrently, single-page mode, settings from the
official vLLM recipe (recipes.vllm.ai/baidu/Unlimited-OCR): prompt
"<image>document parsing.", greedy decoding, max 8,192 tokens, n-gram logits
processor (size 35, window 128), special tokens kept.

Writes data/ocr/unlimited-vllm.parquet (resumes) and results/p2_throughput.json.
"""

import argparse
import asyncio
import base64
import json
import mimetypes
import subprocess
import time
from pathlib import Path

from openai import AsyncOpenAI

from pageproof.ocr.metrics import cer, wer
from pageproof.ocr.runs import OCR_DIR, append_rows, pending_images
from pageproof.ocr.unlimited import to_plain_text

MODEL = "baidu/Unlimited-OCR"
PROMPT = "<image>document parsing."


def data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


async def ocr_one(client: AsyncOpenAI, sem: asyncio.Semaphore, item: tuple) -> dict:
    image_id, page_id, path, text = item
    async with sem:
        t = time.perf_counter()
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": data_url(path)}},
            ]}],
            max_tokens=8192,
            temperature=0.0,
            extra_body={"skip_special_tokens": False,
                        "vllm_xargs": {"ngram_size": 35, "window_size": 128}},
        )
        secs = time.perf_counter() - t
    raw = resp.choices[0].message.content or ""
    plain = to_plain_text(raw)
    return {"image_id": image_id, "page_id": page_id, "source": "unlimited-ocr-vllm",
            "raw": raw, "plain": plain, "cer": cer(text, plain), "wer": wer(text, plain),
            "seconds": secs, "completion_tokens": resp.usage.completion_tokens if resp.usage else None}


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--batch", type=int, default=128, help="rows per Parquet append")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pending images (smoke test)")
    ap.add_argument("--gpu-price", type=float, default=0.0, help="USD per GPU-hour, for cost per 1,000 pages")
    args = ap.parse_args()

    out = OCR_DIR / "unlimited-vllm.parquet"
    todo = pending_images("unlimited-ocr-vllm", out)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} images to do on {gpu_name()}, concurrency {args.concurrency}")

    client = AsyncOpenAI(api_key="EMPTY", base_url=args.base_url, timeout=3600)
    sem = asyncio.Semaphore(args.concurrency)
    start, done, tokens = time.perf_counter(), 0, 0
    for i in range(0, len(todo), args.batch):
        rows = await asyncio.gather(*(ocr_one(client, sem, item) for item in todo[i:i + args.batch]))
        append_rows(out, rows)
        done += len(rows)
        tokens += sum(r["completion_tokens"] or 0 for r in rows)
        wall = time.perf_counter() - start
        print(f"  {done}/{len(todo)}  {done / wall:.2f} pages/s  {tokens / wall:.0f} tokens/s")

    wall = time.perf_counter() - start
    pps = done / wall if wall else 0.0
    report = {
        "gpu": gpu_name(), "pages": done, "wall_seconds": round(wall, 1),
        "pages_per_second": round(pps, 3), "output_tokens_per_second": round(tokens / wall, 1) if wall else 0,
        "concurrency": args.concurrency, "gpu_price_per_hour": args.gpu_price,
        "usd_per_1000_pages": round(args.gpu_price / 3600 / pps * 1000, 4) if pps and args.gpu_price else None,
        "note": "Excludes server start-up and model load; measured over this run only.",
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/p2_throughput.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
