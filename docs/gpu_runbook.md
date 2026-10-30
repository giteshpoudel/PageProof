# Rented GPU runbook

One GPU session covers all of P2's GPU work: Unlimited-OCR on the whole corpus with vLLM (2,640 images),
throughput and cost, and the rest of the Mac vs CUDA parity check (D-07). Expect 1–3 hours on an RTX 4090,
about $0.50–2 at Community Cloud prices. Stop the pod as soon as you finish.

## 1. Rent

- RunPod → Deploy a pod → **RTX 4090 (24 GB)**, Community Cloud (~$0.34/hr).
- **Container image:** `vllm/vllm-openai:unlimited-ocr` (the official image from the model's vLLM recipe;
  the architecture is not in a stable vLLM wheel yet). Container disk 40 GB.
- **Volume:** 30 GB mounted at `/workspace`, so weights and outputs survive a restart.
- **Start command:** leave it as a plain shell (e.g. `sleep infinity`); we start the server by hand.
- Set a spending limit on the account first. Don't expose port 8000: the server has no API key and
  only needs to be reachable from inside the pod.
- **SSH:** add the Mac mini's *public* key (`~/.ssh/id_ed25519.pub`; create one with `ssh-keygen -t ed25519` if it doesn't exist) under RunPod → Settings → SSH Public Keys
  before deploying, and enable "SSH over exposed TCP". The pod's Connect tab then shows the IP and port.

## 2. Copy the repo and corpus up (from the Mac)

The corpus is built on the Mac and copied up, so both machines use byte-identical images.

```bash
# from the Mac, in the PageProof folder (use the pod's SSH host and port from RunPod's "Connect" tab)
rsync -av -e "ssh -p <port>" --exclude .env --exclude .git --exclude .venv --exclude models \
    --exclude 'data/corpus/pdf' --exclude data/index --exclude demo/bundle --exclude build \
    ./ root@<pod-ip>:/workspace/PageProof/
```

That sends the code, `data/corpus` (images + Parquet manifests, ~420 MB) and the P0 pages. `.env` (the
DeepSeek key) stays on the Mac: nothing on the pod calls DeepSeek.

## 3. Start the vLLM server (on the pod)

```bash
export HF_HOME=/workspace/hf   # keep the weights on the volume
vllm serve baidu/Unlimited-OCR --revision 8249d20b2822a7d8a4c22c8d8c9069b5ac36bda3 \
    --host 127.0.0.1 --port 8000 --trust-remote-code \
    --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \
    --no-enable-prefix-caching --mm-processor-cache-gb 0 \
    > /workspace/vllm.log 2>&1 &
tail -f /workspace/vllm.log   # wait for "Application startup complete", then Ctrl-C
```

The revision is pinned to the one the Mac runs used.

## 4. Run the corpus (on the pod)

```bash
cd /workspace/PageProof
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv sync --extra gpu                       # client only: openai + our metrics

# smoke test: 4 pages; check the CER column is small and the text looks right
uv run --extra gpu python scripts/gpu_ocr_vllm.py --limit 4
uv run python -c "import duckdb; print(duckdb.sql(\"select image_id, round(cer,4) cer, left(plain, 80) from 'data/ocr/unlimited-vllm.parquet'\"))"

# full run: resumes, so it's safe to stop and restart
uv run --extra gpu python scripts/gpu_ocr_vllm.py --gpu-price 0.34
```

If the smoke test shows `<|ref|>`/`<|det|>` debris in `plain`, stop and fix `to_plain_text` before the full run.
Delete `data/ocr/unlimited-vllm.parquet` after the smoke test so the throughput figure covers the full run only.

## 5. Parity: the remaining P0 pages with Transformers (on the pod)

vLLM and Transformers are different inference engines, so parity uses Transformers on both sides,
the same code the Mac ran (unpatched here, since this is CUDA).

```bash
uv sync --extra vlm
uv run hf download baidu/Unlimited-OCR --revision 8249d20b2822a7d8a4c22c8d8c9069b5ac36bda3 \
    --local-dir models/unlimited-ocr
uv run python scripts/p0_feasibility.py --label cuda --skip-tesseract
```

## 6. Bring results back (from the Mac) and stop the pod

Copy back only the new outputs, so nothing newer on the Mac is overwritten:

```bash
P='ssh -p <port>'; POD=root@<pod-ip>:/workspace/PageProof
rsync -av -e "$P" $POD/data/ocr/unlimited-vllm.parquet data/ocr/
rsync -av -e "$P" $POD/data/p0/ocr/unlimited-cuda data/p0/ocr/
rsync -av -e "$P" $POD/results/p2_throughput.json results/
uv run python scripts/p0_parity.py --a unlimited-mac --b unlimited-cuda
```

Check the files arrived, then **terminate the pod and delete its volume** in RunPod. A stopped pod's
volume still bills ($0.20/GB-month when stopped, so about $6/month for 30 GB; prices checked 2026-10-30), and nothing on it is needed once the outputs are on the Mac.

## 7. Back on the Mac

```bash
uv run python scripts/p2_summary.py                                   # CER per source x level
uv run pageproof index build --source unlimited-ocr-vllm --levels 0-5 # 6 search indexes
uv run python evals/run.py answers --workers 12                        # adds Unlimited-OCR cells (off-peak)
uv run python evals/run.py summary && uv run python evals/curves.py
uv run python demo/export.py --space build/space
```
