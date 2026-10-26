# Rented GPU runbook

How to run the CUDA reference OCR on a rented GPU, cheaply. P0 needs about 30 minutes of GPU time.

## 1. Rent

- RunPod, **RTX 4090 (24 GB)**, Community Cloud (~$0.34/hr). Template: a PyTorch image with CUDA 12.x.
- Add a **20 GB network volume** mounted at `/workspace`, so the model weights (~6.7 GB) survive between sessions.
- Set a spending limit in the account before starting.

## 2. Set up (on the pod)

```bash
cd /workspace
git clone https://github.com/<you>/PageProof.git && cd PageProof   # or rsync the repo up
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
apt-get update && apt-get install -y tesseract-ocr
uv sync --extra vlm
uv run hf download baidu/Unlimited-OCR --revision 8249d20b2822a7d8a4c22c8d8c9069b5ac36bda3 \
    --local-dir models/unlimited-ocr
```

Use the same model revision as the Mac run, so the parity check compares like with like.
On CUDA, `UnlimitedOCR` does not patch the model code: it runs Baidu's original code.

## 3. P0 reference run

Copy the Mac's page images up first, so both machines OCR byte-identical PNGs:

```bash
# from the Mac
rsync -av data/p0/pg1342 data/p0/html <pod>:/workspace/PageProof/data/p0/
# on the pod
uv run python scripts/p0_feasibility.py --label cuda
```

The script reuses the existing PNGs, and writes `data/p0/ocr/unlimited-cuda/` and `results/p0_cuda.csv`.

## 4. Bring results back and compare

```bash
# from the Mac
rsync -av <pod>:/workspace/PageProof/data/p0/ocr/unlimited-cuda data/p0/ocr/
rsync -av <pod>:/workspace/PageProof/results/p0_cuda.csv results/
uv run python scripts/p0_parity.py --a unlimited-mac --b unlimited-cuda
```

## 5. Stop the pod

Stop it as soon as the run finishes. Idle time is the main cost risk; the network volume keeps the weights.
