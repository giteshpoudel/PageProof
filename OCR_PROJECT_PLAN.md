# PageProof: project plan

Ask questions about scanned books and get answers you can check against the page. The research question underneath: **how much does OCR quality decide RAG quality, and where does it break: retrieval or reading?**

Status: plan v2 (2026-10-26). Repo created; nothing built yet.

## Why this project

The portfolio already has text RAG (EDGAR) and an MCP agent (Experimentation Copilot). This one adds what they don't show:

| Data science foundation | AI engineering |
| --- | --- |
| Error propagation: OCR error rate → answer accuracy, as a dose-response curve | Vision-language OCR (Unlimited-OCR) on scans and PDFs |
| Ground truth by construction: clean text rendered as degraded scans | Layout-aware chunking that keeps page and region ids |
| Paired comparisons with bootstrap CIs, repeated runs | Answers that cite a page and show the page image |
| Controlling confounds: LLM memorization, judge bias, retrieval vs reading | GPU batch inference (vLLM), throughput and cost per 1,000 pages |
| Validating a patched model against a reference implementation | |

The headline result is a curve, not one number: answer accuracy against measured character error rate (CER), one line per OCR source.

## Experimental design

These choices make the curve trustworthy. Each one closes a hole a reviewer would find.

1. **Severity, not a mix of types.** Degradation levels 0–5 turn one severity knob that scales every effect together (noise, blur, skew, JPEG quality, DPI). A separate small grid varies one effect at a time, to show which effect hurts most. The x-axis is each page's *measured* CER, binned, not the level number.
2. **Memorization control.** Gutenberg books are in most LLMs' training data, so a model can answer from memory and hide OCR damage. Two guards:
   - a **closed-book baseline**: the same questions with no retrieved context; report it as a floor line on the chart;
   - questions favour details a model is unlikely to recall (numbers, minor names, footnotes, table cells), and closed-book-answerable questions are flagged and reported separately.
3. **Retrieval vs reading.** Three conditions per OCR source × level:
   - *full pipeline*: retrieve, then answer;
   - *oracle page*: give the model the gold page's OCR text, no retrieval (isolates reading);
   - *retrieval only*: recall@k against the gold page (isolates search).
4. **Judge validation.** Hand-label ~60 answers spread across levels; report judge-vs-human agreement (Cohen's κ) before trusting the judge at scale.
5. **Splits.** ~200 verified questions: 50 dev (prompt and chunking tuning), 150 test (touched only for final runs).
6. **Statistics.** 3 runs per cell; paired bootstrap CIs over questions; compare sources at the same page set.

## Hardware: Mac mini for development, a rented GPU for reference runs

Unlimited-OCR ([baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR), MIT, released 2026-06-18) is built on DeepSeek-OCR: about 3B parameters, ~500M active per token (6.7 GB in BF16). It fits in 16 GB, but Baidu only supports NVIDIA GPUs, and Apple Silicon has open issues (checked 2026-10-26):

- [#81](https://github.com/baidu/Unlimited-OCR/issues/81): hard-coded `.cuda()`; CPU works after removing it, MPS crashes mid-generation.
- [#18](https://github.com/baidu/Unlimited-OCR/issues/18): on MPS, `masked_scatter` places image embeddings wrongly, so output is empty. PRs [#56](https://github.com/baidu/Unlimited-OCR/pull/56) and [#57](https://github.com/baidu/Unlimited-OCR/pull/57) patch this; #57 reports working OCR on an M4 Pro.
- [#53](https://github.com/baidu/Unlimited-OCR/issues/53): on MPS, multi-page output collapses after 2–3 pages.

So:

- **Mac mini (M4, 16 GB):** development, single-page OCR with the community patches, Tesseract, indexing, retrieval, the QA agent and evals. Use single-page inference only.
- **Rented NVIDIA GPU:** the reference OCR runs the evals depend on, plus throughput and cost, using the official vLLM path.
- **Parity check (a small result in itself):** OCR the same 50 pages on Mac and on CUDA and report the CER between them. If they match, the Mac is a valid local setup; if not, the write-up says where it diverges.
- **Stretch:** the model's headline feature is multi-page parsing in one pass. On CUDA, compare single-page vs multi-page CER on the same books.

## Data

Public domain (US) only.

- **Synthetic, with exact ground truth:** 5–8 Project Gutenberg books with varied layouts (prose, verse, tables, footnotes). Render from Gutenberg's **HTML** editions (not plain text) so layout survives: HTML → PDF → page images. Strip illustration captions and drop-cap images in P1 (P0 found both leaking into pages). Every page keeps its source text and region boxes, so CER and answer correctness are exact.
- **Real scans, for realism:** the same editions' scans from the Internet Archive, where available. No exact ground truth: use them for qualitative examples, the demo, and the PDF text-layer source.

## OCR sources

| Source | Where it applies | Notes |
| --- | --- | --- |
| Clean text | Synthetic | Upper bound (CER 0) |
| Tesseract | Synthetic + real | Classical baseline |
| Unlimited-OCR | Synthetic + real | The VLM under test |
| PDF text layer | Real scans only | Internet Archive PDFs carry their own OCR layer; synthetic pages have none |

## Phases

| Phase | Work | Done when |
| --- | --- | --- |
| P0 Feasibility | Run Unlimited-OCR on 10 pages: Mac mini (patched, single page) and CUDA (rented GPU or the HF Space). Time a short GPU batch. | Both produce sane Markdown; Mac-vs-CUDA CER measured; rough pages/sec; go / no-go on the Mac path. |
| P1 Corpus | HTML → page renderer, degradation pipeline (severity levels + one-effect grid), page ↔ source-text alignment stored as Parquet. | ~1,500–3,000 synthetic pages in total, each with ground truth; CER of clean text = 0 as a sanity test. |
| P2 OCR runs | Tesseract and Unlimited-OCR on every synthetic page; all sources on real scans. | CER/WER per page × source × level; pages/sec and $ per 1,000 pages on the GPU. |
| P3 Index + QA | Layout-aware chunks keeping page ids; hybrid search (BM25 + vector); answers cite pages and show a page crop. | `pageproof ask` answers with page citations from any OCR source. |
| P4 Evals | ~200 questions from the clean text (LLM-drafted, hand-verified), each with a gold page; judge validated on ~60 hand labels. Run full pipeline, oracle page and closed-book per source × level, 3 runs. | Curves of answer accuracy vs CER per source, with CIs and the closed-book floor; retrieval vs reading split; test split untouched until final runs. |
| P5 Demo | Pre-processed books only (no uploads, to cap cost): question → answer → highlighted page image. | Hosted demo on CPU (Streamlit or the portfolio site). |
| P6 Write-up | README results, "what the evals caught", limitations; portfolio kit filled with measured numbers. | Ready for the portfolio site. |

**First milestone (MVP):** one book, levels 0/3/5, Tesseract only, 30 questions, one curve end to end. It proves the pipeline before scaling to every book and source.

## Stack

Python 3.12 with uv, Unlimited-OCR (Transformers on Mac; vLLM on CUDA), Tesseract, PyMuPDF (also HTML → PDF via its Story engine), Pillow + Augraphy (degradation), DuckDB + Parquet, LanceDB, BM25, sentence-transformers, an OpenAI-compatible LLM (DeepSeek), Pydantic, Typer (CLI), pytest, GitHub Actions.

## Repo layout

```
pageproof/
  corpus/     # fetch books, render pages, degrade, align
  ocr/        # one adapter per source, CER/WER
  index/      # chunking, BM25 + vectors
  qa/         # retrieval, answering, citations
  evals/      # questions, judge, runs, bootstrap
  cli.py
data/         # gitignored; Parquet + images
results/      # committed: metrics tables and charts
tests/
```

## Rented GPU budget

On-demand list prices from 2026 comparisons; check the provider's page before renting.

| GPU | Typical price | Notes |
| --- | --- | --- |
| RTX 4090 (24 GB) | $0.34/hr (RunPod Community) to $0.69/hr (Secure) | Enough for a 3B model; the cheapest sensible choice. |
| L4 (24 GB) | ~$0.44–0.80/hr | Slower than a 4090; widely available. |
| A100 80 GB | ~$1.39/hr (RunPod) to $1.99–2.79/hr (Lambda) | Only if batch throughput matters. |
| H100 | ~$2.89–4.29/hr on-demand | Unnecessary for this model. |

Workload: about 3,000 synthetic pages + 500 real-scan pages + reruns ≈ 5,000–10,000 GPU-OCR pages. Throughput isn't published for this model, so plan on 0.5–2 pages/sec on a 4090 with vLLM batching and measure it in P0:

| Scenario | GPU-hours incl. setup | Cost on a 4090 ($0.34–0.69/hr) |
| --- | --- | --- |
| Fast (2 pages/s, 5k pages) | ~3 h | ~$1–2 |
| Expected | ~6–10 h | ~$2–7 |
| Slow (0.5 pages/s, 10k pages, debugging) | ~15–20 h | ~$5–14 |

Plan on **$10–25 total**, with a hard cap around $50. The main risk is paying for an idle pod: stop pods between runs and keep the model weights (~7 GB) on a small persistent volume.

LLM cost (DeepSeek): ~200 questions × 2 OCR sources × 6 levels × 2 conditions × 3 runs ≈ 14k answer calls, plus the clean-text and closed-book baselines and judge calls. Estimate in P3 from measured tokens per call, before the full run.

## Risks

- **Mac (MPS) bugs:** single-page inference and the patched model; CUDA stays the reference.
- **LLM memorization hides OCR damage:** closed-book baseline and detail-level questions (see Experimental design).
- **Too few test questions for tight CIs:** check CI width after the MVP; add questions before adding books.
- **Overlap with EDGAR:** keep the focus on OCR quality → answer quality, not on yet another RAG pipeline.
- **Licensing:** public-domain texts and scans only; strip Gutenberg's license header from rendered pages.

## Out of scope

User uploads in the public demo, handwriting, non-Latin scripts (maybe later), and fine-tuning the OCR model.
