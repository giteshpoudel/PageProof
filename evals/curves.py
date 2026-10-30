"""Answer accuracy against the gold page's measured CER, the headline curve.

    uv run python evals/curves.py            # all 3 runs; -> results/p4_accuracy_vs_cer.csv and .png

One panel per condition (search, then answer / given the right page), one line per OCR source.
The x-axis is each answer's gold-page CER (that source's OCR at that answer's level), binned, not
the level number, so levels whose pages overlap in CER fall in the same bins. Clean text is CER 0.
CIs are 95% bootstrap intervals that resample questions within a bin.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import typer  # noqa: E402

from pageproof import config  # noqa: E402
from pageproof.index.store import SOURCE_FILES, SOURCE_LABELS, ocr_sources  # noqa: E402

app = typer.Typer()
EVALS = Path(__file__).resolve().parent
ANSWERS = EVALS / "runs" / "answers.jsonl"
QUESTIONS = EVALS / "questions.jsonl"
EDGES = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 1.01]  # CER bin edges (fractions)
SEED = 20261026
# Reference palette (dataviz skill), light mode: slots 1-2, plus text and grid tokens.
BLUE, ORANGE, INK, INK2, MUTED, GRID, SURFACE = ("#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#8a8984",
                                                 "#e8e7e3", "#fcfcfb")
LABELS = {"full": "Search, then answer", "oracle": "Given the right page"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


SOURCE_COLORS = {"tesseract": BLUE, "unlimited-ocr-vllm": ORANGE}


def gold_cer() -> dict[tuple[str, str, int], float]:
    """(source, page_id, level) -> that source's CER on the page's level image."""
    out = {}
    for s in ocr_sources():
        rows = duckdb.sql(f"""
            select i.page_id, i.level, r.cer from '{config.DATA}/ocr/{SOURCE_FILES[s]}' r
            join '{config.DATA}/corpus/images.parquet' i using (image_id) where i.set = 'level'""").fetchall()
        out |= {(s, p, lv): c for p, lv, c in rows}
    return out


def bin_of(cer: float) -> int:
    return next(i for i in range(len(EDGES) - 1) if cer < EDGES[i + 1])


def ci(groups: dict[str, list[bool]], n: int = 2000) -> tuple[float, float, float]:
    qs = list(groups)
    flat = lambda sample: [v for q in sample for v in groups[q]]  # noqa: E731
    point = sum(flat(qs)) / len(flat(qs))
    rng = random.Random(SEED)
    means = sorted(sum(f) / len(f) for f in (flat(rng.choices(qs, k=len(qs))) for _ in range(n)))
    return point, means[int(0.025 * n)], means[int(0.975 * n) - 1]


@app.command()
def main(runs: str = "1,2,3") -> None:
    gold = {q["qid"]: q["gold_pages"] for q in read_jsonl(QUESTIONS)}
    cers = gold_cer()
    rows = [r for r in read_jsonl(ANSWERS) if str(r["run"]) in runs.split(",")]
    closed = [r["verdict"] == "correct" for r in rows if r["mode"] == "closed"]
    sources = [s for s in ocr_sources() if any(r["source"] == s for r in rows)]
    cells = defaultdict(lambda: defaultdict(list))  # (mode, source, bin) -> qid -> verdicts
    cer_of = defaultdict(list)
    for r in rows:
        if r["source"] not in sources:
            continue
        cer = sum(cers[(r["source"], p, r["level"])] for p in gold[r["qid"]]) / len(gold[r["qid"]])
        k = (r["mode"], r["source"], bin_of(cer))
        cells[k][r["qid"]].append(r["verdict"] == "correct")
        cer_of[k].append(cer)
    clean = {m: ci(_by_q([r for r in rows if r["mode"] == m and r["source"] == "clean"])) for m in LABELS}

    lines = ["mode,source,cer_bin,cer_median,answers,questions,accuracy,ci_low,ci_high"]
    series = defaultdict(list)  # (mode, source) -> points
    for m in LABELS:
        acc, lo, hi = clean[m]
        n_clean = sum(r["mode"] == m and r["source"] == "clean" for r in rows)
        lines.append(f"{m},clean,clean,0,{n_clean},150,{acc:.3f},{lo:.3f},{hi:.3f}")
        for s in sources:
            for b in range(len(EDGES) - 1):
                g = cells.get((m, s, b))
                if not g:
                    continue
                n = sum(len(v) for v in g.values())
                vals = sorted(cer_of[(m, s, b)])
                med = vals[len(vals) // 2]
                acc, lo, hi = ci(g)
                lines.append(f"{m},{s},{EDGES[b]:.3f}-{EDGES[b + 1]:.3f},{med:.4f},{n},{len(g)},"
                             f"{acc:.3f},{lo:.3f},{hi:.3f}")
                if n >= 20:  # thinner bins are in the CSV but too noisy to plot
                    x = med if med > 0 else EDGES[b + 1] / 2  # a CER-0 median can't sit on a log axis
                    series[(m, s)].append((x, acc, lo, hi, n))
    closed_acc = sum(closed) / len(closed)
    lines.append(f"closed,none,,,{len(closed)},150,{closed_acc:.3f},,")
    Path("results/p4_accuracy_vs_cer.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(series, sources, clean, closed_acc, runs)


def _by_q(rs: list[dict]) -> dict[str, list[bool]]:
    g = defaultdict(list)
    for r in rs:
        g[r["qid"]].append(r["verdict"] == "correct")
    return g


def plot(series, sources, clean, closed_acc, runs) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=200, facecolor=SURFACE, sharey=True)
    x_clean = 0.0005  # clean text (CER 0) drawn at the left edge of the log axis
    for ax, m in zip(axes, LABELS):
        ax.set_facecolor(SURFACE)
        for s in sources:
            color = SOURCE_COLORS.get(s, MUTED)
            pts = [(x_clean, *clean[m])] + [p[:4] for p in series[(m, s)]]
            xs, ys, lo, hi = zip(*pts)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.14, linewidth=0)
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=SOURCE_LABELS[s])
            ax.annotate(f"{ys[-1]:.0%}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=9)
        ax.axhline(closed_acc, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(x_clean, closed_acc + 0.02, f"No pages (from memory): {closed_acc:.0%}", color=INK2, fontsize=8.5)
        ax.set_xscale("log")
        ax.set_xticks([x_clean, 0.001, 0.01, 0.1, 0.5], ["clean", "0.1%", "1%", "10%", "50%"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(0.00035, 0.9)
        ax.set_ylim(0, 1.04)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(LABELS[m], loc="left", color=INK, fontsize=10.5)
        ax.set_xlabel("Character error rate of the page holding the answer (binned)")
    axes[0].set_ylabel("Answers judged correct")
    axes[0].legend(loc="lower left", frameon=False, labelcolor=INK2)
    fig.suptitle("Answer accuracy vs OCR error, 150 test questions × "
                 + ("3 runs" if runs == "1,2,3" else f"run {runs}"), x=0.01, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig("results/p4_accuracy_vs_cer.png", facecolor=SURFACE)


if __name__ == "__main__":
    app()
