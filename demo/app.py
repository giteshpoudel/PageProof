"""PageProof demo: answers about scanned books that you can check against the page.

    uv run --extra demo streamlit run demo/app.py

Two modes:
- Browse saved results: the 150 test questions, answered at every OCR damage level in the eval runs.
  Needs no API key.
- Ask your own question: retrieves from one index, answers with DeepSeek and highlights the quote
  the model relied on. Needs DEEPSEEK_API_KEY with credit.
"""

import json
from collections import defaultdict
from pathlib import Path

import duckdb
import streamlit as st
from PIL import Image, ImageDraw

from pageproof.config import DATA
from pageproof.qa.locate import locate

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = [("clean", 0, "Clean text")] + [("tesseract", lv, f"Tesseract, damage level {lv}") for lv in range(6)]
MODE_LABELS = {"full": "Search, then answer", "oracle": "Given the right page"}
VERDICT_ICON = {"correct": "✅ correct", "partial": "🟡 partial", "incorrect": "❌ incorrect"}

st.set_page_config(page_title="PageProof", page_icon="📖", layout="wide")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


@st.cache_data
def questions() -> list[dict]:
    return [q for q in read_jsonl(ROOT / "evals" / "questions.jsonl") if q["split"] == "test"]


@st.cache_data
def saved_answers() -> dict[tuple, dict]:
    """(qid, mode, source, level) -> answer from the lowest-numbered run that has it."""
    out = {}
    for r in sorted(read_jsonl(ROOT / "evals" / "runs" / "answers.jsonl"), key=lambda r: -r["run"]):
        out[(r["qid"], r["mode"], r["source"], r["level"])] = r
    return out


@st.cache_data
def pages() -> dict[str, tuple[str, int]]:
    return {p: (t, n) for p, t, n in duckdb.sql(
        f"select page_id, title, page_no from '{DATA}/corpus/pages.parquet'").fetchall()}


@st.cache_data
def image_path(page_id: str, level: int) -> str:
    return str(ROOT / duckdb.sql(f"select path from '{DATA}/corpus/images.parquet' where page_id = ? "
                                 "and \"set\" = 'level' and level = ?", params=[page_id, level]).fetchall()[0][0])


@st.cache_data
def ocr_text(page_id: str, level: int) -> tuple[str, float]:
    row = duckdb.sql(f"""select r.plain, r.cer from '{DATA}/ocr/tesseract.parquet' r
        join '{DATA}/corpus/images.parquet' i using (image_id)
        where i.page_id = ? and i."set" = 'level' and i.level = ?""", params=[page_id, level]).fetchall()
    return (row[0][0], row[0][1]) if row else ("", float("nan"))


@st.cache_data(show_spinner="Finding the passage on the page...")
def highlighted(path: str, quote: str, clean_path: str = "") -> tuple[Image.Image, str]:
    """The page with the quote boxed and cropped around it. Returns (image, how): "found" on this image,
    "from_clean" when the box was found on the clean image of the same page and scaled across (heavy
    damage defeats the matcher; damaged images are the clean one resized and slightly tilted), or "none"."""
    image = Image.open(path).convert("RGB")
    found = locate(image.convert("L"), quote) if quote else None
    how = "found"
    if not found and quote and clean_path and clean_path != path:
        clean = Image.open(clean_path).convert("L")
        hit = locate(clean, quote)
        if hit:
            sx, sy = image.width / clean.width, image.height / clean.height
            x0, y0, x1, y1 = hit.box
            found = type(hit)(box=(int(x0 * sx), int(y0 * sy) - 8, int(x1 * sx), int(y1 * sy) + 8),
                              score=hit.score, matched=hit.matched)
            how = "from_clean"
    if not found:
        return image, "none"
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    x0, y0, x1, y1 = found.box
    ImageDraw.Draw(overlay).rectangle((x0 - 4, y0 - 3, x1 + 4, y1 + 3), fill=(255, 210, 0, 90),
                                      outline=(220, 120, 0, 255), width=3)
    shown = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return shown.crop((0, max(0, y0 - 240), image.width, min(image.height, y1 + 240))), how


HOW_CAPTION = {"found": "Highlighted: {what}",
               "from_clean": "Highlighted: {what} (too damaged to find directly; position taken from the clean page)",
               "none": "Passage not located; showing the whole page"}


def page_label(page_id: str) -> str:
    title, n = pages().get(page_id, ("?", 0))
    return f"{title}, page {n}"


def browse() -> None:
    qs = questions()
    answers = saved_answers()
    if not answers:
        st.warning("No saved eval answers found (evals/runs/answers.jsonl).")
        return
    books = sorted({q["title"] for q in qs})
    with st.sidebar:
        book = st.selectbox("Book", books)
        in_book = [q for q in qs if q["title"] == book]
        q = st.selectbox("Question", in_book, format_func=lambda q: q["question"])
        cond = CONDITIONS[st.select_slider("OCR text the system reads", range(len(CONDITIONS)),
                                           value=len(CONDITIONS) - 1, format_func=lambda i: CONDITIONS[i][2])]
    source, level, label = cond
    st.subheader(q["question"])
    st.caption(f"Reference answer: **{q['answer']}** · from {page_label(q['source_page'])}")

    cols = st.columns(2)
    for col, mode in zip(cols, MODE_LABELS):
        r = answers.get((q["qid"], mode, source, level))
        with col:
            st.markdown(f"**{MODE_LABELS[mode]}** · {label}")
            if not r:
                st.info("Not in the saved runs.")
                continue
            st.markdown(f"> {r['answer']}")
            cited = ", ".join(page_label(p) for p in r["citations"]) or "none"
            st.caption(f"Judge: {VERDICT_ICON[r['verdict']]} · cites {cited}")
            if mode == "full":
                st.caption("Right page in the top 5 search results: " + ("yes" if r["gold_retrieved"] else "**no**"))

    left, right = st.columns([3, 2])
    with left:
        image, how = highlighted(image_path(q["source_page"], level), q["evidence"], image_path(q["source_page"], 0))
        st.image(image, use_container_width=True,
                 caption=HOW_CAPTION[how].format(what="the passage holding the reference answer"))
    with right:
        if source == "tesseract":
            text, cer = ocr_text(q["source_page"], level)
            st.markdown(f"**What Tesseract read** · {cer:.1%} character error rate")
            st.text_area("OCR text", text, height=420, label_visibility="collapsed")
        else:
            st.markdown("**Clean text**: the system reads the book's own text, with no OCR errors.")
    st.markdown("**Same question at every level**")
    rows = []
    for s, lv, lab in CONDITIONS:
        row = {"OCR text": lab}
        if s == "tesseract":
            row["Page CER"] = f"{ocr_text(q['source_page'], lv)[1]:.1%}"
        else:
            row["Page CER"] = "0%"
        for mode, mlabel in MODE_LABELS.items():
            r = answers.get((q["qid"], mode, s, lv))
            row[mlabel] = VERDICT_ICON[r["verdict"]] if r else "—"
        rows.append(row)
    st.dataframe(rows, hide_index=True, use_container_width=True)


def live() -> None:
    from pageproof.qa.answer import ask

    with st.sidebar:
        cond = CONDITIONS[st.select_slider("OCR text to search", range(len(CONDITIONS)), value=0,
                                           format_func=lambda i: CONDITIONS[i][2])]
    source, level, label = cond
    st.caption("Books: Pride and Prejudice, The Adventures of Sherlock Holmes, Paradise Lost, War and Peace, "
               "The Wealth of Nations, On the Origin of Species (65 pages of each).")
    question = st.text_input("Your question", placeholder="How much money did Wickham receive in lieu of the living?")
    if not question:
        return
    try:
        with st.spinner("Searching and answering..."):
            a = ask(question, source, level)
    except Exception as e:  # noqa: BLE001 - shown to the user instead of a stack trace
        st.error(f"Live answers are unavailable right now ({type(e).__name__}). Browse saved results instead.")
        return
    st.markdown(f"> {a.answer}")
    st.caption(f"{label} · cites {', '.join(page_label(p) for p in a.citations) or 'none'}")
    if a.found and a.quote_page:
        image, how = highlighted(image_path(a.quote_page, level), a.quote, image_path(a.quote_page, 0))
        st.image(image, use_container_width=True, caption=HOW_CAPTION[how].format(what=f'"{a.quote}"'))


st.title("PageProof")
st.write("Answers about scanned books that you can check against the page, and how OCR damage changes them.")
mode = st.sidebar.radio("Mode", ["Browse saved results", "Ask your own question"])
if mode == "Browse saved results":
    browse()
else:
    live()
