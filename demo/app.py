"""PageProof demo: answers about scanned books that you can check against the page.

    uv run python demo/export.py                         # build demo/bundle once
    uv run --extra demo streamlit run demo/app.py

Two modes:
- Browse saved results: the 150 test questions, answered at every OCR damage level in the eval runs
  (demo/static/index.html is the same view as a static page, for the free Hugging Face Space).
  Reads only the bundle (demo/export.py), so it runs anywhere with Streamlit and Pillow.
- Ask your own question: retrieves from one index, answers with DeepSeek and highlights the quote
  the model relied on. Shown only where the pageproof package and data are installed, and not when
  PAGEPROOF_LIVE=0 (the hosted Space), so visitors can't spend the API credit.
"""

import importlib.util
import json
import os
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ.get("PAGEPROOF_BUNDLE", HERE / "bundle"))
LIVE = importlib.util.find_spec("pageproof") is not None and os.environ.get("PAGEPROOF_LIVE", "1") != "0"
CONDITIONS = [("clean", 0, "Clean text")] + [("tesseract", lv, f"Tesseract, damage level {lv}") for lv in range(6)]
MODE_LABELS = {"full": "Search, then answer", "oracle": "Given the right page"}
VERDICT_ICON = {"correct": "✅ correct", "partial": "🟡 partial", "incorrect": "❌ incorrect"}
HOW_CAPTION = {"found": "Highlighted: {what}",
               "from_clean": "Highlighted: {what} (too damaged to find directly; position taken from the clean page)",
               "none": "Passage not located; showing the whole page"}

st.set_page_config(page_title="PageProof", page_icon="📖", layout="wide")


@st.cache_data
def data() -> dict:
    return json.loads((BUNDLE / "data.json").read_text())


def page_label(page_id: str) -> str:
    return data()["labels"].get(page_id, page_id)


@st.cache_data
def boxed(path: str, box: tuple[int, int, int, int] | None) -> Image.Image:
    """The page with the box highlighted, cropped around it; the whole page when there's no box."""
    image = Image.open(path).convert("RGB")
    if not box:
        return image
    x0, y0, x1, y1 = box
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle((x0 - 4, y0 - 3, x1 + 4, y1 + 3), fill=(255, 210, 0, 90),
                                      outline=(220, 120, 0, 255), width=3)
    shown = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return shown.crop((0, max(0, y0 - 240), image.width, min(image.height, y1 + 240)))


def browse() -> None:
    d = data()
    qs, answers = d["questions"], d["answers"]
    with st.sidebar:
        book = st.selectbox("Book", sorted({q["title"] for q in qs}))
        q = st.selectbox("Question", [q for q in qs if q["title"] == book], format_func=lambda q: q["question"])
        source, level, label = CONDITIONS[st.select_slider(
            "OCR text the system reads", range(len(CONDITIONS)), value=len(CONDITIONS) - 1,
            format_func=lambda i: CONDITIONS[i][2])]
    page = d["pages"][q["source_page"]]
    st.subheader(q["question"])
    st.caption(f"Reference answer: **{q['answer']}** · from {page_label(q['source_page'])}")

    for col, mode in zip(st.columns(2), MODE_LABELS):
        r = answers.get(f"{q['qid']}|{mode}|{source}|{level}")
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

    shown = page[str(level)]
    left, right = st.columns([3, 2])
    with left:
        st.image(boxed(str(BUNDLE / shown["image"]), tuple(shown["box"]) if shown["box"] else None),
                 use_container_width=True,
                 caption=HOW_CAPTION[shown["how"]].format(what="the passage holding the reference answer"))
    with right:
        if source == "tesseract":
            st.markdown(f"**What Tesseract read** · {shown['cer']:.1%} character error rate")
            st.text_area("OCR text", shown["ocr_text"], height=420, label_visibility="collapsed")
        else:
            st.markdown("**Clean text**: the system reads the book's own text, with no OCR errors. "
                        "The image is the undamaged page.")
    st.markdown("**Same question at every level**")
    rows = []
    for s, lv, lab in CONDITIONS:
        row = {"OCR text": lab, "Page CER": "0%" if s == "clean" else f"{page[str(lv)]['cer']:.1%}"}
        for mode, mlabel in MODE_LABELS.items():
            r = answers.get(f"{q['qid']}|{mode}|{s}|{lv}")
            row[mlabel] = VERDICT_ICON[r["verdict"]] if r else "—"
        rows.append(row)
    st.dataframe(rows, hide_index=True, use_container_width=True)


@st.cache_data(show_spinner="Finding the quote on the page...")
def live_highlight(page_id: str, level: int, quote: str) -> tuple[Image.Image, str]:
    import duckdb

    from pageproof.config import DATA
    from pageproof.qa.locate import locate

    paths = dict(duckdb.sql(f"select level, path from '{DATA}/corpus/images.parquet' where page_id = ? "
                            "and \"set\" = 'level'", params=[page_id]).fetchall())
    root = HERE.parent
    image = Image.open(root / paths[level]).convert("L")
    hit, how = locate(image, quote), "found"
    if not hit and level:
        clean = Image.open(root / paths[0]).convert("L")
        if hit := locate(clean, quote):
            sx, sy = image.width / clean.width, image.height / clean.height
            x0, y0, x1, y1 = hit.box
            return boxed(str(root / paths[level]), (int(x0 * sx), int(y0 * sy) - 8, int(x1 * sx), int(y1 * sy) + 8)), \
                "from_clean"
    return boxed(str(root / paths[level]), hit.box if hit else None), how if hit else "none"


def live() -> None:
    from pageproof.qa.answer import ask

    with st.sidebar:
        source, level, label = CONDITIONS[st.select_slider(
            "OCR text to search", range(len(CONDITIONS)), value=0, format_func=lambda i: CONDITIONS[i][2])]
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
        image, how = live_highlight(a.quote_page, level, a.quote)
        st.image(image, use_container_width=True, caption=HOW_CAPTION[how].format(what=f'"{a.quote}"'))


st.title("PageProof")
st.write("Answers about scanned books that you can check against the page, and how OCR damage changes them.")
if not (BUNDLE / "data.json").exists():
    st.error("No demo bundle found. Build it with `uv run python demo/export.py`.")
elif LIVE and st.sidebar.radio("Mode", ["Browse saved results", "Ask your own question"]) != "Browse saved results":
    live()
else:
    browse()
