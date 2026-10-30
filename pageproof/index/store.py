"""Chunk, embed and store OCR text in LanceDB: one table per OCR source x level.

Tables: clean_L0 (the exact page text: the upper bound), tesseract_L0..L5,
unlimited-ocr-vllm_L0..L5 once the GPU run exists. Each table has a vector
index (bge-small embeddings) and a BM25 full-text index on the same text.
"""

import json
import threading
from functools import lru_cache

import duckdb
import lancedb

from pageproof.config import DATA
from pageproof.index.chunking import Chunk, chunk_page

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-d, same as EDGAR
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# torch models are not safe to call from several threads at once (EDGAR saw Metal aborts).
MODEL_LOCK = threading.Lock()

SOURCE_FILES = {"tesseract": "tesseract.parquet", "unlimited-ocr-vllm": "unlimited-vllm.parquet"}
SOURCE_LABELS = {"clean": "Clean text", "tesseract": "Tesseract", "unlimited-ocr-vllm": "Unlimited-OCR"}
LEVELS = range(6)


def ocr_sources() -> list[str]:
    """OCR sources whose output exists on disk, in a fixed order (Tesseract first)."""
    return [s for s, f in SOURCE_FILES.items() if (DATA / "ocr" / f).exists()]


def indexed_sources() -> list[str]:
    """OCR sources with a search index at every level, i.e. ready for evals."""
    have = set(connect().table_names())
    return [s for s in ocr_sources() if all(table_name(s, lv) in have for lv in LEVELS)]


def table_name(source: str, level: int) -> str:
    return f"{source.replace('-', '_')}_L{level}"


def connect() -> lancedb.DBConnection:
    return lancedb.connect(str(DATA / "index" / "lancedb"))


@lru_cache(maxsize=1)
def _model():
    import os

    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def embed_documents(texts: list[str]):
    with MODEL_LOCK:
        return _model().encode(texts, batch_size=32, normalize_embeddings=True,
                               show_progress_bar=False)


def embed_query(text: str):
    with MODEL_LOCK:
        return _model().encode(QUERY_PREFIX + text, normalize_embeddings=True)


def page_texts(source: str, level: int) -> list[tuple]:
    """(image_id, page_id, book_id, title, page_no, text) for one source at one level."""
    corpus = DATA / "corpus"
    if source == "clean":
        rows = duckdb.sql(f"""
            select i.image_id, p.page_id, p.book_id, p.title, p.page_no, p.blocks_json
            from '{corpus}/pages.parquet' p join '{corpus}/images.parquet' i using (page_id)
            where i.set = 'level' and i.level = 0 order by p.page_id
        """).fetchall()
        # Paragraph blocks joined by blank lines, the shape OCR output has, so clean
        # and OCR text are chunked the same way.
        return [(*r[:5], "\n\n".join(b["text"] for b in json.loads(r[5]))) for r in rows]
    ocr = DATA / "ocr" / SOURCE_FILES[source]
    return duckdb.sql(f"""
        select i.image_id, p.page_id, p.book_id, p.title, p.page_no, r.plain
        from '{ocr}' r join '{corpus}/images.parquet' i using (image_id)
        join '{corpus}/pages.parquet' p on p.page_id = i.page_id
        where i.set = 'level' and i.level = {level} order by p.page_id
    """).fetchall()


def build_index(source: str, level: int) -> int:
    name = table_name(source, level)
    chunks = []
    for image_id, page_id, book_id, title, page_no, text in page_texts(source, level):
        for n, piece in enumerate(chunk_page(text or "")):
            chunks.append(Chunk(chunk_id=f"{name}:{page_id}:c{n}", index=name, source=source,
                                level=level, page_id=page_id, image_id=image_id, book_id=book_id,
                                title=title, page_no=page_no, position=n, text=piece))
    if not chunks:
        raise RuntimeError(f"no pages found for {source} level {level}")
    vectors = embed_documents([c.search_text() for c in chunks])
    rows = [{**c.model_dump(), "search_text": c.search_text(), "vector": v.tolist()}
            for c, v in zip(chunks, vectors)]

    from lancedb.index import FTS

    db = connect()
    table = db.create_table(name, data=rows, mode="overwrite")
    table.create_index("search_text", config=FTS(stem=True, remove_stop_words=True,
                                                 language="English"), replace=True)
    return len(rows)


def open_table(source: str, level: int):
    return connect().open_table(table_name(source, level))
