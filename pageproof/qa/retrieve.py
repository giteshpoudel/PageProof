"""Search one index: vector, keyword (BM25) or hybrid (both, fused by reciprocal rank).

Same design as EDGAR, where hybrid beat vector by +0.23 recall@5.
"""

from pydantic import BaseModel

from pageproof.index.chunking import Chunk
from pageproof.index.store import embed_query, open_table

RRF_K = 60  # standard constant: damps the advantage of rank 1 over rank 2
MODES = ("vector", "keyword", "hybrid")
_EXTRA = ("vector", "search_text", "_distance", "_score", "_relevance_score")


class SearchResult(BaseModel):
    chunk: Chunk
    score: float  # higher is better; scale depends on the mode, so compare ranks across modes


def _rows(rows: list[dict], score_field: str, transform=lambda s: s) -> list[SearchResult]:
    out = []
    for row in rows:
        score = transform(row[score_field])
        out.append(SearchResult(chunk=Chunk(**{k: v for k, v in row.items() if k not in _EXTRA}),
                                score=score))
    return out


def fts_query(question: str) -> str:
    """Drop quotes: LanceDB treats them as phrase queries, which this BM25 index can't run."""
    return question.replace('"', " ").replace("“", " ").replace("”", " ")


def _vector(table, question: str, n: int) -> list[SearchResult]:
    rows = table.search(embed_query(question)).metric("cosine").limit(n).to_list()
    return _rows(rows, "_distance", lambda d: 1 - d)


def _keyword(table, question: str, n: int) -> list[SearchResult]:
    rows = table.search(fts_query(question), query_type="fts").limit(n).to_list()
    return _rows(rows, "_score")


def rrf(*ranked: list[SearchResult]) -> list[SearchResult]:
    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for results in ranked:
        for rank, r in enumerate(results, 1):
            fused[r.chunk.chunk_id] = fused.get(r.chunk.chunk_id, 0.0) + 1 / (RRF_K + rank)
            chunks[r.chunk.chunk_id] = r.chunk
    return [SearchResult(chunk=chunks[i], score=fused[i])
            for i in sorted(fused, key=fused.get, reverse=True)]


def search(question: str, source: str = "tesseract", level: int = 0, k: int = 5,
           mode: str = "hybrid", candidates: int = 30) -> list[SearchResult]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    table = open_table(source, level)
    if mode == "vector":
        return _vector(table, question, k)
    if mode == "keyword":
        return _keyword(table, question, k)
    return rrf(_vector(table, question, candidates), _keyword(table, question, candidates))[:k]
