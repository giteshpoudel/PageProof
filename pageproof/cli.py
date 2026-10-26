"""PageProof command line."""

from pathlib import Path

import typer

app = typer.Typer(help="Question answering over scanned books, with page citations.")
corpus = typer.Typer(help="Build and inspect the synthetic corpus.")
app.add_typer(corpus, name="corpus")


@corpus.command("build")
def corpus_build(root: Path = Path("data/corpus"), pages_per_book: int = 65,
                 loo_pages: int = 10) -> None:
    """Render the books, degrade every page at levels 0-5, write Parquet manifests."""
    from pageproof.corpus.build import build

    build(root, pages_per_book, loo_pages)


@corpus.command("summary")
def corpus_summary(root: Path = Path("data/corpus")) -> None:
    """Counts per book and set, from the Parquet manifests."""
    import duckdb

    con = duckdb.connect()
    print(con.sql(f"""
        select p.title, count(distinct p.page_id) as pages,
               count(*) filter (where i.set = 'level') as level_images,
               count(*) filter (where i.set = 'loo') as loo_images,
               round(sum(len(p.text)) filter (where i.level = 0 and i.set = 'level') / 1000.0, 1) as kchars
        from '{root}/images.parquet' i join '{root}/pages.parquet' p using (page_id)
        group by p.title order by p.title
    """))


ocr = typer.Typer(help="Run OCR sources over the corpus.")
app.add_typer(ocr, name="ocr")


@ocr.command("tesseract")
def ocr_tesseract(workers: int = 0) -> None:
    """Tesseract on every corpus image (resumes; writes data/ocr/tesseract.parquet)."""
    from pageproof.ocr.runs import run_tesseract

    run_tesseract(workers or None)


index = typer.Typer(help="Build the search indexes (one per OCR source x level).")
app.add_typer(index, name="index")


@index.command("build")
def index_build(source: str = "tesseract", levels: str = "0-5") -> None:
    """Chunk, embed and index OCR text. --source clean indexes the exact page text (level 0)."""
    from pageproof.index.store import build_index

    lo, _, hi = levels.partition("-")
    for level in ([0] if source == "clean" else range(int(lo), int(hi or lo) + 1)):
        print(f"{source} L{level}: {build_index(source, level)} chunks")


@app.command()
def search(question: str, source: str = "tesseract", level: int = 0, k: int = 5,
           mode: str = "hybrid") -> None:
    """Show the passages retrieval finds, without calling the LLM."""
    from pageproof.qa.retrieve import search as run

    for i, r in enumerate(run(question, source, level, k, mode), 1):
        print(f"[{i}] {r.chunk.page_id}  {r.chunk.title}, p.{r.chunk.page_no}  score {r.score:.4f}")
        print(f"    {r.chunk.text[:160]}...")


@app.command()
def ask(question: str, source: str = "tesseract", level: int = 0, k: int = 5,
        mode: str = "hybrid", crop: bool = True) -> None:
    """Answer from one OCR source at one degradation level, citing pages; save a highlighted page crop."""
    import re

    from pageproof.config import DATA
    from pageproof.qa.answer import ask as run
    from pageproof.qa.locate import save_crop

    a = run(question, source, level, k, mode)
    print(f"\n{a.answer}\n")
    for page_id in a.citations:
        r = next(p for p in a.passages if p.chunk.page_id == page_id)
        print(f"  cites {page_id}: {r.chunk.title}, page {r.chunk.page_no}")
    if a.quote:
        print(f'  quote: "{a.quote}"')
    if crop and a.found and a.quote_page:
        image = next(p.chunk for p in a.passages if p.chunk.page_id == a.quote_page)
        import duckdb

        path = duckdb.sql(f"select path from '{DATA}/corpus/images.parquet' "
                          f"where image_id = '{image.image_id}'").fetchall()[0][0]
        slug = re.sub(r"\W+", "-", question.lower()).strip("-")[:40]
        out = DATA / "crops" / f"{slug}_{source}_L{level}.png"
        found = save_crop(path, a.quote, out)
        print(f"  page image: {out}" + ("" if found else " (quote not located; whole page)"))
    print(f"  [{a.model}, {a.prompt_tokens} prompt + {a.completion_tokens} completion tokens]")


if __name__ == "__main__":
    app()
