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


if __name__ == "__main__":
    app()
