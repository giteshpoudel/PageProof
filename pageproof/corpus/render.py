"""Render Project Gutenberg HTML editions into page images with exact ground truth.

The HTML is laid out into a PDF with PyMuPDF's Story engine. Because the PDF is
ours, its text layer is the exact text on each page: that is the ground truth.
Each page is then rasterized to PNG, which is what the OCR engines see.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
import requests
from bs4 import BeautifulSoup

GUTENBERG_HTML = "https://www.gutenberg.org/cache/epub/{id}/pg{id}-images.html"
USER_AGENT = "PageProof research project (personal, non-commercial)"

# A book-like page: 5.5 x 8.5 in (US trade paperback), 0.75 in margins.
PAGE = pymupdf.Rect(0, 0, 5.5 * 72, 8.5 * 72)
MARGIN = 54
BASE_DPI = 200  # resolution of the clean page images
CSS = """
* { font-family: serif; }
body { font-size: 11pt; line-height: 1.35; }
h1, h2, h3 { font-weight: bold; text-align: center; }
h1 { font-size: 16pt; } h2 { font-size: 14pt; } h3 { font-size: 12pt; }
p { text-align: justify; margin-top: 0; margin-bottom: 6pt; }
"""


@dataclass
class Page:
    book_id: int
    page_no: int  # 1-based, in the rendered PDF
    image: str  # path to the PNG
    text: str  # exact text on the page (ground truth)


def fetch_html(book_id: int, cache_dir: Path) -> str:
    """Download a book's HTML edition once and keep it in the cache."""
    path = cache_dir / f"pg{book_id}.html"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        r = requests.get(GUTENBERG_HTML.format(id=book_id), timeout=60,
                         headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
    return path.read_text(encoding="utf-8")


# Elements that are not the book's running text: illustrations and their
# captions, and the print edition's page numbers ("{4}").
_DROP_CLASSES = {"caption", "figcenter", "figleft", "figright", "fig", "illustration",
                 "pagenum"}


def clean_html(html: str) -> str:
    """Keep the book body: drop Gutenberg's header/footer, illustrations, captions and page numbers."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body or soup
    for el in body.select("#pg-header, #pg-footer, script, style, figure"):
        el.decompose()
    for el in body.find_all(class_=lambda c: c is not None):
        if el.decomposed:
            continue
        if _DROP_CLASSES & set(el.get("class") or []):
            el.decompose()
    for img in body.find_all("img"):
        alt = (img.get("alt") or "").strip()
        # A drop cap is an image of the chapter's first letter; keep the letter.
        if 0 < len(alt) <= 2:
            img.replace_with(alt)
        else:
            img.decompose()
    return body.decode_contents() if body is not soup else str(soup)


def render_pdf(html: str, out_pdf: Path) -> int:
    """Lay the HTML out into book pages; returns the page count."""
    story = pymupdf.Story(html=html, user_css=CSS)
    writer = pymupdf.DocumentWriter(str(out_pdf))
    body = PAGE + (MARGIN, MARGIN, -MARGIN, -MARGIN)
    pages, more = 0, True
    while more:
        device = writer.begin_page(PAGE)
        more, _ = story.place(body)
        story.draw(device)
        writer.end_page()
        pages += 1
    writer.close()
    return pages


def render_book(book_id: int, out_dir: Path, first_page: int, n_pages: int,
                dpi: int = 200) -> list[Page]:
    """Render pages [first_page, first_page + n_pages) of a book to PNG + ground truth."""
    book_dir = out_dir / f"pg{book_id}"
    book_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = book_dir / "book.pdf"
    if not pdf_path.exists():
        render_pdf(clean_html(fetch_html(book_id, out_dir / "html")), pdf_path)

    doc = pymupdf.open(pdf_path)
    pages = []
    for i in range(first_page - 1, min(first_page - 1 + n_pages, doc.page_count)):
        page = doc[i]
        image = book_dir / f"page_{i + 1:04d}.png"
        if not image.exists():  # keep existing images so every machine OCRs the same bytes
            page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY).save(image)
        pages.append(Page(book_id, i + 1, str(image), page.get_text("text")))
    doc.close()

    with open(book_dir / "pages.jsonl", "w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    return pages
