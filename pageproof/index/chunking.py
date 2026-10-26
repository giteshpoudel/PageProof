"""Split each page's OCR text into chunks that never cross a page boundary.

A chunk belongs to exactly one page, so every retrieved passage cites one page
and can be shown on that page's image. Chunks are built from the OCR engine's
own paragraphs (blank-line separated blocks, which both Tesseract and
Unlimited-OCR emit per layout region), merged up to a target size.
"""

import re

from pydantic import BaseModel

TARGET_CHARS = 700  # merge paragraphs until a chunk reaches this
MAX_CHARS = 1200  # split a longer paragraph at sentence ends


class Chunk(BaseModel):
    chunk_id: str  # <index>:<page_id>:c<n>
    index: str  # e.g. tesseract-L3, clean
    source: str
    level: int
    page_id: str
    image_id: str
    book_id: int
    title: str
    page_no: int
    position: int  # chunk number within the page
    text: str

    def search_text(self) -> str:
        """What gets embedded and keyword-indexed: the book and page give context."""
        return f"{self.title}, page {self.page_no}: {self.text}"


def paragraphs(text: str) -> list[str]:
    """Blank-line separated blocks, with line breaks inside a block joined."""
    blocks = re.split(r"\n\s*\n", text)
    out = []
    for b in blocks:
        b = re.sub(r"(\w)-\n(\w)", r"\1\2", b)  # rejoin words hyphenated across lines
        b = re.sub(r"\s*\n\s*", " ", b).strip()
        if b:
            out.append(b)
    return out


def _split_long(p: str) -> list[str]:
    if len(p) <= MAX_CHARS:
        return [p]
    parts, cur = [], ""
    for sentence in re.split(r"(?<=[.!?;:])\s+", p):
        if cur and len(cur) + len(sentence) + 1 > MAX_CHARS:
            parts.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        parts.append(cur)
    # a single sentence longer than MAX_CHARS: hard-split on spaces
    final = []
    for part in parts:
        while len(part) > MAX_CHARS:
            cut = part.rfind(" ", 0, MAX_CHARS)
            cut = cut if cut > 0 else MAX_CHARS
            final.append(part[:cut])
            part = part[cut:].strip()
        if part:
            final.append(part)
    return final


def chunk_page(text: str) -> list[str]:
    pieces = [s for p in paragraphs(text) for s in _split_long(p)]
    chunks, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + len(piece) + 1 > MAX_CHARS:
            chunks.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip() if cur else piece
        if len(cur) >= TARGET_CHARS:
            chunks.append(cur)
            cur = ""
    if cur:
        # a short tail joins the previous chunk when it fits, so no tiny chunks
        if chunks and len(chunks[-1]) + len(cur) + 1 <= MAX_CHARS:
            chunks[-1] = f"{chunks[-1]} {cur}"
        else:
            chunks.append(cur)
    return chunks
