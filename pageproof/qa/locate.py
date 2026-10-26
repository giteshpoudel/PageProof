"""Find a quote on a page image and save a highlighted crop.

Words and boxes come from Tesseract run on the cited image itself (any OCR
source's citation can be shown this way). The quote is matched against every
window of the same number of words, by fuzzy similarity, so OCR errors in the
quote or on the page still land on the right lines.
"""

from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image, ImageDraw
from rapidfuzz import fuzz

from pageproof.ocr.metrics import normalize


@dataclass
class Located:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in image pixels
    score: float  # 0-100 fuzzy similarity between the quote and the matched words
    matched: str


def words_with_boxes(image: Image.Image) -> list[tuple[str, tuple[int, int, int, int]]]:
    d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 3")
    out = []
    for text, x, y, w, h, conf in zip(d["text"], d["left"], d["top"], d["width"], d["height"], d["conf"]):
        if text.strip() and float(conf) >= 0:
            out.append((text, (x, y, x + w, y + h)))
    return out


MIN_SCORE = 70  # below this the quote is treated as not on the page (unrelated text scores ~50)


def locate(image: Image.Image, quote: str, min_score: float = MIN_SCORE) -> Located | None:
    target = normalize(quote).lower()
    n = len(target.split())
    words = words_with_boxes(image)
    if not n or not words:
        return None
    best = None
    for size in {max(1, n - 2), n, n + 2}:  # OCR may split or merge a few words
        for i in range(0, max(1, len(words) - size + 1)):
            window = words[i:i + size]
            text = normalize(" ".join(w for w, _ in window)).lower()
            score = fuzz.ratio(target, text)
            if best is None or score > best[0]:
                best = (score, window)
    score, window = best
    if score < min_score:
        return None
    boxes = [b for _, b in window]
    box = (min(b[0] for b in boxes), min(b[1] for b in boxes),
           max(b[2] for b in boxes), max(b[3] for b in boxes))
    return Located(box=box, score=score, matched=" ".join(w for w, _ in window))


def save_crop(image_path: str, quote: str, out_path: Path, margin: int = 60) -> Located | None:
    """Highlight the quote on the page and save a crop around it (whole page if not found)."""
    image = Image.open(image_path).convert("RGB")
    found = locate(image.convert("L"), quote) if quote else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not found:
        image.save(out_path)
        return None
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    x0, y0, x1, y1 = found.box
    ImageDraw.Draw(overlay).rectangle((x0 - 4, y0 - 3, x1 + 4, y1 + 3),
                                      fill=(255, 210, 0, 90), outline=(220, 120, 0, 255), width=3)
    shown = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    # Crop full page width, a margin above and below the highlighted lines.
    top, bottom = max(0, y0 - margin * 3), min(image.height, y1 + margin * 3)
    shown.crop((0, top, image.width, bottom)).save(out_path)
    return found
