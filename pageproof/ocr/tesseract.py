"""Tesseract OCR: the classical baseline."""

import pytesseract
from PIL import Image

# psm 3 = fully automatic page segmentation (the default for book pages)
CONFIG = "--psm 3"


def ocr_text(image: Image.Image, lang: str = "eng") -> str:
    return pytesseract.image_to_string(image, lang=lang, config=CONFIG)


def ocr_page(image_path: str, lang: str = "eng") -> str:
    return ocr_text(Image.open(image_path), lang=lang)
