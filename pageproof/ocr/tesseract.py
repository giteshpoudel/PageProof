"""Tesseract OCR: the classical baseline."""

import pytesseract
from PIL import Image


def ocr_page(image_path: str, lang: str = "eng") -> str:
    # psm 3 = fully automatic page segmentation (the default for book pages)
    return pytesseract.image_to_string(Image.open(image_path), lang=lang, config="--psm 3")
