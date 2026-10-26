"""Character and word error rates between a reference text and an OCR hypothesis."""

import re
import unicodedata

from rapidfuzz.distance import Levenshtein

# Typographic characters that renderers and OCR engines swap freely.
_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ", "­": "",
    "ﬁ": "fi", "ﬂ": "fl",
})


def normalize(text: str) -> str:
    """NFKC, unify quotes and dashes, collapse all whitespace to single spaces."""
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP)
    # Re-join words hyphenated across a line break ("experi-\nment").
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def cer(reference: str, hypothesis: str) -> float:
    """Edit distance over characters, divided by reference length."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Edit distance over words, divided by reference word count."""
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)
