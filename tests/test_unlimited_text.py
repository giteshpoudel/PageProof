from pageproof.ocr.unlimited import to_plain_text


def test_strips_layout_tags():
    raw = '<|det|>text [160, 96, 415, 115]<|/det|>"What is his name?"'
    assert to_plain_text(raw) == '"What is his name?"'


def test_flattens_superscript_latex():
    raw = "<|det|>text [1, 2, 3, 4]<|/det|>\\( M^{r} \\) & \\( M^{rs} \\) . Bennet"
    assert to_plain_text(raw) == "Mr & Mrs. Bennet"


def test_strips_markdown_emphasis_and_headings():
    assert to_plain_text("## CHAPTER II.\n**Her** mind") == "CHAPTER II.\nHer mind"
