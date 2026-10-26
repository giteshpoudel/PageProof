from pageproof.ocr.metrics import cer, normalize, wer


def test_identical_text_is_zero():
    assert cer("It is a truth", "It is a truth") == 0.0
    assert wer("It is a truth", "It is a truth") == 0.0


def test_whitespace_and_quotes_do_not_count():
    assert cer("“Yes,” she said.\n\nNext", '"Yes," she said. Next') == 0.0


def test_line_break_hyphenation_is_rejoined():
    assert normalize("experi-\nment") == "experiment"


def test_one_substitution():
    assert cer("abcd", "abxd") == 0.25
    assert wer("one two three four", "one too three four") == 0.25


def test_empty_reference():
    assert cer("", "") == 0.0
    assert cer("", "junk") == 1.0
