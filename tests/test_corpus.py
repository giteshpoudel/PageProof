import numpy as np
import pytest
from PIL import Image, ImageDraw

from pageproof.corpus.books import BOOKS
from pageproof.corpus.build import choose_pages
from pageproof.corpus.degrade import EFFECTS, degrade, level_severity, save_degraded
from pageproof.corpus.render import clean_html


@pytest.fixture
def page():
    img = Image.new("L", (400, 300), 255)
    d = ImageDraw.Draw(img)
    for y in range(20, 280, 24):
        d.text((20, y), "It is a truth universally acknowledged", fill=0)
    return img


def test_level_zero_is_unchanged(page):
    assert np.array_equal(np.asarray(degrade(page, 0.0, seed=1)), np.asarray(page))


def test_same_seed_same_pixels(page):
    a, b = degrade(page, 0.8, seed=42), degrade(page, 0.8, seed=42)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_different_seed_different_pixels(page):
    a, b = degrade(page, 0.8, seed=1), degrade(page, 0.8, seed=2)
    assert not np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("effects", [EFFECTS, tuple(e for e in EFFECTS if e != "jpeg")])
def test_saved_file_matches_in_memory_pipeline(page, tmp_path, effects):
    path = save_degraded(page, 1.0, 7, tmp_path / "x", effects)
    with Image.open(path) as saved:
        assert np.array_equal(np.asarray(saved.convert("L")),
                              np.asarray(degrade(page, 1.0, 7, effects)))


def test_severity_spans_zero_to_one():
    assert level_severity(0) == 0.0 and level_severity(5) == 1.0


def test_chosen_pages_skip_front_and_back_matter():
    for book in BOOKS:
        pages = choose_pages(book, 500, 65)
        assert len(pages) == 65 and pages == list(range(pages[0], pages[0] + 65))
        assert pages[0] > 50 and pages[-1] <= 425


def test_clean_html_keeps_drop_cap_letter_and_drops_captions():
    html = ('<body><div class="figcenter"><img alt="" src="a.jpg">'
            '<div class="caption"><p>Mr. &amp; Mrs. Bennet</p></div></div>'
            '<p><span class="letra"><img alt="M" src="m.png"></span>R. BENNET was'
            '<span class="pagenum">{6}</span> among</p></body>')
    text = clean_html(html)
    assert "MR. BENNET was" in text.replace("</span>", "").replace('<span class="letra">', "")
    assert "Bennet</p>" not in text and "{6}" not in text
