"""Scan-like degradation controlled by one severity knob.

Level 0 is the clean render. Levels 1-5 scale five effects together, in the
order a real scan picks them up: skew, blur, low resolution, sensor noise, JPEG
compression. Alone, no effect does much harm (calibration: at most 2% Tesseract
CER at full strength); together they compound. So instead of a one-effect grid,
the corpus has a leave-one-out set: all effects but one, which measures what
each effect adds to the combination.

Every call is seeded per page, so the corpus is byte-for-byte reproducible.
"""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

EFFECTS = ("skew", "blur", "lowres", "noise", "jpeg")
N_LEVELS = 6  # 0 = clean, 5 = worst


@dataclass(frozen=True)
class Strength:
    """Effect strengths at severity 1.0 (level 5). Calibrated in P1, see results/p1_calibration.csv."""

    # First pass put level 5 at 78% median Tesseract CER, a cliff after 11% at
    # level 4. These are 0.9x of that pass, which puts level 5 near 30-40%.
    skew_deg: float = 2.25  # rotation, degrees (random sign)
    blur_sigma: float = 1.44  # Gaussian blur radius, pixels at 200 DPI
    min_dpi: float = 92.0  # resolution falls from 200 DPI to this
    noise_std: float = 34.2  # Gaussian noise, grey levels (0-255)
    salt_pepper: float = 0.0108  # fraction of pixels flipped to black or white
    min_jpeg_quality: int = 20  # JPEG quality falls from 95 to this


DEFAULT = Strength()
BASE_DPI = 200


def _skew(img: Image.Image, s: float, rng: np.random.Generator, k: Strength) -> Image.Image:
    angle = s * k.skew_deg * rng.choice([-1, 1])
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)


def _blur(img: Image.Image, s: float, rng: np.random.Generator, k: Strength) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=s * k.blur_sigma))


def _lowres(img: Image.Image, s: float, rng: np.random.Generator, k: Strength) -> Image.Image:
    dpi = BASE_DPI - s * (BASE_DPI - k.min_dpi)
    scale = dpi / BASE_DPI
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(size, Image.LANCZOS)  # stays small, like a low-DPI scan


def _noise(img: Image.Image, s: float, rng: np.random.Generator, k: Strength) -> Image.Image:
    a = np.asarray(img, dtype=np.float32)
    a = a + rng.normal(0.0, s * k.noise_std, a.shape)
    flips = rng.random(a.shape)
    p = s * k.salt_pepper
    a[flips < p / 2] = 0
    a[(flips >= p / 2) & (flips < p)] = 255
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), mode="L")


def _jpeg(img: Image.Image, s: float, rng: np.random.Generator, k: Strength) -> Image.Image:
    quality = round(95 - s * (95 - k.min_jpeg_quality))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("L")


_APPLY = {"skew": _skew, "blur": _blur, "lowres": _lowres, "noise": _noise, "jpeg": _jpeg}


def degrade(img: Image.Image, severity: float, seed: int, effects=EFFECTS,
            strength: Strength = DEFAULT) -> Image.Image:
    """Apply the given effects at severity in [0, 1]. Severity 0 returns the image unchanged."""
    img = img.convert("L")
    if severity <= 0:
        return img
    rng = np.random.default_rng(seed)
    for name in EFFECTS:  # fixed order, whatever order `effects` lists
        if name in effects:
            img = _APPLY[name](img, severity, rng, strength)
    return img


def save_degraded(img: Image.Image, severity: float, seed: int, path_stem, effects=EFFECTS,
                  strength: Strength = DEFAULT):
    """Same pixels as degrade(), written compactly; returns the file path.

    When JPEG is the last effect, the JPEG bytes are the degraded image, so they
    are written as-is (.jpg) instead of being decoded and stored as PNG.
    """
    from pathlib import Path

    stem = Path(path_stem)
    img = img.convert("L")
    if severity > 0:
        rng = np.random.default_rng(seed)
        for name in EFFECTS:
            if name in effects and name != "jpeg":
                img = _APPLY[name](img, severity, rng, strength)
    if severity > 0 and "jpeg" in effects:
        path = stem.with_suffix(".jpg")
        img.save(path, format="JPEG", quality=round(95 - severity * (95 - strength.min_jpeg_quality)))
    else:
        path = stem.with_suffix(".png")
        img.save(path, format="PNG", optimize=True)
    return path


def level_severity(level: int) -> float:
    return level / (N_LEVELS - 1)
