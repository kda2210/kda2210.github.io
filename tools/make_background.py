#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 kda2210 <kda2210@duck.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate background.webp for kda2210.github.io.

The image is made entirely by this script (soft fractal noise, a glow and a vignette in the
site's own purple palette), so the script is the preferred form for modifying it. Nothing is
copied from any other image. The result is the same on every run (fixed seed).

Usage (from the repository root):
    pip install numpy pillow
    python3 tools/make_background.py      # writes background.webp into the current folder
"""

import numpy as np
from PIL import Image

SIZE = 1600          # pixels; the page scales it with `background-size: cover`
SEED = 2210
OUTPUT = "background.webp"

# Colour stops (level 0..1 -> RGB), based on the page's CSS variables (--void, --plum, ...).
STOPS = [
    (0.00, (14, 4, 26)),
    (0.25, (40, 12, 70)),
    (0.50, (96, 38, 140)),
    (0.75, (150, 82, 190)),
    (1.00, (232, 190, 250)),
]


def fractal(rng, n, exponent):
    """Smooth, tileable noise with a 1/f^exponent amplitude spectrum, scaled to 0..1."""
    spectrum = np.fft.fft2(rng.standard_normal((n, n)))
    fx = np.fft.fftfreq(n)[:, None]
    fy = np.fft.fftfreq(n)[None, :]
    radius = np.sqrt(fx ** 2 + fy ** 2)
    radius[0, 0] = 1.0
    field = np.real(np.fft.ifft2(spectrum / radius ** exponent))
    lo, hi = np.percentile(field, [1, 99])
    return np.clip((field - lo) / (hi - lo), 0.0, 1.0)


def warp(field, dx, dy, amount):
    """Bend `field` by the offsets dx, dy (0..1) using bilinear sampling."""
    n = field.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    sx = (xx + (dx - 0.5) * amount) % n
    sy = (yy + (dy - 0.5) * amount) % n
    x0 = np.floor(sx).astype(int)
    y0 = np.floor(sy).astype(int)
    x1 = (x0 + 1) % n
    y1 = (y0 + 1) % n
    tx = sx - x0
    ty = sy - y0
    top = field[y0, x0] * (1 - tx) + field[y0, x1] * tx
    bottom = field[y1, x0] * (1 - tx) + field[y1, x1] * tx
    return top * (1 - ty) + bottom * ty


def lowpass(field, cutoff):
    """Remove fine, grainy detail (Gaussian low-pass in the frequency domain)."""
    n = field.shape[0]
    f = np.sqrt(np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2)
    return np.real(np.fft.ifft2(np.fft.fft2(field) * np.exp(-((f / cutoff) ** 2))))


def smoothstep(low, high, x):
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def colorize(level):
    xs = [s[0] for s in STOPS]
    return np.stack(
        [np.interp(level, xs, [s[1][c] for s in STOPS]) for c in range(3)], axis=-1
    )


def main():
    rng = np.random.default_rng(SEED)
    n = SIZE

    clouds = fractal(rng, n, 2.0)        # large soft masses
    detail = fractal(rng, n, 1.5)        # finer wisps
    bend_x = fractal(rng, n, 2.4)
    bend_y = fractal(rng, n, 2.4)

    mist = 0.80 * warp(clouds, bend_x, bend_y, 0.30 * n) + 0.20 * warp(detail, bend_y, bend_x, 0.08 * n)
    mist = smoothstep(0.22, 0.82, mist)

    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    glow = np.exp(-(((xx - 0.52) / 0.22) ** 2 + ((yy - 0.20) / 0.18) ** 2))   # light near the top
    haze = np.exp(-(((xx - 0.50) / 0.55) ** 2 + ((yy - 0.55) / 0.60) ** 2))   # broad central haze
    vignette = 1.0 - 0.45 * np.clip(((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.5, 0.0, 1.0)

    level = (0.20 + 0.62 * mist * (0.55 + 0.5 * haze) + 0.55 * glow * (0.7 + 0.3 * mist)) * vignette
    level = np.clip(lowpass(level, 0.10 * 800 / n), 0.0, 1.0)

    rgb = colorize(level)
    rgb += rng.normal(0.0, 1.1, rgb.shape)        # light dithering against colour banding
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    image.save(OUTPUT, "WEBP", quality=82, method=6)
    print(f"wrote {OUTPUT} ({n}x{n})")


if __name__ == "__main__":
    main()
