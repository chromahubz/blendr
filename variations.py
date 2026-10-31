"""Render seeded variations of garden.py (palette, folds, eye, tear) plus a contact sheet.

Run with Blender:      blender -b -P variations.py -- [count]
Or with the bpy module: python variations.py [count]
Writes variations/garden_01.png ... and variations/contact_sheet.png.
"""
import colorsys
import math
import os
import random
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import garden as g  # noqa: E402

OUT = os.path.join(g.HERE, "variations")
SIZE, SAMPLES, COLS = 600, 32, 5

IRISES = {  # inner, mid, outer
    "green": ((0.2, 0.3, 0.14, 1), (0.45, 0.6, 0.35, 1), (0.2, 0.25, 0.17, 1)),
    "ice blue": ((0.1, 0.2, 0.35, 1), (0.4, 0.6, 0.85, 1), (0.08, 0.15, 0.3, 1)),
    "amber": ((0.35, 0.15, 0.02, 1), (0.85, 0.5, 0.08, 1), (0.25, 0.1, 0.02, 1)),
    "grey": ((0.2, 0.2, 0.22, 1), (0.55, 0.56, 0.6, 1), (0.15, 0.15, 0.17, 1)),
    "violet": ((0.2, 0.05, 0.3, 1), (0.55, 0.3, 0.8, 1), (0.15, 0.04, 0.25, 1)),
    "brown": ((0.12, 0.05, 0.02, 1), (0.35, 0.17, 0.06, 1), (0.08, 0.03, 0.01, 1)),
}


def hsv(h, s, v):
    return (*colorsys.hsv_to_rgb(h % 1.0, s, v), 1)


def randomize(i, count):
    rng = random.Random(i)
    h = (i / count + rng.uniform(-0.02, 0.02)) % 1.0  # spread skin hues round the wheel
    g.SKIN = (hsv(h - 0.03, 0.6, 1.0), hsv(h, 0.97, 0.95), hsv(h + 0.02, 1.0, 0.3))
    g.CREVICE = hsv(h, 1.0, 0.12)
    g.TITLE_COLOR = hsv(h + 0.5 + rng.uniform(-0.08, 0.08), 0.95, 1.0)
    g.FILL_COLOR = hsv(h, 0.4, 1.0)[:3]
    g.WORLD_COLOR = hsv(h, 0.5, 0.9)
    g.IRIS = IRISES[rng.choice(sorted(IRISES))]
    g.SWIRL = dict(cx=rng.uniform(-0.2, 0.3), cy=rng.uniform(-0.1, 0.25),
                   f1=rng.uniform(7, 13), f2=rng.uniform(17, 28),
                   phase=rng.uniform(0, 2 * math.pi), amp=rng.uniform(0.05, 0.09))
    k = rng.uniform(0.85, 1.12)
    g.EYE_W, g.EYE_H_TOP, g.EYE_H_BOT = 0.52 * k, 0.21 * k, 0.095 * k
    g.EYE_TILT = math.radians(rng.uniform(-14, 6))
    g.GLANCE = (rng.uniform(-8, 4), rng.uniform(-18, 10))
    g.TEAR_BEND = (rng.uniform(0.08, 0.18), rng.uniform(-0.12, -0.05))
    g.TEAR_END = (rng.uniform(0.1, 0.28), rng.uniform(-0.7, -0.5))


def contact_sheet(paths, out):
    tiles = []
    for p in paths:
        img = bpy.data.images.load(p)
        img.colorspace_settings.name = "Non-Color"  # raw pixels, no colour conversion
        px = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        tiles.append(px.reshape(img.size[1], img.size[0], 4))
    rows = math.ceil(len(tiles) / COLS)
    th, tw = tiles[0].shape[:2]
    sheet = np.ones((rows * th, COLS * tw, 4), dtype=np.float32)
    for n, t in enumerate(tiles):  # Blender images are stored bottom row first
        r, c = rows - 1 - n // COLS, n % COLS
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    img = bpy.data.images.new("contact_sheet", COLS * tw, rows * th)
    img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(sheet.ravel())
    img.filepath_raw, img.file_format = out, "PNG"
    img.save()


def main(count):
    os.makedirs(OUT, exist_ok=True)
    paths = []
    for i in range(1, count + 1):
        randomize(i, count)
        path = os.path.join(OUT, f"garden_{i:02d}.png")
        g.main(path, size=SIZE, samples=SAMPLES, save_blend=False)
        paths.append(path)
    contact_sheet(paths, os.path.join(OUT, "contact_sheet.png"))


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(int(args[0]) if args else 20)
