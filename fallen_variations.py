"""Render seeded variations of fallen.py (tear glow colour, iris, skin, brow, eye shape, gaze)
plus a contact sheet.

Run with Blender:      blender -b -P fallen_variations.py -- [count]
Or with the bpy module: python fallen_variations.py [count]
Writes fallen_variations/fallen_01.png ... and fallen_variations/contact_sheet.png.
"""
import colorsys
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fallen as f  # noqa: E402
import variations  # noqa: E402  (contact_sheet)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fallen_variations")
SIZE, SAMPLES = 600, 24

SKINS = [(0.85, 0.62, 0.5, 1), (0.8, 0.52, 0.36, 1), (0.62, 0.45, 0.3, 1),
         (0.5, 0.32, 0.2, 1), (0.32, 0.18, 0.1, 1), (0.2, 0.11, 0.06, 1)]
BROWS = [(0.13, 0.04, 0.015, 1), (0.02, 0.015, 0.012, 1), (0.06, 0.035, 0.02, 1),
         (0.45, 0.3, 0.12, 1), (0.35, 0.08, 0.02, 1), (0.3, 0.3, 0.3, 1)]


def hsv(h, s, v):
    return (*colorsys.hsv_to_rgb(h % 1.0, s, v), 1)


def randomize(i, count):
    rng = random.Random(100 + i)
    h = (i - 1) / count  # tear glow hue steps round the colour wheel
    f.TEAR_COLOR = hsv(h, 0.95, 1.0)
    ih = h + rng.uniform(-0.08, 0.08)  # iris echoes the glow
    f.IRIS = (hsv(ih, 0.8, 0.2), hsv(ih, 0.8, 0.75), hsv(ih, 0.8, 0.16))
    f.SKIN = rng.choice(SKINS)
    f.BROW_COLOR = rng.choice(BROWS)
    f.BROW_DROP = rng.uniform(0.02, 0.11)
    f.EYE_W = rng.uniform(0.15, 0.19)
    f.EYE_H_TOP, f.EYE_H_BOT = rng.uniform(0.035, 0.06), rng.uniform(0.035, 0.05)
    f.EYE_TILT = math.radians(rng.uniform(-10, 6))
    f.GLANCE = (rng.uniform(-2, 6), rng.uniform(-30, 20))
    f.TEAR_DRIFT = rng.uniform(-0.03, 0.04)
    warm = rng.random() < 0.7
    f.KEY_COLOR = hsv(rng.uniform(0.04, 0.12) if warm else rng.uniform(0.55, 0.65),
                      rng.uniform(0.15, 0.35), 1.0)[:3]


def main(count):
    os.makedirs(OUT, exist_ok=True)
    paths = []
    for i in range(1, count + 1):
        randomize(i, count)
        path = os.path.join(OUT, f"fallen_{i:02d}.png")
        f.main(path, size=SIZE, samples=SAMPLES)
        paths.append(path)
    variations.contact_sheet(paths, os.path.join(OUT, "contact_sheet.png"))


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(int(args[0]) if args else 20)
