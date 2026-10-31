"""Render many unique concepts built on fallen.py: each mixes a tear, an eye, extras and a
skin, plus its own pose and light. Also writes contact sheets and a numbered idea list.

Run with Blender:      blender -b -P fallen_concepts.py -- [count]
Or with the bpy module: python fallen_concepts.py [count]
Writes fallen_concepts/concept_001.png ..., sheet_1.png ... and ideas.md.
"""
import colorsys
import itertools
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fallen as f  # noqa: E402
import variations  # noqa: E402  (contact_sheet)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fallen_concepts")
SIZE, SAMPLES, PER_SHEET = 560, 20, 20

TEARS = {  # name: (kind, glow colour)
    "radioactive green": ("glow", (0.3, 1.0, 0.04, 1)),
    "toxic cyan": ("glow", (0.02, 0.9, 1.0, 1)),
    "hot pink glow": ("glow", (1.0, 0.05, 0.5, 1)),
    "lava": ("glow", (1.0, 0.25, 0.01, 1)),
    "ultraviolet": ("glow", (0.45, 0.05, 1.0, 1)),
    "electric blue": ("glow", (0.05, 0.25, 1.0, 1)),
    "golden glow": ("glow", (1.0, 0.75, 0.1, 1)),
    "blood": ("blood", None),
    "crystal water": ("water", None),
    "molten gold": ("gold", None),
    "mercury": ("chrome", None),
    "black ink": ("ink", None),
    "milk": ("milk", None),
    "honey": ("honey", None),
}
NO_VEINS = 0.4995
EYES = {  # name: (eye_material overrides, iris hue or None for natural/tear-matched)
    "natural": ({}, None),
    "bloodshot": ({"flush_color": (0.7, 0.1, 0.08, 1), "flush_amount": 0.9, "vein_min": 0.43}, None),
    "vampire": ({"sclera": (0.75, 0.25, 0.22, 1), "flush_color": (0.45, 0.02, 0.02, 1),
                 "flush_amount": 1.0, "vein_min": 0.42, "pupil": "slit"}, 0.0),
    "demon": ({"sclera": (0.02, 0.02, 0.02, 1), "flush_amount": 0.0, "vein_min": NO_VEINS,
               "pupil": "slit", "iris_glow": 2.0}, 0.08),
    "void": ({"sclera": (0.01, 0.01, 0.01, 1), "flush_amount": 0.0, "vein_min": NO_VEINS,
              "pupil": "none"}, "black"),
    "glowing": ({"iris_glow": 5.0}, "tear"),
    "blind": ({"pupil": "none", "vein_min": 0.45}, "milky"),
    "cat": ({"pupil": "slit"}, 0.2),
}
EXTRAS = [(), ("horns",), ("flowers",), ("piercing",), ("eyeshadow",), ("cracks",),
          ("freckles",), ("lashes",), ("horns", "piercing"), ("flowers", "lashes"),
          ("cracks", "eyeshadow"), ("freckles", "lashes"), ("horns", "cracks")]
SKINS = {
    "pale": (0.85, 0.62, 0.5, 1), "peach": (0.8, 0.52, 0.36, 1), "olive": (0.62, 0.45, 0.3, 1),
    "tan": (0.5, 0.32, 0.2, 1), "brown": (0.32, 0.18, 0.1, 1), "deep brown": (0.2, 0.11, 0.06, 1),
    "devil red": (0.6, 0.08, 0.05, 1), "alien green": (0.35, 0.6, 0.3, 1),
    "ice blue": (0.45, 0.6, 0.8, 1), "corpse grey": (0.5, 0.52, 0.5, 1),
    "porcelain": (0.92, 0.88, 0.85, 1),
}
BROWS = [(0.13, 0.04, 0.015, 1), (0.02, 0.015, 0.012, 1), (0.06, 0.035, 0.02, 1),
         (0.45, 0.3, 0.12, 1), (0.35, 0.08, 0.02, 1), (0.3, 0.3, 0.3, 1), (0.85, 0.85, 0.82, 1)]
NATURAL_IRIS_HUES = [0.3, 0.58, 0.08, 0.12, 0.5]  # green, blue, brown, hazel, teal


def hsv(h, s, v):
    return (*colorsys.hsv_to_rgb(h % 1.0, s, v), 1)


def iris_from_hue(h, sat=0.8):
    return hsv(h, sat, 0.2), hsv(h, sat, 0.75), hsv(h, sat, 0.16)


def concepts(count, seed=2026):
    """Unique (tear, eye) pairs, each with its own extras and skin. The first two are the
    vampire and devil ideas."""
    rng = random.Random(seed)
    pairs = list(itertools.product(TEARS, EYES))
    rng.shuffle(pairs)
    pinned = [("blood", "vampire", (), "pale"), ("blood", "demon", ("horns",), "devil red")]
    pairs = [p for p in pairs if p not in [(t, e) for t, e, _, _ in pinned]]
    out = list(pinned)
    for tear, eye in pairs[:max(0, count - len(pinned))]:
        out.append((tear, eye, rng.choice(EXTRAS), rng.choice(list(SKINS))))
    return out[:count]


def apply(i, tear, eye, extras, skin):
    rng = random.Random(1000 + i)
    kind, glow = TEARS[tear]
    f.TEAR_KIND, f.TEAR_COLOR = kind, glow or f.TEAR_COLOR
    look, iris = EYES[eye]
    f.EYE_LOOK = dict(look)
    if iris is None:
        f.IRIS = iris_from_hue(rng.choice(NATURAL_IRIS_HUES), rng.uniform(0.4, 0.8))
    elif iris == "tear":
        f.IRIS = (glow[:3] + (1,),) * 3 if glow else iris_from_hue(rng.random())
    elif iris == "black":
        f.IRIS = ((0.01, 0.01, 0.01, 1),) * 3
    elif iris == "milky":
        f.IRIS = ((0.7, 0.74, 0.76, 1), (0.82, 0.85, 0.86, 1), (0.6, 0.64, 0.66, 1))
    else:
        f.IRIS = iris_from_hue(iris + rng.uniform(-0.03, 0.03), 0.9)
    f.EXTRAS = extras
    f.SKIN = SKINS[skin]
    f.SEED = i
    f.BROW_COLOR = rng.choice(BROWS)
    f.EYESHADOW = hsv(rng.random(), 0.8, 0.5)
    f.METAL = rng.choice([(0.9, 0.9, 0.92, 1), (1.0, 0.75, 0.3, 1)])
    f.HORN_COLOR = rng.choice([(0.06, 0.02, 0.015, 1), (0.5, 0.05, 0.03, 1), (0.85, 0.8, 0.7, 1)])
    f.BROW_DROP = rng.uniform(0.02, 0.11)
    f.EYE_W = rng.uniform(0.15, 0.19)
    f.EYE_H_TOP, f.EYE_H_BOT = rng.uniform(0.035, 0.06), rng.uniform(0.035, 0.05)
    f.EYE_TILT = math.radians(rng.uniform(-10, 6))
    f.GLANCE = (rng.uniform(-2, 6), rng.uniform(-30, 20))
    f.TEAR_DRIFT = rng.uniform(-0.03, 0.04)
    warm = rng.random() < 0.7
    f.KEY_COLOR = hsv(rng.uniform(0.04, 0.12) if warm else rng.uniform(0.55, 0.65),
                      rng.uniform(0.15, 0.35), 1.0)[:3]


def describe(n, tear, eye, extras, skin):
    bits = [f"{tear} tear", f"{eye} eye"] + list(extras) + [f"{skin} skin"]
    return f"{n:03d}. " + ", ".join(bits)


def main(count):
    os.makedirs(OUT, exist_ok=True)
    ideas = concepts(count)
    with open(os.path.join(OUT, "ideas.md"), "w") as fh:
        fh.write("# Fallen eye concepts\n\n" + "\n".join(describe(n, *c) for n, c in enumerate(ideas, 1)) + "\n")
    paths = []
    for n, idea in enumerate(ideas, 1):
        path = os.path.join(OUT, f"concept_{n:03d}.png")
        if not os.path.exists(path):  # lets an interrupted batch resume
            apply(n, *idea)
            f.main(path, size=SIZE, samples=SAMPLES)
        paths.append(path)
    for k in range(0, len(paths), PER_SHEET):
        variations.contact_sheet(paths[k:k + PER_SHEET],
                                 os.path.join(OUT, f"sheet_{k // PER_SHEET + 1}.png"))


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(int(args[0]) if args else 100)
