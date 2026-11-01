"""Render many unique devils (horns and fangs) built on fallen.py: each mixes a hellish tear,
an eye, a skin and a horn colour, sometimes glowing magma cracks or piercings, plus its own
pose and light. Also writes contact sheets and a numbered list.

Run with Blender:      blender -b -P devil_variations.py -- [count]
Or with the bpy module: python devil_variations.py [count]
Writes devil_variations/devil_001.png ..., sheet_1.png ... and ideas.md.
"""
import itertools
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fallen as f  # noqa: E402
import fallen_concepts as fc  # noqa: E402  (eye looks, iris and colour helpers)
import variations  # noqa: E402  (contact_sheet)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "devil_variations")
SIZE, SAMPLES, PER_SHEET = 520, 16, 20

TEARS = {  # name: (kind, glow colour)
    "lava": ("glow", (1.0, 0.25, 0.01, 1)),
    "white-hot lava": ("glow", (1.0, 0.6, 0.2, 1)),
    "hellfire red": ("glow", (1.0, 0.04, 0.01, 1)),
    "blue hellfire": ("glow", (0.05, 0.3, 1.0, 1)),
    "green hellfire": ("glow", (0.3, 1.0, 0.04, 1)),
    "purple hellfire": ("glow", (0.5, 0.05, 1.0, 1)),
    "molten gold": ("gold", None),
    "blood": ("blood", None),
    "black oil": ("ink", None),
}
EYES = ["demon", "vampire", "void", "cat", "glowing"]
SKINS = {
    "crimson": (0.6, 0.08, 0.05, 1), "scarlet": (0.8, 0.12, 0.06, 1),
    "maroon": (0.3, 0.03, 0.03, 1), "ember": (0.75, 0.25, 0.06, 1),
    "charcoal": (0.06, 0.05, 0.05, 1), "obsidian": (0.02, 0.02, 0.025, 1),
    "purple": (0.3, 0.08, 0.35, 1), "deep blue": (0.06, 0.1, 0.3, 1),
    "ash": (0.35, 0.33, 0.32, 1), "bone": (0.8, 0.75, 0.65, 1),
}
HORNS = {
    "black": (0.02, 0.015, 0.015, 1), "bone": (0.85, 0.8, 0.7, 1),
    "blood red": (0.45, 0.03, 0.02, 1), "gold": (0.9, 0.6, 0.15, 1),
    "charcoal": (0.12, 0.1, 0.09, 1),
}


def ideas(count, seed=666):
    """Unique (tear, eye, skin, horns) combinations; #1 is the classic lava devil."""
    rng = random.Random(seed)
    combos = list(itertools.product(TEARS, EYES, SKINS, HORNS))
    rng.shuffle(combos)
    first = ("lava", "demon", "crimson", "black")
    combos = [first] + [c for c in combos if c != first]
    out = []
    for n, combo in enumerate(combos[:count], 1):
        r = random.Random(n)
        extras = ("horns", "fangs") + (("cracks",) if r.random() < 0.35 else ()) \
            + (("piercing",) if r.random() < 0.2 else ())
        out.append(combo + (extras,))
    return out


def apply(n, tear, eye, skin, horns, extras):
    fc.apply(n, "blood", eye, extras, "devil red")  # eye look, iris, pose, light, brow
    kind, glow = TEARS[tear]
    f.TEAR_KIND, f.TEAR_COLOR = kind, glow or f.TEAR_COLOR
    if eye == "glowing" and glow:
        f.IRIS = (glow,) * 3  # the eye burns the same colour as the tear
    f.SKIN = SKINS[skin]
    f.HORN_COLOR = HORNS[horns]
    rng = random.Random(5000 + n)
    f.HORN_SIZE = rng.uniform(0.8, 1.15)
    f.BROW_DROP = rng.uniform(0.07, 0.12)  # devils always frown
    f.BROW_COLOR = rng.choice([(0.02, 0.015, 0.012, 1), (0.06, 0.02, 0.01, 1), (0.3, 0.3, 0.3, 1)])
    f.CRACK_GLOW = (glow or (1.0, 0.3, 0.02, 1)) if "cracks" in extras else None
    f.KEY_COLOR = fc.hsv(rng.uniform(0.0, 0.1), rng.uniform(0.2, 0.5), 1.0)[:3]  # firelight


def describe(n, tear, eye, skin, horns, extras):
    more = [("glowing magma cracks" if e == "cracks" else e) for e in extras if e not in ("horns", "fangs")]
    return f"{n:03d}. " + ", ".join([f"{tear} tear", f"{eye} eye", f"{skin} skin", f"{horns} horns"] + more)


def main(count):
    os.makedirs(OUT, exist_ok=True)
    todo = ideas(count)
    with open(os.path.join(OUT, "ideas.md"), "w") as fh:
        fh.write("# Devil variations\n\nAll have horns and fangs.\n\n"
                 + "\n".join(describe(n, *c) for n, c in enumerate(todo, 1)) + "\n")
    paths = []
    for n, idea in enumerate(todo, 1):
        path = os.path.join(OUT, f"devil_{n:03d}.png")
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
