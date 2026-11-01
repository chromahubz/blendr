"""The vampire take on fallen.py: blood tear, blood-red slit-pupil eye, snarling mouth with fangs.

Run with Blender:      blender -b -P fallen_fangs.py -- fallen_fangs.png [--no-title]
Or with the bpy module: python fallen_fangs.py fallen_fangs.png [--no-title]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fallen_concepts as fc  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    paths = [a for a in args if not a.startswith("--")]
    fc.apply(1, "blood", "vampire", ("fangs",), "pale")
    fc.f.main(os.path.abspath(paths[0] if paths else "fallen_fangs.png"), title="--no-title" not in args)
