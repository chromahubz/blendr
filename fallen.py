"""Painterly 3D close-up of a glaring green eye shedding a radioactive tear
(after Cabanel's "The Fallen Angel", 1847), with the "garden" title.

Run with Blender:      blender -b -P fallen.py -- fallen.png [--no-title]
Or with the bpy module: python fallen.py fallen.png [--no-title]
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import garden as g  # noqa: E402  (eye, tear and title builders)

EYE = np.array([-0.02, 0.05])
EYE_TILT = math.radians(-4)
EYE_W, EYE_H_TOP, EYE_H_BOT = 0.17, 0.045, 0.04  # narrowed: upper lid cuts the iris
GLANCE = (2, -24)  # eyeball rotation in degrees: side-eye toward the viewer's left
BROW_DROP = 0.07  # how far the brow slants down toward the nose (anger)

# Look. fallen_variations.py overrides these.
SKIN = (0.8, 0.52, 0.36, 1)
BROW_COLOR = (0.13, 0.04, 0.015, 1)  # dark auburn
IRIS = ((0.04, 0.22, 0.04, 1), (0.25, 0.75, 0.12, 1), (0.03, 0.17, 0.03, 1))  # green
TEAR_COLOR = (0.3, 1.0, 0.04, 1)  # radioactive green
TEAR_DRIFT = 0.02  # sideways drift of the tear as it runs down
KEY_COLOR = (1.0, 0.84, 0.66)
smoothstep = g.smoothstep


def eye_local(x, y):
    dx, dy = x - EYE[0], y - EYE[1]
    c, s = math.cos(-EYE_TILT), math.sin(-EYE_TILT)
    return c * dx - s * dy, s * dx + c * dy


def almond_q(x, y):
    u, v = eye_local(x, y)
    hv = np.where(v > 0, EYE_H_TOP, EYE_H_BOT)
    pinch = np.maximum(1 - (u / EYE_W) ** 2, 0.05) ** 0.5
    return np.sqrt((u / EYE_W) ** 2 + (v / hv) ** 2 / pinch), v


def brow_line(x):
    return 0.21 - BROW_DROP * smoothstep(-0.3, 0.15, x)  # slants down toward the nose: a frown


def face(x, y):
    """Height of the face plus paint masks (lid redness, eyebrow, lash line)."""
    q, v = almond_q(x, y)
    up = smoothstep(-0.015, 0.015, v)  # soft upper/lower split (no ledges at the corners)
    u, _ = eye_local(x, y)
    qe = np.sqrt((u / EYE_W) ** 2 + (v / np.where(v > 0, EYE_H_TOP, EYE_H_BOT)) ** 2)  # plain ellipse for the lid rings
    z = 0.35 * np.sqrt(np.clip(1 - ((x + 0.1) / 1.1) ** 2 - ((y - 0.05) / 0.9) ** 2, 0, 1))
    z -= 0.22 * smoothstep(0.3, 0.6, x)  # far side of the face turns away
    z += 0.2 * np.exp(-((x - 0.25) / 0.08) ** 2) * smoothstep(0.25, 0.05, y)  # nose bridge
    yb = brow_line(x)
    z += 0.18 * np.exp(-((y - yb) / 0.07) ** 2) * smoothstep(-0.6, -0.3, x) * smoothstep(0.3, 0.15, x)
    z -= 0.05 * np.exp(-((x - 0.17) / 0.018) ** 2) * np.exp(-((y - 0.13) / 0.06) ** 2)  # furrow
    z -= 0.11 * np.exp(-(((x - EYE[0]) / 0.2) ** 2 + ((y - EYE[1]) / 0.1) ** 2))  # socket
    z += 0.04 * np.exp(-((qe - 1.2) / 0.25) ** 2)  # lids
    z += 0.05 * np.exp(-((qe - 1.75) / 0.25) ** 2) * up  # heavy upper fold
    z += 0.03 * np.exp(-((qe - 1.3) / 0.2) ** 2) * (1 - up)  # puffy lower lid
    z -= 0.07 * smoothstep(1.35, 1.0, q)  # rim curls softly into the socket
    z += 0.06 * np.exp(-(((x + 0.1) / 0.25) ** 2 + ((y + 0.12) / 0.12) ** 2))  # cheek
    z += 0.002 * np.sin(x * 40 + np.sin(y * 30)) * np.sin(y * 37)  # skin unevenness

    red = np.exp(-((qe - 1.12) / 0.12) ** 2) * (1 - up)
    red += 0.8 * np.exp(-(((u - EYE_W) / 0.04) ** 2 + (v / 0.04) ** 2))  # inner corner
    red += 0.4 * np.exp(-((qe - 1.1) / 0.08) ** 2) * up  # flushed upper lid
    brow = (np.exp(-((y - yb - 0.015) / (0.04 + 0.02 * smoothstep(-0.3, 0.12, x))) ** 4)
            * smoothstep(-0.42, -0.3, x) * smoothstep(0.19, 0.12, x)
            * (0.8 + 0.2 * np.sin(x * 37 + np.sin(y * 23))))
    lashes = np.exp(-((q - 1.03) / 0.035) ** 2) * (0.35 + 0.65 * up)
    return z, q >= 1.0, np.stack([np.clip(red, 0, 1), np.clip(brow, 0, 1), lashes], axis=1)


def face_z(x, y):
    return float(face(np.array([x]), np.array([y]))[0][0])


def grid_mesh(name, xs, ys, fn):
    X, Y = np.meshgrid(xs, ys)
    x, y = X.ravel(), Y.ravel()
    z, keep, masks = fn(x, y)
    n, m = len(ys), len(xs)
    idx = np.arange(n * m).reshape(n, m)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    ok = keep[a] & keep[b] & keep[c] & keep[d]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(np.stack([x, y, z], axis=1).tolist(), [], np.stack([a, b, c, d], 1)[ok].tolist())
    attr = mesh.color_attributes.new("masks", "FLOAT_COLOR", "POINT")
    rgba = np.concatenate([masks, np.ones((len(x), 1))], axis=1).astype(np.float32)
    attr.data.foreach_set("color", rgba.ravel())
    mesh.shade_smooth()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def skin_material(name, base):
    mat, nt, bsdf = g.node_mat(name)
    n, link = nt.nodes, nt.links
    attr = n.new("ShaderNodeAttribute")
    attr.attribute_name = "masks"
    sep = n.new("ShaderNodeSeparateColor")
    link.new(attr.outputs["Color"], sep.inputs[0])
    color = None
    for channel, paint in (("Red", (0.55, 0.13, 0.09, 1)), ("Green", BROW_COLOR),
                           ("Blue", (0.04, 0.02, 0.015, 1))):
        mix = n.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        if color is None:
            mix.inputs["A"].default_value = base
        else:
            link.new(color, mix.inputs["A"])
        mix.inputs["B"].default_value = paint
        link.new(sep.outputs[channel], mix.inputs["Factor"])
        color = mix.outputs["Result"]
    link.new(color, bsdf.inputs["Base Color"])
    pores = n.new("ShaderNodeTexNoise")
    pores.inputs["Scale"].default_value = 900.0
    bump = n.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    link.new(pores.outputs["Fac"], bump.inputs["Height"])
    link.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Subsurface Weight"].default_value = 0.25
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
    bsdf.inputs["Subsurface Scale"].default_value = 0.03
    return mat


def build_face():
    obj = grid_mesh("Face", np.linspace(-0.5, 0.55, 700), np.linspace(-0.45, 0.5, 640), face)
    obj.data.materials.append(skin_material("Face skin", SKIN))
    sub = obj.modifiers.new("Smooth", "SUBSURF")
    sub.levels, sub.render_levels = 0, 1


def build_eye():
    g.IRIS = IRIS
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=1.0)
    eye = bpy.context.active_object
    eye.name = "Eye"
    eye.scale = (EYE_W, EYE_W, 0.1)  # flattened so it stays behind the low lower lid
    # Sink it just below the lowest point of the lid rim (ellipsoid is ~0.11 deep once turned).
    X, Y = np.meshgrid(np.linspace(-0.25, 0.25, 300), np.linspace(-0.15, 0.25, 300))
    z, _, _ = face(X.ravel(), Y.ravel())
    q, _ = almond_q(X.ravel(), Y.ravel())
    rim_min = z[(q > 1) & (q < 1.05)].min()
    eye.location = (EYE[0], EYE[1] + 0.005, rim_min - 0.12)
    eye.rotation_euler = (math.radians(GLANCE[0]), math.radians(GLANCE[1]), 0)
    bpy.ops.object.shade_smooth()
    eye.data.materials.append(g.eye_material())


def build_tear():
    mb = bpy.data.metaballs.new("Tear")
    mb.resolution = mb.render_resolution = 0.003
    obj = bpy.data.objects.new("Tear", mb)
    obj.visible_shadow = False
    bpy.context.collection.objects.link(obj)
    def ball(x, y, r):
        el = mb.elements.new()
        el.co = (x, y, face_z(x, y) + r * 0.9)
        el.radius = r * 1.4

    c, s = math.cos(EYE_TILT), math.sin(EYE_TILT)
    for t in np.linspace(-0.8, 0.6, 40):  # wet film along the lower lid
        u, v = t * EYE_W, -EYE_H_BOT * math.sqrt(1 - t * t) - 0.012
        ball(EYE[0] + c * u - s * v, EYE[1] + s * u + c * v, 0.006 + 0.004 * (t + 0.8))
    x0, y0 = EYE[0] + 0.05, EYE[1] - 0.06
    for t in np.linspace(0, 1, 40):  # swollen bead spilling over the lid, running down the cheek
        x, y = x0 + TEAR_DRIFT * t + 0.008 * math.sin(t * 5), y0 - 0.2 * t
        ball(x, y, 0.02 * (1 - 0.45 * t) * (1 + 0.15 * math.sin(t * 18)))
    ball(x0 + TEAR_DRIFT, y0 - 0.215, 0.019)  # gathering drop at the end
    obj.data.materials.append(radioactive_material())


def radioactive_material():
    """Glowing toxic liquid; the glow also lights the lid and cheek around it."""
    mat, _, bsdf = g.node_mat("Radioactive tear")
    bsdf.inputs["Base Color"].default_value = TEAR_COLOR
    bsdf.inputs["Emission Color"].default_value = TEAR_COLOR
    bsdf.inputs["Emission Strength"].default_value = 6.0
    bsdf.inputs["Transmission Weight"].default_value = 0.4
    bsdf.inputs["Roughness"].default_value = 0.05
    bsdf.inputs["Coat Weight"].default_value = 1.0
    return mat


def build_brow(seed=3):
    """Individual brow hairs: sweeping toward the temple, standing up near the nose."""
    rng = np.random.default_rng(seed)
    curve = bpy.data.curves.new("Brow", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.0016
    curve.bevel_resolution = 2
    placed = 0
    while placed < 380:
        x0 = rng.uniform(-0.42, 0.13)
        w = 0.035 + 0.012 * smoothstep(-0.3, 0.12, x0)
        y0 = brow_line(x0) + 0.03 + rng.uniform(-0.8, 0.8) * w  # sits above the lid fold
        inner = smoothstep(0.0, 0.17, x0)
        ang = math.radians(195 - 85 * inner + rng.uniform(-12, 12))
        length, bend = rng.uniform(0.035, 0.06), rng.uniform(-0.25, 0.25)
        ts = np.linspace(0, 1, 4)
        xs = x0 + length * ts * np.cos(ang + bend * ts)
        ys = y0 + length * ts * np.sin(ang + bend * ts)
        if almond_q(xs, ys)[0].min() < 1.9:  # keep every hair clear of the eye
            continue
        placed += 1
        spl = curve.splines.new("NURBS")
        spl.points.add(3)
        for k, (p, x, y) in enumerate(zip(spl.points, xs, ys)):
            t = ts[k]
            p.co = (x, y, face_z(x, y) + 0.002 + 0.004 * t, 1)
            p.radius = 1.0 - 0.8 * t
        spl.use_endpoint_u = True
        spl.order_u = 3
    obj = bpy.data.objects.new("Brow", curve)
    bpy.context.collection.objects.link(obj)
    mat, _, bsdf = g.node_mat("Brow hair")
    bsdf.inputs["Base Color"].default_value = BROW_COLOR
    bsdf.inputs["Roughness"].default_value = 0.4
    obj.data.materials.append(mat)


def build_lights_and_camera(scene):
    def area(name, loc, size, energy, color):
        data = bpy.data.lights.new(name, "AREA")
        data.size, data.energy, data.color = size, energy, color
        obj = bpy.data.objects.new(name, data)
        obj.location = loc
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = (-Vector(loc)).to_track_quat("-Z", "Y")
        scene.collection.objects.link(obj)

    area("Key", (-3.0, 1.2, 1.8), 1.5, 650, KEY_COLOR)
    area("Fill", (2.0, -0.5, 2.0), 3.0, 15, (0.6, 0.7, 1.0))
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.03, 0.018, 1)
    scene.world = world
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_fit, cam_data.sensor_height, cam_data.lens = "VERTICAL", 36, 72
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = (0.02, 0.04, 2.1)  # tight on the eye and brow
    scene.collection.objects.link(cam)
    scene.camera = cam


def build_compositor(scene, size):
    """Kuwahara filter for brush strokes, then a bloom so the radioactive tear glows."""
    nt = bpy.data.node_groups.new("Painterly", "CompositorNodeTree")
    nt.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = nt.nodes.new("CompositorNodeRLayers")
    kw = nt.nodes.new("CompositorNodeKuwahara")
    kw.inputs["Size"].default_value = max(4, size // 60)
    glare = nt.nodes.new("CompositorNodeGlare")
    glare.inputs["Type"].default_value = "Bloom"
    glare.inputs["Threshold"].default_value = 3.0  # only the glowing tear blooms
    glare.inputs["Strength"].default_value = 0.8
    glare.inputs["Size"].default_value = 0.75
    out = nt.nodes.new("NodeGroupOutput")
    nt.links.new(rl.outputs["Image"], kw.inputs["Image"])
    nt.links.new(kw.outputs[0], glare.inputs["Image"])
    nt.links.new(glare.outputs["Image"], out.inputs[0])
    scene.compositing_node_group = nt
    scene.render.compositor_device = "CPU"  # no GPU needed


def canvas_finish(path):
    """Warm varnish, vignette and a fine canvas weave over the render."""
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    f = 2.1 * 1024 / h  # same weave density at any resolution
    weave = 1 + 0.025 * np.sin(xx * f) * np.sin(yy * f)
    rng = np.random.default_rng(1)
    weave += rng.normal(0, 0.015, (h, w))
    vig = 1 - 0.35 * (((xx / w - 0.5) * 1.6) ** 2 + ((yy / h - 0.5) * 1.4) ** 2)
    px[..., :3] *= (weave * vig)[..., None] * np.array([1.0, 0.95, 0.84], np.float32)
    img.pixels.foreach_set(np.clip(px, 0, 1).ravel())
    img.filepath_raw, img.file_format = path, "PNG"
    img.save()


def main(out_png, size=1024, samples=64, title=True):
    scene = g.reset_scene()
    build_face()
    build_eye()
    build_brow()
    build_tear()
    if title:
        g.TITLE_COLOR = (0.32, 0.15, 0.03, 1)  # dark old gold, reads against the skin
        g.build_title()
        obj = bpy.data.objects["Title"]
        obj.location = (0.02, 0.33, 0.7)
        obj.scale = (0.3, 0.3, 0.3)
    build_lights_and_camera(scene)

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = size * 3 // 4, size
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    build_compositor(scene, size)
    scene.render.filepath = out_png
    bpy.ops.render.render(write_still=True)
    canvas_finish(out_png)


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(os.path.abspath(args[0] if args else "fallen.png"), title="--no-title" not in args)
