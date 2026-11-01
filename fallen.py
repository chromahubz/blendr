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
TEAR_KIND = "glow"  # glow | blood | water | gold | chrome | ink | milk | honey
EYE_LOOK = {}  # keyword overrides for garden.eye_material (sclera, pupil, iris_glow...)
EXTRAS = ()  # any of: horns, flowers, piercing, eyeshadow, cracks, freckles, lashes, fangs
EYESHADOW = (0.3, 0.02, 0.25, 1)
HORN_COLOR = (0.06, 0.02, 0.015, 1)
METAL = (0.9, 0.9, 0.92, 1)  # piercing metal
SEED = 0  # randomness for extras
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


MOUTH = np.array([0.22, -0.62])  # used with "fangs": the camera pulls back to show it
MOUTH_W, MOUTH_H_TOP, MOUTH_H_BOT = 0.17, 0.055, 0.07  # a snarl: lips drawn back


def mouth_q(x, y):
    u, v = x - MOUTH[0], y - MOUTH[1]
    hv = np.where(v > 0, MOUTH_H_TOP, MOUTH_H_BOT)
    pinch = np.maximum(1 - (u / MOUTH_W) ** 2, 0.05) ** 0.5
    return np.sqrt((u / MOUTH_W) ** 2 + (v / hv) ** 2 / pinch), u, v


def mouth_edge(u, top=True):
    """v of the mouth opening's upper / lower edge at u (approximate, for placing teeth)."""
    a = max(1 - (u / MOUTH_W) ** 2, 0.0)
    return (MOUTH_H_TOP if top else -MOUTH_H_BOT) * a ** 0.75


def dist_to_segment(x, y, p0, p1):
    p0, p1 = np.array(p0), np.array(p1)
    d = p1 - p0
    t = np.clip(((x - p0[0]) * d[0] + (y - p0[1]) * d[1]) / d.dot(d), 0, 1)
    return np.hypot(x - (p0[0] + t * d[0]), y - (p0[1] + t * d[1]))


def lower_face(x, y):
    """Nose tip, snarl crease, lips and the open mouth (only with fangs).
    Returns height, keep-face mask, lip paint and nostril shadow paint."""
    mq, u, v = mouth_q(x, y)
    up = smoothstep(-0.01, 0.01, v)
    dome = lambda ry: 0.35 * np.sqrt(np.clip(1 - ((x + 0.1) / 1.1) ** 2 - ((y - 0.05) / ry) ** 2, 0, 1))
    z = dome(1.4) - dome(0.9)  # the face carries on down to the chin
    z -= 0.2 * np.exp(-((x - 0.25) / 0.08) ** 2) * smoothstep(-0.25, -0.4, y)  # nose bridge ends
    z += 0.08 * np.exp(-((u / 0.3) ** 2 + ((v + 0.02) / 0.2) ** 2))  # muzzle
    z += 0.05 * np.exp(-((mq - 1.3) / 0.22) ** 2) * up  # upper lip, drawn back
    z += 0.065 * np.exp(-((mq - 1.35) / 0.25) ** 2) * (1 - up)  # full lower lip
    z -= 0.07 * smoothstep(1.3, 1.0, mq)  # lips roll into the mouth
    z -= 0.015 * np.exp(-(u / 0.02) ** 2) * smoothstep(0.06, 0.12, v) * smoothstep(0.3, 0.2, v)  # philtrum
    z += 0.18 * np.exp(-(((x - 0.27) / 0.09) ** 2 + ((y + 0.29) / 0.075) ** 2))  # nose tip
    z += 0.1 * np.exp(-(((x - 0.16) / 0.06) ** 2 + ((y + 0.33) / 0.05) ** 2))  # nose wing
    nq = np.sqrt(((x - 0.225) / 0.04) ** 2 + ((y + 0.365) / 0.02) ** 2)
    z -= 0.05 * smoothstep(1.6, 0.6, nq)  # nostril: a shaded hollow, not a hole
    z -= 0.03 * np.exp(-(dist_to_segment(x, y, (0.13, -0.34), (0.03, -0.6)) / 0.018) ** 2)  # snarl crease
    z += 0.03 * np.exp(-((x / 0.12) ** 2 + ((y + 0.38) / 0.1) ** 2))  # cheek bunched above it
    lips = np.exp(-((mq - 1.25) / 0.2) ** 2) * (mq < 1.8)
    nostril = 0.75 * smoothstep(1.2, 0.5, nq)
    return z, mq >= 1.0, lips, nostril


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
    # Optional extra paint: eyeshadow, porcelain cracks, freckles.
    zero = np.zeros_like(x)
    shadow = (up * smoothstep(1.0, 1.15, qe) * smoothstep(2.1, 1.4, qe)
              if "eyeshadow" in EXTRAS else zero)
    if "cracks" in EXTRAS:
        # Two families of wandering lines, broken into segments: reads as craquelure.
        a = np.abs(np.sin(x * 19 + 3 * np.sin(y * 7 + x * 3)))
        b = np.abs(np.sin(y * 23 + 3 * np.sin(x * 9 - y * 2)))
        gate_a = smoothstep(0.1, 0.4, np.sin(y * 11 + x * 5))
        gate_b = smoothstep(0.1, 0.4, np.sin(x * 13 - y * 4))
        cracks = np.maximum(smoothstep(0.09, 0.0, a) * gate_a, smoothstep(0.09, 0.0, b) * gate_b)
        cracks *= q >= 1.05
    else:
        cracks = zero
    freckles = zero.copy()
    if "freckles" in EXTRAS and len(x) > 1:
        rng = np.random.default_rng(SEED + 11)
        for fx, fy, fr in zip(rng.uniform(-0.4, 0.45, 220), rng.uniform(-0.4, 0.1, 220),
                              rng.uniform(0.003, 0.007, 220)):
            near = (np.abs(x - fx) < 3 * fr) & (np.abs(y - fy) < 3 * fr)
            freckles[near] += np.exp(-((x[near] - fx) ** 2 + (y[near] - fy) ** 2) / fr ** 2)
        freckles *= q >= 1.2
    keep = q >= 1.0
    if "fangs" in EXTRAS:
        zl, keep_l, lips, nostril = lower_face(x, y)
        z, keep, red, lashes = z + zl, keep & keep_l, red + lips, lashes + nostril
    return z, keep, np.stack([np.clip(red, 0, 1), np.clip(brow, 0, 1), lashes, shadow,
                                  cracks, np.clip(freckles, 0, 0.8)], axis=1)


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
    for k, attr_name in enumerate(("masks", "masks2")):
        attr = mesh.color_attributes.new(attr_name, "FLOAT_COLOR", "POINT")
        rgba = np.concatenate([masks[:, 3 * k:3 * k + 3], np.ones((len(x), 1))], axis=1)
        attr.data.foreach_set("color", rgba.astype(np.float32).ravel())
    mesh.shade_smooth()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def skin_material(name, base):
    mat, nt, bsdf = g.node_mat(name)
    n, link = nt.nodes, nt.links
    color = None
    layers = (("masks", ((0.55, 0.13, 0.09, 1), BROW_COLOR, (0.04, 0.02, 0.015, 1))),
              ("masks2", (EYESHADOW, (0.06, 0.04, 0.03, 1), (0.35, 0.17, 0.08, 1))))
    for attr_name, paints in layers:
        attr = n.new("ShaderNodeAttribute")
        attr.attribute_name = attr_name
        sep = n.new("ShaderNodeSeparateColor")
        link.new(attr.outputs["Color"], sep.inputs[0])
        for channel, paint in zip(("Red", "Green", "Blue"), paints):
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
    if "fangs" in EXTRAS:  # wider frame down to the chin
        obj = grid_mesh("Face", np.linspace(-0.45, 0.7, 900), np.linspace(-1.0, 0.45, 1000), face)
    else:
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
    eye.data.materials.append(g.eye_material(**EYE_LOOK))


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
    if "fangs" in EXTRAS:  # a drip running off the fang and down the chin
        fx, fy = MOUTH[0] + 0.07, MOUTH[1] - 0.085
        for t in np.linspace(0, 1, 30):
            ball(fx + 0.01 * math.sin(t * 4), fy - 0.17 * t, 0.012 * (1 - 0.3 * t) * (1 + 0.15 * math.sin(t * 15)))
        ball(fx, fy - 0.18, 0.014)
    obj.data.materials.append(tear_material())


def tear_material():
    """The tear's liquid. "glow" is radioactive: it also lights the lid and cheek around it."""
    if TEAR_KIND == "water":
        return g.water_material()
    mat, _, bsdf = g.node_mat("Tear " + TEAR_KIND)
    i = bsdf.inputs
    i["Roughness"].default_value = 0.05
    i["Coat Weight"].default_value = 1.0
    if TEAR_KIND == "glow":
        i["Base Color"].default_value = TEAR_COLOR
        i["Emission Color"].default_value = TEAR_COLOR
        i["Emission Strength"].default_value = 6.0
        i["Transmission Weight"].default_value = 0.4
    elif TEAR_KIND == "blood":
        i["Base Color"].default_value = (0.25, 0.0, 0.005, 1)
        i["Subsurface Weight"].default_value = 0.3
        i["Subsurface Radius"].default_value = (1.0, 0.1, 0.05)
    elif TEAR_KIND in ("gold", "chrome"):
        i["Base Color"].default_value = (1.0, 0.7, 0.25, 1) if TEAR_KIND == "gold" else (0.9, 0.9, 0.92, 1)
        i["Metallic"].default_value = 1.0
        i["Roughness"].default_value = 0.1
    elif TEAR_KIND == "ink":
        i["Base Color"].default_value = (0.01, 0.01, 0.012, 1)
    elif TEAR_KIND == "milk":
        i["Base Color"].default_value = (0.95, 0.93, 0.88, 1)
        i["Subsurface Weight"].default_value = 0.6
        i["Roughness"].default_value = 0.15
    elif TEAR_KIND == "honey":
        i["Base Color"].default_value = (0.9, 0.45, 0.05, 1)
        i["Transmission Weight"].default_value = 0.9
        i["IOR"].default_value = 1.5
    return mat


def simple_mat(name, color, roughness=0.4, metallic=0.0):
    mat, _, bsdf = g.node_mat(name)
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def build_horns():
    """Two ridged devil horns curling out of the forehead, either side of the title."""
    mat, nt, bsdf = g.node_mat("Horn")
    bsdf.inputs["Base Color"].default_value = HORN_COLOR
    bsdf.inputs["Roughness"].default_value = 0.3
    bsdf.inputs["Coat Weight"].default_value = 0.6
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.inputs["Scale"].default_value = 60.0
    wave.inputs["Distortion"].default_value = 3.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    for bx, by, d in ((-0.22, 0.24, -1), (0.26, 0.26, 1)):
        curve = bpy.data.curves.new("Horn", "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = 0.045
        curve.bevel_resolution = 4
        spl = curve.splines.new("NURBS")
        pts = [(0, 0, 0, 1.0), (0.02, 0.04, 0.08, 0.8), (0.06, 0.08, 0.13, 0.55),
               (0.11, 0.1, 0.15, 0.3), (0.15, 0.08, 0.15, 0.04)]  # stays inside the frame
        spl.points.add(len(pts) - 1)
        z0 = face_z(bx, by) - 0.01
        for p, (dx, dy, dz, r) in zip(spl.points, pts):
            p.co = (bx + d * dx, by + dy, z0 + dz, 1)
            p.radius = r
        spl.use_endpoint_u = True
        spl.order_u = 4
        obj = bpy.data.objects.new("Horn", curve)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(mat)


def build_flowers():
    """Little flowers sprouting from the cheek under the eye: a literal garden."""
    import colorsys
    rng = np.random.default_rng(SEED + 5)
    meshes = {}
    for name, verts in (("Petal", 16), ("Center", 12), ("Leaf", 16)):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=verts, ring_count=verts // 2, radius=1.0)
        proto = bpy.context.active_object
        bpy.ops.object.shade_smooth()
        meshes[name] = proto.data
        bpy.data.objects.remove(proto)

    for mesh in meshes.values():
        mesh.materials.append(None)  # one slot; each object picks its own material

    def place(mesh, mat, loc, scale, rot):
        obj = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.collection.objects.link(obj)
        obj.location, obj.scale, obj.rotation_euler = loc, scale, (0, 0, rot)
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = mat

    leaf_mat = simple_mat("Leaf", (0.05, 0.3, 0.05, 1), 0.35)
    center_mat = simple_mat("Flower center", (1.0, 0.75, 0.1, 1), 0.5)
    for _ in range(10):
        fx, fy = rng.uniform(-0.25, 0.3), rng.uniform(-0.33, -0.04)
        z = face_z(fx, fy) + 0.006
        hue = rng.uniform(0, 1)
        petal_mat = simple_mat("Petal", (*colorsys.hsv_to_rgb(hue, 0.45, 1.0), 1), 0.45)
        size = rng.uniform(0.8, 1.3)
        a0 = rng.uniform(0, 2 * math.pi)
        for k in range(2):
            la = a0 + math.pi * k + 0.6
            place(meshes["Leaf"], leaf_mat,
                  (fx + 0.025 * size * math.cos(la), fy + 0.025 * size * math.sin(la), z - 0.002),
                  (0.026 * size, 0.01 * size, 0.003), la)
        for k in range(6):
            pa = a0 + k * math.pi / 3
            place(meshes["Petal"], petal_mat,
                  (fx + 0.015 * size * math.cos(pa), fy + 0.015 * size * math.sin(pa), z),
                  (0.017 * size, 0.009 * size, 0.004), pa)
        place(meshes["Center"], center_mat, (fx, fy, z + 0.003), (0.008 * size,) * 2 + (0.005,), 0)


def build_piercing():
    """Rings through the outer end of the brow, and a stud under it."""
    mat = simple_mat("Metal", METAL, 0.12, 1.0)
    for k, bx in enumerate((-0.17, -0.12)):
        by = float(brow_line(bx)) + 0.01
        bpy.ops.mesh.primitive_torus_add(major_radius=0.03, minor_radius=0.005,
                                         major_segments=48, minor_segments=12)
        ring = bpy.context.active_object
        ring.location = (bx, by, face_z(bx, by) + 0.012)  # above the brow hairs
        ring.rotation_euler = (math.radians(80), 0, math.radians(10 * k))
        bpy.ops.object.shade_smooth()
        ring.data.materials.append(mat)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.006, location=(-0.22, -0.05, face_z(-0.22, -0.05) + 0.004))
    bpy.ops.object.shade_smooth()
    bpy.context.active_object.data.materials.append(mat)


def build_fangs():
    """Teeth in the snarling mouth, two long fangs hanging over the lower lip."""
    X, Y = np.meshgrid(np.linspace(MOUTH[0] - 0.25, MOUTH[0] + 0.25, 300),
                       np.linspace(MOUTH[1] - 0.12, MOUTH[1] + 0.1, 200))
    z, _, _ = face(X.ravel(), Y.ravel())
    mq = mouth_q(X.ravel(), Y.ravel())[0]
    rim = z[(mq > 1) & (mq < 1.05)].min()
    bpy.ops.mesh.primitive_plane_add(size=0.45, location=(MOUTH[0], MOUTH[1] + 0.05, rim - 0.08))
    bpy.context.active_object.data.materials.append(simple_mat("Mouth", (0.06, 0.005, 0.005, 1), 0.3))
    ivory = simple_mat("Teeth", (0.92, 0.88, 0.78, 1), 0.2)
    for top, row in ((True, np.linspace(-0.12, 0.12, 7)), (False, np.linspace(-0.1, 0.1, 6))):
        for u in row:
            v = mouth_edge(u, top) + (-0.018 if top else 0.014)
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(MOUTH[0] + u, MOUTH[1] + v, rim - 0.02))
            tooth = bpy.context.active_object
            tooth.scale = (0.032, 0.04 if top else 0.03, 0.03)
            tooth.modifiers.new("Round", "BEVEL").width = 0.008
            bpy.ops.object.shade_smooth()
            tooth.data.materials.append(ivory)
    for u in (-0.075, 0.075):
        base = Vector((MOUTH[0] + u, MOUTH[1] + mouth_edge(u) - 0.01, rim - 0.01))
        tx, ty = MOUTH[0] + u * 0.95, MOUTH[1] - 0.085
        tip = Vector((tx, ty, face_z(tx, ty) + 0.012))  # over the lower lip
        d = tip - base
        bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.018, radius2=0.0008, depth=d.length,
                                        location=(base + tip) / 2)
        fang = bpy.context.active_object
        fang.rotation_mode = "QUATERNION"
        fang.rotation_quaternion = d.to_track_quat("-Z", "Y")  # cone tip points along -Z
        bpy.ops.object.shade_smooth()
        fang.data.materials.append(ivory)


def build_lashes():
    """Long dark lashes fanning up off the upper lid."""
    rng = np.random.default_rng(SEED + 7)
    curve = bpy.data.curves.new("Lashes", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.0013
    c, s = math.cos(EYE_TILT), math.sin(EYE_TILT)
    for t in np.linspace(-0.92, 0.92, 70):
        u = t * EYE_W
        v = EYE_H_TOP * (1 - t * t) ** 0.75 + 0.003
        du, dv = 0.7 * t, 1.0
        norm = math.hypot(du, dv)
        length = rng.uniform(0.03, 0.05) * (1 - 0.4 * abs(t))
        spl = curve.splines.new("NURBS")
        spl.points.add(3)
        for k, p in enumerate(spl.points):
            tt = k / 3
            lu, lv = u + du / norm * length * tt, v + dv / norm * length * tt * (1 - 0.3 * tt)
            x, y = EYE[0] + c * lu - s * lv, EYE[1] + s * lu + c * lv
            p.co = (x, y, face_z(x, y) + 0.003 + 0.025 * tt * tt, 1)
            p.radius = 1.0 - 0.8 * tt
        spl.use_endpoint_u = True
        spl.order_u = 3
    obj = bpy.data.objects.new("Lashes", curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(simple_mat("Lashes", (0.015, 0.01, 0.01, 1), 0.4))


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
    if "fangs" in EXTRAS:
        cam.location = (0.12, -0.27, 2.9)  # pulled back to show the mouth
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
    for extra, build in (("horns", build_horns), ("flowers", build_flowers),
                         ("piercing", build_piercing), ("lashes", build_lashes), ("fangs", build_fangs)):
        if extra in EXTRAS:
            build()
    if title:
        g.TITLE_COLOR = (0.32, 0.15, 0.03, 1)  # dark old gold, reads against the skin
        g.build_title()
        obj = bpy.data.objects["Title"]
        obj.location = (0.02, 0.33, 0.7)
        obj.scale = (0.3, 0.3, 0.3)
        if "fangs" in EXTRAS:  # wider frame: up in the corner over the far side of the face
            obj.location, obj.scale = (0.37, 0.19, 0.7), (0.28, 0.28, 0.28)
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
