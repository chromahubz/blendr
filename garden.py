"""Glossy pink flesh-and-eye scene with "garden" title, built procedurally in Blender.

Run with Blender:      blender -b -P garden.py -- out.png
Or with the bpy module: python garden.py out.png
Also saves garden.blend next to the image so you can open and tweak it.
"""
import math
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "fonts", "UnifrakturMaguntia-Book.ttf")
TITLE = "garden"

# Eye opening (almond) in world XY.
EYE = np.array([-0.05, -0.08])
EYE_TILT = math.radians(-6)
EYE_W, EYE_H_TOP, EYE_H_BOT = 0.42, 0.17, 0.075


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def almond_q(x, y):
    """~1 on the eyelid edge, <1 inside the opening, >1 outside."""
    dx, dy = x - EYE[0], y - EYE[1]
    c, s = math.cos(-EYE_TILT), math.sin(-EYE_TILT)
    u, v = c * dx - s * dy, s * dx + c * dy
    hv = np.where(v > 0, EYE_H_TOP, EYE_H_BOT)
    pinch = np.maximum(1 - (u / EYE_W) ** 2, 0.05) ** 0.5  # pointed corners
    return np.sqrt((u / EYE_W) ** 2 + (v / hv) ** 2 / pinch), v


def height(x, y):
    q, v = almond_q(x, y)
    upper = smoothstep(-0.02, 0.05, v)

    # Puffy lids, a crease above the upper lid, and the rim curling into the socket.
    z = 0.10 * np.exp(-((q - 1.15) / 0.3) ** 2)
    z += 0.07 * np.exp(-((q - 1.9) / 0.3) ** 2) * upper
    z -= 0.14 * smoothstep(1.2, 1.0, q)

    # Big swirling folds arcing around the eye, fading out near it.
    cx, cy = 0.1, 0.05
    r = np.sqrt(((x - cx) / 1.0) ** 2 + ((y - cy) / 0.72) ** 2)
    th = np.arctan2(y - cy, x - cx)
    swirl = np.sin(r * 10.0 - 1.2 + 0.8 * np.sin(th * 2.0 + r * 3.0))
    swirl += 0.4 * np.sin(r * 23.0 + 1.5 * np.sin(th * 3.0 - r * 4.0))
    z += 0.07 * swirl * smoothstep(1.6, 2.8, q)

    # Soft organic lumps everywhere, a big mound lower-left.
    z += 0.015 * np.sin(x * 13 + np.sin(y * 7)) * np.sin(y * 11 + np.sin(x * 5))
    z += 0.12 * np.exp(-((x + 0.7) ** 2 + (y + 0.75) ** 2) / 0.25)
    return z


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def node_mat(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    return mat, nt, nt.nodes["Principled BSDF"]


def flesh_material():
    mat, nt, bsdf = node_mat("Flesh")
    n, link = nt.nodes, nt.links
    coord = n.new("ShaderNodeTexCoord")
    sep = n.new("ShaderNodeSeparateXYZ")
    link.new(coord.outputs["Object"], sep.inputs[0])
    # Diagonal gradient: pale pink lower-left -> hot red upper-right.
    add = n.new("ShaderNodeMath")
    add.operation = "ADD"
    link.new(sep.outputs["X"], add.inputs[0])
    link.new(sep.outputs["Y"], add.inputs[1])
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.15
    ramp.color_ramp.elements[0].color = (1.0, 0.42, 0.52, 1)
    ramp.color_ramp.elements[1].position = 0.85
    ramp.color_ramp.elements[1].color = (0.62, 0.01, 0.07, 1)
    mid = ramp.color_ramp.elements.new(0.5)
    mid.color = (0.95, 0.12, 0.28, 1)
    mapr = n.new("ShaderNodeMapRange")
    mapr.inputs["From Min"].default_value = -1.6
    mapr.inputs["From Max"].default_value = 1.6
    link.new(add.outputs[0], mapr.inputs["Value"])
    link.new(mapr.outputs[0], ramp.inputs[0])
    # Deepen crevices toward red.
    ao = n.new("ShaderNodeAmbientOcclusion")
    ao.inputs["Distance"].default_value = 0.15
    mix = n.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["B"].default_value = (0.55, 0.0, 0.06, 1)
    inv = n.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    link.new(ao.outputs["AO"], inv.inputs[1])
    link.new(inv.outputs[0], mix.inputs["Factor"])
    link.new(ramp.outputs["Color"], mix.inputs["A"])
    link.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    link.new(mix.outputs["Result"], bsdf.inputs["Subsurface Radius"])
    bsdf.inputs["Roughness"].default_value = 0.22
    bsdf.inputs["Subsurface Weight"].default_value = 0.15
    bsdf.inputs["Subsurface Scale"].default_value = 0.05
    bsdf.inputs["Coat Weight"].default_value = 0.8
    bsdf.inputs["Coat Roughness"].default_value = 0.04
    return mat


def eye_material():
    mat, nt, bsdf = node_mat("Eye")
    n, link = nt.nodes, nt.links
    coord = n.new("ShaderNodeTexCoord")
    sep = n.new("ShaderNodeSeparateXYZ")
    link.new(coord.outputs["Object"], sep.inputs[0])
    comb = n.new("ShaderNodeCombineXYZ")
    link.new(sep.outputs["X"], comb.inputs["X"])
    link.new(sep.outputs["Y"], comb.inputs["Y"])
    radial = n.new("ShaderNodeVectorMath")
    radial.operation = "LENGTH"
    link.new(comb.outputs[0], radial.inputs[0])
    ramp = n.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (0.01, 0.01, 0.01, 1)
    els[1].position, els[1].color = 0.44, (0.93, 0.91, 0.9, 1)
    for pos, col in [(0.15, (0.01, 0.01, 0.01, 1)), (0.17, (0.2, 0.3, 0.14, 1)),
                     (0.3, (0.45, 0.6, 0.35, 1)), (0.39, (0.2, 0.25, 0.17, 1)),
                     (0.42, (0.06, 0.07, 0.05, 1))]:
        e = els.new(pos)
        e.color = col
    link.new(radial.outputs["Value"], ramp.inputs[0])
    # Iris fibres.
    noise = n.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 40.0
    link.new(coord.outputs["Object"], noise.inputs["Vector"])
    fib = n.new("ShaderNodeMix")
    fib.data_type = "RGBA"
    fib.blend_type = "MULTIPLY"
    fib.inputs["Factor"].default_value = 0.3
    link.new(ramp.outputs["Color"], fib.inputs["A"])
    link.new(noise.outputs["Color"], fib.inputs["B"])
    # Back hemisphere is plain sclera.
    back = n.new("ShaderNodeMath")
    back.operation = "LESS_THAN"
    back.inputs[1].default_value = 0.0
    link.new(sep.outputs["Z"], back.inputs[0])
    out = n.new("ShaderNodeMix")
    out.data_type = "RGBA"
    out.inputs["B"].default_value = (0.93, 0.91, 0.9, 1)
    link.new(back.outputs[0], out.inputs["Factor"])
    link.new(fib.outputs["Result"], out.inputs["A"])
    link.new(out.outputs["Result"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.3
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.0
    return mat


def water_material():
    mat, nt, bsdf = node_mat("Tears")
    bsdf.inputs["Base Color"].default_value = (1.0, 0.92, 0.95, 1)
    bsdf.inputs["Transmission Weight"].default_value = 0.85
    bsdf.inputs["Roughness"].default_value = 0.0
    bsdf.inputs["IOR"].default_value = 1.33
    return mat


def title_material():
    mat, nt, bsdf = node_mat("Title")
    lime = (0.08, 0.55, 0.0, 1)
    bsdf.inputs["Base Color"].default_value = lime
    bsdf.inputs["Emission Color"].default_value = lime
    bsdf.inputs["Emission Strength"].default_value = 0.15
    bsdf.inputs["Roughness"].default_value = 0.3
    return mat


def build_flesh(n=520, extent=1.4):
    xs = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(xs, xs)
    x, y = X.ravel(), Y.ravel()
    z = height(x, y)
    q, _ = almond_q(x, y)
    keep = q >= 1.0
    idx = np.arange(n * n).reshape(n, n)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    ok = keep[a] & keep[b] & keep[c] & keep[d]  # cut the eye opening
    faces = np.stack([a, b, c, d], axis=1)[ok]
    mesh = bpy.data.meshes.new("Flesh")
    mesh.from_pydata(np.stack([x, y, z], axis=1).tolist(), [], faces.tolist())
    mesh.shade_smooth()
    obj = bpy.data.objects.new("Flesh", mesh)
    bpy.context.collection.objects.link(obj)
    sol = obj.modifiers.new("Lid thickness", "SOLIDIFY")
    sol.thickness = 0.06
    sol.offset = -1
    sub = obj.modifiers.new("Smooth", "SUBSURF")
    sub.levels, sub.render_levels = 0, 1
    obj.data.materials.append(flesh_material())
    return obj


def build_eye():
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=1.0)
    eye = bpy.context.active_object
    eye.name = "Eye"
    eye.scale = (0.33, 0.33, 0.33)
    eye.location = (EYE[0], EYE[1], -0.27)
    eye.rotation_euler = (math.radians(-4), math.radians(-12), 0)  # glance left
    bpy.ops.object.shade_smooth()
    eye.data.materials.append(eye_material())


def build_tears():
    mb = bpy.data.metaballs.new("Tears")
    mb.resolution = mb.render_resolution = 0.006
    obj = bpy.data.objects.new("Tears", mb)
    bpy.context.collection.objects.link(obj)

    def ball(x, y, r, lift=0.6):
        el = mb.elements.new()
        el.co = (x, y, float(height(np.array([x]), np.array([y]))[0]) + r * lift)
        el.radius = r * 1.4

    # Wet film along the lower lid.
    for t in np.linspace(-0.75, 0.75, 45):
        u = t * EYE_W
        v = -EYE_H_BOT * math.sqrt(max(1 - t * t, 0.0)) - 0.025
        c, s = math.cos(EYE_TILT), math.sin(EYE_TILT)
        ball(EYE[0] + c * u - s * v, EYE[1] + s * u + c * v, 0.022 * (1 - 0.5 * abs(t)))
    # Tear bead spilling over the lid and streaming down-right.
    path = [(0.05, -0.19, 0.05), (0.07, -0.25, 0.035), (0.09, -0.32, 0.03),
            (0.11, -0.4, 0.028), (0.14, -0.48, 0.032), (0.17, -0.55, 0.026),
            (0.2, -0.62, 0.03), (0.24, -0.68, 0.024)]
    for (x0, y0, r0), (x1, y1, r1) in zip(path, path[1:]):
        for t in np.linspace(0, 1, 5, endpoint=False):
            ball(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, r0 + (r1 - r0) * t)
    # Loose droplets and a splash further down.
    for x, y, r in [(0.3, -0.75, 0.028), (0.36, -0.79, 0.02), (0.44, -0.8, 0.026),
                    (0.5, -0.86, 0.018), (0.58, -0.84, 0.022), (0.27, -0.84, 0.016),
                    (0.4, -0.9, 0.014), (0.65, -0.9, 0.015), (-0.02, -0.3, 0.018)]:
        ball(x, y, r)
    obj.data.materials.append(water_material())


def build_title():
    curve = bpy.data.curves.new("Title", "FONT")
    curve.body = TITLE
    curve.font = bpy.data.fonts.load(FONT)
    curve.align_x, curve.align_y = "CENTER", "CENTER"
    curve.size = 0.34
    curve.extrude = 0.015
    curve.bevel_depth = 0.004
    obj = bpy.data.objects.new("Title", curve)
    bpy.context.collection.objects.link(obj)
    obj.location = (0.0, 0.8, 0.25)
    obj.data.materials.append(title_material())


def build_lights_and_camera(scene):
    def area(name, loc, size, size_y, energy, color, target=(0, 0, 0)):
        data = bpy.data.lights.new(name, "AREA")
        data.shape = "RECTANGLE"
        data.size, data.size_y = size, size_y
        data.energy, data.color = energy, color
        obj = bpy.data.objects.new(name, data)
        obj.location = loc
        scene.collection.objects.link(obj)
        d = np.array(target) - np.array(loc)
        obj.rotation_euler = (0, 0, 0)
        obj.rotation_mode = "QUATERNION"
        from mathutils import Vector
        obj.rotation_quaternion = Vector(d).to_track_quat("-Z", "Y")

    area("Key", (-2.0, 2.2, 3.0), 2.5, 1.2, 350, (1.0, 0.95, 0.95))
    area("Fill", (2.5, -1.5, 1.5), 3.0, 3.0, 120, (1.0, 0.6, 0.65))
    # Long thin strip for the wet streak highlights.
    area("Streak", (0.5, 2.5, 2.5), 4.0, 0.15, 250, (1.0, 1.0, 1.0))

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.9, 0.45, 0.5, 1)
    bg.inputs["Strength"].default_value = 0.2
    scene.world = world

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 72
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = (0.0, 0.0, 4.0)
    scene.collection.objects.link(cam)
    scene.camera = cam


def main(out_png):
    scene = reset_scene()
    build_flesh()
    build_eye()
    build_tears()
    build_title()
    build_lights_and_camera(scene)

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = 1024
    scene.view_settings.view_transform = "Standard"
    scene.render.filepath = out_png

    bpy.ops.wm.save_as_mainfile(filepath=os.path.splitext(out_png)[0] + ".blend")
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(os.path.abspath(args[0] if args else "garden.png"))
