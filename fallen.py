"""Painterly 3D take on the glaring, tearful eye peering over a forearm
(after Cabanel's "The Fallen Angel", 1847), with the "garden" title.

Run with Blender:      blender -b -P fallen.py -- fallen.png
Or with the bpy module: python fallen.py fallen.png
"""
import math
import os
import random
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import garden as g  # noqa: E402  (eye, tear and title builders)

EYE = np.array([-0.02, 0.05])
EYE_TILT = math.radians(-4)
EYE_W, EYE_H_TOP, EYE_H_BOT = 0.17, 0.06, 0.045  # narrowed, glaring
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
    return 0.2 - 0.11 * smoothstep(-0.3, 0.15, x)  # slants down toward the nose: a frown


def hairline(x):
    return (0.33 + 0.03 * np.sin(x * 9) - 0.35 * smoothstep(-0.5, -0.7, x)
            - 0.45 * smoothstep(0.33, 0.48, x))


def face(x, y):
    """Height of the face plus paint masks (lid redness, eyebrow, hair)."""
    q, v = almond_q(x, y)
    z = 0.35 * np.sqrt(np.clip(1 - ((x + 0.1) / 1.1) ** 2 - ((y - 0.05) / 0.9) ** 2, 0, 1))
    z -= 0.22 * smoothstep(0.3, 0.6, x)  # far side of the face turns away
    z += 0.2 * np.exp(-((x - 0.25) / 0.08) ** 2) * smoothstep(0.25, 0.05, y)  # nose bridge
    yb = brow_line(x)
    z += 0.14 * np.exp(-((y - yb) / 0.06) ** 2) * smoothstep(-0.6, -0.3, x) * smoothstep(0.3, 0.15, x)
    z -= 0.05 * np.exp(-((x - 0.17) / 0.018) ** 2) * np.exp(-((y - 0.13) / 0.06) ** 2)  # furrow
    z -= 0.11 * np.exp(-(((x - EYE[0]) / 0.2) ** 2 + ((y - EYE[1]) / 0.1) ** 2))  # socket
    z += 0.04 * np.exp(-((q - 1.2) / 0.25) ** 2)  # lids
    z += 0.03 * np.exp(-((q - 1.75) / 0.2) ** 2) * (v > 0)  # heavy upper fold
    z += 0.03 * np.exp(-((q - 1.3) / 0.2) ** 2) * (v < 0)  # puffy lower lid
    z -= 0.06 * smoothstep(1.2, 1.0, q)  # rim curls into the socket
    z += 0.06 * np.exp(-(((x + 0.1) / 0.25) ** 2 + ((y + 0.12) / 0.12) ** 2))  # cheek
    z += 0.006 * np.sin(x * 40 + np.sin(y * 30)) * np.sin(y * 37)  # skin unevenness

    u, _ = eye_local(x, y)
    red = np.exp(-((q - 1.12) / 0.12) ** 2) * (v < 0.01)
    red += 0.8 * np.exp(-(((u - EYE_W) / 0.04) ** 2 + (v / 0.04) ** 2))  # inner corner
    red += 0.5 * np.exp(-((q - 1.06) / 0.06) ** 2) * (v > 0)  # upper lash line
    brow = (np.exp(-((y - yb - 0.012) / (0.03 + 0.015 * smoothstep(-0.3, 0.12, x))) ** 2)
            * smoothstep(-0.36, -0.26, x) * smoothstep(0.17, 0.1, x)
            * (0.8 + 0.2 * np.sin(x * 37 + np.sin(y * 23))))
    hair = smoothstep(hairline(x) - 0.03, hairline(x) + 0.04, y)
    return z, q >= 1.0, np.stack([np.clip(red, 0, 1), np.clip(brow, 0, 1), hair], axis=1)


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
    for channel, paint in (("Red", (0.55, 0.13, 0.09, 1)), ("Green", (0.1, 0.045, 0.025, 1)),
                           ("Blue", (0.13, 0.04, 0.018, 1))):
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
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Subsurface Weight"].default_value = 0.25
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
    bsdf.inputs["Subsurface Scale"].default_value = 0.03
    return mat


def build_face():
    obj = grid_mesh("Face", np.linspace(-1.0, 1.0, 600), np.linspace(-0.5, 1.15, 500), face)
    obj.data.materials.append(skin_material("Face skin", (0.78, 0.56, 0.42, 1)))
    sub = obj.modifiers.new("Smooth", "SUBSURF")
    sub.levels, sub.render_levels = 0, 1


def build_eye():
    g.IRIS = ((0.12, 0.16, 0.2, 1), (0.42, 0.52, 0.62, 1), (0.12, 0.15, 0.19, 1))  # grey-blue
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=1.0)
    eye = bpy.context.active_object
    eye.name = "Eye"
    eye.scale = (0.17, 0.17, 0.1)  # flattened so it stays behind the low lower lid
    # Sink it just below the lowest point of the lid rim (ellipsoid is ~0.11 deep once turned).
    X, Y = np.meshgrid(np.linspace(-0.25, 0.25, 300), np.linspace(-0.15, 0.25, 300))
    z, _, _ = face(X.ravel(), Y.ravel())
    q, _ = almond_q(X.ravel(), Y.ravel())
    rim_min = z[(q > 1) & (q < 1.05)].min()
    eye.location = (EYE[0], EYE[1] + 0.005, rim_min - 0.12)
    eye.rotation_euler = (math.radians(4), math.radians(-14), 0)  # side-eye toward the viewer's left
    bpy.ops.object.shade_smooth()
    eye.data.materials.append(g.eye_material())


def build_tear():
    mb = bpy.data.metaballs.new("Tear")
    mb.resolution = mb.render_resolution = 0.003
    obj = bpy.data.objects.new("Tear", mb)
    obj.visible_shadow = False
    bpy.context.collection.objects.link(obj)
    x0, y0 = EYE[0] + 0.035, EYE[1] - 0.05
    for t in np.linspace(0, 1, 14):  # bead on the lid, running down the cheek
        x, y, r = x0 + 0.01 * t, y0 - 0.075 * t, 0.016 * (1 - 0.3 * t)
        el = mb.elements.new()
        el.co = (x, y, face_z(x, y) + r * 0.9)
        el.radius = r * 1.4
    obj.data.materials.append(g.water_material())


def build_hair(rng):
    mat, nt, bsdf = g.node_mat("Hair")
    n, link = nt.nodes, nt.links
    info = n.new("ShaderNodeObjectInfo")
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.05, 0.014, 0.006, 1)
    ramp.color_ramp.elements[1].color = (0.3, 0.085, 0.03, 1)
    link.new(info.outputs["Random"], ramp.inputs[0])
    link.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Coat Weight"].default_value = 0.2

    placed = 0
    while placed < 260:
        cx, cy = rng.uniform(-0.85, 0.85), rng.uniform(-0.3, 1.1)
        if cy < hairline(cx) + 0.04:
            continue
        placed += 1
        curve = bpy.data.curves.new("Curl", "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = rng.uniform(0.012, 0.022)
        curve.bevel_resolution = 3
        spl = curve.splines.new("NURBS")
        npts = 14
        spl.points.add(npts - 1)
        big, a0 = rng.uniform(0.06, 0.13), rng.uniform(0, 2 * math.pi)
        turns, spin = rng.uniform(0.8, 1.6), rng.choice([-1, 1])
        for k, p in enumerate(spl.points):
            t = k / (npts - 1)
            ang = a0 + spin * t * turns * 2 * math.pi
            r = big * (1 - 0.65 * t)
            x, y = cx + r * math.cos(ang), cy + r * math.sin(ang)
            p.co = (x, y, face_z(x, y) + 0.01 + 0.03 * (1 - t), 1)
            p.radius = 1.2 - 0.8 * t
        spl.use_endpoint_u = True
        spl.order_u = 4
        obj = bpy.data.objects.new("Curl", curve)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(mat)


def build_wing(rng):
    mat, _, bsdf = g.node_mat("Wing")
    bsdf.inputs["Base Color"].default_value = (0.05, 0.16, 0.65, 1)
    bsdf.inputs["Roughness"].default_value = 0.45
    for k in range(9):
        x, y = rng.uniform(0.5, 0.82), rng.uniform(-0.16, 0.03)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0)
        f = bpy.context.active_object
        f.name = "Feather"
        f.scale = (rng.uniform(0.12, 0.2), rng.uniform(0.03, 0.045), 0.02)
        f.location = (x, y, face_z(x, y) + 0.08 + 0.01 * k)
        f.rotation_euler = (0, 0, math.radians(rng.uniform(-30, -10)))
        bpy.ops.object.shade_smooth()
        f.data.materials.append(mat)


def build_arm(name, p0, p1, radius, zc):
    """A muscular limb lying across the frame, axis from p0 to p1 in XY."""
    p0, p1 = np.array(p0), np.array(p1)
    axis = p1 - p0
    length = np.linalg.norm(axis)
    along, side = axis / length, np.array([-axis[1], axis[0]]) / length  # side points "up"
    ts, phis = np.linspace(0, 1, 320), np.linspace(-0.3, math.pi, 140)
    T, P = np.meshgrid(ts, phis)
    s = T * length
    r = radius * (1 + 0.05 * np.sin(s * 4 + 1) + 0.03 * np.sin(s * 9) + 0.02 * np.sin(P * 3 + s * 5))
    cx = p0[0] + along[0] * s + side[0] * r * np.cos(P)
    cy = p0[1] + along[1] * s + side[1] * r * np.cos(P)
    cz = zc + r * np.sin(P)
    n, m = P.shape
    idx = np.arange(n * m).reshape(n, m)
    faces = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(np.stack([cx.ravel(), cy.ravel(), cz.ravel()], 1).tolist(), [], faces.tolist())
    attr = mesh.color_attributes.new("masks", "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", np.tile([0, 0, 0, 1], n * m).astype(np.float32))
    mesh.shade_smooth()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(skin_material(name + " skin", (0.82, 0.62, 0.45, 1)))


def build_lights_and_camera(scene):
    def area(name, loc, size, energy, color):
        data = bpy.data.lights.new(name, "AREA")
        data.size, data.energy, data.color = size, energy, color
        obj = bpy.data.objects.new(name, data)
        obj.location = loc
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = (-Vector(loc)).to_track_quat("-Z", "Y")
        scene.collection.objects.link(obj)

    area("Key", (-3.0, 1.2, 1.8), 1.5, 650, (1.0, 0.84, 0.66))
    area("Fill", (2.0, -0.5, 2.0), 3.0, 15, (0.6, 0.7, 1.0))
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.03, 0.018, 1)
    scene.world = world
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_fit, cam_data.sensor_height, cam_data.lens = "VERTICAL", 36, 72
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = (0, 0.05, 4.0)
    scene.collection.objects.link(cam)
    scene.camera = cam


def build_compositor(scene, size):
    """Kuwahara filter: smooths detail into brush-stroke-like patches."""
    nt = bpy.data.node_groups.new("Painterly", "CompositorNodeTree")
    nt.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = nt.nodes.new("CompositorNodeRLayers")
    kw = nt.nodes.new("CompositorNodeKuwahara")
    kw.inputs["Size"].default_value = max(3, size // 90)
    out = nt.nodes.new("NodeGroupOutput")
    nt.links.new(rl.outputs["Image"], kw.inputs["Image"])
    nt.links.new(kw.outputs[0], out.inputs[0])
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
    weave = 1 + 0.015 * np.sin(xx * f) * np.sin(yy * f)
    rng = np.random.default_rng(1)
    weave += rng.normal(0, 0.015, (h, w))
    vig = 1 - 0.35 * (((xx / w - 0.5) * 1.6) ** 2 + ((yy / h - 0.5) * 1.4) ** 2)
    px[..., :3] *= (weave * vig)[..., None] * np.array([1.0, 0.95, 0.84], np.float32)
    img.pixels.foreach_set(np.clip(px, 0, 1).ravel())
    img.filepath_raw, img.file_format = path, "PNG"
    img.save()


def main(out_png, size=1024, samples=64):
    scene = g.reset_scene()
    rng = random.Random(7)
    build_face()
    build_eye()
    build_tear()
    build_hair(rng)
    build_wing(rng)
    build_arm("Forearm", (-1.1, -0.58), (1.1, -0.5), 0.4, 0.45)
    build_arm("Shoulder", (-1.2, -0.95), (0.7, -1.45), 0.42, 0.95)
    g.TITLE_COLOR = (0.85, 0.68, 0.38, 1)  # old gold
    g.build_title()
    bpy.data.objects["Title"].location = (0.0, 0.7, 0.75)
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
    main(os.path.abspath(args[0] if args else "fallen.png"))
