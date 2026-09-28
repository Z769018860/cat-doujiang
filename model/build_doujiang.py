"""Build a reference based, editable Doujiang cat blockout as GLB.

Pure Python 3 standard library. Coordinates: head at -X, left side +Y, Z up.
This is an art blockout, not a scan or a production rig.
"""
import json
import math
import struct
from pathlib import Path

WHITE = (0.90, 0.89, 0.83)
TABBY = (0.30, 0.27, 0.22)
LIGHT = (0.48, 0.40, 0.30)
DARK = (0.16, 0.15, 0.14)
EAR = (0.55, 0.34, 0.32)
GREEN = (0.44, 0.55, 0.27)
PINK = (0.60, 0.36, 0.35)

meshes = []


def ellipsoid(name, center, radius, color, paint=None, rings=18, sides=32):
    """UV ellipsoid with per vertex coat coloring; paint receives normalized XYZ."""
    vertices, colors, triangles = [], [], []
    for i in range(rings + 1):
        phi = math.pi * i / rings
        for j in range(sides + 1):
            theta = 2 * math.pi * j / sides
            xyz = (math.sin(phi) * math.cos(theta),
                   math.sin(phi) * math.sin(theta), math.cos(phi))
            vertices.append(tuple(center[k] + radius[k] * xyz[k] for k in range(3)))
            colors.append(paint(xyz) if paint else color)
    for i in range(rings):
        for j in range(sides):
            a = i * (sides + 1) + j
            triangles += [a, a + sides + 1, a + 1,
                          a + 1, a + sides + 1, a + sides + 2]
    meshes.append((name, vertices, colors, triangles))


def body_paint(p):
    x, y, z = p
    # White belly and legs; dark saddle above with a white shoulder gap.
    threshold = -0.06 + 0.09 * math.sin(5 * x + 2 * y)
    if z < threshold or (x < -0.36 and x > -0.62):
        return WHITE
    stripe = math.sin(29 * x + 5 * y + 4 * z)
    return DARK if stripe > 0.70 else (LIGHT if stripe < -0.65 else TABBY)


def face_paint(p):
    x, y, z = p
    if z < 0.15 or x < -0.65 and abs(y) < 0.30 + 0.12 * z:
        return WHITE
    if math.sin(18 * y + 8 * x) > 0.65:
        return DARK
    return TABBY


def tapered_segment(name, start, end, r0, r1, color, sides=12):
    """Cylinder mesh with independent material, suitable for later separation."""
    axis = tuple(end[k] - start[k] for k in range(3))
    norm = math.sqrt(sum(v * v for v in axis))
    d = tuple(v / norm for v in axis)
    u = (-d[1], d[0], 0)
    un = math.sqrt(sum(v * v for v in u))
    if un < 0.0001:
        u = (1, 0, 0)
    else:
        u = tuple(v / un for v in u)
    v = (d[1] * u[2] - d[2] * u[1],
         d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0])
    positions = []
    for origin, radius in ((start, r0), (end, r1)):
        for j in range(sides):
            t = 2 * math.pi * j / sides
            positions.append(tuple(origin[k] + radius * (u[k] * math.cos(t) + v[k] * math.sin(t)) for k in range(3)))
    indices = []
    for j in range(sides):
        nxt = (j + 1) % sides
        indices += [j, nxt, sides + j, nxt, sides + nxt, sides + j]
    meshes.append((name, positions, [color] * len(positions), indices))


# Relative proportions are estimated from the supplied photographs.
ellipsoid('body_saddle_white_belly', (0.12, 0, 0.79), (0.69, 0.30, 0.35), WHITE, body_paint)
ellipsoid('chest_white', (-0.46, 0, 0.72), (0.34, 0.28, 0.36), WHITE)
ellipsoid('head_tabby_white_blaze', (-0.75, 0, 1.08), (0.225, 0.22, 0.205), WHITE, face_paint)
ellipsoid('muzzle_left', (-0.955, 0.075, 0.995), (0.095, 0.090, 0.075), WHITE)
ellipsoid('muzzle_right', (-0.955, -0.075, 0.995), (0.095, 0.090, 0.075), WHITE)
ellipsoid('pink_dark_nose', (-1.035, 0, 1.035), (0.040, 0.050, 0.029), PINK)

for sign, side in ((1, 'left'), (-1, 'right')):
    y = sign * 0.148
    ellipsoid('iris_' + side, (-0.942, y, 1.113), (0.018, 0.048, 0.052), GREEN)
    ellipsoid('pupil_' + side, (-0.959, y, 1.113), (0.009, 0.016, 0.041), DARK)
    tapered_segment('ear_outer_' + side, (-0.77, sign * 0.144, 1.22),
                    (-0.73, sign * 0.20, 1.42), 0.092, 0.006, TABBY)
    tapered_segment('ear_inner_' + side, (-0.811, sign * 0.151, 1.253),
                    (-0.773, sign * 0.193, 1.397), 0.049, 0.003, EAR)
    for x, label in ((-0.46, 'front'), (0.55, 'hind')):
        yy = sign * 0.225
        top = 0.68 if x < 0 else 0.72
        tapered_segment(label + '_leg_' + side, (x, yy, top), (x + 0.03, yy, 0.15),
                        0.110 if x < 0 else 0.145, 0.077, WHITE)
        ellipsoid(label + '_white_paw_' + side, (x - 0.045, yy, 0.105),
                  (0.130, 0.086, 0.064), WHITE)
    for n in range(3):
        # A few deliberately restrained whiskers, which remain separate objects.
        tapered_segment('whisker_%s_%d' % (side, n),
                        (-0.995, sign * 0.085, 1.013 + 0.018 * n),
                        (-1.14 + 0.025 * n, sign * 0.35, 0.97 + 0.045 * n),
                        0.0023, 0.0005, WHITE, 5)

tail_points = [(0.70, 0, 0.83), (0.91, 0, 0.77), (1.08, 0.015, 0.66),
               (1.19, 0.025, 0.55), (1.31, 0.03, 0.57)]
for k in range(len(tail_points) - 1):
    tapered_segment('tail_ring_%d' % k, tail_points[k], tail_points[k + 1],
                    0.079 - k * 0.009, 0.071 - k * 0.009,
                    DARK if k % 2 else LIGHT)
ellipsoid('tail_dark_tip', tail_points[-1], (0.060, 0.052, 0.053), DARK)


def glb(output):
    buffer = bytearray()
    views, accessors, nodes, mesh_defs = [], [], [], []

    def accessor(data, component, kind, count, target, bounds=None):
        while len(buffer) % 4:
            buffer.append(0)
        offset = len(buffer)
        buffer.extend(data)
        vi = len(views)
        views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(data), 'target': target})
        ac = {'bufferView': vi, 'componentType': component, 'count': count, 'type': kind}
        if bounds:
            ac['min'], ac['max'] = bounds
        accessors.append(ac)
        return len(accessors) - 1

    for name, positions, colors, indices in meshes:
        pos_flat = [v for point in positions for v in point]
        color_flat = [v for rgb in colors for v in (*rgb, 1.0)]
        bounds = ([min(p[k] for p in positions) for k in range(3)],
                  [max(p[k] for p in positions) for k in range(3)])
        pos = accessor(struct.pack('<%sf' % len(pos_flat), *pos_flat), 5126, 'VEC3', len(positions), 34962, bounds)
        col = accessor(struct.pack('<%sf' % len(color_flat), *color_flat), 5126, 'VEC4', len(colors), 34962)
        idx = accessor(struct.pack('<%sI' % len(indices), *indices), 5125, 'SCALAR', len(indices), 34963)
        mesh_defs.append({'name': name, 'primitives': [{'attributes': {'POSITION': pos, 'COLOR_0': col},
                                                        'indices': idx, 'material': 0}]})
        nodes.append({'name': name, 'mesh': len(mesh_defs) - 1})
    doc = {'asset': {'version': '2.0', 'generator': 'cat-doujiang procedural blockout'},
           'scene': 0, 'scenes': [{'nodes': list(range(len(nodes)))}], 'nodes': nodes,
           'meshes': mesh_defs, 'buffers': [{'byteLength': len(buffer)}],
           'bufferViews': views, 'accessors': accessors,
           'materials': [{'name': 'vertex_coat', 'pbrMetallicRoughness': {
               'baseColorFactor': [1, 1, 1, 1], 'metallicFactor': 0, 'roughnessFactor': 0.9},
               'doubleSided': True}]}
    js = json.dumps(doc, separators=(',', ':')).encode('utf8')
    js += b' ' * ((-len(js)) % 4)
    buffer.extend(b'\0' * ((-len(buffer)) % 4))
    total = 12 + 8 + len(js) + 8 + len(buffer)
    output.write_bytes(struct.pack('<4sII', b'glTF', 2, total) +
                       struct.pack('<I4s', len(js), b'JSON') + js +
                       struct.pack('<I4s', len(buffer), b'BIN\0') + buffer)
    print('%s: %d bytes, %d editable mesh objects' % (output, total, len(meshes)))


if __name__ == '__main__':
    glb(Path(__file__).with_name('doujiang_blockout.glb'))
