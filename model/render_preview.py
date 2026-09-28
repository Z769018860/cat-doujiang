"""Render the generated GLB as a two-view inspection sheet (requires numpy/matplotlib)."""
import json
import struct
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

folder = Path(__file__).parent
blob = (folder / 'doujiang_blockout.glb').read_bytes()
length = struct.unpack_from('<I', blob, 12)[0]
doc = json.loads(blob[20:20 + length])
binary = blob[28 + length:]
views = doc['bufferViews']
accessors = doc['accessors']


def read(index):
    a = accessors[index]
    view = views[a['bufferView']]
    dtype = {5126: '<f4', 5125: '<u4', 5123: '<u2', 5121: 'u1'}[a['componentType']]
    width = {'VEC3': 3, 'VEC4': 4, 'SCALAR': 1}[a['type']]
    start = view.get('byteOffset', 0) + a.get('byteOffset', 0)
    values = np.frombuffer(binary, dtype=dtype, count=a['count'] * width,
                           offset=start).reshape(-1, width)
    return values.astype('f4') / 255 if a.get('normalized') else values


fig = plt.figure(figsize=(13, 6), facecolor='#f3f0e9')
for slot, (label, elevation, azimuth) in enumerate((('Three-quarter view', 19, -124),
                                                      ('Back and tail', 67, -94)), 1):
    ax = fig.add_subplot(1, 2, slot, projection='3d')
    ax.set_facecolor('#f3f0e9')
    for mesh in doc['meshes']:
        primitive = mesh['primitives'][0]
        xyz = read(primitive['attributes']['POSITION'])
        colors = read(primitive['attributes']['COLOR_0'])[:, :3]
        faces = read(primitive['indices']).reshape(-1, 3)
        # Subsample for a compact, repeatable inspection render.
        step = max(1, len(faces) // 1800)
        faces = faces[::step]
        tris = xyz[faces]
        normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-8)
        light = np.clip((normal @ np.array([-0.4, -0.5, 0.76]) + 0.6) * 0.5, 0.38, 1.0)
        rgb = np.clip(colors[faces].mean(axis=1) * light[:, None], 0, 1)
        ax.add_collection3d(Poly3DCollection(tris, facecolors=rgb, linewidths=0,
                                              edgecolors='none', zsort='average'))
    ax.set(xlim=(-1.2, 1.48), ylim=(-0.75, 0.75), zlim=(0, 1.55))
    ax.set_box_aspect((2.68, 1.5, 1.55), zoom=1.35)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_axis_off()
    ax.set_title(label, fontsize=17, pad=0)
fig.suptitle('Doujiang | 3D model study', fontsize=19)
fig.text(0.5, 0.035, 'Geometry preview | photographic coat, groom, rig and animation are future work',
         ha='center', color='#746f68', fontsize=11)
fig.savefig(folder / 'doujiang_preview.png', dpi=155, facecolor=fig.get_facecolor())
print(folder / 'doujiang_preview.png')
