"""Check the exported desktop pet GLB's editable body and render data."""
import json
import struct
from pathlib import Path

import numpy as np
import trimesh

path = Path(__file__).with_name('doujiang_textured.glb')
blob = path.read_bytes()
assert blob[:4] == b'glTF' and struct.unpack_from('<I', blob, 8)[0] == len(blob)
json_len = struct.unpack_from('<I', blob, 12)[0]
doc = json.loads(blob[20:20+json_len])
payload = blob[28+json_len:]

def values(index):
    acc = doc['accessors'][index]
    view = doc['bufferViews'][acc['bufferView']]
    kind = {5126: '<f4', 5123: '<u2', 5121: 'u1'}[acc['componentType']]
    components = {'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'SCALAR': 1}[acc['type']]
    offset = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
    return np.frombuffer(payload, dtype=kind, count=acc['count']*components,
                         offset=offset).reshape(-1, components)

body = doc['meshes'][0]['primitives'][0]
attr = body['attributes']
assert {'POSITION', 'NORMAL', 'COLOR_0', 'TEXCOORD_0'} <= attr.keys()
xyz, normals = values(attr['POSITION']), values(attr['NORMAL'])
uv = values(attr['TEXCOORD_0']) / 65535
faces = values(body['indices']).reshape(-1, 3)
assert np.isfinite(xyz).all() and np.isfinite(normals).all()
assert np.allclose(np.linalg.norm(normals, axis=1), 1, atol=0.002)
assert ((uv >= 0) & (uv <= 1)).all()
assert len(doc['images']) == len(doc['textures']) == 1
assert trimesh.Trimesh(vertices=xyz, faces=faces, process=False).is_watertight
assert any(mesh['name'] == 'short_fur_wisps' for mesh in doc['meshes'])
print(f'PASS: {len(xyz)} body vertices, {len(faces)} faces, smooth normals, '
      f'UV and embedded texture, watertight body, separate fur mesh')
