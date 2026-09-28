"""Construct a single continuous cat surface by implicit sculpting.

Requires numpy and scikit-image. Reuses the compact glTF writer in
build_doujiang.py. This is a sculptable mesh, not an anatomical scan.
"""
from pathlib import Path
import numpy as np
from skimage.measure import marching_cubes
import build_doujiang as base

base.meshes.clear()


def soft_union(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def egg(x, y, z, center, radius):
    # Ellipsoid distance approximation, in model units.
    return (np.sqrt(((x-center[0])/radius[0])**2 +
                    ((y-center[1])/radius[1])**2 +
                    ((z-center[2])/radius[2])**2) - 1) * min(radius)


step = 0.021
lo = np.array([-1.21, -0.60, -0.035])
hi = np.array([1.45, 0.60, 1.55])
axis = [np.arange(a, b, step, dtype=np.float32) for a, b in zip(lo, hi)]
x, y, z = np.meshgrid(*axis, indexing='ij', sparse=True)

pieces = [
    ((0.10, 0, 0.75), (0.66, 0.285, 0.32), 0.10),  # torso
    ((0.53, 0, 0.68), (0.31, 0.30, 0.31), 0.16),   # rounded haunches
    ((-0.45, 0, 0.74), (0.255, 0.245, 0.33), 0.16),# chest
    ((-0.65, 0, 0.97), (0.16, 0.17, 0.24), 0.13), # neck
    ((-0.78, 0, 1.105), (0.214, 0.208, 0.19), 0.09), # skull
    ((-0.956, 0, 1.025), (0.09, 0.118, 0.076), 0.05), # muzzle
]
for sx in (-1, 1):
    for fore, px in ((True, -0.46), (False, 0.53)):
        sy = sx * (0.196 if fore else 0.202)
        pieces.extend([
            ((px, sy, 0.48), (0.115 if fore else 0.15, 0.11, 0.26), 0.13),
            ((px + (0.01 if fore else 0.06), sy, 0.255),
             (0.090 if fore else 0.10, 0.084, 0.17), 0.105),
            ((px - 0.05, sy, 0.105), (0.132, 0.095, 0.070), 0.09),
        ])

surface = None
for center, radius, blend in pieces:
    d = egg(x, y, z, center, radius)
    surface = d if surface is None else soft_union(surface, d, blend)

# Smooth tapered curved tail, grown out of the haunch without a visible joint.
for i in range(13):
    t = i / 12
    center = (0.76 + 0.55*t, 0.018 + 0.033*t, 0.735 - 0.17*np.sin(np.pi*t))
    r = 0.078 - 0.022*t
    surface = soft_union(surface, egg(x, y, z, center, (r*1.3, r, r)), 0.065)

# Ears rise from the skull, with smooth blending at their bases.
for sign in (-1, 1):
    for i in range(8):
        t = i / 7
        center = (-0.775 + 0.025*t, sign*(0.142+0.042*t), 1.232+0.175*t)
        r = 0.083*(1-t) + 0.006
        surface = soft_union(surface, egg(x, y, z, center, (r*0.72, r, 0.041)), 0.032)

vertices, faces, _, _ = marching_cubes(surface, 0, spacing=(step, step, step))
vertices += lo


def coat(p):
    xx, yy, zz = p
    # Photos show a mostly white underside, dark saddle, separate shoulder
    # patch, a white neck band, and a mostly dark ringed tail.
    if xx > 0.76 and zz < 0.77:
        return base.DARK if np.sin((xx-0.76)*38) > 0.1 else base.LIGHT
    if xx < -0.67:
        if zz < 1.075 or xx < -0.90 and abs(yy) < 0.09:
            return base.WHITE
        return base.DARK if np.sin(42*yy + 8*xx) > 0.78 else base.TABBY
    if zz < 0.80 or abs(yy) > 0.14 and zz < 0.85:
        return base.WHITE
    if -0.43 < xx < -0.29:
        return base.WHITE
    stripe = np.sin(32*xx + 4.5*yy + 6*zz + 1.4*np.sin(8*xx))
    return base.DARK if stripe > 0.70 else (base.LIGHT if stripe < -0.6 else base.TABBY)


base.meshes.append(('continuous_body_head_legs_tail_ears',
                    vertices.tolist(), [coat(p) for p in vertices],
                    faces.astype('int32').flatten().tolist()))

# Separate surfaces intentionally retained for the eyes, nose and fine whiskers.
base.ellipsoid('nose', (-1.035, 0, 1.028), (0.032, 0.044, 0.028), base.PINK,
               rings=10, sides=18)
for sign, side in ((1, 'left'), (-1, 'right')):
    sy = sign * 0.136
    for suffix, cx, scale, color in (
        ('rim', -0.990, (0.012, 0.045, 0.042), base.DARK),
        ('iris', -1.002, (0.009, 0.034, 0.034), base.GREEN),
        ('pupil', -1.011, (0.005, 0.012, 0.028), base.DARK),
        ('glint', -1.017, (0.003, 0.006, 0.006), base.WHITE),
    ):
        base.ellipsoid(side+'_'+suffix, (cx, sy, 1.13), scale, color,
                       rings=10, sides=18)
    for i in range(3):
        base.tapered_segment(side+'_whisker_'+str(i),
                             (-0.986, sign*0.08, 1.020+i*0.015),
                             (-1.12+i*0.018, sign*0.31, 0.978+i*0.032),
                             0.0018, 0.0003, base.WHITE, 5)

output = Path(__file__).with_name('doujiang_continuous.glb')
base.glb(output)
print('continuous surface:', len(vertices), 'vertices,', len(faces), 'triangles')
