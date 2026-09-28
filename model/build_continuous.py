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
        center = (-0.775 + 0.025*t, sign*(0.142+0.035*t), 1.232+0.140*t)
        r = 0.074*(1-t) + 0.006
        surface = soft_union(surface, egg(x, y, z, center, (r*0.72, r, 0.041)), 0.032)

vertices, faces, _, _ = marching_cubes(surface, 0, spacing=(step, step, step))
vertices += lo


def coat(p):
    xx, yy, zz = p
    # Spatial coat map traced from image(8), image(10), and the overhead
    # image(20260928-043052). The far side is an estimate where occluded.
    white = np.array((0.87, 0.85, 0.81))
    tabby = np.array((0.37, 0.33, 0.285))
    shadow = np.array((0.20, 0.19, 0.18))

    def smooth(a, b, val):
        t = np.clip((val-a)/(b-a), 0, 1)
        return t*t*(3-2*t)

    # Ringed tail and nearly black tip, as seen clearly in 043209.
    if xx > 0.83 and (zz > 0.50 or xx > 0.92):
        wave = np.sin(35*(xx-0.80) + 2*yy)
        return tuple(shadow if xx > 1.19 or wave > 0.27 else tabby)

    # White muzzle, broad lower cheeks and the narrow central white blaze.
    if xx < -0.69:
        blaze = 1 - smooth(0.052, 0.095, abs(yy) + 0.05*(zz-1.12))
        cheek = 1 - smooth(1.055, 1.12, zz + 0.018*np.sin(30*yy))
        white_mix = max(blaze, cheek)
        forehead_stripe = 0.5 + 0.5*np.cos(45*yy + 9*xx)
        fur = tabby*(1-0.42*forehead_stripe) + shadow*(0.42*forehead_stripe)
        if zz > 1.23 and abs(yy) > 0.12:
            fur = tabby*0.72
        return tuple(fur*(1-white_mix) + white*white_mix)

    # Neck and chest are white. The shoulder tabby patch is separate from
    # the main saddle by a distinct white strip visible in both side photos.
    shoulder = smooth(-0.63, -0.56, xx) * (1-smooth(-0.405, -0.33, xx))
    shoulder *= smooth(0.77, 0.92, zz) * smooth(0.10, 0.19, abs(yy))
    shoulder *= (1-smooth(1.02, 1.12, zz))
    saddle = smooth(-0.255, -0.17, xx) * (1-smooth(0.57, 0.73, xx))
    saddle *= smooth(0.73 + 0.03*np.sin(11*xx + 6*yy), 0.85, zz)
    saddle = max(saddle, shoulder)
    if zz < 0.6:
        saddle = 0
    wave = np.sin(35*xx + 4*yy + 3.5*zz + 0.7*np.sin(10*xx))
    stripe = smooth(0.42, 0.76, wave)
    fur = tabby*(1-0.38*stripe) + shadow*(0.38*stripe)
    # Reduced broad striping on the shoulder patch, stronger on torso.
    return tuple(white*(1-saddle) + fur*saddle)


base.meshes.append(('continuous_body_head_legs_tail_ears',
                    vertices.tolist(), [coat(p) for p in vertices],
                    faces.astype('int32').flatten().tolist()))

# Separate surfaces intentionally retained for the eyes, nose and fine whiskers.
base.ellipsoid('nose', (-1.035, 0, 1.028), (0.026, 0.035, 0.022), base.PINK,
               rings=10, sides=18)
for sign, side in ((1, 'left'), (-1, 'right')):
    sy = sign * 0.110
    for suffix, cx, scale, color in (
        ('rim', -0.987, (0.058, 0.038, 0.034), base.DARK),
        ('iris', -1.040, (0.012, 0.028, 0.027), base.GREEN),
        ('pupil', -1.051, (0.006, 0.010, 0.023), base.DARK),
        ('glint', -1.057, (0.003, 0.005, 0.005), base.WHITE),
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
