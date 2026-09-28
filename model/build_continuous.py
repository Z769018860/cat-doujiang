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
    ((0.10, 0, 0.73), (0.65, 0.315, 0.37), 0.12), # fuller torso
    ((0.53, 0, 0.67), (0.32, 0.335, 0.34), 0.17), # fuller haunches
    ((-0.45, 0, 0.72), (0.265, 0.26, 0.34), 0.16),# chest
    ((-0.65, 0, 0.97), (0.16, 0.17, 0.24), 0.13), # neck
    ((-0.78, 0, 1.102), (0.221, 0.225, 0.190), 0.09), # rounded skull
    ((-0.956, 0, 1.025), (0.09, 0.118, 0.076), 0.05), # muzzle
]
for sx in (-1, 1):
    for fore, px in ((True, -0.46), (False, 0.53)):
        sy = sx * (0.196 if fore else 0.202)
        pieces.extend([
            ((px, sy, 0.47), (0.135 if fore else 0.16, 0.12, 0.245), 0.14),
            ((px + (0.01 if fore else 0.06), sy, 0.245),
             (0.100 if fore else 0.11, 0.091, 0.16), 0.115),
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
        center = (-0.775 + 0.025*t, sign*(0.150+0.030*t), 1.225+0.112*t)
        r = 0.072*(1-t) + 0.006
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

    # Shoulder island: a tapered oval, descending towards the visible outer
    # foreleg. The curved white collar behind it separates it from the saddle.
    # The dark shoulder is a separate side patch, terminating above the
    # white foreleg. Its upper edge wraps over the shoulder in the top photo.
    shoulder_cx = -0.42 + 0.025*np.sin(8*zz + 3*yy)
    shoulder_cz = 0.995 - 0.025*np.sin(8*xx)
    shoulder_distance = ((xx-shoulder_cx)/0.155)**2 + ((zz-shoulder_cz)/0.196)**2
    shoulder = 1-smooth(0.72, 1.13, shoulder_distance)
    shoulder *= smooth(0.065, 0.15, abs(yy))

    # Main saddle observed in the side views: starts after the collar, curves
    # across the shoulder and bulges lower in the middle, then climbs back to
    # leave a broad white hindquarter. It is not a rectangular x/z threshold.
    front = -0.135 + 0.045*np.sin(8*zz + 2*yy)
    rear = 0.60 - 0.13*(1-smooth(0.69, 0.94, zz))
    lower = 0.69 - 0.025*np.exp(-((xx-0.18)/0.28)**2)
    lower += 0.16*smooth(0.36, 0.62, xx)
    lower += 0.014*np.sin(17*xx+7*yy)
    saddle = smooth(front-0.035, front+0.045, xx)
    saddle *= 1-smooth(rear-0.06, rear+0.06, xx)
    saddle *= smooth(lower-0.035, lower+0.055, zz)
    saddle = max(saddle, shoulder)
    wave = np.sin(36*xx + 6*yy + 4*zz + 0.8*np.sin(9*xx))
    stripe = smooth(0.42, 0.87, wave)
    fur = tabby*(1-0.21*stripe) + shadow*(0.21*stripe)
    # Reduced broad striping on the shoulder patch, stronger on torso.
    return tuple(white*(1-saddle) + fur*saddle)


base.meshes.append(('continuous_body_head_legs_tail_ears',
                    vertices.tolist(), [coat(p) for p in vertices],
                    faces.astype('int32').flatten().tolist()))

# Separate surfaces intentionally retained for the eyes, nose and fine whiskers.
base.ellipsoid('nose', (-1.035, 0, 1.028), (0.019, 0.027, 0.016), base.PINK,
               rings=10, sides=18)
for sign, side in ((1, 'left'), (-1, 'right')):
    sy = sign * 0.110
    for suffix, cx, scale, color in (
        ('rim', -1.015, (0.023, 0.037, 0.028), base.DARK),
        ('iris', -1.034, (0.009, 0.029, 0.023), base.GREEN),
        ('pupil', -1.043, (0.005, 0.011, 0.020), base.DARK),
        ('glint', -1.048, (0.002, 0.004, 0.004), base.WHITE),
    ):
        base.ellipsoid(side+'_'+suffix, (cx, sy, 1.13), scale, color,
                       rings=10, sides=18)
    base.ellipsoid(side+'_inner_ear', (-0.824, sign*0.177, 1.274),
                   (0.014, 0.037, 0.051), (0.43, 0.29, 0.29),
                   rings=10, sides=18)
    for i in range(3):
        base.tapered_segment(side+'_whisker_'+str(i),
                             (-0.986, sign*0.08, 1.020+i*0.015),
                             (-1.12+i*0.018, sign*0.31, 0.978+i*0.032),
                             0.0018, 0.0003, base.WHITE, 5)

output = Path(__file__).with_name('doujiang_continuous.glb')
base.glb(output)
print('continuous surface:', len(vertices), 'vertices,', len(faces), 'triangles')
