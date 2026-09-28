"""Create editable left/right coat atlas and a texture-bearing GLB.

Run after build_continuous.py. This is a hand mapped painting based on the
photographs, not a photogrammetric projection or an inferred hidden flank.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import build_continuous as sculpt

folder = Path(__file__).parent
size = 256
rng = np.random.default_rng(1999)
atlas = np.zeros((size, size, 3), dtype=np.float32)

for side in (0, 1):
    # Stored as two adjacent independently editable panels. On the GLB,
    # positive Y is the left panel and negative Y the right panel.
    yy = 0.23 if side == 0 else -0.23
    for py in range(size):
        zz = (1 - (py + 0.5) / size) * 1.585 - 0.035
        for col in range(size // 2):
            xx = ((col + 0.5) / (size / 2)) * 2.66 - 1.21
            atlas[py, side * (size // 2) + col] = ((1, 1, 1) if xx < -0.67
                                                     else sculpt.coat((xx, yy, zz)))

# Low amplitude fiber variation adds scale without moving the patch edges.
grain = rng.normal(0, 0.009, (size, size))
grain += 0.006 * np.sin(np.arange(size)[None, :] * 2.4 +
                        np.arange(size)[:, None] * 0.25)
atlas = np.clip(atlas + grain[..., None], 0, 1)
image = Image.fromarray((atlas * 255).astype('uint8'), 'RGB')
texture = folder / 'doujiang_coat_atlas.png'
image.save(texture, optimize=True)
sculpt.base.glb(folder / 'doujiang_textured.glb', texture_path=texture)
print('editable coat atlas:', texture)
