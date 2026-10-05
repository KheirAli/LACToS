#!/usr/bin/env python3
"""Wire X-ray DPS reconstructions into the layout evaluate_ddad_fe.py expects."""
import os, glob, argparse
import numpy as np
from PIL import Image
ap=argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--subcat', required=True)
ap.add_argument('--n_draws', type=int, default=10)
ap.add_argument('--zero_gt', action='store_true', help='normal set: emit all-zero GT masks')
ap.add_argument('--gt_out', default=None, help='where to write zero GT masks')
ap.add_argument('--img_out', default=None, help='stage source images for superpixel generation')
ap.add_argument('--img_src', default=None)
a=ap.parse_args()
PKG=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
dst=f'{PKG}/recons/results_patches_ddad_native_xray_{a.subcat}'
made, skipped = 0, []
for d in sorted(glob.glob(f'{a.src}/*')):
    if not os.path.isdir(d): continue
    stem=os.path.basename(d)
    recons=[f'{d}/recon/m{k}.png' for k in range(a.n_draws)]
    if not (all(map(os.path.exists,recons)) and os.path.exists(f'{d}/label/m0.png')):
        skipped.append(stem); continue
    base=f'{dst}/samples_{stem}/ddad_native_0_4/inpainting'
    for sub in ('recon','label','input'): os.makedirs(f'{base}/{sub}', exist_ok=True)
    for k,rp in enumerate(recons):
        ln=f'{base}/recon/{k}_0_00000.png'
        if os.path.lexists(ln): os.remove(ln)
        os.symlink(os.path.abspath(rp), ln)
    for s,ln in ((f'{d}/label/m0.png', f'{base}/label/0_00000.png'),
                 (f'{d}/input/m0.png', f'{base}/input/0_00000.png')):
        if os.path.exists(s):
            if os.path.lexists(ln): os.remove(ln)
            os.symlink(os.path.abspath(s), ln)
    if a.zero_gt and a.gt_out:
        os.makedirs(a.gt_out, exist_ok=True)
        Image.fromarray(np.zeros((256,256),np.uint8)).save(f'{a.gt_out}/{stem}_mask.png')
    if a.img_out and a.img_src:
        os.makedirs(a.img_out, exist_ok=True)
        src=f'{a.img_src}/{stem}.png'
        if os.path.exists(src):
            ln=f'{a.img_out}/{stem}.png'
            if os.path.lexists(ln): os.remove(ln)
            os.symlink(os.path.realpath(src), ln)
    made+=1
print(f'wired {made} -> {dst}')
if skipped: print(f'skipped {len(skipped)}')
