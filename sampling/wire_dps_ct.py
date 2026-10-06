#!/usr/bin/env python3
"""Expose dps_ct_bundle CT reconstructions in the layout evaluate.py expects.

ood/data.py::load_reconstructions scans
  {results_dir}/{sample_name}/{test_origin}_{patch}_{bottom_suffix}/inpainting/recon/*.png
and load_label_image wants label/0_00000.png. For the CT target that is
  results_patches_ddad_native_CT_scan/samples_image_XXXX/ddad_native_0_4/inpainting/...

Symlinks only -- the sampler outputs are never copied or modified. Use `cp -rL` on the
result to make a self-contained copy.

  python sampling/wire_dps_ct.py --src <dps outputs dir> --dst recons/results_patches_ddad_native_CT_<name>
"""
import os, glob, json, argparse, datetime

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True, help='sampling/dps output dir with image_XXXX/{recon,label,input}/m*.png')
ap.add_argument('--dst', required=True, help='new recon dir, e.g. recons/results_patches_ddad_native_CT_<name>')
ap.add_argument('--force', action='store_true', help='allow writing into an existing --dst')
ap.add_argument('--n_draws', type=int, default=10, help='use draws m0..m{n-1}')
ap.add_argument('--sigma', type=float, default=0.05)
ap.add_argument('--scale', type=float, default=0.4, help="DPS scale used for sampling (zeta' = scale/2)")
ap.add_argument('--images', default=None, help='comma list of image stems to wire (default: all)')
args = ap.parse_args()
if os.path.exists(args.dst) and not args.force:
    raise SystemExit(f'{args.dst} exists; refusing to overwrite (pass --force). The released CT_scan set must not be replaced.')

src = os.path.abspath(args.src)
made, skipped = 0, []
want = set(args.images.split(',')) if args.images else None
for d in sorted(glob.glob(os.path.join(src, 'image_*'))):
    stem = os.path.basename(d)                       # image_0042
    if want is not None and stem not in want:
        continue
    recons = [os.path.join(d, 'recon', f'm{k}.png') for k in range(args.n_draws)]
    if not all(os.path.exists(p) for p in recons):
        skipped.append(stem); continue
    label = os.path.join(d, 'label', 'm0.png')
    inp   = os.path.join(d, 'input', 'm0.png')
    if not os.path.exists(label):
        skipped.append(stem); continue

    base = os.path.join(args.dst, f'samples_{stem}', 'ddad_native_0_4', 'inpainting')
    for sub in ('recon', 'label', 'input'):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    for k, rp in enumerate(recons):
        link = os.path.join(base, 'recon', f'{k}_0_00000.png')
        if os.path.islink(link) or os.path.exists(link): os.remove(link)
        os.symlink(rp, link)
    for srcf, link in ((label, os.path.join(base, 'label', '0_00000.png')),
                       (inp,   os.path.join(base, 'input', '0_00000.png'))):
        if os.path.exists(srcf):
            if os.path.islink(link) or os.path.exists(link): os.remove(link)
            os.symlink(srcf, link)
    made += 1

os.makedirs(args.dst, exist_ok=True)
json.dump({
    'sampler': 'sampling/dps/sample_fast.py',
    'test_origin': 'ddad_native',
    'effective_sigma': args.sigma,
    'hyperparams': {'sigma': args.sigma, 'conditioning': 'ps', 'scale': args.scale,
                    "zeta_prime": args.scale / 2, 'grad_norm': 'per_sample',
                    'steps': 1000, 'sampler': 'ddpm', 'operator': 'inpainting (mask=ones -> denoising)'},
    'ckpt': 'weights/diffusion/ct_dps_model050000.pt (sampling/dps/configs/ct_model_config.yaml)',
    'image_dir': 'data/mvtec_ad/CT/test/random',
    'n_reconstructions_per_sample': args.n_draws,
    'generated': datetime.datetime.now().isoformat(timespec='seconds'),
    'note': 'DPS CT reconstructions wired by sampling/wire_dps_ct.py',
    'samples': [f'image_{i:04d}' for i in range(100)],
}, open(os.path.join(args.dst, 'recon_metadata.json'), 'w'), indent=1)

print(f'wired {made} samples ({args.n_draws} draws each) -> {args.dst}')
if skipped:
    print(f'incomplete, skipped {len(skipped)}: {skipped[:6]}{"..." if len(skipped)>6 else ""}')
