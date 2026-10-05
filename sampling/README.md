# Sampling: generating diffusion reconstructions

Stage 1 of LACToS is a single step: **draw M diffusion reconstructions conditioned on the test image.**
Everything downstream (features, autoencoder, superpixel typical-set score, metrics) is the same for
every dataset and only reads the reconstructions from `recons/`. Any conditional diffusion sampler that
writes this layout can be plugged in. The two samplers below are the ones that produced the released sets.

The evaluation consumes **M stochastic reconstructions per test image** stored as

```
recons/results_patches_ddad_native_<category>_<subcategory><suffix>/
  samples_<id>/<test_origin>_<patch>_<bottom_suffix>/inpainting/{input,label,recon}/*.png
```

The reconstructions behind the paper are released (`scripts/download_assets.sh`). You only need this
stage to regenerate them or to run LACToS on new data. Exact hyperparameters per dataset, including the
ones that were not recorded, are in [docs/HYPERPARAMETERS.md §5](../docs/HYPERPARAMETERS.md).

Samplers behind the released reconstructions:

| Dataset | Sampler | Code | Diffusion checkpoint (not released) |
|---|---|---|---|
| MVTec AD (15 categories) | DDAD-guided conditioned denoising (w=2, t*=250, skip 25, 40 draws) | `tools/generate_recons.py` + `configs/recon/ddad_native_*.yaml` + `DDAD/` | DDAD UNet per category, `weights/diffusion/mvtec/<cat>/<epoch>` (**not yet released**) |
| CT (100 slices) | DPS denoising: y = x + σn, σ = 0.05, ζ′ = 0.2, 10 draws | `sampling/dps/sample_fast.py` + `sampling/wire_dps_ct.py` | `weights/diffusion/ct_dps_model050000.pt` |
| X-ray (DvXray scissors) | DPS patch inpainting | `sampling/dps/` + `sampling/wire_dps_xray.py` | `weights/diffusion/xray_dps_model500000.pt` |
| Faces (random22, moreface28) | DPS patch inpainting, FFHQ prior, σ = 0.5 | `sampling/dps/` (DPS codebase) | `weights/diffusion/ffhq_10m.pt` (public DPS model) |

## A. DDAD-guided denoising (MVTec)

1. Get the DDAD UNet for the category (`weights/diffusion/mvtec/<category>/<epoch>`). **These per-category
   MVTec models are not in the current release yet** (see docs/DATA.md for the epoch of each). To train
   your own, use `DDAD/main.py --train True` with `DDAD/config_<cat>.yaml`. The public DDAD checkpoints
   (`DDAD/README.md`) work for new experiments, but are not verified to be the models behind the released
   reconstructions.
2. Generate reconstructions:

```bash
python tools/generate_recons.py --recon_config configs/recon/ddad_native_cable.yaml
# or flat CLI, e.g. 40 draws per image as in the paper:
python tools/generate_recons.py --sampler ddad_native --ckpt weights/diffusion/mvtec/cable/3000 \
    --image_dir data/mvtec_ad/cable/test/bent_wire --samples 000 001 002 003 004 005 006 007 008 009 \
    --num_seeds 40 \
    --out_root recons/results_patches_ddad_native_cable_bent_wire_1 \
    --sampler_param w=2 --sampler_param test_trajectoy_steps=250
```

Draw k is seeded with `torch.manual_seed(k)`. A `recon_metadata.json` with every setting is written
next to the samples.

## B. DPS (CT denoising; X-ray and faces patch inpainting)

`sampling/dps/` is the DPS sampler (Chung et al., ICLR 2023) with batched N-measurements-per-patch
sampling:

```
sampling/dps/sample_batch.py     N measurements per patch, batched (patch-inpainting layout)
sampling/dps/sample_fast.py      sharded / batched variant with --scale and --grad_norm
sampling/dps/sample_condition.py single image / single measurement
sampling/dps/guided_diffusion/   UNet, DDPM sampler, DPS conditioning, measurement operators
sampling/dps/configs/            model / diffusion / task configs (CT and X-ray)
sampling/dps/run_ct_reconstruction.sh   example full run (16 patches per image, multi-GPU)
```

Run it from `sampling/dps/` (it imports `guided_diffusion`, `util`, `data` relative to that folder),
for example:

```bash
cd sampling/dps
python sample_fast.py --model_config configs/xray_model_config.yaml \
    --task_config configs/xray_inpainting_config.yaml --data_root <images> \
    --save_dir outputs/xray --num_measurements 10 --batch 20 --scale 1.0 --grad_norm per_sample
```

**CT, exactly as released** (100 slices, 10 draws each; ζ′ = scale / 2 with per-sample normalized steps;
seeds are per (image, draw), so shards and reruns give identical draws):

```bash
cd sampling/dps
python sample_fast.py --model_config configs/ct_model_config.yaml \
    --task_config configs/ctgrid/sigma0.05.yaml --data_root ../../data/mvtec_ad/CT/test/random \
    --save_dir outputs/CT_sigma0.05_zeta0.2 --num_measurements 10 --batch 25 \
    --scale 0.4 --grad_norm per_sample            # add --shard g --num_shards N to split over GPUs
cd ../..
python sampling/wire_dps_ct.py --src sampling/dps/outputs/CT_sigma0.05_zeta0.2 \
    --dst recons/results_patches_ddad_native_CT_mine --sigma 0.05
```

`sample_fast.py` checks that the measurement mask is all ones, so the "inpainting" operator in the task
config is a pure denoising measurement y = x + σn. Other σ values: `configs/ctgrid/sigma{0.02,0.1,0.2,0.4}.yaml`.

Then convert the outputs into the evaluation layout:

```bash
python sampling/wire_dps_xray.py --src sampling/dps/outputs/xray --subcat <name> --n_draws 10
python sampling/wire_dps_ct.py   --src <dps CT outputs> --dst recons/results_patches_ddad_native_CT_<name>
```

Both scripts create symlinks. `wire_dps_ct.py` refuses to write into an existing directory unless you
pass `--force`, so the released `CT_scan` set cannot be overwritten by accident.

**Faces.** The paper's face reconstructions were produced with the public DPS FFHQ model
(`ffhq_10m.pt`) and the patch-inpainting procedure above at σ = 0.5. The FFHQ model/task configs come
from the upstream DPS repository. The DPS guidance scale for faces and X-ray was not recorded
(docs/HYPERPARAMETERS.md §5, §7).

## Then

Evaluate a new reconstruction set with the same flags as `scripts/run_reproduce.sh` (for example
`--category <cat> --all_subcategories --results_root ./recons/results_patches_ddad_native --results_suffix <suffix>`).
