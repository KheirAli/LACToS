# Sampling: generating diffusion reconstructions

The evaluation consumes **M stochastic reconstructions per test image** stored as

```
recons/results_patches_ddad_native_<category>_<subcategory><suffix>/
  samples_<id>/<test_origin>_<patch>_<bottom_suffix>/inpainting/{input,label,recon}/*.png
```

The reconstructions behind the paper are released (`scripts/download_assets.sh`). You only need this
stage to regenerate them or to run LACToS on new data. Exact hyperparameters per dataset, including the
ones that were not recorded, are in [docs/HYPERPARAMETERS.md §5](../docs/HYPERPARAMETERS.md).

Two samplers were used:

| Dataset | Sampler | Code | Diffusion checkpoint (not released) |
|---|---|---|---|
| MVTec AD (15 categories) | DDAD-native conditioned denoising (w=2, t*=250, skip 25, 40 draws) | `tools/generate_recons.py` + `configs/recon/ddad_native_*.yaml` + `DDAD/` | DDAD UNet per category (`weights/ddad_unet/<cat>/<epoch>`) |
| CT (100 slices) | DDAD-native (same settings, 40 draws) | same, `configs/recon/ddad_native_ct_sigma03.yaml` / `ddad_native_xray.yaml` (CT despite the name) | DDAD UNet `combined/2000` (CT) |
| X-ray (DvXray scissors) | DPS patch inpainting | `sampling/dps/` + `sampling/wire_dps_xray.py` | guided-diffusion X-ray UNet (`model500000.pt`) |
| Faces (random22, moreface28) | DPS patch inpainting, FFHQ prior, σ = 0.5 | `sampling/dps/` (DPS codebase) | `ffhq_10m.pt` (public DPS release) |

## A. DDAD-native (MVTec, CT)

1. Train or download a DDAD UNet per category. Training uses `DDAD/main.py --train True` with
   `DDAD/config_<cat>.yaml`; the original DDAD checkpoints are linked from `DDAD/README.md`. Put them
   at `weights/ddad_unet/<category>/<epoch>`.
2. Generate reconstructions:

```bash
python tools/generate_recons.py --recon_config configs/recon/ddad_native_cable.yaml
# or flat CLI, e.g. 40 draws per image as in the paper:
python tools/generate_recons.py --sampler ddad_native --ckpt weights/ddad_unet/cable/3000 \
    --image_dir data/mvtec_ad/cable/test/bent_wire --samples 000 001 002 003 004 005 006 007 008 009 \
    --num_seeds 40 \
    --out_root recons/results_patches_ddad_native_cable_bent_wire_1 \
    --sampler_param w=2 --sampler_param test_trajectoy_steps=250
```

Draw k is seeded with `torch.manual_seed(k)`. A `recon_metadata.json` with every setting is written
next to the samples.

## B. DPS patch inpainting (X-ray, faces; also CT ablations)

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
