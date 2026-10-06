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

## Quick commands (one per dataset)

```bash
GPU=0 bash sampling/sample_mvtec_ddad.sh <category> weights/diffusion/mvtec/<category>/<epoch>   # MVTec, DDAD-guided, 40 draws
GPU=0 bash sampling/sample_ct_dps.sh mine                                                        # CT, DPS denoising, 10 draws
GPU=0 bash sampling/sample_patch_dps.sh xray                                                     # X-ray, DPS patch inpainting
GPU=0 bash sampling/sample_patch_dps.sh faces                                                    # faces, DPS patch inpainting
```

All four write to `recons_mine/` (set `RECON_ROOT` to change it) in the layout LACToS reads, and print the
Step-3 command that evaluates the result, e.g. `RECON_ROOT=recons_mine bash scripts/run_reproduce.sh cable`.
The full walk-through is [docs/PIPELINE.md](../docs/PIPELINE.md); the sections below explain what each script
runs.

## Samplers behind the released reconstructions

| Dataset | Sampler | Script | Diffusion model |
|---|---|---|---|
| MVTec AD (15 categories) | DDAD-guided conditioned denoising (w=2, t*=250, skip 25, eta 1, 40 draws) | `sample_mvtec_ddad.sh` → `tools/generate_recons.py` | DDAD UNet per category, `weights/diffusion/mvtec/<cat>/<epoch>` (**not yet released**) |
| CT (100 slices) | DPS denoising: y = x + σn, σ = 0.05, ζ′ = 0.2, 10 draws | `sample_ct_dps.sh` → `dps/sample_fast.py` + `wire_dps_ct.py` | `weights/diffusion/ct_dps_model050000.pt` |
| X-ray (DvXray scissors) | DPS superpixel-patch inpainting, 6 patches x 4 draws | `sample_patch_dps.sh xray` → `dps/sample_batch.py` | `weights/diffusion/xray_dps_model500000.pt` |
| Faces (random22, moreface28) | DPS superpixel-patch inpainting, σ = 0.5, 12 patches x 4 draws | `sample_patch_dps.sh faces` → `dps/sample_batch.py` | `weights/diffusion/ffhq_10m.pt` (public DPS model) |

Re-running a sampler with the recorded seeds gives statistically equivalent, not bit-identical, draws
(different GPU/library versions); the released reconstructions are the ones behind the paper numbers.

## A. MVTec: DDAD-guided denoising (`sample_mvtec_ddad.sh`)

1. Put the category's DDAD UNet at `weights/diffusion/mvtec/<category>/<epoch>`. These per-category models
   are not in the current release yet (epochs in docs/DATA.md). To train your own:
   `DDAD/main.py --train True` with `DDAD/config_<cat>.yaml`. The public DDAD checkpoints (`DDAD/README.md`)
   are fine for new experiments but are not verified to be the models behind the released reconstructions.
2. `GPU=0 bash sampling/sample_mvtec_ddad.sh <category> <ckpt> [num_seeds=40] [suffix=_1]` loops over every
   test subcategory and calls, per subcategory,

```bash
python tools/generate_recons.py --sampler ddad_native --ckpt <ckpt> --image_dir data/mvtec_ad/<cat>/test/<sub> \
    --samples <all image ids> --num_seeds 40 --out_root recons_mine/results_patches_ddad_native_<cat>_<sub>_1 \
    --sampler_param w=2 --sampler_param test_trajectoy_steps=250 --sampler_param skip=25 --sampler_param eta=1.0
```

Draw k is seeded with `torch.manual_seed(k)`; a `recon_metadata.json` with every setting is written next to
the samples. YAML alternative: `python tools/generate_recons.py --recon_config configs/recon/ddad_native_cable.yaml`.

## B. DPS code (`sampling/dps/`)

```
sampling/dps/sample_fast.py      batched / sharded DPS for the denoising setup (CT); --scale, --grad_norm
sampling/dps/sample_batch.py     N measurements per superpixel patch (X-ray, faces); --box_coords, --mask_prob
sampling/dps/sample_condition.py single image / single measurement
sampling/dps/guided_diffusion/   UNet, DDPM sampler, DPS conditioning, measurement operators
sampling/dps/configs/            model configs (ct, xray, faces), diffusion_config.yaml (1000 DDPM steps),
                                 task configs: ctgrid/sigma*.yaml (CT), patch_inpainting_config.yaml (X-ray, faces)
```

The Python scripts import `guided_diffusion`, `util` and `data` relative to `sampling/dps/`, so run them from
that folder; the shell scripts in `sampling/` do this for you.

### CT (`sample_ct_dps.sh`)

`GPU=0 bash sampling/sample_ct_dps.sh [name=mine] [sigma=0.05] [scale=0.4] [draws=10]` runs

```bash
cd sampling/dps
python sample_fast.py --model_config configs/ct_model_config.yaml --task_config configs/ctgrid/sigma0.05.yaml \
    --data_root ../../data/mvtec_ad/CT/test/random --save_dir outputs/... --num_measurements 10 --batch 25 \
    --scale 0.4 --grad_norm per_sample                    # SHARDS=N splits over GPUs with --shard/--num_shards
cd ../.. && python sampling/wire_dps_ct.py --src sampling/dps/outputs/... \
    --dst recons_mine/results_patches_ddad_native_CT_<name> --sigma 0.05 --scale 0.4
```

ζ′ = scale / 2 with per-sample normalised steps. `sample_fast.py` checks that the measurement mask is all
ones, so the "inpainting" operator in the task config is a pure denoising measurement y = x + σn. Seeds are
per (image, draw), so shards and reruns draw the same noise. Other σ: `configs/ctgrid/sigma{0.02,0.1,0.2,0.4}.yaml`.
`wire_dps_ct.py` creates symlinks and refuses to write into an existing folder unless `--force` is given.

### X-ray and faces (`sample_patch_dps.sh`)

`GPU=0 bash sampling/sample_patch_dps.sh <xray|faces> [sigma] [image_dir] [suffix=_1]`, per test image:

1. `super_pixel_generation.py` writes the superpixel label map `mask.png`;
2. for each patch (X-ray tops 8, 13, 16, 17, 20, 21; faces tops 8, 12–14, 16–23, each with its 64-px column
   band), `sample_batch.py --box_coords <top> 4 <left> <right> --mask_prob 0.9 --seed 3 --num_measurements 4`
   with `patch_inpainting_config.yaml` (σ substituted);
3. all patch reconstructions of the image are copied into
   `recons_mine/results_patches_ddad_native_<xray_scissors|faces_random><suffix>/samples_<id>/ddad_native_0_4/`.

Defaults: X-ray σ = 0.3, faces σ = 0.5. The DPS guidance scale of the released X-ray and faces sets was not
recorded (the config uses the upstream DPS default 0.5), and the superpixel-box mask code may differ from the
version used in 2026-08, so this reproduces the procedure rather than identical reconstructions
(docs/HYPERPARAMETERS.md §5, §7). The X-ray DPS *denoising* variant (like CT) is available through
`sample_fast.py` with `configs/xraygrid/sigma*.yaml` and `sampling/wire_dps_xray.py`.

## Then: run LACToS on the samples

```bash
RECON_ROOT=recons_mine GPU=0 bash scripts/run_reproduce.sh <category | CT | xray | faces>   # CT: add CT_SUB=<name>
bash scripts/score_metrics.sh && python scripts/summarize_results.py
```

See [docs/PIPELINE.md](../docs/PIPELINE.md) for the complete walk-through.
