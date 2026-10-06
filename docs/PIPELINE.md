# Step-by-step: from the datasets to LACToS scores

LACToS runs in four steps. Each step writes a folder that the next step reads.

```
Step 0  download        data/ (test images + GT)  weights/ (AEs, backbones, diffusion priors)
Step 1  sample          recons_mine/results_patches_*/samples_<id>/ddad_native_0_4/inpainting/{recon,label,input}/
Step 2  check layout    (quick sanity check of the Step-1 folders)
Step 3  run LACToS      results_eval_reproduction/{json,maps,logs}/
Step 4  metrics         results_eval_reproduction/metrics/ + comparison table
```

To reproduce the paper numbers **without sampling**, download the released reconstructions and skip
Steps 1-2 (see "Shortcut" at the end).

---

## Step 0: download the datasets and weights

All assets are zip files with SHA-256 checksums (`SHA256SUMS`); see [DATA.md](DATA.md) for what each
contains and where it comes from.

| Zip | Contents | Size | Needed for |
|---|---|---:|---|
| `lactos_data.zip` | `data/`: test images + GT masks (15 MVTec categories, CT, X-ray, faces) | 1.6 GB | all steps |
| `lactos_weights.zip` | `weights/ae`, `weights/backbones`, `weights/torch`: 18 autoencoders + ResNet-101 backbones | 0.6 GB | Step 3 |
| `lactos_diffusion.zip` | `weights/diffusion/`: diffusion priors (CT, X-ray, FFHQ) | 0.8 GB | Step 1 |
| `lactos_recons_<target>.zip` | released reconstructions, one zip per target | 0.1–1.1 GB | shortcut only |

```bash
bash scripts/download_assets.sh weights data diffusion       # what Steps 1-3 need
# or download the zips manually and unzip them in the repository root:
unzip lactos_data.zip && unzip lactos_weights.zip && unzip lactos_diffusion.zip
bash scripts/check_env.sh                                   # checksums, files, python env, unit tests
```

---

## Step 1: sample M diffusion reconstructions per test image

Every sampler writes to `recons_mine/` (override with `RECON_ROOT=...`), using the folder names Step 3
expects. The released `recons/` folder is never touched. Commands below are the settings of the
released sets; all sampler hyperparameters are in [HYPERPARAMETERS.md §5](HYPERPARAMETERS.md).

### MVTec AD (DDAD-guided denoising, 40 draws)

Needs the DDAD UNet of the category at `weights/diffusion/mvtec/<category>/<epoch>` (not yet released;
epochs in [DATA.md](DATA.md)).

```bash
GPU=0 bash sampling/sample_mvtec_ddad.sh cable weights/diffusion/mvtec/cable/3000
#  -> recons_mine/results_patches_ddad_native_cable_<subcategory>_1/   (every test subcategory)
```

Settings: w = 2, t* = 250, skip = 25, eta = 1, draw k seeded with k. Quick test: `MAX_IMAGES=1`.

### CT (DPS denoising, σ = 0.05, ζ′ = 0.2, 10 draws)

```bash
GPU=0 bash sampling/sample_ct_dps.sh mine                    # all 100 slices
SHARDS=4 bash sampling/sample_ct_dps.sh mine                 # split over GPUs 0-3 (identical draws)
#  -> recons_mine/results_patches_ddad_native_CT_mine/
```

Arguments: `[name] [sigma] [scale] [draws]`; ζ′ = scale / 2. Other σ: `sampling/dps/configs/ctgrid/`.
Quick test: `START=0 END=0`.

### X-ray and faces (DPS superpixel-patch inpainting)

```bash
GPU=0 bash sampling/sample_patch_dps.sh xray                 # sigma 0.3, 6 patches x 4 draws per image
GPU=0 bash sampling/sample_patch_dps.sh faces                # sigma 0.5, 12 patches x 4 draws per image
#  -> recons_mine/results_patches_ddad_native_xray_scissors_1/ , ..._faces_random_1/
```

Arguments: `<xray|faces> [sigma] [image_dir] [suffix]`. The DPS guidance scale of the released X-ray and
faces sets was not recorded, so these two samplers reproduce the procedure, not necessarily identical
reconstructions. This script has not yet been run end-to-end on the release (the MVTec/DDAD and CT scripts
have been smoke-tested).

### Your own sampler

Any conditional diffusion sampler can be used. Write M reconstructions per test image as

```
<RECON_ROOT>/results_patches_ddad_native_<category>_<subcategory><suffix>/
    samples_<image id>/ddad_native_0_4/inpainting/recon/<k>_0_00000.png   (k = 0..M-1)
                                                 /label/0_00000.png       (the test image itself)
                                                 /input/0_00000.png       (the measurement; optional)
```

where `<image id>` is the file name of the test image in `data/mvtec_ad/<category>/test/<subcategory>/`
(its GT mask is looked up by the same id).

---

## Step 2: check the sampled folders

```bash
find recons_mine -maxdepth 1 -type d                              # one folder per (category, subcategory)
d=recons_mine/results_patches_ddad_native_CT_mine
ls $d | head -3                                                   # samples_image_0000, ...
ls $d/samples_image_0000/ddad_native_0_4/inpainting/recon | wc -l # = number of draws M
```

---

## Step 3: run LACToS on the sampled reconstructions

The same script evaluates the released reconstructions or your own. Point it at your Step-1 output with
`RECON_ROOT` (and `CT_SUB` for the CT folder name):

```bash
RECON_ROOT=recons_mine GPU=0 bash scripts/run_reproduce.sh cable              # MVTec category
RECON_ROOT=recons_mine CT_SUB=mine GPU=0 bash scripts/run_reproduce.sh CT
RECON_ROOT=recons_mine GPU=0 bash scripts/run_reproduce.sh xray faces
```

Per target this computes ResNet-101 features → autoencoder latents → superpixels → the typical-set
score Δ = (NLL_rgb − H_rgb) + (NLL_lat − H_lat) for every superpixel, and writes per-image delta maps:

```
results_eval_reproduction/json/<target>/   per-subcategory results (pixel AUROC etc. at several σ)
results_eval_reproduction/maps/<target>/   delta maps (npz: delta_map, gt_mask, valid_mask)
results_eval_reproduction/logs/<target>.log
```

Settings per target (bins, superpixel size, AE, backbone) are fixed in `scripts/run_reproduce.sh` and
documented in [HYPERPARAMETERS.md §3](HYPERPARAMETERS.md). Use `OUT=...` to write elsewhere.

---

## Step 4: metrics and comparison

```bash
bash scripts/score_metrics.sh            # AUROC, AP, F1max, AUPRO, SNR, MSE from the delta maps (CPU)
python scripts/summarize_results.py      # side by side with the paper numbers
```

---

## Shortcut: reproduce the paper numbers without sampling

```bash
bash scripts/download_assets.sh weights data bottle cable CT    # any targets, or no arguments for all
GPU=0 bash scripts/run_reproduce.sh bottle cable CT              # reads the released recons/
bash scripts/score_metrics.sh && python scripts/summarize_results.py
```

Expected numbers: [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
