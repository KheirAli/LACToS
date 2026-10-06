# LACToS — local anomaly detection from diffusion reconstructions

LACToS is a method for OOD detection in inverse problems.

<!-- TODO(maintainers): paper title, authors, venue, arXiv link, teaser figure -->

Pixel-level anomaly / OOD localization. For each test image we draw several diffusion
reconstructions, embed image and reconstructions with ResNet-101 pixel features (1792-d),
compress them with a small per-target **PixelAutoEncoder (1792 → 3)**, and score every
superpixel (`ood/superpixels.py`) by how far its RGB and latent histograms fall outside the
*typical set* of the reconstructions (average NLL minus entropy). The superpixel scores
form a delta map, which is smoothed and compared against the ground-truth masks.

```
test image ──► M conditional diffusion reconstructions (shipped)
           ──► ResNet-101 pixel features ──► PixelAutoEncoder (1792→3)
           ──► superpixels ──► typical-set score  Δ = (NLL_rgb − H_rgb) + (NLL_lat − H_lat)
           ──► Gaussian smoothing (σ=5) ──► pixel AUROC / AP / F1max / AUPRO / SNR
```

**How the RGB and latent terms are combined.** For each superpixel, the RGB values and the
autoencoder latents of the reconstructions each get their own histogram, and the score is the sum
of the two typical-set deviations:

    Δ = (NLL_rgb − H_rgb) + (NLL_lat − H_lat)

This sum is exactly the joint typical-set score under a factorized model p(rgb, z) = p(rgb) · p(z)
(joint entropy = H_rgb + H_lat, joint NLL = NLL_rgb + NLL_lat). We use the addition for
computational reasons: a full joint histogram over (R, G, B, z₁, z₂, z₃) would have 32⁶ ≈ 10⁹ bins
per superpixel, while each superpixel only provides ~25 pixels × M reconstructions (~10³ samples),
so it could neither be stored nor estimated reliably.

**One algorithm for every dataset.** LACToS has a single pipeline: (1) draw M diffusion
reconstructions conditioned on the test image, (2) embed and compress them, (3) score superpixels with
the typical-set deviation above. Steps 2–3 are identical for all datasets. Step 1 only needs *some*
conditional diffusion sampler that returns M plausible in-distribution versions of the test image; the
released reconstruction sets were produced with:

| Data | Conditional sampler | Draws per image |
|---|---|---:|
| MVTec AD (15 categories) | DDAD-guided denoising (w = 2, t* = 250, DDAD UNet per category) | 40 (pill 32) |
| CT | DPS denoising, y = x + σn with σ = 0.05, ζ′ = 0.2 (CT diffusion prior) | 10 |
| X-ray (DvXray), faces | DPS patch inpainting (Chung et al., 2023) | 16–48 |

Sampler details and how to regenerate or add reconstructions: [sampling/README.md](sampling/README.md).
Diffusion models: the CT, X-ray and FFHQ priors are released (`weights/diffusion/`); the per-category
MVTec DDAD UNets are **not yet released** (epochs listed in [docs/DATA.md](docs/DATA.md)). None of them
are needed to reproduce the numbers, because the reconstructions themselves are released.

Evaluated on all 15 MVTec-AD categories, a CT anomaly set (100 slices), baggage
X-ray (DvXray, scissors class, 72 patients) and masked faces (50 images).

## Results (pixel ROC-AUC, eval σ = 5)

| bottle | cable | capsule | carpet | grid | hazelnut | leather | metal_nut | pill | screw | tile | toothbrush | transistor | wood | zipper | **MVTec mean** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| .9771 | .9569 | .9553 | .9259 | .9126 | .9688 | .9842 | .9580 | .9193 | .9231 | .9007 | .9477 | .9432 | .8912 | .9666 | **.9420** |

CT .9463 · X-ray (b1, 32 patients) .9528 (reproduces as .9526) · faces (random22) .9105.
Full metric table (AUROC, SNR, AP, MSE, F1max, AUPRO) and the verification log:
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Workflow at a glance

| Stage | What | Entry point | Needed to reproduce the paper? |
|---|---|---|---|
| 0 | Get data, weights and reconstructions | `scripts/download_assets.sh` → [docs/DATA.md](docs/DATA.md) | yes |
| 1 | Conditional diffusion reconstructions (M per test image) | `sampling/sample_mvtec_ddad.sh`, `sampling/sample_ct_dps.sh`, `sampling/sample_patch_dps.sh` ([sampling/README.md](sampling/README.md)) | no (released) |
| 2 | Train the per-target PixelAutoEncoder | `embedder/train_pixel_ae_interp.py` (seeded), `embedder/train_embedder.py` (original recipe) | no (released) |
| 3 | Run LACToS on the reconstructions (released or `RECON_ROOT=recons_mine`) | `scripts/run_reproduce.sh` → `evaluate.py`, `embedder/evaluate_ddad_fe.py` | yes |
| 4 | Metrics + comparison with the paper | `scripts/score_metrics.sh`, `scripts/summarize_results.py` | yes |
| 5 | Tests | `pytest -m "not requires_ckpt" tests/`, `scripts/check_env.sh` | recommended |

**Step-by-step guide (download → sample → run LACToS → metrics): [docs/PIPELINE.md](docs/PIPELINE.md).**
Every hyperparameter of every stage: **[docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md)**.
Verified numbers and determinism notes: **[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)**.

## 1. Install

```bash
git clone https://github.com/KheirAli/LACToS.git && cd LACToS
conda env create -f environment.yml          # exact reference env: python 3.10.20, torch 2.11.0 (cu130)
conda activate lactos
# or: pip install -r requirements.txt  (install the torch build matching your CUDA first)
```

## 2. Download assets (weights, test data, reconstructions)

| File | Contents | Size | Download |
|---|---|---:|---|
| `lactos_weights.tar.gz` | 18 autoencoders + ResNet-101 backbones (needed to run LACToS) | 0.6 GB | [Google Drive](https://drive.google.com/file/d/1N6PpLGLLnyDuDG8ecb0trPyS_rbsLM-x/view?usp=drive_link) |
| `lactos_diffusion.tar` | diffusion priors for CT, X-ray, FFHQ (only needed to re-sample) | 0.9 GB | [Google Drive](https://drive.google.com/file/d/10jF91HOyhw6VAEEBWKM7GoLMwdcBisaB/view?usp=drive_link) |
| `lactos_data.zip`, `lactos_recons_<target>.zip` | test images + GT masks; released reconstructions | 1.6 GB; 7 GB | coming soon |

```bash
bash scripts/download_assets.sh weights diffusion    # downloads from Google Drive, checks SHA-256, extracts here
# or download manually and extract in the repository root:
tar xzf lactos_weights.tar.gz && tar xf lactos_diffusion.tar
bash scripts/check_env.sh                            # checksums, files, env, unit tests
```

What each file is and where it came from: [docs/DATA.md](docs/DATA.md).

## 3. Reproduce the paper numbers

Evaluation only: it reads the shipped reconstructions and checkpoints and is deterministic
(seeded through `tools/seeded_run.py`, `ABL_SEED=0`).

```bash
GPU=0 bash scripts/run_reproduce.sh                  # all 20 targets, ~3-6 h on one RTX A6000 (~15 GB VRAM)
GPU=0 bash scripts/run_reproduce.sh toothbrush CT    # any subset
bash scripts/score_metrics.sh                        # full metric table from the delta-map dumps (CPU)
python scripts/summarize_results.py                  # reproduced vs. reference numbers
```

Targets: `bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush
transistor wood zipper CT xray xray_b2 faces faces_moreface`. Run several in parallel with different
`GPU=` values; outputs (`results_eval_reproduction/{json,maps,logs,metrics}`) do not collide.

Every flag, default and per-target setting is documented in
**[docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md)**.

## 4. Use it on your own data

* **Score your own heat maps with our metrics.** `python tools/metrics/full_metric_table.py --maps_dir DIR --out OUT.json`
  reads `.npz` files with `delta_map` (float32 H×W), `gt_mask` (uint8), `valid_mask` (uint8).
* **New category.** Produce reconstructions in the `recons/` layout (`tools/generate_recons.py`,
  needs a DDAD UNet), train an AE with `embedder/train_pixel_ae_interp.py --latent_dim 3 --num_epochs 30
  --seed 0 --n_restarts 3`, then run `evaluate.py` with the MVTec flags from `scripts/run_reproduce.sh`.

## Repository layout

```
evaluate.py                 main evaluation CLI (scorer x config dispatch)
super_pixel_generation.py   superpixel mask generation (cached under figures*/, regenerated on demand)
ood/                        core package: data loading, superpixels, embeddings, typical-set scoring, metrics, samplers
embedder/                   evaluate_ddad_fe.py (backbone-weight injection), train_pixel_ae_interp.py / train_embedder.py (AE training)
sampling/                   Step 1: sample_mvtec_ddad.sh, sample_ct_dps.sh, sample_patch_dps.sh, DPS code (sampling/dps/), README
tools/                      seeded_run.py, generate_recons.py (DDAD-guided sampler driver), prepare_xray_dataset.py, metrics/
configs/                    evaluation YAMLs; configs/recon/ = sampler YAMLs for tools/generate_recons.py
scripts/                    run_reproduce.sh, score_metrics.sh, summarize_results.py, check_env.sh, download/make assets
DDAD/                       DDAD implementation (Mousakhan et al.): UNet training + DDAD-guided sampling backend (MVTec)
tests/                      pytest suite (`pytest -m "not requires_ckpt"` needs no checkpoints)
docs/                       PIPELINE.md (step by step), HYPERPARAMETERS.md, DATA.md, REPRODUCIBILITY.md
reference_results/          JSON outputs of the verified run (compare with your own)
```

## Citation

See [CITATION.cff](CITATION.cff). <!-- TODO(maintainers): BibTeX -->

## Acknowledgements and licences

* **DDAD** — Mousakhan et al., *Anomaly Detection with Conditioned Denoising Diffusion Models*, 2023 (`DDAD/`, reconstruction sampler).
* **DPS** — Chung et al., *Diffusion Posterior Sampling*, ICLR 2023 (CT, X-ray and face reconstructions).
* **MVTec AD** — Bergmann et al.; data under CC BY-NC-SA 4.0.

Code licence: **not yet chosen** — a LICENSE file will be added before the public release. <!-- TODO(maintainers): choose licence; check DDAD upstream licence for DDAD/ -->
