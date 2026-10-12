# Reproducibility report

All paper numbers come from **evaluation only**: the released diffusion reconstructions and
checkpoints are fixed inputs, and no evaluation run subsamples reconstructions. Results are
deterministic up to floating-point summation order on the GPU.

## Verified run (2026-10-05)

* Code: this repository. Assets: the release zip files (`weights.sha256` for the weights).
* Environment of this verification: Python 3.9.25, torch 2.8.0+cu128, torchvision 0.23.0, numpy 1.23.5,
  scipy 1.9.3, scikit-image 0.22.0, scikit-learn 1.6.1, NVIDIA RTX A6000.
* The original paper runs used the environment in `environment.yml` (Python 3.10.20, torch 2.11.0+cu130,
  torchvision 0.26.0, numpy 2.2.6, scipy 1.15.3, scikit-image 0.25.2, scikit-learn 1.7.2). The numbers
  below reproduce across these two quite different stacks.
* Command: `GPU=… bash scripts/run_reproduce.sh` (all 20 targets, three GPUs in parallel),
  then `bash scripts/score_metrics.sh && python scripts/summarize_results.py`.
* The four targets that exercise different code paths (toothbrush, CT, xray, faces_moreface) were
  re-run from a clean checkout with the release directory layout and an empty superpixel cache.
  They match the full run to about 1e-8.
* Outputs: [`reference_results/`](../reference_results) (per-subcategory JSONs and metric JSONs).
  `OUT=reference_results python scripts/summarize_results.py` prints the tables below.

### A. Pixel ROC-AUC at eval σ = 5 (mean over defect subcategories)

| Target | Reproduced | Paper | Δ |
|---|---:|---:|---:|
| bottle | 0.9771 | .9771 | 0 |
| cable | 0.9569 | .9569 | 0 |
| capsule | 0.9554 | .9553 | +0.0001 |
| carpet | 0.9259 | .9259 | 0 |
| grid | 0.9126 | .9126 | 0 |
| hazelnut | 0.9688 | .9688 | 0 |
| leather | 0.9841 | .9842 | −0.0001 |
| metal_nut | 0.9580 | .9580 | 0 |
| pill | 0.9193 | .9193 | 0 |
| screw | 0.9232 | .9231 | +0.0001 |
| tile | 0.9006 | .9007 | −0.0001 |
| toothbrush | 0.9478 | .9477 | +0.0001 |
| transistor | 0.9432 | .9432 | 0 |
| wood | 0.8912 | .8912 | 0 |
| zipper | 0.9666 | .9666 | 0 |
| **MVTec mean** | **0.9420** | **.9420** | 0 |
| CT (100 slices, DPS σ=0.05) | 0.9463 | .9463 | 0 |
| X-ray b1 (32 patients) | 0.9526 | .9528 | −0.0002 |
| faces random22 | 0.9105 | .9105 | 0 |

The ±0.0001 differences are 4th-decimal rounding of the reference values. For example, toothbrush
is 0.947798 in both the 2026-09-22 and 2026-10-05 runs. X-ray: superpixel target size 30, the
setting in `run_reproduce.sh`, gives 0.9526; target size 10 gives 0.9527. Neither reproduces 0.9528
exactly. The difference is below the precision reported in the paper's tables.

### B. Full metric table (image-averaged; `tools/metrics/full_metric_table.py`, σ = 5)

| Dataset | AUROC | SNR | AP | MSE | F1max (img) | F1max (pooled) | AUPRO (FPR ≤ 0.3) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MVTec (15 cats) | 94.13 | 4.81 | 49.25 | 0.0491 | 51.52 | 46.11 | 79.41 |
| CT (100) | 94.63 | 1.84 | 48.53 | 0.2475 | 49.55 | 37.64 | 82.70 |
| faces (50 = random22 + moreface28) | 84.09 | 2.83 | 29.85 | 0.0665 | 34.26 | 28.15 | 45.63 |
| X-ray (72 = b1 + b2) | 94.22 | 2.79 | 11.67 | 0.0253 | 18.24 | 10.37 | 72.09 |

Every entry matches the reference to within 0.03 points. CT is the DPS-denoising set (σ = 0.05, ζ′ = 0.2,
10 draws); its numbers match the original 2026-09-25 run (`cts005z02full`) exactly. An earlier CT set
made with the DDAD-guided sampler (40 draws) gives 0.9378 and is not part of the release. Pooled-mode rows are printed by
`summarize_results.py` as well.

## Determinism notes

* `tools/seeded_run.py` seeds python, numpy and torch (`ABL_SEED=0`). No paper run actually draws
  random numbers, because `--max_reconstructions` is never smaller than the number of shipped draws.
* cuDNN determinism flags are not set. Convolution summation order can change the last bits of
  features, and this shows up as rare 1e-4 changes in per-category AUROC.
* Superpixels are recomputed from the test images when `figures*/` is absent. This is deterministic,
  so the cache is not shipped.
* Two different ImageNet backbones are used, matching the original runs. See docs/DATA.md.

## Weight checksums

`sha256sum -c weights.sha256` (also run by `scripts/check_env.sh`).

## What is *not* reproducible from this repository alone

* **Reconstructions.** They are released as data. Re-running a sampler with the recorded seeds gives
  statistically equivalent but not bit-identical draws: regenerating CT image_0011 with the DDAD sampler
  and seeds 0-1 differs from the original draw by 1.0 grey level on average (correlation 0.9985), versus
  2.3 between two different seeds, because GPU kernels and library versions changed. Regenerating them needs the diffusion checkpoints
  (not released) and, for faces and X-ray, an external DPS patch-inpainting codebase. Sampler settings
  are recorded in docs/HYPERPARAMETERS.md §5. Some are marked UNKNOWN because no metadata was saved.
* **AE training.** `embedder/train_pixel_ae_interp.py` (seeded) and `embedder/train_embedder.py`
  (original recipe) are included. The released checkpoints are the ones behind every number
  above.
