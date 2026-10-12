# LACToS method

This document describes the algorithmic details kept out of the main [README](../README.md). In the repository, the main landing page should be named `README.md` at its root.

## Overview

LACToS performs pixel-level anomaly localization using a collection of conditional diffusion reconstructions of each test image. The scoring stage is shared across datasets; the diffusion reconstruction model varies with the image domain and observation process.

1. Generate multiple in-distribution reconstructions conditioned on the test observation.
2. Extract ResNet-101 pixel features (1792 dimensions) and compress them using a per-target PixelAutoEncoder (1792 to 3 dimensions).
3. Partition the image into superpixels and compare RGB and latent-value histograms to the corresponding distributions from the reconstructions.
4. Map superpixel scores to pixels and smooth the anomaly map before metric computation.

## Typical-set deviation

For each superpixel, LACToS computes typical-set deviations for RGB and learned latent features. The two contributions are added:

$
\Delta = (\mathrm{NLL}_{\mathrm{RGB}} - H_{\mathrm{RGB}})
       + (\mathrm{NLL}_{\mathrm{latent}} - H_{\mathrm{latent}}).$

Here NLL denotes negative log-likelihood under reconstruction-derived histograms, and \(H\) denotes the corresponding entropy. The expression can be interpreted as a joint typical-set deviation **under a factorized RGB/latent model**; it is not an estimate of a full six-dimensional joint histogram. The original implementation uses separate histograms because a joint histogram at 32 bins per dimension would require \(32^6\) bins per superpixel, far more than the available samples support.

The original README reports Gaussian smoothing with \(\sigma=5\) for the anomaly map. Per-target settings and other numerical choices belong in [Hyperparameters](HYPERPARAMETERS.md).

## Reconstruction backends

| Domain | Reconstruction method | Notes |
|---|---|---|
| MVTec AD | DDAD-guided denoising | Different UNet per category; the referenced per-category checkpoints are not yet released |
| CT | DPS denoising| CT diffusion prior available in the diffusion-weight archive |
| X-ray and faces | DPS denoising | Diffusion priors available in the diffusion-weight archive |

Precomputed reconstructions can be used without rerunning diffusion sampling. See [Sampling](../sampling/README.md) for scripts and reconstruction formats, and [Data and assets](DATA.md) for checkpoint availability.

## Implementation pointers

- `ood/superpixels.py` — superpixel processing
- `ood/` — feature, scoring, data, and metric code
- `embedder/train_pixel_ae_interp.py` — pixel-autoencoder training
- `embedder/evaluate_ddad_fe.py` — feature-model evaluation
- `evaluate.py` — scoring entry point
- `scripts/run_reproduce.sh` — configured evaluation invocation

For execution steps see [Pipeline](PIPELINE.md); for validated metrics see [Reproducibility](REPRODUCIBILITY.md). All implementation details here are based on the supplied project README; they have not been independently checked against the full repository source.
