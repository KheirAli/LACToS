# LACToS: Local Anomaly Detection from Diffusion Reconstructions

**LACToS** localizes out-of-distribution (OOD) content in images using multiple diffusion reconstructions of the observed image. It produces a pixel-level anomaly map from a clean-data diffusion prior, without requiring training examples of every possible anomaly.

The repository includes experiments on **MVTec AD (15 categories), CT, baggage X-ray, and face images**. For the method and mathematical details, see [Method](docs/METHOD.md); for the sampling backends, see [Sampling](sampling/README.md).

## Quick start

**Requirements:** Conda, a compatible NVIDIA GPU, and a working CUDA installation. The reference environment is specified in [`environment.yml`](environment.yml).

```bash
# 1. Clone and install
git clone https://github.com/KheirAli/LACToS.git
cd LACToS
conda env create -f environment.yml
conda activate lactos

# 2. Download the pretrained feature models and benchmark data
bash scripts/download_assets.sh weights data

# 3. Check the environment and downloaded files
bash scripts/check_env.sh

# 4. Run one benchmark (once its diffusion reconstructions are available)
GPU=0 bash scripts/run_reproduce.sh toothbrush
```

Results are written under `results_eval_reproduction/` (including anomaly maps, logs, and result files).

**Important — reconstruction availability:** LACToS evaluation also requires precomputed diffusion reconstructions. The reconstruction archives are marked *coming soon* in the supplied release information, so the final command cannot be run from the currently listed public downloads alone. Once published, place them in the layout described in [Data and assets](docs/DATA.md). Alternatively, generate reconstructions using [Sampling](sampling/README.md); some sampling weights, including the MVTec DDAD UNets, are not yet publicly released.

## Run LACToS

After installing the environment and downloading the assets and reconstructions required for your selected target:

```bash
# Run a single category
GPU=0 bash scripts/run_reproduce.sh bottle

# Run a subset
GPU=0 bash scripts/run_reproduce.sh toothbrush CT

# Run all configured targets
GPU=0 bash scripts/run_reproduce.sh

# Compute metrics from the generated anomaly maps
bash scripts/score_metrics.sh
python scripts/summarize_results.py
```

For the complete sequence of inputs, outputs, and options, see [Pipeline guide](docs/PIPELINE.md). Configuration choices and hyperparameters are in [Hyperparameters](docs/HYPERPARAMETERS.md); reported results and reproduction checks are in [Reproducibility](docs/REPRODUCIBILITY.md).

## Download models and data

| Asset | Purpose | Download |
|---|---|---|
| Feature models (`lactos_weights.tar.gz`) | ResNet-101 and pretrained pixel autoencoders used by LACToS | [Google Drive](https://drive.google.com/file/d/1N6PpLGLLnyDuDG8ecb0trPyS_rbsLM-x/view?usp=drive_link) |
| Benchmark data (`lactos_data.zip`) | Test images and ground-truth masks | [Google Drive](https://drive.google.com/file/d/1YS7Z4Sd6dTJSaipUpTjY4X4SSqj38Snv/view?usp=sharing) |
| Diffusion priors (`lactos_diffusion.tar`) | CT, X-ray, and face reconstruction generation; **not needed** when reconstructions are already available | [Google Drive](https://drive.google.com/file/d/10jF91HOyhw6VAEEBWKM7GoLMwdcBisaB/view?usp=drive_link) |
| Precomputed reconstructions (`lactos_recons_<target>.zip`) | Input reconstruction sets for the scoring pipeline | **Not yet released** |

To download only the assets needed for LACToS scoring, use `bash scripts/download_assets.sh weights data`. To obtain the available diffusion priors as well, run:

```bash
bash scripts/download_assets.sh diffusion
```

Manual download, expected file locations, checksums, and dataset licenses are described in [Data and assets](docs/DATA.md). Do not assume third-party datasets or weights may be redistributed beyond their license terms.

## Supported applications

| Application | Datasets / inputs | Reconstruction approach |
|---|---|---|
| Industrial anomaly localization | MVTec AD, 15 categories | DDAD-guided diffusion |
| Medical imaging | CT anomaly images | Diffusion posterior sampling (DPS) |
| Security imaging | DvXray baggage images | DPS inpainting |
| Face-image anomalies | Masked face images | DPS inpainting |

LACToS takes a set of plausible reconstructions and outputs localized anomaly scores. To process a **new dataset**, first prepare reconstructions and the necessary feature-model assets, then follow the dataset setup and scoring instructions in [Pipeline guide](docs/PIPELINE.md) and [Sampling](sampling/README.md). The repository does not currently document a verified, one-command inference interface for arbitrary user images.

## Documentation

- [Method](docs/METHOD.md) — algorithm, mathematical formulation, and implementation pointers
- [Pipeline](docs/PIPELINE.md) — setup, reconstruction inputs, execution, and output formats
- [Sampling](sampling/README.md) — generating diffusion reconstructions
- [Data and assets](docs/DATA.md) — downloads, directory layout, and model availability
- [Hyperparameters](docs/HYPERPARAMETERS.md) — configurations and per-dataset settings
- [Reproducibility](docs/REPRODUCIBILITY.md) — metrics, results, and verification notes

## Repository structure

```text
ood/                Core anomaly-scoring and data-processing code
embedder/           Pixel-autoencoder training and feature evaluation
sampling/           Conditional diffusion reconstruction methods
configs/            Evaluation and reconstruction configurations
scripts/            Download, execution, and reporting scripts
tools/              Dataset preparation and supporting utilities
docs/               Method, usage, asset, and reproducibility guides
tests/              Automated tests
reference_results/  Reference outputs for comparison
```

For a basic test that does not require model checkpoints:

```bash
pytest -m "not requires_ckpt" tests/
```

## Citation

Citation metadata is maintained in [`CITATION.cff`](CITATION.cff). Please use the publication details listed there when citing this work.

## Acknowledgments and licensing

This work builds on [DDAD](DDAD/) (Mousakhan et al.), Diffusion Posterior Sampling (Chung et al., ICLR 2023), and MVTec AD (Bergmann et al.). Respect their original licenses and attribution requirements.

**Release note:** The supplied repository documentation does not yet identify a finalized top-level software license. Confirm the license and third-party redistribution permissions before publishing the repository or model archives.
