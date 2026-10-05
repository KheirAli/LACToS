# LACToS hyperparameter reference

This document lists every hyperparameter that affects the numbers reported for LACToS: what the code
sets by default, what the paper runs used, and where each value is set. It was compiled by reading the
code, configs, scripts, checkpoints and reconstruction metadata, and checked against the verified
reproduction of 2026-10-05 (docs/REPRODUCIBILITY.md).

**Provenance notes.** Evidence columns sometimes cite internal lab scripts and notes that are *not* part
of this repository (`run_train_ae.sh`, `rerun_all15_fixed_ae.sh`, `repro_orig_ae.sh`,
`xray_finetune_mine.sh`, `ddad_da_finetune.py`, `tools/sigma_sampling_20260916/*`, `gen_jobs.py`,
`REPRODUCE_RESULTS.md`, `RUNBOOK.md`). They are cited only to document how a released checkpoint or
reconstruction set was produced. Everything needed to *reproduce the numbers* is in this repository
and its release assets.

Conventions:

* `file:line` points at the **live** code. `evaluate.py` lines 1-~800, `ood/embeddings.py` lines
  1-260, `ood/superpixels.py` lines 1-246 and 384-462, and parts of `ood/scoring.py`, `ood/metrics.py`
  and `ood/data.py` are commented-out legacy copies. Python ignores them, and so does this document.
* "Paper" means the invocation in `scripts/run_reproduce.sh` followed by `scripts/score_metrics.sh`
  (`tools/metrics/full_metric_table.py`). Section 7 lists every place where the docs disagree with those
  scripts.
* **UNKNOWN — reason** marks a value that could not be determined from the repository.
* **Inferred** marks a value derived indirectly, for example from BatchNorm `num_batches_tracked`
  counters in checkpoints. The derivation is shown next to each one.

---

## 1. Pipeline overview

LACToS scores a test image `x` as follows.

**(A) Reconstruction (offline; outputs are shipped, `--skip_sampling`).** A conditional diffusion
sampler produces M stochastic reconstructions of `x`. This is the only dataset-dependent stage; stages
B–F are identical for all datasets.
* MVTec uses **DDAD-guided** conditioned denoising (`ood/samplers/ddad_native.py`, via
  `tools/generate_recons.py`).
* CT uses **DPS denoising** (`sampling/dps/sample_fast.py`): measurement y = x + σn with σ = 0.05,
  ζ′ = 0.2, CT diffusion prior, 10 draws per slice.
* Faces and x-ray use **DPS patch inpainting**.
* All sets are stored in the same `ddad_native_0_4` directory layout, so the evaluation reads them
  identically.

**(B) Feature embedding.** Each reconstruction and the input image pass through a ResNet-101.
* The backbone is either torchvision ImageNet weights or weights injected through `DDAD_FE_WEIGHTS`.
* Features are taken from layers 1-3, averaged over 3x3 patches, bilinearly upsampled to 256x256,
  L2-normalised per layer, and concatenated into **1792 channels**.

**(C) Feature reduction.** A per-target **PixelAutoEncoder** (1x1-conv MLP, 1792->512->256->64->3 with
`tanh` output) maps each pixel feature to 3 dimensions. Only the encoder is used at test time. These AEs
are trained offline (Section 4).

**(D) Superpixels.** The input image is over-segmented. The live code uses a *compact watershed on a
regular marker grid*, not recursive SLIC (Section 2.4). A cached coarse SLIC `mask.png` is loaded, but it
does not affect the typical-set score.

**(E) Typical-set scoring.** For each superpixel, the M reconstructions give two histograms:
* an RGB histogram (`bins_rgb` per channel; one grey channel for CT);
* a 3-D histogram of the AE latent (`bins_pca` per dimension).

Both are optionally smoothed in bin space (`smooth_sigma`). The superpixel score is

`delta = (NLL_rgb(x) − H_rgb) + (NLL_feat(x) − H_feat)`

The RGB and latent histograms are built separately and their deviations are added. This equals the
joint typical-set score under a factorized model p(rgb, z) = p(rgb)·p(z); the addition is used for
computational reasons, since a joint 6-D histogram (32⁶ ≈ 10⁹ bins per superpixel) cannot be stored
or estimated from the ~10³ samples a superpixel provides.

where NLL is the mean negative log₂-likelihood of the input pixels under the histogram and H is the
histogram's entropy, both in bits.

**(F) Map post-processing and metrics.** The delta map is resized to 256x256 and dumped to `.npz`. At
evaluation it is Gaussian-smoothed in pixel space with the **eval sigma** (5 for all reported numbers),
then scored with pixel AUROC/AP, F1max, AUPRO (FPR <= 0.3), SNR and mask-MSE/PSNR.

---

## 2. Per-stage parameter tables

### 2.0 Entry points, CLI flags and how they reach `cfg`

`evaluate.py` CLI: `evaluate.py:1745-1817`. Every flag has `default=None` (or `store_true`) except
`--results_suffix` (`""`) and `--scorer` (`typical_set`). A flag is applied only when it is given, so the
YAML value applies otherwise.

| CLI flag | type / default (`evaluate.py`) | maps to | effect / notes |
|---|---|---|---|
| `--config` | str, required (1747) | YAML loaded by `load_config` | base config |
| `--category` | str, None (1748) | `category` (falls back to `data.category`, else `"cable"`, 1820) | prefix of the recon directory glob |
| `--subcategory` | str, None (1749) | one subcategory | |
| `--all_subcategories` | flag (1750) | `discover_subcategories(results_root, category, suffix)` (1140) | globs `{results_root}_{category}_*{suffix}`. **Every** matching directory becomes a subcategory, so keep only the released recon dirs under `recons/` |
| `--results_root` | str, None (1751) | `cfg.data.results_root` (1809) | recon dir = `{results_root}_{category}_{subcat}{suffix}` (1117) |
| `--results_suffix` | str, `""` (1752) | `cfg.data.results_suffix` (1810) | an empty value cannot override the YAML (`if args.results_suffix:`) |
| `--gt_root` | str, None (1753) | `cfg.data.gt_root` (1811) | GT template candidates `{gt_root}/{cat}/{sub}/`, `{gt_root}/{sub}/`, `{gt_root}/` (1124). `dirname(gt_root)/test/{sub}` becomes `image_dir` for superpixel generation (1492). **Ignored for CT** (CT masks come from `OOD_CT_MASK_ROOT`) |
| `--sample_name` / `--sample_names` | str / list, None (1754-1755) | samples to run | default: every `samples_*` directory |
| `--skip_sampling` | flag (1756) | **no-op** | sampling is always disabled: `cfg.sampling.enabled=False` is set unconditionally (1791) |
| `--no_plots` | flag (1757) | `cfg.eval.save_plots=False` (1795) | speed only |
| `--device` | str, None (1758) | `cfg.embeddings.device` (1799) | YAML default `cuda` |
| `--output_dir` | str, None (1759) | `cfg.eval.output_dir` (1800) | |
| `--backbone` | str, None (1760) | `cfg.embeddings.backbone` (1801) | resnet18/50/101/152. Anything except resnet101 changes the 1792 dimension and breaks the AEs |
| `--n_pca` | int, None (1761) | `cfg.pca.n_components` (1802) | used as the **AE latent_dim** (AE path) or the PCA rank (PCA path). Must equal the AE checkpoint's latent size (3) |
| `--bins_rgb` | int, None (1762) | `cfg.scoring.bins_rgb` (1803) | RGB bins per channel |
| `--bins_pca` | int, None (1763) | `cfg.scoring.bins_pca` (1804) | latent bins per dimension |
| `--superpixel_target_size` | int, None (1764) | `cfg.superpixels.target_size` (1813) | sets the watershed marker grid step (2.4) |
| `--smooth_sigma` | float, None (1765) | `cfg.scoring.smooth_sigma` (1805) | Gaussian σ **in histogram-bin units**. This is not the eval map smoothing |
| `--scorer` | `typical_set` (1766-1767); choices `typical_set`, `local_gaussian`, `feature_cos_delta` | `cfg.scoring.algorithm` (1792) | paper: `typical_set` |
| `--rgb_only` | flag (1768) | `cfg.scoring.rgb_only` (1793) | ablation only: RGB term alone |
| `--save_heatmaps DIR` | None (1771) | `cfg.eval.heatmap_dir` (1794) | writes σ=5 overlays |
| `--sigma_rohan` | float, None (1773) | `cfg.scoring.sigma_rohan` (1806) | read **only** by `local_gaussian` (1347-1351). **Ignored by `typical_set`** |
| `--autoencoder_path` | str, None (1776) | `cfg.scoring.autoencoder_path` (1808) | if set, AE path; otherwise per-image PCA path (1330-1338) |
| `--max_reconstructions` | int, None (1777) | `cfg.sampling.max_reconstructions` (1815) | random subset without replacement, `np.random.choice` (1202), **only if** more draws are available than the cap |
| `--max_samples` | int, None (1779) | `cfg.eval.max_samples` (1817) | first N samples per subcategory (1499) |
| `--embed_batch_size` | int, None (1782) | `cfg.embeddings.batch_size` (1797) | default 16 when absent (1333). Changes memory only |

**`embedder/evaluate_ddad_fe.py`** (85 lines) takes the same CLI: it imports `evaluate`
and calls `evaluate.main()` (line 85). Its only change is a monkeypatch,
`evaluate.ResNetPixelEmbedder = DDADFinetunedEmbedder` (line 81). That class builds the normal torchvision
embedder and then calls `self.extractor.load_state_dict(torch.load($DDAD_FE_WEIGHTS), strict=False)`
(lines 62-78):
* It unwraps `state_dict`, `model_state_dict` and `model` containers and the `module.` prefix (46-55).
* It raises an error if any layer1-3 key is missing. Unexpected `layer4`/`fc` keys are ignored.
* If `DDAD_FE_WEIGHTS` is unset it warns and keeps the torchvision weights.

Layers, patch size, AE usage and every other setting are the same as in `evaluate.py`. Its docstring
says not to pass `--autoencoder_path` and to use PCA, but every paper run passes an AE (the docstring is outdated).

**Environment variables** (all `os.environ` reads on the evaluation path):

| env var | read at | default | effect | paper |
|---|---|---|---|---|
| `DDAD_FE_WEIGHTS` | `evaluate_ddad_fe.py:43`; `train_pixel_ae_interp.py:164` (default of `--weights`) | `""` (torchvision weights) | backbone state dict loaded into layers 1-3 | MVTec 14: `weights/backbones/resnet101_imagenet1k_v1.pth`; xray: `weights/backbones/xray_ddad_lastlayer_feat4.pth` |
| `DUMP_DELTA_DIR` | `evaluate.py:1443` | unset (no dump) | writes `{sub}__{sample}.npz` with `delta_map` (256², unsmoothed, NaN kept), `gt_mask`, `valid_mask` (all ones if none) | `$OUT/maps/<target>` |
| `OOD_CT_MASK_ROOT` | `evaluate.py:1256`, `ood/data.py:314` | `./data/id_new_warped_images` | CT GT from `masks_ood_med_med/`, CT valid region from `masks_body_ood_default_large/` | `$ROOT/data/id_new_warped_images` |
| `ABL_SEED` | `tools/seeded_run.py:5` | `0` | seeds `np.random`, `torch.manual_seed` (CPU and CUDA) and `random` once, before the script runs (line 6). cuDNN determinism flags are **not** set | 0 (no effect on paper numbers: no run subsamples, see 2.1) |
| `TIMING_JSON` | `evaluate.py:864` | unset | per-stage timing records | — |
| `OOD_BENCH_NOCACHE` | `evaluate.py:968, 987` | unset | disables the embedder/AE cache (benchmarking) | — |
| `OOD_NO_FOLD` | `evaluate.py:1033` | unset | disables the folded first-conv path. Same result up to float associativity | — |
| `OOD_LEGACY_EMBED` | `evaluate.py:1033, 1095`; `ood/embeddings.py:300` | unset | old unfold/decoder path. Same result | — |
| `PYTHONPATH`, `CUDA_VISIBLE_DEVICES`, `PYTHONUNBUFFERED`, `GPU`, `OUT` | `scripts/run_reproduce.sh` | — | plumbing | — |

### 2.1 Stage A: reconstruction loading (evaluation side)

| Parameter | CLI / YAML / env | Default in code | Paper value | Meaning / effect |
|---|---|---|---|---|
| recon dir | `--results_root` + `--category` + `--results_suffix` | — (`evaluate.py:1117-1122`) | Section 3 | `{root}_{cat}_{sub}{suffix}/samples_*/` |
| `test_origin` | `data.test_origin` | none, required | `ddad_native`; moreface: `Combined_0.5_sigma_batched` | recon subdir prefix `{test_origin}_{patch}_{bottom_suffix}` (`ood/data.py:166-210`) |
| `bottom_suffix` | `data.bottom_suffix` | none, required | `"4"` everywhere | recon subdir suffix |
| `num_patches` | `sampling.num_patches` | 24 (`evaluate.py:1191`) | 1 (DDAD-guided YAMLs); 24 (moreface YAML) | patch indices `0..num_patches-1` are scanned and missing ones skipped with a warning. Moreface: only patches 8, 12-14, 16-23 (12 patches) exist at σ=0.5, so 12x4 = 48 draws. The `Combined_1.0_*` dirs are ignored |
| label image | — | first of patch 0..49, `0_00000.png` or `00000.png` (`ood/data.py:227`) | — | input image `x` |
| `max_reconstructions` (M cap) | `--max_reconstructions` / `sampling.max_reconstructions` | None (all draws) | MVTec 50, pill 50, CT 40, xray 40, faces none | cap only. **No paper run subsamples**: MVTec dirs hold 40 draws, pill_3 32, CT 10, xray b1 24 (25 samples) or 16 (7 samples), b2 32, faces 48, moreface 48 (counted on disk). So `ABL_SEED` and `np.random.choice` never run |
| `sampling.enabled` | YAML | forced `False` (1791) | False | — |
| other `sampling.*` (model, diffusion, conditioning, measurement, patch_size, gpu_ids, seed) | YAML | — | — | **not read** by evaluation; documentation only ("kept for schema completeness") |

### 2.2 Stage B: ResNet-101 pixel embedding

`ood/embeddings.py:312-400` (`ResNetPixelEmbedder`). It is built by `get_embedder` (`evaluate.py:967`),
which passes `backbone`, `layers`, `use_patch_context`, `patchify_size` and `proj_dim_per_layer`
**only**. `out_size` and `use_imagenet_norm` therefore keep their code defaults.

| Parameter | YAML key | Default in code | Paper value | Meaning |
|---|---|---|---|---|
| backbone | `embeddings.backbone` | `resnet18` (constructor, :325) | `resnet101` (all YAMLs) | torchvision registry: resnet101 means **`IMAGENET1K_V2`** (:319) |
| backbone weights | `DDAD_FE_WEIGHTS` | torchvision V2 | MVTec 14: `feat0.pth`; xray: `feat4.pth`; pill, CT, faces: none (V2) | **`feat0.pth` is bitwise identical to torchvision `IMAGENET1K_V1`** (`resnet101-63fe2227.pth`): checked here for bottle and carpet, 522/522 tensors equal. So the "ImageNet anchor" for MVTec is **V1**, while pill, CT and faces run on **V2**. `feat4.pth`: see 4.3 |
| layers | `embeddings.layers` | `("layer1","layer2","layer3")` | same | 256+512+1024 = **1792 channels**; native 64², 32², 16² at 256² input |
| patch context | `embeddings.use_patch_context`, `patchify_size` | True, 3 | True, 3 | 3x3 mean over the feature map with zero padding (`avg_pool2d`, `count_include_pad=True`, :288-309) |
| per-layer projection | `embeddings.proj_dim_per_layer` | None | None | — |
| upsampling | — | bilinear, `align_corners=False`, to the input size (`out_size=None`) (:396) | 256x256 | — |
| normalisation | — | per-layer L2 over channels (`F.normalize`, :397); ImageNet mean/std input norm (:363-366, :382-383); inputs in [-1,1] mapped to [0,1] (:379) | same | — |
| embed batch | `--embed_batch_size` / `embeddings.batch_size` | 16 (`evaluate.py:1333`) | 16 | memory only |

PCA path (not used in the paper; every target passes an AE): `embed_and_project`, `ood/embeddings.py:629+`.
It fits a per-image SVD on the **label image's** features, keeps `n_pca` components and projects the
reconstructions without clipping. The docstring says "clip to [-1,1]", but `project_to_pca` does not clip.

### 2.3 Stage C: PixelAutoEncoder (evaluation side)

| Parameter | Source | Value | Notes |
|---|---|---|---|
| architecture | `evaluate.py:934-962` | enc 1792->512->256->64->3 (Conv-BN-ReLU x3, last conv + `Tanh`); dec mirror | kernel sizes are read from the checkpoint (`load_autoencoder`, :995-1007). Every paper AE is all-1x1 (checked) |
| latent_dim | `--n_pca` / `pca.n_components` | 3 | YAML default 5 is overridden by `--n_pca 3` in every paper run |
| part used | `embed_with_ae` (:1092) | encoder only; latent `z ∈ (-1,1)³` per pixel | the first 1x1 conv is folded per layer at native resolution (:1028-1089). Exact up to float associativity |

### 2.4 Stage D: superpixels

**Live implementation**: `ood/superpixels.py:281-295` `recursive_subdivide`. Despite its name and
docstrings, it does **not** recurse with SLIC:

1. **Edge map** (`_build_edge_map`, :260-268):
   * convert to grey and rescale robustly to the 1st-99th percentile;
   * apply CLAHE (`clip_limit=0.02`);
   * apply Gaussian σ=1 to both the grey image and the CLAHE image;
   * `grad = rescale(0.35·sobel(gray_smoothed) + 0.65·sobel(clahe_smoothed))`;
   * raise to the power `max(1, alpha_grad/5)`.
2. **Markers** (`_make_grid_markers`, :271-279): a regular grid with step `max(2, int(sqrt(target_size)))`
   starting at `step//2`.
3. **Compact watershed** (:286): `skimage.segmentation.watershed(edge, markers, compactness=compactness/1000)`.

`var_threshold`, `min_pixels`, `max_sub` and `max_depth` are accepted and **ignored**. The coarse SLIC
`labels` argument is used only to build `parent_map`, which `compute_delta_map_gpu` never receives, so the
**coarse SLIC mask does not influence the typical-set delta map or the metrics**. The pre-existing failing
unit test `tests/test_superpixels.py:40` (`..._no_split_with_large_min_pixels`) fails for this reason.

| Parameter | YAML key (CLI) | Default in code | Paper value | Meaning |
|---|---|---|---|---|
| target size | `superpixels.target_size` (`--superpixel_target_size`) | 300 (function default, :283); YAML 100 (MVTec), 50 (CT, xray), 10 (faces) | **30 for all 20 targets** | grid step `int(sqrt(30)) = 5`, i.e. 51x51 = **2601 markers**, about 25 px per superpixel on 256² (for reference: 10 gives step 3 and 7225 markers; 50 gives step 7 and 1369; 100 gives step 10 and 676) |
| compactness | `superpixels.compactness` | 1.0 (:282) | **1.0** (MVTec 14 + pill, `experiment_ddad_native.yaml:45`); **12.0** (CT, xray, faces YAMLs) | watershed compactness = value/1000 (0.001 or 0.012) |
| alpha_grad | `superpixels.alpha_grad` | 1.0 (:283) | **1.0** (MVTec + pill); **10.0** (CT, xray, faces) | edge exponent `max(1, alpha/5)`: 1 or 2 |
| var_threshold, min_pixels, max_sub, max_depth | `superpixels.*` | 150, 16, 6, 4 | 0.0, 20, 6, 4 | **ignored** by the live code |
| coarse SLIC mask | `super_pixel_generation.py` `generate_mask_files` (:103-106), CLI defaults (:152-155) | `n_segments=50`, `compactness=10.0`, `sigma=1.0`, `topk=20`; image resized to 256² | cached `figures*/.../mask.png` | read from `{figures_dir}/{cat}/{sub}/{sample}/mask.png` (`./figures_CT/{sample}` for CT, :1487) and regenerated with these defaults if missing (:1227). **No effect on scores** (see above). `topk` only builds an unused binary mask |
| superpixel-level label | `eval.sp_anomaly_threshold` | 0.5 (`ood/metrics.py:239`) | 0.5 | a superpixel counts as anomalous if more than 50% of its pixels are GT-positive (only affects the `sp_*` metrics) |

### 2.5 Stage E: typical-set scorer

`ood/scoring.py:373-536` `compute_delta_map_gpu`, called at `evaluate.py:1362-1369`.

| Parameter | YAML key (CLI) | Default in code | Paper value | Meaning / effect |
|---|---|---|---|---|
| algorithm | `scoring.algorithm` (`--scorer`) | `typical_set` | `typical_set` | — |
| `bins_rgb` | `scoring.bins_rgb` | 32 (:380); YAML 32 | 32; **CT 64** | uint8 is quantised as `(v·bins)//256` (:43-45). RGB is a joint 3-D histogram (`bins³`). **CT only** (`category == "CT"`, `evaluate.py:1272`): grey = mean of RGB, 1-D histogram. X-ray is grey-scale but is scored as 3-D RGB |
| `bins_pca` | `scoring.bins_pca` | 8 (:381); YAML 16 | 32; **CT 64** | joint `bins^n_pca` histogram over the AE latent |
| latent quantisation range | — | per-image **1st/99th percentile of each latent dimension over all reconstruction pixels** (:446-449) | same | `quantize_pca_adaptive` (:54-72). Input-image values outside the range are clamped to the edge bins |
| `smooth_sigma` | `scoring.smooth_sigma` (`--smooth_sigma`) | 0.1 (:382); YAML 0.1 | **1.0** | separable Gaussian over every bin axis. Kernel radius `ceil(3σ)` (7 taps at σ=1), zero-padded `conv1d`, so mass leaks at the histogram borders (:79-100). **Dense path only** |
| dense/sparse switch | `MAX_HIST_ELEMENTS` | 2e8 (:36) | — | dense if `n_sp × n_bins <= 2e8`, otherwise sparse sort+unique **with no smoothing** (:497-505). At target size 30 (about 2601 superpixels): RGB 32³ and latent 32³ give 8.5e7, dense and smoothed. **CT**: grey 64 is dense; latent 64³ gives 6.8e8, **sparse, so the CT latent histogram is not smoothed** |
| `eps` | — (YAML `scoring.eps` is **not read**) | 1e-12, function default (:385) | 1e-12 | added to the histogram before normalising (:142) and inside log₂ (:154, :169) |
| PMF storage | — | dense PMF cast to **float16** (:144) | — | probabilities below fp16 range become 0, so per-pixel NLL is capped at `-log2(1e-12) ≈ 39.9` bits. Sparse path: unseen bin gives `p = eps/(N_sp+eps)` (:218-221) |
| entropy | — | `H = −Σ q log2(q+eps)` per superpixel, bits (:152-154) | — | over the smoothed PMF built from all M·|SP| reconstruction pixels |
| NLL | — | mean over the superpixel's input pixels of `−log2 p(bin(x))` (:157-172) | — | target = input/label image (YAML `use_label_as_target` is **not read**; GPU path always uses the label) |
| `min_pixels` (scoring) | `scoring.min_pixels` | 2 (:383); YAML 2 | 2 | superpixels smaller than this get NaN (:416) |
| histogram counts | — | int16 (:134-136) | — | overflows if one bin of one superpixel exceeds 32767 counts. Not reachable at 25 px x 48 draws |
| `rgb_only` | `--rgb_only` | False | False | ablation |
| `sigma_rohan`, `n_realizations`, `lg_metric` | `scoring.*` (`--sigma_rohan`) | —, 1000, `typicality_unsigned` | sigma_rohan 1.0 passed for CT and faces | **local_gaussian only** (`evaluate.py:1346-1358`; `ood/scoring_local_gaussian.py:31-40`, RNG `seed=0` default at :39). Not used by any paper number |

### 2.6 Stage F: resizing, dumping and evaluation smoothing

| Parameter | Source | Default | Paper | Meaning |
|---|---|---|---|---|
| output grid | `evaluate.py:1412-1435` | 256x256 | 256x256 | delta map: bilinear with anti-aliasing. GT, superpixel labels and valid mask: nearest |
| GT downsampling | `data.gt_mask.downsample_factor` | 4 (`ood/data.py:268`) | MVTec 4 (`[::4,::4]`, then nearest resize to 256: e.g. bottle 900 -> 225 -> 256); CT/xray/faces 1 | xray GT is 800x600, resized to 256² with nearest (aspect not preserved, same as the recon resize) |
| CT GT / valid | `evaluate.py:1243-1289` | — | GT `masks_ood_med_med/image_XXXX_mask.png` (512², `[::2,::2]` to 256²); valid = `masks_body_ood_default_large` (512², nearest to 256²) | metrics only use pixels inside the body mask |
| in-run eval sigmas | `eval.delta_smooth_sigmas` | — | MVTec YAML `[null,1,5,10,20]`; others `[null,5,10,20]` | `evaluate_delta_map` (`ood/metrics.py:257-263`): NaN to 0, `scipy.ndimage.gaussian_filter(σ, mode="nearest")` for the **pixel** metrics. The superpixel metrics always use the unsmoothed map |
| **reported eval sigma** | `full_metric_table.py --sigma` (131) | 5.0 | **5.0** | same filter (NaN to 0, `gaussian_filter(σ=5, mode="nearest")`, 115-117) on the dumped map |

---

## 3. Per-target paper settings (all 20 targets)

Common to every target (`scripts/run_reproduce.sh` `COMMON`, line 41, or explicit flags):
`--n_pca 3`, `--superpixel_target_size 30`, `--smooth_sigma 1`, `--no_plots`, `--skip_sampling`.
Also: scorer `typical_set`, `scoring.min_pixels` 2, eps 1e-12, layers 1-3, patch 3, eval
sigma **5** (`full_metric_table.py` default, used by `scripts/score_metrics.sh`), `ABL_SEED=0`. Every run goes
through `seeded_run.py`.

### 3a. Backbone, AE, recon directory, config

| # | Target | Driver | Config | Backbone weights | AE checkpoint | Recon dir (`--results_root` + suffix) |
|---|---|---|---|---|---|---|
| 1-14 | bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, screw, tile, toothbrush, transistor, wood, zipper | `evaluate_ddad_fe.py` | `experiment_ddad_native.yaml` | `weights/backbones/resnet101_imagenet1k_v1.pth` (= torchvision IMAGENET1K_V1) | `weights/ae/<cat>.pth`; carpet and wood: same naming (corrected seeded recipe) | `recons/results_patches_ddad_native_<cat>_<sub>_1` (all subdirs including `good`/`combined`; those two are excluded at scoring) |
| 15 | pill | `evaluate.py` | `experiment_ddad_native.yaml` | torchvision IMAGENET1K_V2 (no injection) | `weights/ae/pill.pth` | `recons/results_patches_ddad_native_pill_<sub>_3` (color, combined, contamination, crack, faulty_imprint, pill_type, scratch; no `good_3`) |
| 16 | CT (100 slices) | `evaluate.py` | `experiment_ddad_native_xray.yaml` (the CT eval config despite its name) | torchvision V2 | `weights/ae/CT.pth` | `recons/results_patches_ddad_native_CT_scan` (`--subcategory scan`; 100 slices x 10 DPS-denoising draws) |
| 17 | xray b1 (32 patients) | `evaluate_ddad_fe.py` | `experiment_ddad_native_xray_real.yaml` | `weights/backbones/xray_ddad_lastlayer_feat4.pth` | `weights/ae/xray.pth` | `recons/results_patches_ddad_native_xray_scissors_1` (`--subcategory scissors`) |
| 18 | xray b2 (40 patients) | same | same | same | same | `recons/results_patches_ddad_native_xray_scissors_b2_500k` |
| 19 | faces random22 | `evaluate.py` | `experiment_ddad_native_faces.yaml` | torchvision V2 | `weights/ae/faces.pth` | `recons/results_patches_ddad_native_faces_random_1` (24 sample dirs; `samples_5` and `samples_98` have no GT and are skipped, leaving 22) |
| 20 | faces moreface28 | `evaluate.py` | `experiment_dps_faces_moreface.yaml` | torchvision V2 | `weights/ae/faces.pth` | `recons/results_patches_dps_guided_faces_random` (`test_origin=Combined_0.5_sigma_batched`, `num_patches` 24, 12 present) |

### 3b. Numeric settings

| Target | bins_rgb | bins_pca | n_pca | SP target size | SP compactness / alpha_grad (YAML) | smooth_sigma (hist) | sigma_rohan | max_reconstructions (flag -> draws actually used) | grey-scale RGB | eval sigma | GT root (`--gt_root`) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MVTec 14 | 32 | 32 | 3 | 30 | 1.0 / 1.0 | 1 | — (ignored anyway) | 50 -> **40** | no | 5 | `data/mvtec_ad/<cat>/ground_truth` (downsample 4) |
| pill | 32 | 32 | 3 | 30 | 1.0 / 1.0 | 1 | — | 50 -> **32** | no | 5 | `data/mvtec_ad/pill/ground_truth` |
| CT | **64** | **64** | 3 | 30 | 12.0 / 10.0 | 1 (RGB dense; **latent sparse, so unsmoothed**) | 1.0 (ignored) | 40 -> **10** | **yes** (1-D, 64 bins) | 5 | passed but unused; masks via `OOD_CT_MASK_ROOT`, valid = body mask |
| xray b1 | 32 | 32 | 3 | 30 | 12.0 / 10.0 | 1 | — | 40 -> **24** (25 patients) / **16** (7 patients) | no | 5 | `data/mvtec_ad/xray/ground_truth` (downsample 1) |
| xray b2 | 32 | 32 | 3 | 30 | 12.0 / 10.0 | 1 | — | 40 -> **32** | no | 5 | same |
| faces random22 | 32 | 32 | 3 | 30 | 12.0 / 10.0 | 1 | 1.0 (ignored) | none -> **48** | no | 5 | `data/mvtec_ad/faces/ground_truth` |
| faces moreface28 | 32 | 32 | 3 | 30 | 12.0 / 10.0 | 1 | 1.0 (ignored) | none -> **48** (12 patches x 4) | no | 5 | `data/faces_moreface28/ground_truth` |

Pooled sets in `scripts/score_metrics.sh`: **xray full72** = b1 ∪ b2; **faces all50** = random22 ∪ moreface28.
MVTec rows in the main table are the plain mean of the 15 per-category metric values
(`scripts/summarize_results.py`).

---

## 4. AE and backbone training

### 4.1 `embedder/train_pixel_ae_interp.py` (current trainer): flags and defaults

| Flag | Default (line) | Meaning |
|---|---|---|
| `--weights` | `$DDAD_FE_WEIGHTS` or `""` (164) | backbone state dict, injected like the eval driver. Empty means torchvision V2 |
| `--data_dir`, `--save_model` | required (166-167) | nominal training images (all image files, recursive); output `.pth` (state_dict only) |
| `--latent_dim` | 3 (168) | |
| `--num_epochs` | 30 (169) | |
| `--batch_size` | 4 (170) | |
| `--lr` | 1e-3 (171) | **Adam** (261), constant LR unless `--lr_schedule cosine` (191) |
| `--seed` | 0 (172) | `set_seed`: python/numpy/torch/CUDA, `cudnn.deterministic=True`, `benchmark=False` (36-45). Fixed train/val split with `Generator(seed)` (217) |
| `--n_restarts` | 1 (174) | restart r uses seed `seed+r` (249). Keeps the run with the lowest validation selection loss |
| `--max_images` | 0 = all (176) | seeded random subset (`randperm`, `Generator(seed)`, 213) |
| `--ae_loss` | `cos` (178) | training loss: `cos` = 1−cos; `mse_cos`; `mse<W>_cos`; `wmse_cos`; `mse_var` = MSE + 0.1·ReLU(0.33−var(z)) (54-77) |
| `--select_loss` | `mse_var` (180) | validation loss for checkpoint selection (per epoch, best kept, 285-297). The comment at :285 ("SAME loss used for training") is stale |
| `--feat_noise` | 0.0 (183) | VarAE input noise |
| `--augment` | `none` (187) | none: `Resize(256)+ToTensor`; `geo`: hflip + affine(±10°, ±5%, 0.95-1.05); `geo_intensity` adds ColorJitter 0.1 |
| `--lr_schedule` | `none` (191) | `cosine` = CosineAnnealingLR(T_max=epochs) |
| `--train_last_layer`, `--backbone_lr` | off, 1e-4 (193-200) | joint tuning of `layer3[-1]` (ablation) |
| fixed | — | validation split = `max(1, int(0.1·N))` (215); DataLoader `num_workers=2`, `shuffle=True` with a seeded generator; embedder resnet101 layer1-3, patch 3 (146-150); AE = `evaluate.PixelAutoEncoder` |

`embedder/train_embedder.py` (root; original recipe, used for the "origAE" family):
* hard-coded torchvision resnet101 **V2** (falls back to `pretrained=True` on old torchvision), layers
  1-3, patch context always on;
* `--resnet_name`, `--use_patch_context` and `--proj_dim_per_layer` are parsed and **ignored**;
* decoder output L2-normalised (`F.normalize`, line 184);
* training loss 1−cos (258), checkpoint selection MSE + 0.1·ReLU(0.33−var z) (302-305);
* Adam lr 1e-3 (234), batch 4 (196), `--num_epochs` default **10** (195), latent 3 (192), 10% validation
  via unseeded `random_split` (211-213), **no seed**;
* defaults `--data_dir .../faces/extras` and `--save_model models/pixel_autoencoder_faces.pth` (189-190).

### 4.2 Released AE checkpoints

The checkpoints hold **only** `state_dict` tensors (40 keys): no args, config or epoch. Recipes therefore
come from scripts and docs. **Best epoch is inferred** from the encoder BN `num_batches_tracked`
counter (`nbt`). It counts training steps of the saved restart, since evaluation-mode passes do not
increment it. The derivation assumes batch 4, a 10% validation split and `steps/epoch = ceil(n_train/4)`,
with the standard MVTec-AD `train/good` counts. Training images are not in the package.

| Target | AE file | Recipe (evidence) | Backbone during AE training | Training data | Loss / selection | Seed / restarts / max_images | nbt -> inferred best epoch |
|---|---|---|---|---|---|---|---|
| bottle | `weights/ae/bottle.pth` | `run_train_ae.sh` KIND=pretrained -> `train_pixel_ae_interp.py --weights feat0 --latent_dim 3 --num_epochs 30`, other flags default. **UNKNOWN which trainer revision**: `rerun_all15_fixed_ae.sh` describes these as the pre-"corrected" (unseeded) trainer | feat0 (V1), per KIND mapping | MVTec `bottle/train/good` (209) | UNKNOWN (probably cos/cos; best epochs near 30 fit a selection loss that keeps falling) | UNKNOWN (unseeded per `rerun_all15_fixed_ae.sh:5-7`) | 1392 = 29 x 48 -> **ep 29/30** |
| cable | `.../cable.pth` | same | feat0 | 224 | same | same | 1530 = 30 x 51 -> **30** |
| capsule | `.../capsule.pth` | same | feat0 | 219 | | | 1500 = 30 x 50 -> **30** |
| grid | `.../grid.pth` | same | feat0 | 264 | | | 1740 = 29 x 60 -> **29** |
| hazelnut | `.../hazelnut.pth` | same | feat0 | 391 | | | 2640 = 30 x 88 -> **30** |
| leather | `.../leather.pth` | same | feat0 | 245 | | | 1512 = 27 x 56 -> **27** |
| metal_nut | `.../metal_nut.pth` | same | feat0 | 220 | | | 1500 = 30 x 50 -> **30** |
| screw | `.../screw.pth` | same | feat0 | 320 | | | 2088 = 29 x 72 -> **29** |
| tile | `.../tile.pth` | same | feat0 | 230 | | | 1404 = 27 x 52 -> **27** |
| toothbrush | `.../toothbrush.pth` | same | feat0 | 60 | | | 406 = 29 x 14 -> **29** |
| transistor | `.../transistor.pth` | same | feat0 | 213 | | | 1440 = 30 x 48 -> **30** |
| zipper | `.../zipper.pth` | same | feat0 | 240 | | | 1566 = 29 x 54 -> **29** |
| carpet | `weights/ae/carpet.pth` | `rerun_all15_fixed_ae.sh` (`CATS=carpet`): `train_pixel_ae_interp.py --weights feat0 --latent_dim 3 --num_epochs 30 --seed 0 --n_restarts 3` (lines 21, 59-62) | feat0 (V1) | 280 | `cos` / `mse_var` (defaults) | seed 0, 3 restarts (seeds 0, 1, 2), all images | 63 = 1 x 63 -> **epoch 1** (mse_var selected the first epoch) |
| wood | `weights/ae/wood.pth` | same | feat0 | 247 | `cos` / `mse_var` | 0 / 3 / all | 56 = 1 x 56 -> **epoch 1** |
| pill | `weights/ae/pill.pth` | UNKNOWN. Most consistent with `embedder/train_embedder.py` (`repro_orig_ae.sh` calls it the "ORIGINAL-recipe": `--resnet_name resnet101 --latent_dim 3 --num_epochs 10 --batch_size 4`) | UNKNOWN (train_embedder means V2, which matches eval) | MVTec `pill/train/good` (267) | cos / mse_var (if train_embedder) | unseeded (if train_embedder) | 122 = 2 x 61 -> **epoch 2** |
| CT | `weights/ae/CT.pth` | UNKNOWN (likely `embedder/train_embedder.py`) | UNKNOWN (likely V2) | UNKNOWN. Inferred N ≈ 3,068-3,071 CT slices (691 is prime, so epoch 1 and n_train 2761-2764) | UNKNOWN | UNKNOWN | 691 -> **epoch 1** (inferred) |
| faces | `weights/ae/faces.pth` | UNKNOWN (`embedder/train_embedder.py` defaults point at this file and `faces/extras`) | UNKNOWN (likely V2) | `faces/extras`. Inferred N ≈ 65,001-65,004 if epoch 1 (n_train 58,501-58,504), matching "65k extras". Epoch 2 with N ≈ 32.5k is also arithmetically possible | UNKNOWN | UNKNOWN | 14626 = 1 x 14626 (or 2 x 7313) -> **epoch 1 (most likely)** |
| xray | `weights/ae/xray.pth` | `xray_finetune_mine.sh:60-63`: `train_pixel_ae_interp.py --weights feat4 --latent_dim 3 --num_epochs 30 --seed 0 --n_restarts 2 --max_images 500 --ae_loss mse_cos` | `feat4.pth` | `xray/train/good` capped at 500 (seeded) | `mse_cos` / `mse_var` (default) | seed 0, 2 restarts, 500 images | 3164 = 28 x 113 -> **epoch 28** (n_train 450) |

Not used for paper numbers: `xray_mine/mixed_k3.pth` (encoder conv0 3x3) and `xray_mine/spatial_k3.pth`
(all 3x3), which are spatial-AE ablations.

REPRODUCE_RESULTS.md §6 ("fully seeded end-to-end") describes a **different** recipe
(`--latent_dim 3 --num_epochs 30 --seed 0 --n_restarts 3`, cos/mse_var; xray `mse_cos --n_restarts 2
--max_images 500`; faces capped at 500 images). Its numbers differ from the released checkpoints (for
example CT 0.8716 vs 0.9378, both measured on the earlier DDAD CT reconstructions), and its driver `train_and_evaluate_seeded.sh` is not in the package.

### 4.3 Backbone fine-tuning (feat0 / feat4)

| Item | Finding |
|---|---|
| saved configs | Only the weights were saved (no args/json/log files). The DA fine-tune script `ddad_da_finetune.py` is not released |
| feat0 (14 MVTec) | identical to torchvision ResNet-101 `IMAGENET1K_V1` (verified bitwise). It is the frozen copy DDAD saves before domain adaptation (`DDAD/feature_extractor.py:82`). No fine-tuning hyperparameters apply |
| feat4 (xray) | `xray_finetune_mine.sh`: `ddad_da_finetune.py --category xray --unet_ckpt <DDAD X-ray UNet>/xray/2000 --da_epochs 4 --seed 0 --train_scope last_layer` (conda env `ddad_env`). **Checkpoint contents**: only the 9 weight tensors of **`layer3.22`** (the last bottleneck of layer3) differ from V1, but **BN running statistics differ in every layer** (bn1, layer1-4; 208 tensors). The backbone was therefore in train mode, and "last-layer" fine-tuning also re-estimated BN statistics network-wide on x-ray data. `num_batches_tracked = 2496` |
| DDAD-DA reference hyperparameters (in-repo `DDAD/feature_extractor.py` + `DDAD/config*.yaml`) | AdamW **lr 1e-4** (hard-coded, :75); `DA_batch_size` 16, split into 8 targets + 8 inputs (:87-89); `DA_epochs` 4; `w_DA` 3; `DLlambda` 0.1; loss = Σ_layers [1−cos(f(x̂), f(x))] + λ[1−cos(f(x), f₀(x))] + λ[1−cos(f(x̂), f₀(x̂))] (:13-23); reconstruction uses `test_trajectoy_steps`/`skip` (250/25). The `_DA` variants are unused by `Reconstruction`. Whether `ddad_da_finetune.py` used these exact values: **UNKNOWN (script not shipped)** |

---

## 5. Reconstruction (sampler) hyperparameters

Sources: `recon_metadata.json` where present (written by `tools/generate_recons.py:173-188`, overwritten
on each invocation), otherwise indirect evidence (flagged). DDAD-guided sampler (code name `ddad_native`) defaults
(`ood/samplers/ddad_native.py:29-33, 49-50`): `beta_start=1e-4`, `beta_end=0.02`, `trajectory_steps=1000`,
`test_trajectoy_steps(t*)=250`, `skip=25`, `eta=1.0`, `w=2.0`. Additive-DPS defaults
(`ood/samplers/additive_dps.py:33-54`): σ=0.1, scale 0.5, skip 25, eta 1 (not used for any paper set; its
`DDAD_DPS/` dependency is not in the package). `generate_recons.py` defaults: `num_seeds` 20,
`bottom_suffix` "4" (:106-108). Draw k uses `torch.manual_seed(k)` and `np.random.seed(k)`, k = 0..N−1
(`ddad_native.py:55-60`, `generate_recons.py:161-166`). Inputs are resized to 256² bicubic and scaled to
[-1,1] (`ood/samplers/io.py:54-58`).

DDAD-guided algorithm (`DDAD/reconstruction.py`):
* `x_t* = sqrt(ᾱ_t*)·x + sqrt(1−ᾱ_t*)·ε`;
* reverse DDIM-style over `range(0, t*, skip)`, i.e. 10 steps for 250/25;
* conditioning `ε̂ = ε − sqrt(1−ᾱ)·w·(y_t − x_t)`, with `y0 = x` (the input itself);
* `eta` = 1.

Effective σ = `sqrt(1−ᾱ_250)` = **0.68987**.

| Dataset (dir) | Sampler | Checkpoint (as recorded) | w | t* | skip (reverse steps) | eta | β | Draws / sample | Source |
|---|---|---|---|---|---|---|---|---|---|
| MVTec 14 + pill `_1` (`recons/results_patches_ddad_native_<cat>_<sub>_1`) | DDAD-guided | `<DDAD ckpt root>/<cat>/<epoch>`. **Epochs**: bottle 1000, cable 3000, capsule 1500, carpet 2500, grid 2000, hazelnut 2000, leather 2000, metal_nut 3000, pill 1000, screw 2000, tile 1000, toothbrush 2000, transistor 2000, wood 2000, zipper 1000 | 2.0 | 250 | 25 (10) | 1.0 | linear 1e-4..0.02, T=1000 | 40 (seeds 0-39; verified 40 on disk for all 1725 samples) | metadata |
| pill `_3` (paper pill) | UNKNOWN (dir layout `ddad_native_0_4`) | UNKNOWN | ? | ? | ? | ? | ? | 32 (on disk) | **no metadata** |
| CT (`recons/results_patches_ddad_native_CT_scan`, 100 slices `image_0000-0099`) | **DPS denoising** (`sampling/dps/sample_fast.py`): y = x + σn (all-ones mask), **σ = 0.05** (`configs/ctgrid/sigma0.05.yaml`), **ζ′ = 0.2** (`--scale 0.4 --grad_norm per_sample`), 1000 DDPM steps, linear β, learned variance | `weights/diffusion/ct_dps_model050000.pt` (`ct_model_config.yaml`) | — | — | — (1000) | — | linear 1e-4..0.02, T=1000 | 10 (seed `3·1000003 + image_idx·1009 + draw`) | `recon_metadata.json`; generated 2026-09-25, re-verified 2026-10-05 |
| xray b1 (`..._xray_scissors_1`) | UNKNOWN from metadata. Tooling (`tools/sigma_sampling_20260916/collect_sigma_sets.py:2,12`) says x-ray sets were flattened from DPS patch dirs `Xray_guided/samples_<P>/Combined_<σ>_sigma_batched_<top>_4` | UNKNOWN (b2 note says `guided_diffusion/Xray_small/model500000.pt`) | — | — | — | — | — | 24 (25 patients), 16 (7 patients) = 6 or 4 patches x 4 | **no metadata** |
| xray b2 (`..._xray_scissors_b2_500k`) | same evidence: DPS patch inpainting, "sigma-0.3 recons" (`gen_jobs.py:20`), model `Xray_small/model500000.pt` ("same as the b2 main recons", `sample_patch.sh:5`) | `<guided-diffusion ckpts>/Xray_small/model500000.pt` (inferred) | — | — | 1000 DDPM | — | — | 32 (8 patches x 4, inferred) | **no metadata**. σ=0.3 inferred from comments only |
| faces random22 (`..._faces_random_1`) | DPS patch inpainting, FFHQ prior (`ffhq_10m.pt`), σ=0.5 (RUNBOOK, `scripts/run_reproduce.sh`), flattened 12 patches x 4 | `ffhq_10m.pt` (RUNBOOK §5) | — | — | 1000 DDPM | — | — | 48 | **no metadata** |
| faces moreface28 (`recons/results_patches_dps_guided_faces_random`) | DPS patch inpainting, σ=0.5 (dir name `Combined_0.5_sigma_batched`); patch tops 8, 12-14, 16-23 | `ffhq_10m.pt` (assumed, same pipeline) | — | — | 1000 DDPM | — | — | 48 used (4 per patch); σ=1.0 dirs present but ignored | dir layout only |

DPS patch-sampling settings that appear in `tools/sigma_sampling_20260916/sample_patch.sh:2-4`. That
script was written for the σ-ablation; per its own comment it "mirrors" the main faces/x-ray sampling
scripts from 2026-08-27. Its settings are: `--seed 3`, `--mask_prob 0.9`, `--num_measurements 4`, box
bottom 4, `--box_coords <top> 4 <left> <right>` with 64-px column bands, superpixel mask from
`super_pixel_generation.py` defaults, and 1000 DDPM steps. **DPS guidance scale / ζ: UNKNOWN**, because
they live in the external DPS repo's `configs/inpainting_config*.yaml`. The eval YAMLs' inert `sampling`
blocks (`conditioning: {method: ps, scale: 0.5}`, `mask_len_range [128,129]`, `mask_prob 0.9`, noise σ
0.1/0.5) are **not** verified to match.

DDAD UNet training hyperparameters (DDAD/config*.yaml, upstream DDAD defaults; not re-verified against
the actual checkpoint runs): `image_size 256`, `batch_size 32`, `learning_rate 3e-4`, `weight_decay 0.05`,
`trajectory_steps 1000`, `seed 42`, UNet `UNetModel(256, 64, dropout=0, n_heads=4, in_channels=3)`
(`ood/samplers/io.py:43`). The checkpoint epoch is in the path. **UNKNOWN**: training-run logs are not
shipped.

---

## 6. Metric definitions

Reported numbers come from `tools/metrics/full_metric_table.py`, run on the `DUMP_DELTA_DIR` npz files
by `scripts/score_metrics.sh`. Defaults: `--sigma 5.0` (131), `--skip_sub good combined` (130). Subcategories
whose name contains `good` or `combined` are excluded (substring match on the part of the filename before
`__`, 111-113). Each map has NaN set to 0 and is smoothed with `gaussian_filter(σ=5, mode="nearest")`.
Only pixels with `valid_mask` set are used (CT: body mask; otherwise all pixels).

| Metric | Definition (file:line) |
|---|---|
| pooled AUROC / AUPR | `sklearn.roc_auc_score` / `average_precision_score` on all valid pixels of all images (66-67) |
| pooled F1max | one global threshold. Scores are binned into **8192** histogram bins between global min and max; F1 = 2TP/(2TP+FP+FN) maximised over bin edges (26-34, 68). Approximate (bin-quantised thresholds) |
| AUPRO | 600 thresholds at FPR levels `linspace(0, 0.3, 601)[1:]`, taken as quantiles of all normal valid pixels (37-42). Regions = **8-connected** components of `GT ∩ valid` per image (45). PRO = mean over all regions of the per-region TPR. Trapezoid of PRO vs actual FPR, normalised by **0.3** (`fpr_limit`). The curve starts at (0, PRO at the first threshold) rather than (0, 0) (55-58) |
| per-image AUROC / AP / F1max | computed per image (images with both classes only), then averaged (83-101). F1max uses the per-image best threshold (8192 bins) |
| per-image SNR (table "SNR") | `(mean(s[GT]) − mean(s[¬GT])) / (std(s[¬GT]) + 1e-8)` on the smoothed raw scores (92) |
| per-image snr_paper | min-max normalise per image (+1e-12), then `mean(GT)/(mean(¬GT)+1e-8)` (93-96) |
| per-image mask-MSE / PSNR | `mean((minmax(s) − GT)²)` per image (97); `psnr_db = mean_i 10·log10(1/MSE_i)` (102) |
| pooled SNR / mask-MSE | SNR over all pooled pixels (75); MSE of per-image min-max maps pooled over pixels (76-81) |
| MVTec aggregate | plain mean of the 15 per-category values (`scripts/summarize_results.py`) |
| headline "px ROC-AUC σ5" (Section A of `scripts/summarize_results.py`, REPRODUCE_RESULTS §5) | from `evaluate.py`'s own `evaluate_delta_map` (`ood/metrics.py:235-369`): per-image px AUROC (manual ROC/trapezoid) at σ=5, averaged per subcategory, then averaged over subcategories excluding good/combined |
| `evaluate.py` JSON keys | `sp_roc_auc`/`sp_ap` (superpixel-level: mean unsmoothed score per superpixel, label = GT fraction > 0.5); `px_roc_auc`/`px_ap`; `snr` (same formula as above, 309); `snr_paper` (326); **`psnr` actually holds the mask MSE** (360) |

---

## 7. Known caveats and open questions

These notes describe the released code and checkpoints as they are. None of them changes the
reproduced numbers (docs/REPRODUCIBILITY.md).

### 7.1 Things that are easy to get wrong

* **Two ImageNet backbones.** The 14 MVTec categories load `weights/backbones/resnet101_imagenet1k_v1.pth`
  (= torchvision IMAGENET1K_V1) through `DDAD_FE_WEIGHTS`. Pill, CT and faces use `evaluate.py`'s
  torchvision default, **IMAGENET1K_V2** (`ood/embeddings.py:319`). `scripts/run_reproduce.sh` pins V2
  by pointing `TORCH_HOME` at `weights/torch/`.
* **X-ray backbone.** `xray_ddad_lastlayer_feat4.pth` changes the `layer3.22` weights *and* the BatchNorm
  running statistics of every layer. The DA fine-tune ran the backbone in train mode.
* **Superpixel settings differ by dataset.** `superpixels.compactness` / `alpha_grad` are 1 / 1 for MVTec
  and pill and 12 / 10 for CT, X-ray and faces (YAML). `target_size` is 30 everywhere. The live
  implementation is `ood/superpixels.py::recursive_subdivide` (2.4). The YAML keys `var_threshold`,
  `min_pixels`, `max_sub` and `max_depth` are accepted but not used by it. The cached coarse mask from
  `super_pixel_generation.py` is not used by the score.
* **`--all_subcategories` scores every matching recon dir.** Do not leave extra
  `recons/results_patches_ddad_native_<cat>_*` folders (for example your own sampling experiments) next to
  the released ones, or they will be scored as extra subcategories. CT is run with `--subcategory scan`
  for this reason.
* **`experiment_ddad_native_xray.yaml` is the CT evaluation config**; the X-ray config is
  `experiment_ddad_native_xray_real.yaml`.
* **Recon directory names describe layout, not sampler.** The X-ray and faces sets live in
  `..._ddad_native_...` directories but were produced by DPS patch inpainting (Section 5).
* **CT latent histogram is not smoothed.** 64³ latent bins exceed the dense-histogram limit, so the sparse
  path runs and `smooth_sigma` does not apply to the CT latent term (2.5).
* **JSON key naming.** In `evaluate.py`'s per-sample JSON, `psnr` holds the mask MSE.

### 7.2 Flags and keys that have no effect on the paper path

* `--skip_sampling` (sampling is always off in evaluation), `--sigma_rohan` (only for the `local_gaussian`
  scorer), `--gt_root` for CT (masks come from `OOD_CT_MASK_ROOT`), `--max_reconstructions` (never smaller
  than the shipped number of draws, so `ABL_SEED` has no effect).
* YAML `scoring.eps`, `scoring.use_label_as_target`, `pca.n_components` (overridden by `--n_pca 3`),
  `sampling.*` other than `num_patches` / `max_reconstructions`, and `baselines.*`.
* `embedder/train_embedder.py --resnet_name`, `--use_patch_context` and `--proj_dim_per_layer` are parsed
  and ignored.
* The `embedder/evaluate_ddad_fe.py` docstring predates the AE path. Every paper run passes
  `--autoencoder_path`.

### 7.3 X-ray: 0.9526 vs. 0.9528

The released configuration (superpixel target size 30) gives 0.9526 on X-ray b1. Target size 10 gives
0.9527. The published 0.9528 was produced by an earlier run whose exact setting is not recorded. The full
72-patient metric table matches the paper (94.22 AUROC, 72.09 AUPRO).

### 7.4 Not recorded (UNKNOWN)

* Training recipe (loss, selection loss, seed, restarts) of the 12 `weights/ae/<mvtec cat>.pth` AEs and of
  the pill, CT and faces AEs. Only the best epoch can be inferred (4.2). Carpet, wood and X-ray were trained
  with the seeded command lines given in 4.2.
* Sampler settings for pill (`_3`), X-ray b1/b2 and faces: no `recon_metadata.json` was saved (5). The DPS
  guidance scale for faces and X-ray in particular.
* Hyperparameters of the X-ray DA fine-tune beyond its command line (4.3).
* DDAD UNet training logs for each checkpoint epoch.
