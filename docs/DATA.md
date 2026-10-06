# Data, reconstructions and weights

Nothing below is stored in git. `scripts/download_assets.sh` fetches the release zip files,
checks them against `SHA256SUMS` and unpacks them into the repository root:

```
weights/
  ae/<target>.pth                     18 per-target PixelAutoEncoders (1792 -> 3), ~8.6 MB each
  backbones/resnet101_imagenet1k_v1.pth     ResNet-101, == torchvision IMAGENET1K_V1 (resnet101-63fe2227)
  backbones/xray_ddad_lastlayer_feat4.pth   ResNet-101, DDAD domain-adaptation fine-tune (last layer, epoch 4) for X-ray
  torch/hub/checkpoints/resnet101-cd907fc2.pth  torchvision IMAGENET1K_V2 (used by evaluate.py; TORCH_HOME points here)
data/
  mvtec_ad/<target>/{test,ground_truth}/    test images + GT masks (15 MVTec categories, CT, xray, faces)
  id_new_warped_images/{masks_ood_med_med,masks_body_ood_default_large}/   CT GT masks + body (valid-region) masks
  faces_moreface28/{test,ground_truth}/random/  the 28 additional faces (+ mapping.csv)
recons/
  results_patches_ddad_native_<category>_<subcategory><suffix>/samples_<id>/<origin>_<patch>_4/inpainting/{input,recon,label}/
  results_patches_dps_guided_faces_random/
```

## Release zip files

| File | Contents | Size |
|---|---|---:|
| `lactos_data.zip` | `data/`: test images + GT masks (15 MVTec categories, CT, X-ray, faces) | 1.6 GB |
| `lactos_weights.zip` | `weights/ae`, `weights/backbones`, `weights/torch`: 18 AEs + ResNet-101 backbones | 0.6 GB |
| `lactos_diffusion.zip` | `weights/diffusion/`: diffusion priors for DPS sampling (CT, X-ray, FFHQ) | 0.8 GB |
| `lactos_recons_<cat>.zip` (x15) | MVTec reconstructions per category (`_1` suffix; pill `_3`) | 0.12–0.65 GB each, 5.7 GB total |
| `lactos_recons_CT.zip` | `results_patches_ddad_native_CT_scan` (100 slices x 10 DPS-denoising draws, σ=0.05, ζ′=0.2) | 0.09 GB |
| `lactos_recons_xray.zip` | `..._xray_scissors_1` (32 patients) + `..._xray_scissors_b2_500k` (40 patients) | 0.12 GB |
| `lactos_recons_faces.zip` | `..._faces_random_1` (24 faces, 22 with masks) | 0.12 GB |
| `lactos_recons_faces_moreface.zip` | `results_patches_dps_guided_faces_random` (28 faces) | 1.1 GB |

Unzip every file in the repository root (`unzip <file>.zip`); the folders above are created in place.
Every file is < 2 GB. `SHA256SUMS` lists the checksum of each zip. Maintainers build them with
`scripts/make_release_assets.sh`.

## Weight inventory (what produced which number)

| Target(s) | AE (`weights/ae/`) | Backbone | Loaded by |
|---|---|---|---|
| bottle cable capsule grid hazelnut leather metal_nut screw tile toothbrush transistor zipper | `<cat>.pth` | `backbones/resnet101_imagenet1k_v1.pth` via `DDAD_FE_WEIGHTS` | `embedder/evaluate_ddad_fe.py` |
| carpet, wood | `<cat>.pth` (corrected seeded recipe: cos loss, mse_var selection, best of 3 restarts) | same | same |
| pill | `pill.pth` | torchvision IMAGENET1K_V2 (`torch/hub/...cd907fc2.pth`) | `evaluate.py` |
| CT | `CT.pth` | torchvision IMAGENET1K_V2 | `evaluate.py` |
| xray (b1, b2) | `xray.pth` | `backbones/xray_ddad_lastlayer_feat4.pth` via `DDAD_FE_WEIGHTS` | `embedder/evaluate_ddad_fe.py` |
| faces (random22, moreface28) | `faces.pth` | torchvision IMAGENET1K_V2 | `evaluate.py` |

Note the two ImageNet backbones: the 14-category MVTec path loads **V1** weights explicitly;
the plain `evaluate.py` path (pill, CT, faces) builds torchvision's default **V2** weights.
This matches the runs behind the paper numbers. SHA-256 of every file: `weights.sha256`.

Diffusion models (only needed to *regenerate* reconstructions, not to reproduce the numbers):

| File (`weights/diffusion/`) | Used for | Config |
|---|---|---|
| `ct_dps_model050000.pt` | CT DPS denoising | `sampling/dps/configs/ct_model_config.yaml` |
| `xray_dps_model500000.pt` | X-ray DPS | `sampling/dps/configs/xray_model_config.yaml` |
| `ffhq_10m.pt` | faces DPS (public DPS FFHQ model, Chung et al.) | DPS upstream FFHQ config |

**MVTec diffusion models (not yet in the release).** The MVTec reconstructions were generated with one
DDAD UNet per category (~130 MB each). These checkpoints are not part of the current release; they will be
added to `weights/diffusion/mvtec/<category>/<epoch>` once uploaded. They are **not needed to reproduce the
numbers**: the reconstructions they produced are released (`recons/`). The checkpoint epoch of each model,
as recorded in each set's `recon_metadata.json`:

| Category | Epoch | | Category | Epoch | | Category | Epoch |
|---|---:|---|---|---:|---|---|---:|
| bottle | 1000 | | grid | 2000 | | screw | 2000 |
| cable | 3000 | | hazelnut | 2000 | | tile | 1000 |
| capsule | 1500 | | leather | 2000 | | toothbrush | 2000 |
| carpet | 2500 | | metal_nut | 3000 | | transistor | 2000 |
| pill | not recorded | | wood | 2000 | | zipper | 1000 |

The public DDAD checkpoints linked from `DDAD/README.md` have not been verified to be these models; use them
only for new experiments, not as a substitute when regenerating the released MVTec reconstructions.

## Provenance and licences

| Data | Source | Licence / redistribution |
|---|---|---|
| MVTec AD (15 categories: test images, GT masks; reconstructions contain test images as `label/`) | Bergmann et al., CVPR 2019 / IJCV 2021, https://www.mvtec.com/company/research/datasets/mvtec-ad | CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike) |
| X-ray (scissors, 72 patients) | DvXray, negative samples | TODO(maintainers): confirm licence permits redistribution |
| CT (100 slices + masks) | TODO(maintainers): name the source dataset | TODO |
| Faces (22 + 28 test images, masks) | TODO(maintainers): name the source | TODO: confirm rights to redistribute the images |
| FFHQ diffusion prior (sampling only, not shipped) | Karras et al.; DPS `ffhq_10m.pt` | per upstream |

If a dataset cannot be redistributed, remove it from `lactos_data.zip` / the recon zips
and document how to obtain it here instead.
