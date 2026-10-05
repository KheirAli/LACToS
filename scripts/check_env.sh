#!/bin/bash
# Sanity-check a fresh clone BEFORE running: assets (with checksums), python env, unit tests.
#   bash scripts/check_env.sh            # full check (sha256 of the 21 weight files takes ~10 s)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; ok=0; bad=0
chk() { if [ -e "$1" ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "  MISSING: $1"; fi; }
R=recons/results_patches_ddad_native
echo "== weights (sha256, docs/REPRODUCIBILITY.md)"
if [ -d weights ]; then sha256sum -c --quiet weights.sha256 && echo "  21/21 weight files match" || bad=$((bad+1)); else echo "  MISSING: weights/ (bash scripts/download_assets.sh weights)"; bad=$((bad+1)); fi
echo "== reconstructions"
for c in bottle cable capsule carpet grid hazelnut leather metal_nut screw tile toothbrush transistor wood zipper; do
  n=$(ls -d ${R}_${c}_*_1 2>/dev/null | wc -l); [ "$n" -gt 0 ] && ok=$((ok+1)) || { bad=$((bad+1)); echo "  MISSING: ${R}_${c}_*_1"; }; done
n=$(ls -d ${R}_pill_*_3 2>/dev/null | wc -l); [ "$n" -gt 0 ] && ok=$((ok+1)) || { bad=$((bad+1)); echo "  MISSING: ${R}_pill_*_3"; }
chk ${R}_CT_scan; chk ${R}_xray_scissors_1; chk ${R}_xray_scissors_b2_500k; chk ${R}_faces_random_1; chk recons/results_patches_dps_guided_faces_random
echo "   $(ls -d ${R}_*_1 ${R}_*_3 2>/dev/null | wc -l) recon dirs; CT samples: $(ls -d ${R}_CT_scan/samples_* 2>/dev/null | wc -l) (expect 100)"
echo "== data"
for c in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper CT xray faces; do chk data/mvtec_ad/$c/ground_truth; done
chk data/id_new_warped_images/masks_ood_med_med; chk data/id_new_warped_images/masks_body_ood_default_large; chk data/faces_moreface28/ground_truth/random
echo "== files ok=$ok missing=$bad"
echo "== python env"; python - <<'PY'
import sys
try:
    import torch, torchvision, numpy, scipy, skimage, sklearn, yaml, PIL
    print(f"  python {sys.version.split()[0]}  torch {torch.__version__}  torchvision {torchvision.__version__}  numpy {numpy.__version__}  scipy {scipy.__version__}  scikit-image {skimage.__version__}  scikit-learn {sklearn.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}  ({torch.cuda.device_count()} GPU)")
    print("  reference env: python 3.10.20 torch 2.11.0+cu130 torchvision 0.26.0 numpy 2.2.6 scipy 1.15.3 scikit-image 0.25.2 scikit-learn 1.7.2 (environment.yml)")
except Exception as e: print("  IMPORT ERROR:", e)
PY
echo "== unit tests (no checkpoints needed)"
PYTHONPATH="$ROOT" python -m pytest -q --no-header -p no:cacheprovider -m "not requires_ckpt" tests 2>&1 | tail -3
