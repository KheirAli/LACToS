#!/bin/bash
# Step 1 (MVTec): DDAD-guided reconstructions for every test subcategory of one MVTec category.
#
#   bash sampling/sample_mvtec_ddad.sh <category> <ddad_unet_ckpt> [num_seeds=40] [suffix=_1]
#   e.g. GPU=0 bash sampling/sample_mvtec_ddad.sh cable weights/diffusion/mvtec/cable/3000
#
# Writes   $RECON_ROOT/results_patches_ddad_native_<category>_<subcategory><suffix>/samples_<id>/ddad_native_0_4/
#          inpainting/{recon,label,input}/  + recon_metadata.json
# (RECON_ROOT defaults to recons_mine/, kept separate from the released recons/), which is exactly what
# Step 3 reads:  RECON_ROOT=recons_mine bash scripts/run_reproduce.sh <category>
# Paper settings: w=2, t*=250 (test_trajectoy_steps), skip=25, eta=1, 40 seeds (draw k uses seed k).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
CAT="${1:?category}"; CKPT="${2:?DDAD UNet checkpoint}"; N="${3:-40}"; SUFFIX="${4:-_1}"
RECON_ROOT="${RECON_ROOT:-recons_mine}"; OUT_PREFIX="$RECON_ROOT/results_patches_ddad_native"
export CUDA_VISIBLE_DEVICES="${GPU:-0}" PYTHONPATH="$ROOT:${PYTHONPATH:-}"
TEST="data/mvtec_ad/$CAT/test"
[ -d "$TEST" ] || { echo "missing $TEST (bash scripts/download_assets.sh data)"; exit 1; }
for SUBDIR in "$TEST"/*/; do
  SUB=$(basename "$SUBDIR")
  IDS=$(ls "$SUBDIR" | grep -E '\.png$' | sed 's/\.png$//' | sort | head -n "${MAX_IMAGES:-100000}")   # MAX_IMAGES: quick test
  OUT="${OUT_PREFIX}_${CAT}_${SUB}${SUFFIX}"
  echo "== $CAT/$SUB: $(echo $IDS | wc -w) images x $N draws -> $OUT"
  python tools/generate_recons.py --sampler ddad_native --ckpt "$CKPT" \
      --image_dir "$SUBDIR" --samples $IDS --num_seeds "$N" --out_root "$OUT" \
      --sampler_param w=2 --sampler_param test_trajectoy_steps=250 --sampler_param skip=25 --sampler_param eta=1.0
done
echo "Done. Next (Step 3): RECON_ROOT=$RECON_ROOT bash scripts/run_reproduce.sh $CAT"
