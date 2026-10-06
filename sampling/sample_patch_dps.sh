#!/bin/bash
# Step 1 (X-ray, faces): DPS superpixel-patch inpainting, following the procedure of the released sets.
#
#   GPU=0 bash sampling/sample_patch_dps.sh <xray|faces> [sigma] [image_dir] [suffix=_1]
#     xray : sigma default 0.3, images default data/mvtec_ad/xray/test/scissors,  6 patches
#     faces: sigma default 0.5, images default data/mvtec_ad/faces/test/random,  12 patches
#
# Per image and patch (top index, column band): superpixel mask (super_pixel_generation.py) ->
#   sample_batch.py --box_coords <top> 4 <left> <right> --mask_prob 0.9 --seed 3 --num_measurements 4
# The patch outputs are then flattened into the standard layout
#   $RECON_ROOT/results_patches_ddad_native_<xray_scissors|faces_random><suffix>/samples_<id>/ddad_native_0_4/inpainting/
# so Step 3 is:  RECON_ROOT=recons_mine bash scripts/run_reproduce.sh <xray|faces>
#
# Caveat: the DPS guidance scale of the released X-ray/faces sets was not recorded (patch_inpainting_config.yaml
# uses the upstream default 0.5), and the superpixel-box mask code may differ from the version used in 2026-08.
# This script reproduces the procedure, not necessarily bit-identical reconstructions.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOM="${1:?xray|faces}"; RECON_ROOT="${RECON_ROOT:-recons_mine}"; SUFFIX="${4:-_1}"
case "$DOM" in
  xray)  SIG="${2:-0.3}"; IMG="${3:-data/mvtec_ad/xray/test/scissors}"; MC=configs/xray_model_config.yaml; DST="xray_scissors"
         TOPS="8:0:64 13:64:128 16:0:64 17:64:128 20:0:64 21:64:128" ;;
  faces) SIG="${2:-0.5}"; IMG="${3:-data/mvtec_ad/faces/test/random}"; MC=configs/faces_model_config.yaml; DST="faces_random"
         TOPS="8:0:64 12:0:64 13:64:128 14:128:192 16:0:64 17:64:128 18:128:192 19:192:256 20:0:64 21:64:128 22:128:192 23:192:256" ;;
  *) echo "domain must be xray or faces"; exit 1 ;;
esac
cd "$ROOT"; IMG="$(cd "$IMG" && pwd)"; OUTDIR="$(mkdir -p "$RECON_ROOT" && cd "$RECON_ROOT" && pwd)/results_patches_ddad_native_${DST}${SUFFIX}"
RAW="$ROOT/sampling/dps/outputs/patch_${DOM}_sigma${SIG}"; mkdir -p "$RAW"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
for f in "$IMG"/*.png; do
  ID=$(basename "$f" .png); TMP=$(mktemp -d); mkdir -p "$TMP/img"; cp "$f" "$TMP/img/"
  python "$ROOT/super_pixel_generation.py" --input_image "$TMP/img/$ID.png" --output_dir "$TMP" > /dev/null
  sed "s/^\(\s*sigma:\s*\).*/\1${SIG}/" "$ROOT/sampling/dps/configs/patch_inpainting_config.yaml" > "$TMP/task.yaml"
  for T in $TOPS; do
    IFS=: read -r TOP L R <<< "$T"
    SAVE="$RAW/samples_$ID/Combined_${SIG}_sigma_batched_${TOP}_4"
    [ "$(ls "$SAVE/inpainting/recon" 2>/dev/null | grep -c png)" -ge 4 ] && continue   # resume
    ( cd "$ROOT/sampling/dps" && python sample_batch.py --model_config "$MC" --diffusion_config configs/diffusion_config.yaml \
        --task_config "$TMP/task.yaml" --save_dir "$SAVE" --data_root "$TMP/img" --seed 3 \
        --box_coords "$TOP" 4 "$L" "$R" --mask_prob 0.9 --mask_path "$TMP/mask.png" --num_measurements 4 ) > "$TMP/log" 2>&1 \
      || { echo "sampling failed for $ID top $TOP:"; tail -5 "$TMP/log"; exit 1; }
  done
  # flatten all patches of this image into the evaluation layout
  B="$OUTDIR/samples_$ID/ddad_native_0_4/inpainting"; mkdir -p "$B/recon" "$B/label" "$B/input"; j=0
  for P in "$RAW/samples_$ID"/Combined_${SIG}_sigma_batched_*_4/inpainting; do
    for r in "$P"/recon/*.png; do cp "$r" "$B/recon/${j}_0_00000.png"; j=$((j+1)); done
  done
  first=$(ls -d "$RAW/samples_$ID"/Combined_${SIG}_sigma_batched_*_4/inpainting | head -1)
  cp "$(ls "$first"/label/*.png | head -1)" "$B/label/0_00000.png"; cp "$(ls "$first"/input/*.png | head -1)" "$B/input/0_00000.png"
  echo "  $ID: $j reconstructions"; rm -rf "$TMP"
done
echo "Done -> $OUTDIR. Next (Step 3): RECON_ROOT=$RECON_ROOT bash scripts/run_reproduce.sh $DOM"
