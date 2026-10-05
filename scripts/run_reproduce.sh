#!/bin/bash
# Reproduce the paper numbers (EVALUATION ONLY) on all 20 targets from the released
# reconstructions + checkpoints. See docs/HYPERPARAMETERS.md for every flag below.
#
#   GPU=0 bash scripts/run_reproduce.sh                      # all targets (~3-6 h on one A6000)
#   GPU=0 bash scripts/run_reproduce.sh toothbrush CT faces  # any subset
#
# Targets: bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile
#          toothbrush transistor wood zipper CT xray xray_b2 faces faces_moreface
#
# Outputs (under $OUT, default ./results_eval_reproduction):
#   json/<target>/   per-subcategory JSON (+ *_summary.json)  -> px ROC-AUC (table 3a)
#   maps/<target>/   delta-map dumps (npz: delta_map, gt_mask, valid_mask) -> scripts/score_metrics.sh
#   logs/<target>.log
# Then:  bash scripts/score_metrics.sh && python scripts/summarize_results.py
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OOD_CT_MASK_ROOT="$ROOT/data/id_new_warped_images"    # CT GT + body masks
export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-$ROOT/weights/torch}"     # ships resnet101-cd907fc2.pth (IMAGENET1K_V2) -> fully offline
export ABL_SEED="${ABL_SEED:-0}"                               # tools/seeded_run.py seeds python/numpy/torch
GPU="${GPU:-0}"; OUT="${OUT:-$ROOT/results_eval_reproduction}"; mkdir -p "$OUT/json" "$OUT/maps" "$OUT/logs"

W=weights; AE=$W/ae; BB=$W/backbones
RECON=./recons/results_patches_ddad_native        # prefix: dir = ${RECON}_<category>_<subcategory><suffix>
DATA="$ROOT/data/mvtec_ad"
SEEDED="python tools/seeded_run.py"
IMAGENET_V1="$BB/resnet101_imagenet1k_v1.pth"     # == torchvision ResNet101_Weights.IMAGENET1K_V1 (resnet101-63fe2227.pth)
XRAY_FT="$BB/xray_ddad_lastlayer_feat4.pth"       # DDAD domain-adaptation fine-tune, last layer only, epoch 4

# shared typical-set / superpixel settings (MVTec, xray, faces)
COMMON="--n_pca 3 --bins_pca 32 --bins_rgb 32 --superpixel_target_size 30 --smooth_sigma 1 --no_plots --skip_sampling"

run_mvtec() { local cat="$1"   # 14 MVTec cats (all but pill): ImageNet-V1 backbone via DDAD_FE_WEIGHTS + per-cat AE, recon suffix _1
  DUMP_DELTA_DIR="$OUT/maps/mvtec/$cat" DDAD_FE_WEIGHTS="$IMAGENET_V1" CUDA_VISIBLE_DEVICES="$GPU" \
  $SEEDED embedder/evaluate_ddad_fe.py --config configs/experiment_ddad_native.yaml \
    --category "$cat" --all_subcategories --results_root "$RECON" --gt_root "$DATA/$cat/ground_truth" \
    --autoencoder_path "$AE/$cat.pth" $COMMON --max_reconstructions 50 --results_suffix _1 --output_dir "$OUT/json/mvtec/$cat"; }
run_pill() {                   # pill: plain evaluate.py (torchvision IMAGENET1K_V2 backbone), recon suffix _3
  DUMP_DELTA_DIR="$OUT/maps/mvtec/pill" CUDA_VISIBLE_DEVICES="$GPU" \
  $SEEDED evaluate.py --config configs/experiment_ddad_native.yaml \
    --category pill --all_subcategories --results_root "$RECON" --gt_root "$DATA/pill/ground_truth" \
    --autoencoder_path "$AE/pill.pth" $COMMON --max_reconstructions 50 --results_suffix _3 --output_dir "$OUT/json/mvtec/pill"; }
run_CT() {                     # CT (100 slices): IMAGENET1K_V2 backbone, CT AE, 64/64 bins, 40 DDAD-native draws
  DUMP_DELTA_DIR="$OUT/maps/CT/ct100" CUDA_VISIBLE_DEVICES="$GPU" \
  $SEEDED evaluate.py --config configs/experiment_ddad_native_xray.yaml \
    --category CT --subcategory scan --results_root "$RECON" --gt_root "$DATA/CT/ground_truth" \
    --autoencoder_path "$AE/CT.pth" --n_pca 3 --bins_pca 64 --bins_rgb 64 --sigma_rohan 1.0 \
    --superpixel_target_size 30 --smooth_sigma 1 --no_plots --skip_sampling --max_reconstructions 40 --output_dir "$OUT/json/CT/ct100"; }
run_xray() { local tag="$1" suffix="$2"   # xray/scissors: DDAD last-layer fine-tuned backbone + xray AE. b1 = 32 patients (_1), b2 = 40 patients (_b2_500k)
  DUMP_DELTA_DIR="$OUT/maps/xray/$tag" DDAD_FE_WEIGHTS="$XRAY_FT" CUDA_VISIBLE_DEVICES="$GPU" \
  $SEEDED embedder/evaluate_ddad_fe.py --config configs/experiment_ddad_native_xray_real.yaml \
    --category xray --subcategory scissors --results_root "$RECON" --gt_root "$DATA/xray/ground_truth" \
    --autoencoder_path "$AE/xray.pth" $COMMON --max_reconstructions 40 --results_suffix "$suffix" --output_dir "$OUT/json/xray/$tag"; }
run_faces() {                  # faces random22: IMAGENET1K_V2 backbone, faces AE, DPS recons (FFHQ prior)
  DUMP_DELTA_DIR="$OUT/maps/faces/random22" CUDA_VISIBLE_DEVICES="$GPU" \
  $SEEDED evaluate.py --config configs/experiment_ddad_native_faces.yaml \
    --category faces --all_subcategories --results_root "$RECON" --gt_root "$DATA/faces/ground_truth" \
    --autoencoder_path "$AE/faces.pth" $COMMON --sigma_rohan 1.0 --results_suffix _1 --output_dir "$OUT/json/faces/random22"; }
run_faces_moreface() {         # faces moreface28 (the other 28 faces of the full-50 set)
  DUMP_DELTA_DIR="$OUT/maps/faces/moreface28" CUDA_VISIBLE_DEVICES="$GPU" \
  $SEEDED evaluate.py --config configs/experiment_dps_faces_moreface.yaml \
    --category faces --all_subcategories --results_root ./recons/results_patches_dps_guided \
    --gt_root "$ROOT/data/faces_moreface28/ground_truth" \
    --autoencoder_path "$AE/faces.pth" $COMMON --sigma_rohan 1.0 --output_dir "$OUT/json/faces/moreface28"; }

ALL=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper CT xray faces faces_moreface xray_b2)
TARGETS=("$@"); [ "${#TARGETS[@]}" -eq 0 ] && TARGETS=("${ALL[@]}")
for t in "${TARGETS[@]}"; do
  echo "════════ $t (GPU $GPU) $(date '+%F %T') ════════"
  case "$t" in
    pill)           run_pill ;;
    CT)             run_CT ;;
    xray|xray_b1)   run_xray b1 _1 ;;
    xray_b2)        run_xray b2 _b2_500k ;;
    faces)          run_faces ;;
    faces_moreface) run_faces_moreface ;;
    bottle|cable|capsule|carpet|grid|hazelnut|leather|metal_nut|screw|tile|toothbrush|transistor|wood|zipper) run_mvtec "$t" ;;
    *) echo "  unknown target '$t'"; continue ;;
  esac > "$OUT/logs/$t.log" 2>&1 && echo "  ok  (log: $OUT/logs/$t.log)" || echo "  [FAILED] see $OUT/logs/$t.log"
done
echo "Done -> $OUT.  Next: bash scripts/score_metrics.sh && python scripts/summarize_results.py"
