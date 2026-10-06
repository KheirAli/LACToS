#!/bin/bash
# Step 1 (CT): DPS denoising reconstructions, exactly the released setting by default.
#
#   GPU=0 bash sampling/sample_ct_dps.sh [name=mine] [sigma=0.05] [scale=0.4] [draws=10]
#   multi-GPU: SHARDS=4 bash sampling/sample_ct_dps.sh   (runs shard g on GPU g, g=0..SHARDS-1)
#   subset:    START=0 END=9 bash sampling/sample_ct_dps.sh   (image_0000..image_0009)
#
# Measurement y = x + sigma*n (all-ones inpainting mask, verified inside sample_fast.py).
# DPS step zeta' = scale/2 with per-sample normalisation  (released: sigma 0.05, scale 0.4 -> zeta' 0.2).
# Seeds are per (image, draw), so sharding and reruns give identical draws.
# Writes $RECON_ROOT/results_patches_ddad_native_CT_<name>/ (RECON_ROOT default recons_mine/)
#   -> Step 3: RECON_ROOT=recons_mine CT_SUB=<name> bash scripts/run_reproduce.sh CT
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:-mine}"; SIGMA="${2:-0.05}"; SCALE="${3:-0.4}"; DRAWS="${4:-10}"; SHARDS="${SHARDS:-1}"
RECON_ROOT="${RECON_ROOT:-recons_mine}"
TASK="configs/ctgrid/sigma${SIGMA}.yaml"
cd "$ROOT/sampling/dps"
[ -f "$TASK" ] || { echo "no task config $TASK (available: $(ls configs/ctgrid))"; exit 1; }
[ -f ../../weights/diffusion/ct_dps_model050000.pt ] || { echo "missing CT prior (bash scripts/download_assets.sh diffusion)"; exit 1; }
SAVE="outputs/CT_sigma${SIGMA}_scale${SCALE}_${NAME}"; mkdir -p "$SAVE" logs
for ((g=0; g<SHARDS; g++)); do
  GPUID=$(( SHARDS > 1 ? g : ${GPU:-0} ))
  CUDA_VISIBLE_DEVICES=$GPUID python sample_fast.py --model_config configs/ct_model_config.yaml \
      --task_config "$TASK" --data_root ../../data/mvtec_ad/CT/test/random --save_dir "$SAVE" \
      --num_measurements "$DRAWS" --batch 25 --scale "$SCALE" --grad_norm per_sample \
      --shard $g --num_shards "$SHARDS" --start "${START:-0}" --end "${END:-99}" > "logs/ct_${NAME}_shard$g.log" 2>&1 &
done
wait
cd "$ROOT"
python sampling/wire_dps_ct.py --src "sampling/dps/$SAVE" --dst "$RECON_ROOT/results_patches_ddad_native_CT_${NAME}" \
    --n_draws "$DRAWS" --sigma "$SIGMA" --scale "$SCALE"
echo "Done. Next (Step 3): RECON_ROOT=$RECON_ROOT CT_SUB=${NAME} bash scripts/run_reproduce.sh CT"
