#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CT reconstruction with DPS (self-contained bundle version).
# All paths are relative to this bundle directory, so it runs anywhere.
#
# Per image: the 16 patches of the 4x4 grid (64x64 each, 256x256 image) are
# launched in parallel over the GPUs listed in GPUS, then we wait.
# Each patch job draws NUM_MEASUREMENTS noisy measurements and reconstructs
# each one (batched on one GPU).
#
# Output:
#   ${BASE_SAVE_DIR}/samples_CT_random_image_XXXX_${step}/Combined_${SIGMA}_sigma_batched_${patch}_4/inpainting/recon/
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$(readlink -f "$0")")"
ROOT="$(pwd)"

# --- Configuration (edit here) ---
SIGMA=0.2                      # must match measurement.noise.sigma in configs/ct_inpainting_config.yaml
BASE_SAVE_DIR="${ROOT}/outputs/CT_guided_${SIGMA}"
IMAGES_DIR="${ROOT}/data/CT_images"
MASK_PATH="${ROOT}/data/mask.png"
MODEL_CONFIG="${ROOT}/configs/ct_model_config.yaml"
TASK_CONFIG="${ROOT}/configs/ct_inpainting_config.yaml"
DIFFUSION_CONFIG="${ROOT}/configs/diffusion_config.yaml"
CKPT_DIR="${ROOT}/checkpoints"
MODEL_STEPS=(50000)            # -> checkpoints/model050000.pt
SEED=3
NUM_MEASUREMENTS=4
MASK_PROB=0.9
START_SAMPLE=${START_SAMPLE:-0}   # first image index to process (resume support)
END_SAMPLE=${END_SAMPLE:-99}      # last image index to process

# GPUs to spread the 16 patch jobs over (round-robin). Each job needs ~5-6 GB.
# Override from the command line:  GPUS="0 1" bash run_ct_reconstruction.sh
GPUS=(${GPUS:-0 1 2 3})

SAMPLES_DIR="${ROOT}/temp_samples_ct"
CATEGORY="CT"
SUBCATEGORY="random"

# --- Patch grid: "top bottom left right" for patch_id = row*4 + col ---
get_patches() {
    local bottom=$1
    local i=0
    for row in 0 1 2 3; do
        for col in 0 1 2 3; do
            local gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
            echo "$row $col $i $bottom $(( col * 64 )) $(( col * 64 + 64 )) $gpu"
            i=$(( i + 1 ))
        done
    done
}

process_patches_batch() {
    local bottom=$1 sample_name=$2 seed=$3 model_config=$4
    echo "Processing patches with bottom=$bottom for sample=$sample_name"
    while IFS= read -r patch; do
        read -r row col top _ left right gpu <<< "$patch"
        CUDA_VISIBLE_DEVICES=$gpu python3 sample_batch.py \
            --model_config="${model_config}" \
            --diffusion_config="${DIFFUSION_CONFIG}" \
            --task_config="${TASK_CONFIG}" \
            --save_dir="${BASE_SAVE_DIR}/samples_${sample_name}/Combined_${SIGMA}_sigma_batched_${top}_${bottom}" \
            --data_root="${SAMPLES_DIR}" \
            --seed "${seed}" \
            --box_coords "$top" "$bottom" "$left" "$right" \
            --mask_prob "${MASK_PROB}" \
            --mask_path "${MASK_PATH}" \
            --num_measurements "${NUM_MEASUREMENTS}" &
    done < <(get_patches "$bottom")
    wait
}

process_sample_batch() {
    local filepath=$1 model_step=$2 model_config=$3
    local filename; filename=$(basename "${filepath}")
    local sample_name="${CATEGORY}_${SUBCATEGORY}_${filename%.*}_${model_step}"
    echo "=========================================="
    echo "Processing sample: ${filepath}  ->  ${sample_name}"
    echo "=========================================="
    rm -f "${SAMPLES_DIR}"/*
    cp "${filepath}" "${SAMPLES_DIR}/${filename}"
    process_patches_batch 4 "$sample_name" "${SEED}" "${model_config}"
    rm -f "${SAMPLES_DIR}/${filename}"
    echo "✓ Completed sample: ${filename}"
}

# --- Main ---
echo "Starting CT batch processing in ${ROOT} on GPUs: ${GPUS[*]}"
mkdir -p "${SAMPLES_DIR}" "${BASE_SAVE_DIR}"
rm -f "${SAMPLES_DIR}"/*

for model_step in "${MODEL_STEPS[@]}"; do
    model_path="${CKPT_DIR}/model$(printf '%06d' "${model_step}").pt"
    if [ ! -f "${model_path}" ]; then
        echo "✗ Skipping step ${model_step}: checkpoint not found at ${model_path}"; continue
    fi
    tmp_config=$(mktemp "${ROOT}/.model_config_XXXXX.yaml")
    cp "${MODEL_CONFIG}" "${tmp_config}"
    sed -i "s|^model_path:.*|model_path: ${model_path}|" "${tmp_config}"
    echo "Using checkpoint: ${model_path}"

    for sample_file in "${IMAGES_DIR}"/*.png; do
        sample_num=$(basename "${sample_file}" .png)
        sample_num=$((10#${sample_num#image_}))
        if [ "${sample_num}" -lt "${START_SAMPLE}" ] || [ "${sample_num}" -gt "${END_SAMPLE}" ]; then continue; fi
        process_sample_batch "${sample_file}" "${model_step}" "${tmp_config}"
    done
    rm -f "${tmp_config}"
done

rm -rf "${SAMPLES_DIR}"
echo "All CT samples processed. Outputs in ${BASE_SAVE_DIR}"
