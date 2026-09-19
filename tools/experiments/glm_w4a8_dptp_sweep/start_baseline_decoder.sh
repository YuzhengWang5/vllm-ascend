#!/usr/bin/env bash
set -euo pipefail

attempt_tag=${1:?unique attempt tag}
host_run_dir=${GLM_HOST_RUN_DIR:?set GLM_HOST_RUN_DIR to the host run directory}
container_run_dir=${GLM_RUN_DIR:?set GLM_RUN_DIR to the same run directory inside the container}
container_script_dir=${GLM_CONTAINER_SCRIPT_DIR:?set GLM_CONTAINER_SCRIPT_DIR inside the container}
dp_size=${GLM_DP_SIZE:?set GLM_DP_SIZE}
tp_size=${GLM_TP_SIZE:?set GLM_TP_SIZE}
max_local_batch=${GLM_MAX_LOCAL_BATCH:?set GLM_MAX_LOCAL_BATCH}
buckets=${GLM_BUCKETS:-}
dram_gib_per_dp=${GLM_DRAM_GIB_PER_DP:?set GLM_DRAM_GIB_PER_DP}
hbm_util=${GLM_HBM_UTIL:-0.92}
load_strategy=${GLM_SAFETENSORS_LOAD_STRATEGY:-lazy}
container=${GLM_DECODER_CONTAINER:-glm5_iaas_decoder}
devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

mkdir -p "${host_run_dir}/logs"
npu-smi info > "${host_run_dir}/logs/baseline1_${attempt_tag}_npu_before.txt"
docker exec -d \
    -e ASCEND_RT_VISIBLE_DEVICES="${devices}" \
    -e GLM_DP_SIZE="${dp_size}" \
    -e GLM_TP_SIZE="${tp_size}" \
    -e GLM_MAX_LOCAL_BATCH="${max_local_batch}" \
    -e GLM_BUCKETS="${buckets}" \
    -e GLM_DRAM_GIB_PER_DP="${dram_gib_per_dp}" \
    -e GLM_IAAS_BLOCKS=1 \
    -e GLM_HBM_UTIL="${hbm_util}" \
    -e GLM_SAFETENSORS_LOAD_STRATEGY="${load_strategy}" \
    -e GLM_RUN_DIR="${container_run_dir}" \
    "${container}" bash -lc \
    "source /usr/local/Ascend/cann-9.1.0/set_env.sh; exec bash '${container_script_dir}/serve_decoder.sh' baseline1 '${attempt_tag}' > '${container_run_dir}/logs/baseline1_${attempt_tag}_server.log' 2>&1"
echo "started baseline1/${attempt_tag} on ${devices}; dp=${dp_size} tp=${tp_size}; max local batch=${max_local_batch}; pool=${dram_gib_per_dp} GiB/DP"
