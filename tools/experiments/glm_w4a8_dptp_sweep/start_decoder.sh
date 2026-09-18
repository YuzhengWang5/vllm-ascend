#!/usr/bin/env bash
set -euo pipefail

variant=${1:?iaas or baseline1}
attempt_tag=${2:?unique attempt tag}
dp_size=${GLM_DP_SIZE:?set GLM_DP_SIZE}
tp_size=${GLM_TP_SIZE:?set GLM_TP_SIZE}
max_local_batch=${GLM_MAX_LOCAL_BATCH:?set GLM_MAX_LOCAL_BATCH}
buckets=${GLM_BUCKETS:-}
dram_gib_per_dp=${GLM_DRAM_GIB_PER_DP:?set GLM_DRAM_GIB_PER_DP}
iaas_blocks=${GLM_IAAS_BLOCKS:?set GLM_IAAS_BLOCKS}
hbm_util=${GLM_HBM_UTIL:-0.92}
remote_host=${GLM_REMOTE_HOST:-7.150.13.62}
remote_store_port=${GLM_REMOTE_STORE_PORT:-29740}
remote_profile_device=${GLM_REMOTE_PROFILE_DEVICE:-0}
profiler_dir=${GLM_PROFILER_DIR:-}
case "${variant}" in
    iaas|baseline1) ;;
    *) echo "unsupported variant: ${variant}" >&2; exit 2 ;;
esac

run_dir=/root/wyz/wyz-workspace/dsamem-eurosys27/exps/20260919_012520_glm-w4a8-dptp/runs/20260919_012520_dp_tp_sweep
container=${GLM_DECODER_CONTAINER:-glm_w4a8_sweep_decoder_claude}
devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
npu-smi info > "${run_dir}/logs/${variant}_${attempt_tag}_npu_before.txt"
docker exec -d \
    -e ASCEND_RT_VISIBLE_DEVICES="${devices}" \
    -e GLM_DP_SIZE="${dp_size}" \
    -e GLM_TP_SIZE="${tp_size}" \
    -e GLM_MAX_LOCAL_BATCH="${max_local_batch}" \
    -e GLM_BUCKETS="${buckets}" \
    -e GLM_DRAM_GIB_PER_DP="${dram_gib_per_dp}" \
    -e GLM_IAAS_BLOCKS="${iaas_blocks}" \
    -e GLM_HBM_UTIL="${hbm_util}" \
    -e GLM_REMOTE_HOST="${remote_host}" \
    -e GLM_REMOTE_STORE_PORT="${remote_store_port}" \
    -e GLM_REMOTE_PROFILE_DEVICE="${remote_profile_device}" \
    -e GLM_PROFILER_DIR="${profiler_dir}" \
    -e GLM_RUN_DIR="/workspace/exps/20260919_012520_glm-w4a8-dptp/runs/20260919_012520_dp_tp_sweep" \
    "${container}" bash -lc \
    "source /usr/local/Ascend/cann-9.1.0/set_env.sh; exec bash /workspace/exps/20260919_012520_glm-w4a8-dptp/runs/20260919_012520_dp_tp_sweep/scripts/serve_decoder.sh '${variant}' '${attempt_tag}' > '/workspace/exps/20260919_012520_glm-w4a8-dptp/runs/20260919_012520_dp_tp_sweep/logs/${variant}_${attempt_tag}_server.log' 2>&1"
echo "started ${variant}/${attempt_tag} decoder on ${devices}; dp=${dp_size} tp=${tp_size}; max local batch=${max_local_batch}; pool=${dram_gib_per_dp} GiB/DP; IaaS blocks=${iaas_blocks}"
