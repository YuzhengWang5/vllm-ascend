#!/usr/bin/env bash
set -euo pipefail

variant=${1:?iaas or baseline1}
attempt_tag=${2:?unique attempt tag}
dp_size=${GLM_DP_SIZE:?set GLM_DP_SIZE}
tp_size=${GLM_TP_SIZE:?set GLM_TP_SIZE}
max_local_batch=${GLM_MAX_LOCAL_BATCH:-4}
bucket_spec=${GLM_BUCKETS:-}
dram_gib_per_dp=${GLM_DRAM_GIB_PER_DP:-48}
iaas_blocks=${GLM_IAAS_BLOCKS:-4300}
hbm_util=${GLM_HBM_UTIL:-0.92}
remote_host=${GLM_REMOTE_HOST:-7.150.13.62}
remote_store_port=${GLM_REMOTE_STORE_PORT:-29740}
if ! [[ ${dp_size} =~ ^[1-9][0-9]*$ && ${tp_size} =~ ^[1-9][0-9]*$ && ${max_local_batch} =~ ^[1-9][0-9]*$ && ${dram_gib_per_dp} =~ ^[1-9][0-9]*$ && ${iaas_blocks} =~ ^[1-9][0-9]*$ ]]; then
    echo "DP, TP, max local batch, DRAM GiB and IaaS blocks must be positive integers" >&2
    exit 2
fi
if (( dp_size * tp_size != 16 )); then
    echo "DP*TP must equal the 16 decoder dies" >&2
    exit 2
fi
run_dir=${GLM_RUN_DIR:?set GLM_RUN_DIR to run path inside container}
source_dir=/workspace/src/worktrees/glm-w4a8-dptp
model=/workspace/exps/20260915_102819_glm5_w4a8_dp16_ep16/runs/20260915_102819_glm5_w4a8_dp16_ep16/logs/model_mount
gate=${run_dir}/${variant}_${attempt_tag}_bm_init.ready

case "${variant}" in
    iaas|baseline1) ;;
    *) echo "unsupported variant: ${variant}" >&2; exit 2 ;;
esac
if [[ -e ${gate} ]]; then
    echo "refusing to reuse existing pool gate ${gate}" >&2
    exit 3
fi

npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONPATH="${source_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export MF_HOME=/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid
custom_op_root=${source_dir}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="${custom_op_root}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export LD_LIBRARY_PATH="${custom_op_root}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_ENABLE_NZ=1
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=512
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_ENGINE_READY_TIMEOUT_S=2400
export VLLM_RPC_BASE_PATH=/tmp/glm-sweep-${variant}-${attempt_tag}-rpc
export TMPDIR=/tmp/glm-sweep-${variant}-${attempt_tag}-tmp
export TORCH_EXTENSIONS_DIR=/tmp/glm-sweep-torch-ext
mkdir -p "${VLLM_RPC_BASE_PATH}" "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}"
unset ASCEND_CACHE_PATH VLLM_CACHE_ROOT

/usr/local/python3.12.13/bin/python3 - <<'PY'
from types import SimpleNamespace
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import SparseKVOffloadManager
holder = SimpleNamespace()
SparseKVOffloadManager._build_cpp(holder)
print(f"prebuilt sparse_kv_offload: {holder.sparse_kv_offload_cpp.__file__}")
PY

if [[ ${variant} == iaas ]]; then
    /usr/local/python3.12.13/bin/python3 -c \
        'from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.indexer_shm_transport import _load_extension; _load_extension()'
fi

additional_config=$(/usr/local/python3.12.13/bin/python3 - "${variant}" "${gate}" "${dram_gib_per_dp}" "${remote_host}" "${remote_store_port}" <<'PY'
import json
import sys

variant, gate, dram_gib_per_dp, remote_host, remote_store_port = sys.argv[1:]
config = {
    "enable_cpu_binding": True,
    "enable_sparse_li_c8": False,
    "ascend_compilation_config": {
        "enable_static_kernel": False,
        "fuse_norm_quant": False,
    },
    "multistream_overlap_shared_expert": True,
    "weight_nz_mode": 2,
    "enable_flashcomm1": False,
    "enable_shared_expert_dp": True,
    "enable_reduce_sample": True,
    "enable_fused_mc2": 0,
    "sparse_kv_offload_config": {
        "enabled": True,
        "keep_device_kv_cache": False,
        "dram_size_per_dp_GB": int(dram_gib_per_dp),
        "dram_limited_capacity": True,
        "topk_buffer_size": 4096,
        "motivation_baseline": "colocated",
        "motivation_force_oracle_trace": variant == "baseline1",
        "motivation_forced_miss_count": 819 if variant == "baseline1" else 0,
        "remote_indexer_init_gate_file": gate,
    },
}
if variant == "iaas":
    config["sparse_kv_offload_config"].update({
        "remote_indexer_host": remote_host,
        "remote_indexer_transport": "shm",
        "remote_indexer_shm_store": f"tcp://{remote_host}:{remote_store_port}",
        "remote_indexer_profile_device": False,
        "remote_indexer_direct_pack": True,
        "remote_indexer_share_within_tp": True,
        "remote_indexer_gva_tp_fanout": True,
        "remote_indexer_verify_tp_inputs": False,
        "remote_indexer_metadata_once_per_step": True,
        "remote_indexer_service_managed_resident": True,
    })
print(json.dumps(config))
PY
)

hf_overrides='{"use_index_cache":true,"index_topk_freq":1}'
kv_transfer_config='{"kv_connector":"SFAOffloadDecodeBenchConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"main_fill_value":0.015,"indexer_fill_value":1,"indexer_scale_value":0.015,"fill_std":0.0}}'
block_args=()
if [[ ${variant} == iaas ]]; then
    block_args=(--num-gpu-blocks-override "${iaas_blocks}")
fi

capture_sizes=$(/usr/local/python3.12.13/bin/python3 - "${max_local_batch}" "${bucket_spec}" <<'PY'
import json
import sys
maximum = int(sys.argv[1])
buckets = [int(x) for x in sys.argv[2].split(",") if x] if sys.argv[2] else list(range(1, maximum + 1))
if not buckets or buckets != sorted(set(buckets)) or buckets[0] != 1 or buckets[-1] != maximum:
    raise SystemExit("GLM_BUCKETS must be increasing, start at 1, and end at max local batch")
print(json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": buckets}))
PY
)
echo "GLM sweep setup: variant=${variant} dp=${dp_size} tp=${tp_size} local_batch=${max_local_batch} dram_gib_per_dp=${dram_gib_per_dp} iaas_blocks=${iaas_blocks} hbm_util=${hbm_util} capture=${capture_sizes}"

exec /usr/local/python3.12.13/bin/vllm serve "${model}" \
  --served-model-name GLM-5 \
  --host 127.0.0.1 --port 18300 \
  --tensor-parallel-size "${tp_size}" --data-parallel-size "${dp_size}" \
  --enable-expert-parallel --distributed-executor-backend mp \
  --quantization ascend --dtype bfloat16 --seed 1024 --trust-remote-code \
  --max-num-seqs "${max_local_batch}" --max-model-len 131322 --max-num-batched-tokens "${max_local_batch}" \
  --block-size 128 --gpu-memory-utilization "${hbm_util}" \
  --no-enable-prefix-caching --no-async-scheduling \
  --safetensors-load-strategy prefetch \
  --hf-overrides "${hf_overrides}" \
  --additional-config "${additional_config}" \
  --kv-transfer-config "${kv_transfer_config}" \
  --reasoning-parser glm45 \
  --compilation-config "${capture_sizes}" \
  "${block_args[@]}"
