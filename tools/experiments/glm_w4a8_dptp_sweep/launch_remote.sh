#!/usr/bin/env bash
set -euo pipefail

attempt_tag=${1:?unique attempt tag}
dp_size=${GLM_DP_SIZE:?set GLM_DP_SIZE}
tp_size=${GLM_TP_SIZE:?set GLM_TP_SIZE}
max_local_batch=${GLM_MAX_LOCAL_BATCH:-4}
bucket_spec=${GLM_BUCKETS:-}
iaas_blocks=${GLM_IAAS_BLOCKS:-4300}
if ! [[ ${dp_size} =~ ^[1-9][0-9]*$ && ${tp_size} =~ ^[1-9][0-9]*$ && ${max_local_batch} =~ ^[1-9][0-9]*$ && ${iaas_blocks} =~ ^[1-9][0-9]*$ ]]; then
    echo "DP, TP, max local batch and IaaS blocks must be positive integers" >&2
    exit 2
fi
if (( dp_size * tp_size != 16 )); then
    echo "DP*TP must equal 16" >&2
    exit 2
fi
if [[ -n ${bucket_spec} ]]; then
    service_batches=${bucket_spec//,/ }
else
    service_batches=$(seq -s ' ' 1 "${max_local_batch}")
fi
remote_host=${GLM_REMOTE_HOST:-7.150.13.62}
remote_container=${GLM_REMOTE_CONTAINER:-glm_w4a8_sweep_indexer_claude}
remote_source=/workspace/src/remote_glm_sweep_20260919
store_url=tcp://${remote_host}:${GLM_REMOTE_STORE_PORT:-29740}
run_dir=/root/wyz/wyz-workspace/dsamem-eurosys27/exps/20260919_012520_glm-w4a8-dptp/runs/20260919_012520_dp_tp_sweep
gate=${run_dir}/iaas_${attempt_tag}_bm_init.ready
ssh_opts=(-o ConnectTimeout=10 -o StrictHostKeyChecking=no)

if [[ -e ${gate} ]]; then
    echo "refusing to reuse existing gate ${gate}" >&2
    exit 2
fi
ssh "${ssh_opts[@]}" "${remote_host}" 'npu-smi info'

ssh "${ssh_opts[@]}" "${remote_host}" \
    "docker exec -e ASCEND_RT_VISIBLE_DEVICES=0 \
       -e PYTHONPATH='${remote_source}' \
       -e MF_HOME=/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid \
       -e TORCH_EXTENSIONS_DIR=/tmp/glm5-bf16-torch-ext \
       '${remote_container}' bash -lc \
       \"source /usr/local/Ascend/cann-9.1.0/set_env.sh; \
         python -c 'from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.indexer_shm_transport import _load_extension; _load_extension()'\""

for ((rank=0; rank<dp_size; rank++)); do
    service_device=$((rank * (16 / dp_size)))
    decoder_rank=$((dp_size + rank * tp_size))
    world_size=$((dp_size * (tp_size + 1)))
    ssh "${ssh_opts[@]}" "${remote_host}" \
        "docker exec -d \
          -e ASCEND_RT_VISIBLE_DEVICES='${service_device}' \
          -e PYTHONPATH='${remote_source}' \
          -e MF_HOME=/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid \
          -e TORCH_EXTENSIONS_DIR=/tmp/glm5-bf16-torch-ext \
          '${remote_container}' bash -lc \
          'source /usr/local/Ascend/cann-9.1.0/set_env.sh; \
           exec python -m vllm_ascend.distributed.kv_transfer.sparse_kv_offload.remote_indexer_shm_service \
             --store-url ${store_url} --world-size ${world_size} --global-rank ${rank} \
             --decoder-rank ${decoder_rank} --device 0 --layers 78 \
             --batches ${service_batches} --prelude-batches --heads 32 \
             --head-dim 128 --topk 2048 --index-dtype bf16 \
             --cache-blocks ${iaas_blocks} --block-size 128 --block-table-cols 1026 \
             --metadata-once-per-step --dynamic-batch-by-request-bytes \
             --service-managed-resident --synthetic-oracle-topk \
             --forced-resident-miss-count 819 --log-every 64 \
             --pid-file /tmp/glm_sweep_${attempt_tag}_rank${rank}.pid \
             > /tmp/glm_sweep_${attempt_tag}_rank${rank}.log 2>&1'"
done

deadline=$((SECONDS + 180))
while true; do
    ready=$(ssh "${ssh_opts[@]}" "${remote_host}" \
        "docker exec '${remote_container}' bash -lc \
         'alive=0; for f in /tmp/glm_sweep_${attempt_tag}_rank*.pid; do \
            test -f \"\$f\" || continue; p=\$(cat \"\$f\"); \
            kill -0 \"\$p\" 2>/dev/null && alive=\$((alive+1)); \
          done; echo \$alive'")
    if [[ ${ready} -eq ${dp_size} ]]; then
        break
    fi
    if [[ ${SECONDS} -ge ${deadline} ]]; then
        echo "only ${ready}/${dp_size} remote services alive after 180 seconds" >&2
        exit 4
    fi
    sleep 2
done

sync
touch "${gate}"
echo "${dp_size}/${dp_size} remote BF16 services alive; IaaS decoder gate opened"
