#!/usr/bin/env bash
set -euo pipefail

stage=${1:?stage}
shift
run_dir=/workspace/exps/20260914_232704_iaas_tp_shared_indexer/runs/20260918_151300_stage_aicore_utilization
source_dir=/workspace/src/worktrees/iaas_tp_shared_indexer
mode=tp8_no_comm
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ ${args[i]} == --mode ]]; then
    mode=${args[i+1]:?mode value}
  fi
done
if [[ $mode == single ]]; then
  export ASCEND_RT_VISIBLE_DEVICES=0
  ranks=1
else
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  ranks=16
fi
export PYTHONPATH=${source_dir}:/workspace/src/vllm
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_HOST_SOCKET_PORT_RANGE=auto

exec /workspace/src/vllm/.venv/bin/python -m torch.distributed.run \
  --master_addr=127.0.0.1 --master_port=29579 --nproc_per_node="$ranks" \
  "${source_dir}/benchmarks/scripts/fig01_util/bench.py" \
  --stage "$stage" --marker "${run_dir}/logs/active_stage.json" "$@"
