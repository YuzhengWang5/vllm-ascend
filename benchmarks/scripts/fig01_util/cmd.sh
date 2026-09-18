#!/usr/bin/env bash
set -euo pipefail

run_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=/root/wyz/wyz-workspace/dsamem-eurosys27/src/worktrees/iaas_tp_shared_indexer
container=dsamem_fig01_util_20260918
container_run=/workspace/exps/20260914_232704_iaas_tp_shared_indexer/runs/20260918_151300_stage_aicore_utilization
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# Recheck before creating any NPU context. A concurrent service invalidates
# the utilization experiment even if the kernels themselves can still run.
npu-smi info > "${run_dir}/logs/00_preflight_npu_smi.txt"
python "${source_dir}/benchmarks/scripts/fig01_util/check_idle.py" \
  "${run_dir}/logs/00_preflight_npu_smi.txt"

if ! docker container inspect "$container" >/dev/null 2>&1; then
  docker run --runtime=ascend --privileged --network=host --ipc=host -dit \
    --name "$container" \
    -e ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    -v /root/wyz/wyz-workspace/dsamem-eurosys27:/workspace \
    -v /mnt/sfs_turbo:/mnt/sfs_turbo \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    quay.io/ascend/vllm-ascend:v0.26.0rc1-a3 /bin/bash \
    > "${run_dir}/logs/00_container_id.txt"
else
  docker start "$container" > "${run_dir}/logs/00_container_start.txt" || true
fi

marker="${run_dir}/logs/active_stage.json"
stop="${run_dir}/logs/stop_monitor"
python - "$marker" "$stop" <<'PY'
from pathlib import Path
import sys
for name in sys.argv[1:]:
    Path(name).unlink(missing_ok=True)
PY
python "${source_dir}/benchmarks/scripts/fig01_util/monitor.py" \
  --marker "$marker" --stop "$stop" \
  --output "${run_dir}/logs/usage_samples.jsonl" \
  > "${run_dir}/logs/monitor.stdout" 2> "${run_dir}/logs/monitor.stderr" &
monitor_pid=$!
finish_monitor() {
  touch "$stop"
  wait "$monitor_pid" || true
  python - "$marker" "$stop" <<'PY'
from pathlib import Path
import sys
for name in sys.argv[1:]:
    Path(name).unlink(missing_ok=True)
PY
}
trap finish_monitor EXIT

for stage in dense32 dense64 indexer sparse moe; do
  logfile="${run_dir}/logs/${stage}.jsonl"
  if [[ -e "$logfile" ]]; then
    echo "Existing $logfile; archive or remove it before a new formal attempt" >&2
    exit 1
  fi
  npu-smi info > "${run_dir}/logs/${stage}_preflight_npu_smi.txt"
  python "${source_dir}/benchmarks/scripts/fig01_util/check_idle.py" \
    "${run_dir}/logs/${stage}_preflight_npu_smi.txt"
  docker exec "$container" timeout 3600 bash \
    "${container_run}/scripts/run.sh" "$stage" \
    --seconds 20 --batches 4 8 12 16 24 32 48 64 96 128 \
    > "$logfile" 2>&1
  npu-smi info > "${run_dir}/logs/${stage}_post_npu_smi.txt"
  python "${source_dir}/benchmarks/scripts/fig01_util/check_idle.py" \
    "${run_dir}/logs/${stage}_post_npu_smi.txt"
done

for stage in dense32 dense64 sparse; do
  for mode in single tp8_comm; do
    logfile="${run_dir}/logs/ablation_${stage}_${mode}.jsonl"
    if [[ -e "$logfile" ]]; then
      echo "Existing $logfile; archive or remove it before a new formal attempt" >&2
      exit 1
    fi
    npu-smi info > "${run_dir}/logs/ablation_${stage}_${mode}_preflight_npu_smi.txt"
    python "${source_dir}/benchmarks/scripts/fig01_util/check_idle.py" \
      "${run_dir}/logs/ablation_${stage}_${mode}_preflight_npu_smi.txt"
    docker exec "$container" timeout 3600 bash \
      "${container_run}/scripts/run.sh" "$stage" --mode "$mode" \
      --seconds 20 --batches 4 8 16 32 64 128 \
      > "$logfile" 2>&1
    npu-smi info > "${run_dir}/logs/ablation_${stage}_${mode}_post_npu_smi.txt"
    python "${source_dir}/benchmarks/scripts/fig01_util/check_idle.py" \
      "${run_dir}/logs/ablation_${stage}_${mode}_post_npu_smi.txt"
  done
done

finish_monitor
trap - EXIT
python "${source_dir}/benchmarks/scripts/fig01_util/summarize.py" "$run_dir" \
  > "${run_dir}/logs/summary_stdout.txt"
python "${source_dir}/benchmarks/scripts/fig01_util/plot.py" \
  "${run_dir}/logs/stage_aicore_utilization.csv" \
  "${run_dir}/plots/figure_c"
python "${source_dir}/benchmarks/scripts/fig01_util/summarize_ablations.py" "$run_dir" \
  > "${run_dir}/logs/ablation_summary_stdout.txt"
python "${source_dir}/benchmarks/scripts/fig01_util/plot_ablations.py" \
  "${run_dir}/logs/attention_mode_comparison.csv" \
  "${run_dir}/plots/attention_modes"
