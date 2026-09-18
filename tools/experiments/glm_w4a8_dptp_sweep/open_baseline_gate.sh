#!/usr/bin/env bash
set -euo pipefail

attempt_tag=${1:?unique attempt tag}
run_dir=/root/wyz/wyz-workspace/dsamem-eurosys27/exps/20260919_012520_glm-w4a8-dptp/runs/20260919_012520_dp_tp_sweep
gate=${run_dir}/baseline1_${attempt_tag}_bm_init.ready
if [[ -e ${gate} ]]; then
    echo "refusing to reuse existing gate ${gate}" >&2
    exit 2
fi
npu-smi info
sync
touch "${gate}"
echo "Baseline1 decoder gate opened: ${gate}"
