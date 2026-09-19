#!/usr/bin/env bash
set -euo pipefail

attempt_tag=${1:?unique attempt tag}
host_run_dir=${GLM_HOST_RUN_DIR:?set GLM_HOST_RUN_DIR to the host run directory}
gate=${host_run_dir}/baseline1_${attempt_tag}_bm_init.ready
if [[ -e ${gate} ]]; then
    echo "refusing to reuse existing gate ${gate}" >&2
    exit 2
fi
npu-smi info
sync
touch "${gate}"
echo "Baseline1 decoder gate opened: ${gate}"
