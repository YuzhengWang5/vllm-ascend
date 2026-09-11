#!/usr/bin/env bash
set -euo pipefail

source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build_dir=${source_dir}/indexer_shm_transport_build
mf_root=${MF_HOME:-/usr/local/memfabric_hybrid/latest}

mkdir -p "${build_dir}"
bisheng -x asc "${source_dir}/indexer_shm_transport_kernel.cpp" \
    -shared -fPIC -O2 -o "${build_dir}/libindexer_shm_transport_kernel.so" \
    --cce-aicore-arch=dav-c220 \
    -I"${mf_root}/include/smem/device"

echo "${build_dir}/libindexer_shm_transport_kernel.so"
