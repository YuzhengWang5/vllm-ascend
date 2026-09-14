#!/usr/bin/env bash
set -euo pipefail

source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build_dir=${source_dir}/indexer_shm_transport_build
mf_root=${MF_HOME:-/usr/local/memfabric_hybrid/latest}
mf_device_include=${mf_root}/include/smem/device
if [[ ! -f "${mf_device_include}/smem_shm_aicore_base_api.h" ]]; then
    mf_device_include=${mf_root}/include/device
fi
if [[ ! -f "${mf_device_include}/smem_shm_aicore_base_api.h" ]]; then
    echo "MemFabric device headers not found under ${mf_root}" >&2
    exit 1
fi

mkdir -p "${build_dir}"
bisheng -x asc "${source_dir}/indexer_shm_transport_kernel.cpp" \
    -shared -fPIC -O2 -o "${build_dir}/libindexer_shm_transport_kernel.so" \
    --cce-aicore-arch=dav-c220 \
    -I"${mf_device_include}"

echo "${build_dir}/libindexer_shm_transport_kernel.so"
