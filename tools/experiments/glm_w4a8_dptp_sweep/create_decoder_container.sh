#!/usr/bin/env bash
set -euo pipefail

container=glm_w4a8_sweep_decoder_claude
image=sha256:1a9d274a0f509583fb8b020aad8c5ba53fa045e5162ecb3146c18452d30b6b8d
workspace=/root/wyz/wyz-workspace/dsamem-eurosys27
devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
device_args=()
for device in {0..15}; do
    device_args+=(--device "/dev/davinci${device}")
done

npu-smi info
export ASCEND_RT_VISIBLE_DEVICES="${devices}"
docker run -dit --rm --name "${container}" --network host --ipc host \
    --shm-size=1g --privileged \
    --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
    "${device_args[@]}" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "${workspace}:/workspace" \
    -v "${workspace}:${workspace}" \
    -e ASCEND_RT_VISIBLE_DEVICES="${devices}" \
    "${image}" sleep infinity
