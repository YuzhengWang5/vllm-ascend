from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.utils.mem_constants import GiB_bytes

from vllm_ascend.worker.worker import NPUWorker


def _worker():
    return SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        kv_cache_spec={"host-and-device-specs": object()},
    )


def _worker_with_host_layers(num_layers: int):
    return SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        kv_cache_spec={
            f"host-layer-{index}": SimpleNamespace(store_on_host=True)
            for index in range(num_layers)
        },
    )


def _config(*, pool_gib: int, cap_to_pool: bool, keep_device: bool = False):
    return SimpleNamespace(
        enabled=True,
        motivation_baseline="colocated",
        remote_indexer_enabled=False,
        keep_device_kv_cache=keep_device,
        dram_size_per_dp_GB=pool_gib,
        dram_limited_capacity=cap_to_pool,
    )


def _update(worker, config, available_gib: int, ratio: float):
    with (
        patch(
            "vllm_ascend.worker.worker.get_ascend_config",
            return_value=SimpleNamespace(sparse_kv_offload_config=config),
        ),
        patch(
            "vllm_ascend.worker.worker.get_host_device_memory_usage_ratio",
            return_value=ratio,
        ),
    ):
        return NPUWorker.update_available_memory_for_sparse_kv_offload(
            worker, available_gib * GiB_bytes
        )


def test_dram_limited_capacity_caps_device_and_host_components():
    result = _update(
        _worker(),
        _config(pool_gib=40, cap_to_pool=True),
        available_gib=14,
        ratio=8.0,
    )

    assert result == 45 * GiB_bytes


def test_dram_limited_capacity_reserves_per_layer_alignment_overhead():
    num_layers = 78
    alignment_overhead = num_layers * 3 * 2 * 1024 * 1024
    usable_host_bytes = 40 * GiB_bytes - alignment_overhead
    device_bytes = int(usable_host_bytes / 8.0)

    result = _update(
        _worker_with_host_layers(num_layers),
        _config(pool_gib=40, cap_to_pool=True),
        available_gib=14,
        ratio=8.0,
    )

    assert result == int(device_bytes + 8.0 * device_bytes)


def test_dram_limited_capacity_rejects_pool_smaller_than_alignment_reserve():
    with pytest.raises(ValueError, match="alignment reserve"):
        _update(
            _worker_with_host_layers(200),
            _config(pool_gib=1, cap_to_pool=True),
            available_gib=14,
            ratio=8.0,
        )


def test_dram_limited_capacity_is_opt_in():
    with pytest.raises(ValueError, match="Needed dram size"):
        _update(
            _worker(),
            _config(pool_gib=40, cap_to_pool=False),
            available_gib=14,
            ratio=8.0,
        )


def test_dram_limited_capacity_does_not_change_keep_device_mode():
    with pytest.raises(ValueError, match="Needed dram size"):
        _update(
            _worker(),
            _config(pool_gib=10, cap_to_pool=True, keep_device=True),
            available_gib=14,
            ratio=8.0,
        )


def test_large_enough_pool_keeps_original_capacity():
    result = _update(
        _worker(),
        _config(pool_gib=128, cap_to_pool=True),
        available_gib=14,
        ratio=8.0,
    )

    assert result == 126 * GiB_bytes
