"""Tests for the local synthetic-prefill SFA decode connector."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.distributed.kv_transfer.kv_connector.factory import (  # noqa: E402
    KVConnectorFactory,
)
from vllm_ascend.distributed.kv_transfer import register_connector  # noqa: E402
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.decode_bench_connector import (  # noqa: E402
    SFAOffloadDecodeBenchWorker,
    _layer_id,
)


def test_sfa_offload_decode_bench_connector_is_registered():
    with (
        patch.object(KVConnectorFactory, "_registry", {}),
        patch.object(KVConnectorFactory, "register_connector") as mock_register,
    ):
        register_connector()

    mock_register.assert_any_call(
        "SFAOffloadDecodeBenchConnector",
        "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.decode_bench_connector",
        "SFAOffloadDecodeBenchConnector",
    )


def test_layer_id_requires_a_vllm_layer_name():
    assert _layer_id("model.layers.60.self_attn") == 60
    with pytest.raises(ValueError, match="Cannot infer layer id"):
        _layer_id("indexer-cache")


def test_fill_block_rows_supports_int8_indexer_cache():
    worker = SFAOffloadDecodeBenchWorker.__new__(SFAOffloadDecodeBenchWorker)
    worker.manager = SimpleNamespace(kv_cache_config=SimpleNamespace(num_blocks=4))
    cache = torch.zeros((8, 2), dtype=torch.int8)

    worker._fill_block_rows(cache, [1, 3], 7)

    assert torch.equal(cache[0], torch.zeros(2, dtype=torch.int8))
    assert torch.equal(cache[2], torch.full((2,), 7, dtype=torch.int8))
    assert torch.equal(cache[3], torch.full((2,), 7, dtype=torch.int8))
    assert torch.equal(cache[6], torch.full((2,), 7, dtype=torch.int8))
    assert torch.equal(cache[7], torch.full((2,), 7, dtype=torch.int8))
