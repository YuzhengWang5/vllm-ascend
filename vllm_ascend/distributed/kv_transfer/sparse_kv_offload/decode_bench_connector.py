# SPDX-License-Identifier: Apache-2.0
"""Decode-only synthetic prefill for the SFA CPU-offload path.

This connector lets one D-side engine benchmark sparse decode without a
Prefill engine. The scheduler marks all prompt tokens except the final one as
externally computed. Before the first decode forward, the worker fills the
corresponding main MLA blocks in the manager's CPU pool and the indexer blocks
in their normal HBM cache.

The generated values are intentionally synthetic. They make the memory layout
and sparse onload path valid, but model outputs and top-k locality are not
quality measurements.
"""

from __future__ import annotations

import re
import time
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnectorMetadata,
    DecodeBenchConnectorScheduler,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    get_sparse_kv_offload_manager,
)

logger = init_logger(__name__)
_LAYER_ID_RE = re.compile(r"layers\.(\d+)")


def _layer_id(layer_name: str) -> int:
    match = _LAYER_ID_RE.search(layer_name)
    if match is None:
        raise ValueError(f"Cannot infer layer id from KV cache name {layer_name!r}")
    return int(match.group(1))


class SFAOffloadDecodeBenchConnector(KVConnectorBase_V1, SupportsHMA):
    """Fill synthetic prefill state for a sparse-offload decode instance.

    Required configuration::

      --kv-transfer-config '{"kv_connector":"SFAOffloadDecodeBenchConnector",
          "kv_role":"kv_consumer"}'

    together with ``sparse_kv_offload_config.enabled=true`` and
    ``keep_device_kv_cache=false``. ``kv_role=kv_consumer`` selects the real
    D-side SFA execution path; no producer or network peer is created.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        if kv_cache_config is None:
            raise ValueError("SFAOffloadDecodeBenchConnector requires KVCacheConfig")
        super().__init__(vllm_config, role, kv_cache_config)

        init_ascend_config(vllm_config)
        sparse_config = get_ascend_config().sparse_kv_offload_config
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        if not kv_transfer_config.is_kv_consumer or kv_transfer_config.is_kv_producer:
            raise ValueError(
                "SFAOffloadDecodeBenchConnector requires kv_role=kv_consumer; "
                "it is a single-engine D-side benchmark connector."
            )
        if not sparse_config.enabled or sparse_config.keep_device_kv_cache:
            raise ValueError(
                "SFAOffloadDecodeBenchConnector requires sparse_kv_offload_config "
                "with enabled=true and keep_device_kv_cache=false."
            )

        self.connector_scheduler: DecodeBenchConnectorScheduler | None = None
        self.connector_worker: SFAOffloadDecodeBenchWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = DecodeBenchConnectorScheduler(vllm_config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = SFAOffloadDecodeBenchWorker(vllm_config, kv_cache_config)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, DecodeBenchConnectorMetadata):
            raise TypeError(f"Unexpected connector metadata: {type(metadata).__name__}")
        self.connector_worker.start_fill_kv(metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: Any) -> DecodeBenchConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request: Any, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        self.connector_scheduler.request_finished(request)
        return False, None

    def request_finished_all_groups(
        self,
        request: Any,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        self.connector_scheduler.request_finished(request)
        return False, None


class SFAOffloadDecodeBenchWorker:
    """Fill the two storage tiers owned by ``SparseKVOffloadManager``."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        extra = vllm_config.kv_transfer_config
        assert extra is not None
        self.main_fill_value = float(extra.get_from_extra_config("main_fill_value", 0.015))
        self.indexer_fill_value = int(extra.get_from_extra_config("indexer_fill_value", 1))
        self.indexer_scale_value = float(extra.get_from_extra_config("indexer_scale_value", 0.015))
        fill_std = float(extra.get_from_extra_config("fill_std", 0.0))
        if fill_std != 0.0:
            raise ValueError(
                "SFAOffloadDecodeBenchConnector only supports deterministic fills; "
                "set kv_connector_extra_config.fill_std to 0."
            )
        self.manager = None
        self.indexer_caches: list[tuple[torch.Tensor, ...] | None] = []

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.manager = get_sparse_kv_offload_manager()
        main_names = list(self.manager.offload_layer_names)
        if not main_names:
            raise RuntimeError("Sparse KV offload manager did not register main MLA layers")
        if self.manager.tp_rank == 0 and (
            len(self.manager.k_caches_cpu) != len(main_names)
            or len(self.manager.v_caches_cpu) != len(main_names)
        ):
            raise RuntimeError("Sparse KV offload CPU pool/layer count mismatch")

        indexer_by_layer = {
            _layer_id(name): cache
            for name, cache in kv_caches.items()
            if "indexer" in name.lower()
        }
        self.indexer_caches = []
        for main_name in main_names:
            cache = indexer_by_layer.get(_layer_id(main_name))
            if cache is None:
                self.indexer_caches.append(None)
            elif isinstance(cache, torch.Tensor):
                self.indexer_caches.append((cache,))
            else:
                self.indexer_caches.append(tuple(cache))

        logger.warning(
            "SFAOffloadDecodeBenchConnector registered %d main and %d indexer KV layers",
            len(main_names),
            sum(cache is not None for cache in self.indexer_caches),
        )

    def start_fill_kv(self, metadata: DecodeBenchConnectorMetadata) -> None:
        if not metadata.reqs_to_fill:
            return
        if self.manager is None:
            raise RuntimeError("KV caches must be registered before synthetic prefill")

        started_at = time.perf_counter()
        total_blocks = 0
        total_tokens = 0
        for request_id, (block_ids_per_group, _) in metadata.reqs_to_fill.items():
            if len(block_ids_per_group) != 1:
                raise RuntimeError(
                    "SFAOffloadDecodeBenchConnector expects one uniform SFA KV group, "
                    f"got {len(block_ids_per_group)} for request {request_id!r}."
                )
            block_ids = block_ids_per_group[0]
            total_blocks += len(block_ids)
            total_tokens += metadata.reqs_to_fill[request_id][1]
            self._fill_main_cpu_pool(block_ids)
            self._fill_indexer_hbm(block_ids)

        self.manager.prepare_motivation_no_gather_buffers(
            self.main_fill_value,
        )

        torch.npu.synchronize()
        self.manager.tp_group.barrier()
        logger.warning(
            "SFAOffloadDecodeBenchConnector filled synthetic KV: requests=%d "
            "tokens=%d blocks=%d elapsed_ms=%.3f",
            len(metadata.reqs_to_fill),
            total_tokens,
            total_blocks,
            (time.perf_counter() - started_at) * 1000,
        )

    def _fill_main_cpu_pool(self, block_ids: list[int]) -> None:
        assert self.manager is not None
        if self.manager.tp_rank != 0:
            return
        for k_cache, v_cache in zip(self.manager.k_caches_cpu, self.manager.v_caches_cpu):
            self._fill_block_rows(k_cache, block_ids, self.main_fill_value)
            self._fill_block_rows(v_cache, block_ids, self.main_fill_value)

    def _fill_indexer_hbm(self, block_ids: list[int]) -> None:
        for cache_tuple in self.indexer_caches:
            if cache_tuple is None:
                continue
            self._fill_block_rows(cache_tuple[0], block_ids, self.indexer_fill_value)
            for scale_cache in cache_tuple[1:]:
                self._fill_block_rows(scale_cache, block_ids, self.indexer_scale_value)

    def _fill_block_rows(self, tensor: torch.Tensor, block_ids: list[int], value: float | int) -> None:
        if not block_ids or tensor.ndim == 0:
            return
        assert self.manager is not None
        rows_per_block = max(tensor.shape[0] // self.manager.kv_cache_config.num_blocks, 1)
        row_ids = [
            block_id * rows_per_block + offset
            for block_id in block_ids
            for offset in range(rows_per_block)
            if 0 <= block_id * rows_per_block + offset < tensor.shape[0]
        ]
        if not row_ids:
            return
        indices = torch.tensor(sorted(set(row_ids)), dtype=torch.long, device=tensor.device)
        values = torch.full(
            (len(indices), *tensor.shape[1:]),
            value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        # Ascend IndexFill does not implement int8, which is the indexer KV
        # dtype. IndexCopy supports it and preserves the same row semantics.
        tensor.index_copy_(0, indices, values)
