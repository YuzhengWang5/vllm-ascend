"""Fixed-slot MemFabric mailbox for the remote SFA indexer hot path.

The TCP connection remains as the bootstrap and control channel.  Request and
response payloads use one-sided writes into rank-local host-backed BigMemory;
the payload is published by writing a sequence-number header last.
"""

from __future__ import annotations

import ctypes
import struct
import time
from collections.abc import Callable
from typing import Any

import torch

MAILBOX_POOL_BYTES = 2 << 20
MAILBOX_POOL_ID = 23
REQUEST_HEADER_OFFSET = 0
RESPONSE_HEADER_OFFSET = 64
PAYLOAD_OFFSET = 4096
_REQUEST_HEADER = struct.Struct("<Qi10I")
_RESPONSE_HEADER = struct.Struct("<QQ")
_REQUEST_NAMES = (
    "q",
    "q_scale",
    "weights",
    "new_k",
    "new_k_scale",
    "slot_mapping",
    "actual_seq_lengths_query",
    "actual_seq_lengths_key",
    "block_table",
)


def _request_specs(values: tuple[int, ...]) -> tuple[int, list[tuple[str, tuple[int, ...], torch.dtype, int]]]:
    (
        tokens,
        heads,
        dim,
        new_rows,
        slot_count,
        query_len_count,
        key_len_count,
        block_rows,
        block_cols,
        slot_width,
    ) = values
    slot_dtype = {4: torch.int32, 8: torch.int64}.get(slot_width)
    if slot_dtype is None:
        raise ValueError(f"Unsupported slot_mapping element size: {slot_width}")
    raw_specs = (
        ("q", (tokens, heads, dim), torch.int8),
        ("q_scale", (tokens, heads), torch.float16),
        ("weights", (tokens, heads), torch.float16),
        ("new_k", (new_rows, dim), torch.int8),
        ("new_k_scale", (new_rows, 1), torch.float16),
        ("slot_mapping", (slot_count,), slot_dtype),
        ("actual_seq_lengths_query", (query_len_count,), torch.int32),
        ("actual_seq_lengths_key", (key_len_count,), torch.int32),
        ("block_table", (block_rows, block_cols), torch.int32),
    )
    total = 0
    specs = []
    for name, shape, dtype in raw_specs:
        elements = 1
        for dimension in shape:
            elements *= dimension
        size = elements * torch.empty((), dtype=dtype).element_size()
        specs.append((name, shape, dtype, size))
        total += size
    if PAYLOAD_OFFSET + total > MAILBOX_POOL_BYTES:
        raise ValueError(f"MemFabric request payload is too large: {total} bytes")
    return total, specs


def request_header(sequence: int, layer_id: int, tensors: dict[str, torch.Tensor]) -> bytes:
    q = tensors["q"]
    new_k = tensors["new_k"]
    slot_mapping = tensors["slot_mapping"]
    query_lens = tensors["actual_seq_lengths_query"]
    key_lens = tensors["actual_seq_lengths_key"]
    block_table = tensors["block_table"]
    if q.ndim != 3 or q.dtype != torch.int8:
        raise ValueError("MemFabric mailbox expects int8 q[T,H,D]")
    tokens, heads, dim = q.shape
    values = (
        tokens,
        heads,
        dim,
        new_k.shape[0],
        slot_mapping.numel(),
        query_lens.numel(),
        key_lens.numel(),
        block_table.shape[0],
        block_table.shape[1],
        slot_mapping.element_size(),
    )
    _request_specs(values)
    expected = {
        "q_scale": ((tokens, heads), torch.float16),
        "weights": ((tokens, heads), torch.float16),
        "new_k": ((new_k.shape[0], dim), torch.int8),
        "new_k_scale": ((new_k.shape[0], 1), torch.float16),
        "actual_seq_lengths_query": ((query_lens.numel(),), torch.int32),
        "actual_seq_lengths_key": ((key_lens.numel(),), torch.int32),
        "block_table": ((block_table.shape[0], block_table.shape[1]), torch.int32),
    }
    for name, (shape, dtype) in expected.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"Unexpected {name}: {tuple(tensor.shape)}/{tensor.dtype}; expected {shape}/{dtype}")
    if slot_mapping.dtype not in {torch.int32, torch.int64}:
        raise ValueError("MemFabric mailbox expects int32/int64 slot_mapping")
    return _REQUEST_HEADER.pack(sequence, layer_id, *values)


def _copy_tensor_bytes(destination: int, source: torch.Tensor, size: int) -> None:
    if source.device.type != "cpu" or not source.is_contiguous():
        raise ValueError("MemFabric mailbox requires contiguous CPU tensors")
    ctypes.memmove(destination, source.data_ptr(), size)


class _MailboxBase:
    def __init__(
        self,
        *,
        bm_rank: int,
        logical_rank: int,
        store_url: str,
        device: int,
        timeout_s: float,
        defer_join: bool = False,
        defer_create: bool = False,
    ) -> None:
        import memfabric_hybrid as mf
        from memfabric_hybrid import bm

        self.mf = mf
        self.bm = bm
        self.timeout_s = timeout_s
        mf.set_log_level(2)
        rc = mf.initialize()
        if rc != 0:
            raise RuntimeError(f"mf.initialize failed: rc={rc}: {mf.get_last_err_msg()}")
        config = bm.BmConfig()
        config.rank_id = bm_rank
        config.auto_ranking = False
        config.start_store = bm_rank == 0
        timeout = max(1, int(timeout_s))
        config.init_timeout = timeout
        config.create_timeout = timeout
        config.operation_timeout = timeout
        config.set_nic(f"tcp://127.0.0.1:{30000 + logical_rank}")
        rc = bm.initialize(store_url, 2, device, config)
        if rc != 0:
            raise RuntimeError(f"bm.initialize failed: rc={rc}: {mf.get_last_err_msg()}")
        self.bm_rank = bm_rank
        self.handle = None
        self._joined = False
        self._header = torch.empty(_REQUEST_HEADER.size, dtype=torch.uint8)
        self._response_header = torch.empty(_RESPONSE_HEADER.size, dtype=torch.uint8)
        if not defer_create:
            self.create(join=not defer_join)

    def create(self, *, join: bool = True) -> None:
        if self.handle is None:
            self.handle = self.bm.create2(
                id=MAILBOX_POOL_ID,
                local_dram_size=MAILBOX_POOL_BYTES,
                max_dram_size=MAILBOX_POOL_BYTES,
                data_op_type=self.bm.BmDataOpType.DEVICE_RDMA,
            )
        if join:
            self.join()

    def join(self) -> None:
        if self._joined:
            return
        if self.handle is None:
            raise RuntimeError("MemFabric mailbox must be created before join")
        rc = self.handle.join()
        if rc != 0:
            raise RuntimeError(f"BigMemory.join failed: rc={rc}: {self.mf.get_last_err_msg()}")
        self.local_gva = self.handle.peer_rank_ptr(self.bm_rank, self.bm.BmMemType.HOST)
        self.peer_gva = self.handle.peer_rank_ptr(1 - self.bm_rank, self.bm.BmMemType.HOST)
        self.local_va = self.handle.gva_to_va(self.local_gva, self.bm.BmMemType.LOCAL_HOST)
        if not self.local_gva or not self.peer_gva or not self.local_va:
            raise RuntimeError("MemFabric mailbox returned an invalid local/peer address")
        ctypes.memset(self.local_va, 0, MAILBOX_POOL_BYTES)
        self._joined = True

    def _remote_write(self, source: int, destination: int, size: int) -> None:
        rc = self.handle.copy_data(
            source,
            destination,
            size,
            self.bm.BmCopyType.H2G,
            0,
        )
        if rc != 0:
            raise RuntimeError(f"MemFabric mailbox write failed: rc={rc}: {self.mf.get_last_err_msg()}")

    def _publish_request_header(self, header: bytes) -> None:
        ctypes.memmove(self._header.data_ptr(), header, len(header))
        self._remote_write(
            self._header.data_ptr(),
            self.peer_gva + REQUEST_HEADER_OFFSET,
            len(header),
        )

    def _publish_response(self, sequence: int, ok: bool) -> None:
        header = _RESPONSE_HEADER.pack(sequence, int(ok))
        ctypes.memmove(self._response_header.data_ptr(), header, len(header))
        self._remote_write(
            self._response_header.data_ptr(),
            self.peer_gva + RESPONSE_HEADER_OFFSET,
            len(header),
        )


class MemfabricMailboxClient(_MailboxBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(bm_rank=0, **kwargs)
        # HYBM classifies torch pinned allocations as device-visible memory;
        # H2G expects an ordinary local-host source address.
        self._packed = torch.empty(1, dtype=torch.uint8)

    def select(
        self,
        sequence: int,
        layer_id: int,
        tensors: dict[str, torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        header = request_header(sequence, layer_id, tensors)
        values = _REQUEST_HEADER.unpack(header)[2:]
        total, specs = _request_specs(values)
        if self._packed.numel() < total:
            self._packed = torch.empty(total, dtype=torch.uint8)
        offset = 0
        for name, _shape, _dtype, size in specs:
            _copy_tensor_bytes(self._packed.data_ptr() + offset, tensors[name], size)
            offset += size
        self._remote_write(self._packed.data_ptr(), self.peer_gva + PAYLOAD_OFFSET, total)
        self._publish_request_header(header)

        response_sequence = ctypes.c_uint64.from_address(self.local_va + RESPONSE_HEADER_OFFSET)
        deadline = time.monotonic() + self.timeout_s
        while response_sequence.value != sequence:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"MemFabric response timeout: sequence={sequence}, observed={response_sequence.value}"
                )
        _, ok = _RESPONSE_HEADER.unpack_from(
            (ctypes.c_ubyte * _RESPONSE_HEADER.size).from_address(self.local_va + RESPONSE_HEADER_OFFSET)
        )
        if not ok:
            raise RuntimeError(f"Remote MemFabric indexer failed: sequence={sequence}")
        output_bytes = output.numel() * output.element_size()
        if PAYLOAD_OFFSET + output_bytes > MAILBOX_POOL_BYTES:
            raise ValueError("MemFabric response payload exceeds mailbox pool")
        ctypes.memmove(output.data_ptr(), self.local_va + PAYLOAD_OFFSET, output_bytes)


class MemfabricMailboxServer(_MailboxBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(bm_rank=1, **kwargs)
        self._last_sequence = 0
        self._response_payload = torch.empty(1, dtype=torch.uint8)

    def try_receive(
        self,
        buffer_getter: Callable[[str, tuple[int, ...], torch.dtype], torch.Tensor],
    ) -> dict[str, Any] | None:
        sequence = ctypes.c_uint64.from_address(self.local_va + REQUEST_HEADER_OFFSET).value
        if sequence == self._last_sequence:
            return None
        raw_header = (ctypes.c_ubyte * _REQUEST_HEADER.size).from_address(self.local_va + REQUEST_HEADER_OFFSET)
        values = _REQUEST_HEADER.unpack_from(raw_header)
        request_id, layer_id = values[:2]
        if request_id != sequence:
            return None
        total, specs = _request_specs(values[2:])
        tensors = {}
        offset = 0
        for name, shape, dtype, size in specs:
            tensor = buffer_getter(name, shape, dtype)
            ctypes.memmove(tensor.data_ptr(), self.local_va + PAYLOAD_OFFSET + offset, size)
            tensors[name] = tensor
            offset += size
        assert offset == total
        self._last_sequence = sequence
        return {
            "request_id": request_id,
            "layer_id": layer_id,
            **tensors,
        }

    def respond(self, sequence: int, output: torch.Tensor) -> None:
        size = output.numel() * output.element_size()
        if PAYLOAD_OFFSET + size > MAILBOX_POOL_BYTES:
            raise ValueError("MemFabric response payload exceeds mailbox pool")
        if self._response_payload.numel() < size:
            self._response_payload = torch.empty(size, dtype=torch.uint8)
        ctypes.memmove(self._response_payload.data_ptr(), output.data_ptr(), size)
        self._remote_write(
            self._response_payload.data_ptr(),
            self.peer_gva + PAYLOAD_OFFSET,
            size,
        )
        self._publish_response(sequence, True)

    def respond_error(self, sequence: int) -> None:
        self._publish_response(sequence, False)
