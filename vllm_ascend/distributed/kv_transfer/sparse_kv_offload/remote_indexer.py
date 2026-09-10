"""Synchronous TCP client for a remote SFA indexer.

Each decoder rank owns one persistent connection to one remote NPU worker.
Control messages use the original inspectable torch framing, while the hot
select path uses a fixed-schema header followed by contiguous tensor bytes.
"""

from __future__ import annotations

import io
import socket
import struct
import threading
from collections.abc import Callable
from typing import Any

import torch
import torch_npu
from vllm.logger import logger


PROTOCOL_VERSION = 3
MAX_MESSAGE_BYTES = 1 << 30
_LENGTH = struct.Struct("!Q")
CONTROL_MESSAGE = b"C"
SELECT_MESSAGE = b"S"
_SELECT_HEADER = struct.Struct("!Qi11I")
_SELECT_RESPONSE_HEADER = struct.Struct("!BQ")
_SELECT_OK = 1
_SELECT_ERROR = 0
_SELECT_FLAG_CONTEXT = 1
SELECT_DYNAMIC_TENSORS = (
    "q",
    "q_scale",
    "weights",
    "new_k",
    "new_k_scale",
)
SELECT_CONTEXT_TENSORS = (
    "slot_mapping",
    "actual_seq_lengths_query",
    "actual_seq_lengths_key",
    "block_table",
)
_GRAPH_CLIENTS: dict[int, "RemoteIndexerClient"] = {}


def _run_graph_callback(rank: int, layer_id: int) -> None:
    """Entry point invoked by the replay-safe C++ ACL host callback."""
    client = _GRAPH_CLIENTS.get(rank)
    if client is None:
        raise RuntimeError(f"Remote indexer graph client is not registered: rank={rank}")
    # The destination pinned buffers are created while vLLM initializes under
    # inference mode.  ACL invokes this callback on a report thread which does
    # not inherit thread-local PyTorch grad/inference state, so explicitly
    # restore inference mode before copying the response into those buffers.
    with torch.inference_mode():
        client._select_graph_host(layer_id)


def send_framed(sock: socket.socket, value: Any) -> None:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    payload = buffer.getvalue()
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"Remote indexer message is too large: {len(payload)} bytes")
    sock.sendall(_LENGTH.pack(len(payload)))
    sock.sendall(payload)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray(size)
    view = memoryview(chunks)
    offset = 0
    while offset < size:
        received = sock.recv_into(view[offset:])
        if received == 0:
            raise ConnectionError("Remote indexer connection closed")
        offset += received
    return bytes(chunks)


def recv_framed(sock: socket.socket) -> Any:
    (size,) = _LENGTH.unpack(_recv_exact(sock, _LENGTH.size))
    if size > MAX_MESSAGE_BYTES:
        raise ValueError(f"Remote indexer message is too large: {size} bytes")
    return torch.load(io.BytesIO(_recv_exact(sock, size)), map_location="cpu", weights_only=True)


def _tensor_bytes(tensor: torch.Tensor) -> memoryview:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("Raw transport requires contiguous CPU tensors")
    return memoryview(tensor.numpy()).cast("B")


def _send_views(sock: socket.socket, values: list[bytes | memoryview]) -> None:
    views = [memoryview(value).cast("B") for value in values if len(value)]
    if not hasattr(sock, "sendmsg"):
        for value in views:
            sock.sendall(value)
        return
    while views:
        sent = sock.sendmsg(views)
        if sent == 0:
            raise ConnectionError("Remote indexer connection closed while sending")
        while views and sent >= len(views[0]):
            sent -= len(views.pop(0))
        if sent:
            views[0] = views[0][sent:]


def _recv_into_tensor(sock: socket.socket, tensor: torch.Tensor) -> None:
    view = _tensor_bytes(tensor)
    offset = 0
    while offset < len(view):
        received = sock.recv_into(view[offset:])
        if received == 0:
            raise ConnectionError("Remote indexer connection closed while receiving tensor")
        offset += received


def send_raw_select_request(
    sock: socket.socket,
    request_id: int,
    layer_id: int,
    tensors: dict[str, torch.Tensor],
    *,
    include_context: bool = True,
) -> None:
    q = tensors["q"]
    q_scale = tensors["q_scale"]
    weights = tensors["weights"]
    new_k = tensors["new_k"]
    new_k_scale = tensors["new_k_scale"]
    if q.ndim != 3 or q.dtype != torch.int8:
        raise ValueError(f"Raw select expects int8 q[T,H,D], got {q.shape}/{q.dtype}")
    tokens, heads, dim = q.shape
    if q_scale.shape != (tokens, heads) or q_scale.dtype != torch.float16:
        raise ValueError("Raw select expects FP16 q_scale[T,H]")
    if weights.shape != (tokens, heads) or weights.dtype != torch.float16:
        raise ValueError("Raw select expects FP16 weights[T,H]")
    if new_k.ndim != 2 or new_k.shape[1] != dim or new_k.dtype != torch.int8:
        raise ValueError("Raw select expects int8 new_k[N,D]")
    if new_k_scale.shape != (new_k.shape[0], 1) or new_k_scale.dtype != torch.float16:
        raise ValueError("Raw select expects FP16 new_k_scale[N,1]")
    if include_context:
        slot_mapping = tensors["slot_mapping"]
        query_lens = tensors["actual_seq_lengths_query"]
        key_lens = tensors["actual_seq_lengths_key"]
        block_table = tensors["block_table"]
        if slot_mapping.dtype not in {torch.int32, torch.int64}:
            raise ValueError("Raw select expects int32/int64 slot_mapping")
        if query_lens.dtype != torch.int32 or key_lens.dtype != torch.int32:
            raise ValueError("Raw select expects int32 sequence lengths")
        if block_table.ndim != 2 or block_table.dtype != torch.int32:
            raise ValueError("Raw select expects int32 block_table[B,M]")
        context_shape = (
            slot_mapping.numel(),
            query_lens.numel(),
            key_lens.numel(),
            block_table.shape[0],
            block_table.shape[1],
            slot_mapping.element_size(),
        )
    else:
        context_shape = (0, 0, 0, 0, 0, 0)
    header = _SELECT_HEADER.pack(
        request_id,
        layer_id,
        _SELECT_FLAG_CONTEXT if include_context else 0,
        tokens,
        heads,
        dim,
        new_k.shape[0],
        *context_shape,
    )
    tensor_names = SELECT_DYNAMIC_TENSORS
    if include_context:
        tensor_names += SELECT_CONTEXT_TENSORS
    _send_views(
        sock,
        [
            SELECT_MESSAGE,
            header,
            *(_tensor_bytes(tensors[name]) for name in tensor_names),
        ],
    )


def recv_raw_select_request(
    sock: socket.socket,
    buffer_getter: Callable[[str, tuple[int, ...], torch.dtype], torch.Tensor],
) -> dict[str, Any]:
    values = _SELECT_HEADER.unpack(_recv_exact(sock, _SELECT_HEADER.size))
    request_id, layer_id = values[:2]
    (
        flags,
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
    ) = values[2:]
    if flags & ~_SELECT_FLAG_CONTEXT:
        raise ValueError(f"Unsupported raw select flags: {flags:#x}")
    include_context = bool(flags & _SELECT_FLAG_CONTEXT)
    specs = [
        ("q", (tokens, heads, dim), torch.int8),
        ("q_scale", (tokens, heads), torch.float16),
        ("weights", (tokens, heads), torch.float16),
        ("new_k", (new_rows, dim), torch.int8),
        ("new_k_scale", (new_rows, 1), torch.float16),
    ]
    if include_context:
        slot_dtype = {4: torch.int32, 8: torch.int64}.get(slot_width)
        if slot_dtype is None:
            raise ValueError(
                f"Unsupported slot_mapping element size: {slot_width}"
            )
        specs.extend(
            [
                ("slot_mapping", (slot_count,), slot_dtype),
                ("actual_seq_lengths_query", (query_len_count,), torch.int32),
                ("actual_seq_lengths_key", (key_len_count,), torch.int32),
                ("block_table", (block_rows, block_cols), torch.int32),
            ]
        )
    tensors = {}
    total_bytes = 0
    for name, shape, dtype in specs:
        tensor = buffer_getter(name, shape, dtype)
        total_bytes += tensor.numel() * tensor.element_size()
        if total_bytes > MAX_MESSAGE_BYTES:
            raise ValueError("Raw select payload is too large")
        _recv_into_tensor(sock, tensor)
        tensors[name] = tensor
    return {
        "request_id": request_id,
        "layer_id": layer_id,
        "include_context": include_context,
        **tensors,
    }


def send_raw_select_response(
    sock: socket.socket,
    request_id: int,
    topk: torch.Tensor,
) -> None:
    if topk.dtype != torch.int32 or topk.device.type != "cpu":
        raise ValueError("Raw select response requires a CPU int32 tensor")
    _send_views(
        sock,
        [_SELECT_RESPONSE_HEADER.pack(_SELECT_OK, request_id), _tensor_bytes(topk)],
    )


def send_raw_select_error(sock: socket.socket, request_id: int, error: str) -> None:
    sock.sendall(_SELECT_RESPONSE_HEADER.pack(_SELECT_ERROR, request_id))
    send_framed(sock, {"error": error})


def recv_raw_select_response(
    sock: socket.socket,
    request_id: int,
    output: torch.Tensor,
) -> None:
    status, response_id = _SELECT_RESPONSE_HEADER.unpack(
        _recv_exact(sock, _SELECT_RESPONSE_HEADER.size)
    )
    if response_id != request_id:
        raise RuntimeError(
            f"Remote indexer response id mismatch: expected={request_id}, got={response_id}"
        )
    if status != _SELECT_OK:
        response = recv_framed(sock)
        raise RuntimeError(f"Remote indexer request failed: {response!r}")
    _recv_into_tensor(sock, output)


class RemoteIndexerClient:
    """One rank-local synchronous remote-indexer connection."""

    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        topk: int,
        connect_timeout_s: float,
    ) -> None:
        self.host = host
        self.port = port
        self.rank = rank
        self.topk = topk
        self.connect_timeout_s = connect_timeout_s
        self._socket: socket.socket | None = None
        self._buffers: dict[tuple[str, tuple[int, ...], torch.dtype], torch.Tensor] = {}
        self._lock = threading.Lock()
        self._request_id = 0
        self._enqueue_graph_callback = None
        self._graph_staged: dict[str, torch.Tensor] | None = None
        self._graph_output_cpu: torch.Tensor | None = None
        _GRAPH_CLIENTS[self.rank] = self

    def set_graph_callback_enqueuer(self, enqueuer: Any) -> None:
        self._enqueue_graph_callback = enqueuer

    def reset_cache(self) -> None:
        """Restore the remote cache after graph capture's dummy execution."""
        with self._lock:
            sock = self._connect()
            sock.sendall(CONTROL_MESSAGE)
            send_framed(
                sock,
                {
                    "op": "reset_cache",
                    "version": PROTOCOL_VERSION,
                    "rank": self.rank,
                },
            )
            response = recv_framed(sock)
            if response != {"ok": True, "op": "reset_cache"}:
                raise RuntimeError(
                    "Remote indexer cache reset failed: "
                    f"rank={self.rank}, response={response!r}"
                )

    def fill_blocks(self, block_ids: list[int]) -> None:
        """Initialize newly allocated/reused cache blocks like the connector."""
        if not block_ids:
            return
        with self._lock:
            sock = self._connect()
            sock.sendall(CONTROL_MESSAGE)
            send_framed(
                sock,
                {
                    "op": "fill_blocks",
                    "version": PROTOCOL_VERSION,
                    "rank": self.rank,
                    "block_ids": sorted(set(int(block_id) for block_id in block_ids)),
                },
            )
            response = recv_framed(sock)
            if response != {"ok": True, "op": "fill_blocks"}:
                raise RuntimeError(
                    "Remote indexer block fill failed: "
                    f"rank={self.rank}, response={response!r}"
                )

    def _connect(self) -> socket.socket:
        if self._socket is not None:
            return self._socket
        sock = socket.create_connection(
            (self.host, self.port),
            timeout=self.connect_timeout_s,
        )
        sock.settimeout(None)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_framed(
            sock,
            {
                "op": "hello",
                "version": PROTOCOL_VERSION,
                "rank": self.rank,
            },
        )
        response = recv_framed(sock)
        if response != {
            "ok": True,
            "version": PROTOCOL_VERSION,
            "select_transport": "raw",
        }:
            sock.close()
            raise RuntimeError(f"Remote indexer handshake failed: {response!r}")
        self._socket = sock
        logger.warning(
            "Remote indexer rank %d connected to %s:%d",
            self.rank,
            self.host,
            self.port,
        )
        return sock

    def _cpu_buffer(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        key = (name, tuple(tensor.shape), tensor.dtype)
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=True,
            )
            self._buffers[key] = buffer
        return buffer

    def _stage(
        self,
        name: str,
        tensor: torch.Tensor,
        capturing: bool,
    ) -> torch.Tensor:
        buffer = self._cpu_buffer(name, tensor)
        buffer.copy_(tensor, non_blocking=capturing)
        return buffer

    def select(
        self,
        *,
        layer_id: int,
        q: torch.Tensor,
        q_scale: torch.Tensor,
        weights: torch.Tensor,
        new_k: torch.Tensor,
        new_k_scale: torch.Tensor,
        slot_mapping: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_table: torch.Tensor,
        capturing: bool,
    ) -> torch.Tensor:
        if q.dtype != torch.int8 or new_k.dtype != torch.int8:
            raise ValueError("Remote indexer v0 requires the C8 indexer path")
        staged = {
            "q": self._stage("q", q, capturing),
            "q_scale": self._stage("q_scale", q_scale, capturing),
            # Match BaseDeviceAdaptor: A3's QuantLightningIndexer ABI accepts
            # FP16 weights even though the surrounding model runs in BF16.
            "weights": self._stage("weights", weights.to(torch.float16), capturing),
            "new_k": self._stage("new_k", new_k, capturing),
            "new_k_scale": self._stage("new_k_scale", new_k_scale, capturing),
            "slot_mapping": self._stage("slot_mapping", slot_mapping, capturing),
            "actual_seq_lengths_query": self._stage(
                "actual_seq_lengths_query",
                actual_seq_lengths_query,
                capturing,
            ),
            "actual_seq_lengths_key": self._stage(
                "actual_seq_lengths_key",
                actual_seq_lengths_key,
                capturing,
            ),
            "block_table": self._stage("block_table", block_table, capturing),
        }
        output_shape = (q.shape[0], 1, self.topk)
        output_template = torch.empty(output_shape, dtype=torch.int32, device="cpu")
        output_cpu = self._cpu_buffer("topk", output_template)
        output_npu = torch.empty(output_shape, dtype=torch.int32, device=q.device)
        args = (layer_id, staged, output_cpu)

        if capturing:
            if self._enqueue_graph_callback is None:
                raise RuntimeError("Remote indexer graph callback is not initialized")
            self._graph_staged = staged
            self._graph_output_cpu = output_cpu
            self._enqueue_graph_callback(self.rank, layer_id)
        else:
            torch_npu.npu.synchronize(q.device)
            self._select_host(args)

        output_npu.copy_(output_cpu, non_blocking=capturing)
        return output_npu

    def _select_graph_host(self, layer_id: int) -> None:
        if self._graph_staged is None or self._graph_output_cpu is None:
            raise RuntimeError("Remote indexer graph buffers are not initialized")
        self._select_host((layer_id, self._graph_staged, self._graph_output_cpu))

    def _select_host(self, args: tuple[int, dict[str, torch.Tensor], torch.Tensor]) -> None:
        layer_id, staged, output_cpu = args
        with self._lock:
            sock = self._connect()
            request_id = self._request_id
            self._request_id += 1
            send_raw_select_request(
                sock,
                request_id,
                layer_id,
                staged,
                include_context=layer_id == 0,
            )
            recv_raw_select_response(sock, request_id, output_cpu)
