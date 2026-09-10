"""Trivial synchronous TCP client for a remote SFA indexer.

The v0 transport intentionally favors a small, inspectable implementation over
performance.  Each decoder rank owns one persistent connection to one remote
NPU worker.  Device tensors are staged through pinned host memory, serialized
with ``torch.save``, and the returned top-k ids are copied back to the decoder
NPU.  Later experiments can replace this transport without changing the SFA
selection boundary.
"""

from __future__ import annotations

import io
import socket
import struct
import threading
from typing import Any

import torch
import torch_npu
from vllm.logger import logger


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 1 << 30
_LENGTH = struct.Struct("!Q")
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
        if response != {"ok": True, "version": PROTOCOL_VERSION}:
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
            request = {
                "op": "select",
                "version": PROTOCOL_VERSION,
                "rank": self.rank,
                "request_id": request_id,
                "layer_id": layer_id,
                **staged,
            }
            send_framed(sock, request)
            response = recv_framed(sock)
            if not response.get("ok"):
                raise RuntimeError(
                    "Remote indexer request failed: "
                    f"rank={self.rank}, layer={layer_id}, response={response!r}"
                )
            if response.get("request_id") != request_id:
                raise RuntimeError(
                    "Remote indexer response id mismatch: "
                    f"expected={request_id}, got={response.get('request_id')}"
                )
            topk = response["topk"]
            if topk.shape != output_cpu.shape or topk.dtype != output_cpu.dtype:
                raise RuntimeError(
                    "Remote indexer output mismatch: "
                    f"expected={output_cpu.shape}/{output_cpu.dtype}, "
                    f"got={topk.shape}/{topk.dtype}"
                )
            output_cpu.copy_(topk)
