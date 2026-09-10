"""Fixed-slot shared-memory mailbox between a decoder and its local relay."""

from __future__ import annotations

import ctypes
import mmap
import os
import struct
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from .memfabric_mailbox import (
    _REQUEST_HEADER,
    _copy_tensor_bytes,
    _request_specs,
    request_header,
)

SHM_BYTES = 4 << 20
REQUEST_HEADER_OFFSET = 0
RESPONSE_HEADER_OFFSET = 64
REQUEST_PAYLOAD_OFFSET = 4096
RESPONSE_PAYLOAD_OFFSET = 2 << 20
_RESPONSE_HEADER = struct.Struct("<QQ")


def default_shm_path(rank: int) -> str:
    return f"/dev/shm/vllm_ascend_indexer_rank{rank}.mailbox"


class _LocalShmMailbox:
    def __init__(self, path: str, *, create: bool, timeout_s: float) -> None:
        self.path = path
        self.timeout_s = timeout_s
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        self._fd = os.open(path, flags, 0o600)
        try:
            if create:
                os.ftruncate(self._fd, SHM_BYTES)
            elif os.fstat(self._fd).st_size != SHM_BYTES:
                raise RuntimeError(f"Invalid shared-memory mailbox size: {path}")
            self._mapping = mmap.mmap(
                self._fd,
                SHM_BYTES,
                access=mmap.ACCESS_WRITE,
            )
        except Exception:
            os.close(self._fd)
            raise
        self._address = ctypes.addressof(ctypes.c_ubyte.from_buffer(self._mapping))
        if create:
            ctypes.memset(self._address, 0, SHM_BYTES)

    def close(self, *, unlink: bool = False) -> None:
        self._mapping.close()
        os.close(self._fd)
        if unlink:
            Path(self.path).unlink(missing_ok=True)

    @staticmethod
    def _publish_sequence(address: int, sequence: int) -> None:
        ctypes.c_uint64.from_address(address).value = sequence


class LocalShmMailboxClient(_LocalShmMailbox):
    def __init__(self, path: str, *, timeout_s: float) -> None:
        super().__init__(path, create=False, timeout_s=timeout_s)

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
        if REQUEST_PAYLOAD_OFFSET + total > RESPONSE_PAYLOAD_OFFSET:
            raise ValueError("Shared-memory request exceeds its mailbox slot")
        offset = 0
        for name, _shape, _dtype, size in specs:
            _copy_tensor_bytes(
                self._address + REQUEST_PAYLOAD_OFFSET + offset,
                tensors[name],
                size,
            )
            offset += size

        # Publish the sequence last so the relay never observes a partially
        # written shape header or payload.
        ctypes.memmove(
            self._address + REQUEST_HEADER_OFFSET + 8,
            header[8:],
            len(header) - 8,
        )
        self._publish_sequence(
            self._address + REQUEST_HEADER_OFFSET,
            sequence,
        )

        response_sequence = ctypes.c_uint64.from_address(self._address + RESPONSE_HEADER_OFFSET)
        deadline = time.monotonic() + self.timeout_s
        while response_sequence.value != sequence:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Local relay response timeout: sequence={sequence}, observed={response_sequence.value}"
                )
        _, ok = _RESPONSE_HEADER.unpack_from(
            (ctypes.c_ubyte * _RESPONSE_HEADER.size).from_address(self._address + RESPONSE_HEADER_OFFSET)
        )
        if not ok:
            raise RuntimeError(f"Local relay failed: sequence={sequence}")
        output_bytes = output.numel() * output.element_size()
        if RESPONSE_PAYLOAD_OFFSET + output_bytes > SHM_BYTES:
            raise ValueError("Shared-memory response exceeds its mailbox slot")
        ctypes.memmove(
            output.data_ptr(),
            self._address + RESPONSE_PAYLOAD_OFFSET,
            output_bytes,
        )


class LocalShmMailboxServer(_LocalShmMailbox):
    def __init__(self, path: str, *, timeout_s: float) -> None:
        super().__init__(path, create=True, timeout_s=timeout_s)
        self._last_sequence = 0

    def try_receive(
        self,
        buffer_getter: Callable[[str, tuple[int, ...], torch.dtype], torch.Tensor],
    ) -> dict[str, Any] | None:
        sequence = ctypes.c_uint64.from_address(self._address + REQUEST_HEADER_OFFSET).value
        if sequence == self._last_sequence:
            return None
        raw_header = (ctypes.c_ubyte * _REQUEST_HEADER.size).from_address(self._address + REQUEST_HEADER_OFFSET)
        values = _REQUEST_HEADER.unpack_from(raw_header)
        request_id, layer_id = values[:2]
        if request_id != sequence:
            return None
        total, specs = _request_specs(values[2:])
        if REQUEST_PAYLOAD_OFFSET + total > RESPONSE_PAYLOAD_OFFSET:
            raise ValueError("Shared-memory request exceeds its mailbox slot")
        tensors = {}
        offset = 0
        for name, shape, dtype, size in specs:
            tensor = buffer_getter(name, shape, dtype)
            ctypes.memmove(
                tensor.data_ptr(),
                self._address + REQUEST_PAYLOAD_OFFSET + offset,
                size,
            )
            tensors[name] = tensor
            offset += size
        self._last_sequence = sequence
        return {
            "request_id": request_id,
            "layer_id": layer_id,
            **tensors,
        }

    def try_receive_packed(self) -> dict[str, Any] | None:
        """Return raw request addresses for a zero-copy relay hot path."""
        sequence = ctypes.c_uint64.from_address(self._address + REQUEST_HEADER_OFFSET).value
        if sequence == self._last_sequence:
            return None
        raw_header = (ctypes.c_ubyte * _REQUEST_HEADER.size).from_address(self._address + REQUEST_HEADER_OFFSET)
        header = bytes(raw_header)
        values = _REQUEST_HEADER.unpack(header)
        request_id, layer_id = values[:2]
        if request_id != sequence:
            return None
        total, _ = _request_specs(values[2:])
        if REQUEST_PAYLOAD_OFFSET + total > RESPONSE_PAYLOAD_OFFSET:
            raise ValueError("Shared-memory request exceeds its mailbox slot")
        self._last_sequence = sequence
        return {
            "request_id": request_id,
            "layer_id": layer_id,
            "tokens": values[2],
            "header": header,
            "payload_address": self._address + REQUEST_PAYLOAD_OFFSET,
            "payload_bytes": total,
        }

    @property
    def response_payload_address(self) -> int:
        return self._address + RESPONSE_PAYLOAD_OFFSET

    def publish_response(self, sequence: int, ok: bool = True) -> None:
        header = _RESPONSE_HEADER.pack(sequence, int(ok))
        ctypes.memmove(
            self._address + RESPONSE_HEADER_OFFSET + 8,
            header[8:],
            len(header) - 8,
        )
        self._publish_sequence(
            self._address + RESPONSE_HEADER_OFFSET,
            sequence,
        )

    def respond(self, sequence: int, output: torch.Tensor) -> None:
        size = output.numel() * output.element_size()
        if RESPONSE_PAYLOAD_OFFSET + size > SHM_BYTES:
            raise ValueError("Shared-memory response exceeds its mailbox slot")
        ctypes.memmove(
            self._address + RESPONSE_PAYLOAD_OFFSET,
            output.data_ptr(),
            size,
        )
        self.publish_response(sequence)

    def respond_error(self, sequence: int) -> None:
        self.publish_response(sequence, ok=False)
