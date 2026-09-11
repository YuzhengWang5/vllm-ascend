"""Device-only MemFabric SHM transport for the remote SFA indexer prototype."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import memfabric_hybrid as mf
import torch
import torch_npu
from memfabric_hybrid import shm
from torch.utils.cpp_extension import load
from vllm.logger import logger

ALIGNMENT = 32
LOCAL_MEMORY_BYTES = 4 << 20
MAX_PAYLOAD_BYTES = 1 << 20


def align32(value: int) -> int:
    return (value + ALIGNMENT - 1) & -ALIGNMENT


def _load_extension():
    source_dir = Path(__file__).resolve().parent
    build_dir = source_dir / "indexer_shm_transport_build"
    kernel = build_dir / "libindexer_shm_transport_kernel.so"
    if not kernel.exists():
        subprocess.run(
            ["bash", str(source_dir / "build_indexer_shm_transport.sh")],
            check=True,
        )
    ascend_home = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    torch_npu_root = Path(torch_npu.__file__).resolve().parent
    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"
    return load(
        name="indexer_shm_transport",
        sources=[str(source_dir / "indexer_shm_transport.cpp")],
        extra_cflags=[
            "-O3",
            "-std=c++20",
            "-fPIC",
            f"-I{ascend_home / 'include'}",
            f"-I{torch_npu_root / 'include'}",
        ],
        extra_ldflags=[
            f"-L{ascend_home / 'lib64'}",
            "-lascendcl",
            f"-L{torch_npu_root / 'lib'}",
            "-ltorch_npu",
            f"-L{build_dir}",
            "-lindexer_shm_transport_kernel",
            f"-Wl,-rpath,{build_dir}",
        ],
        verbose=False,
    )


@dataclass(frozen=True)
class TensorLayout:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    num_bytes: int


class PackedTensors:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        if not tensors:
            raise ValueError("PackedTensors requires at least one tensor")
        layouts: list[TensorLayout] = []
        offset = 0
        for name, tensor in tensors.items():
            if tensor.device.type != "npu" or not tensor.is_contiguous():
                raise ValueError(f"{name} must be a contiguous NPU tensor")
            num_bytes = tensor.numel() * tensor.element_size()
            layouts.append(TensorLayout(name, tuple(tensor.shape), tensor.dtype, offset, num_bytes))
            offset += align32(num_bytes)
        if offset > MAX_PAYLOAD_BYTES:
            raise ValueError(f"packed request exceeds 1 MiB: {offset}")
        self.layouts = tuple(layouts)
        self.buffer = torch.empty(offset, dtype=torch.uint8, device="npu")

    def pack(self, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        for layout in self.layouts:
            tensor = tensors[layout.name]
            if tuple(tensor.shape) != layout.shape or tensor.dtype != layout.dtype:
                raise ValueError(f"layout changed for {layout.name}")
            self.buffer[layout.offset : layout.offset + layout.num_bytes].copy_(
                tensor.contiguous().view(torch.uint8).flatten()
            )
        return self.buffer

    def views(self) -> dict[str, torch.Tensor]:
        return {
            layout.name: self.buffer[layout.offset : layout.offset + layout.num_bytes]
            .view(layout.dtype)
            .view(layout.shape)
            for layout in self.layouts
        }


class IndexerShmTransport:
    def __init__(
        self,
        *,
        store_url: str,
        world_size: int,
        global_rank: int,
        device: int,
        decoder_rank: int,
        service_rank: int,
        shm_id: int = 0,
    ) -> None:
        self.world_size = world_size
        self.global_rank = global_rank
        self.device = device
        self.decoder_rank = decoder_rank
        self.service_rank = service_rank
        self._extension = _load_extension()
        # set_device() alone does not run torch_npu's lazy initialization.
        # Materialize the stream/context before HYBM maps its GVA; otherwise
        # torch_npu may create a different ACL context on the first later op,
        # where the SHM mapping is not valid.
        torch.npu.current_stream(device)
        config = shm.ShmConfig()
        # Rank 0 lives on the model-free service node.  It may wait while the
        # decoder loads the 646 GiB checkpoint; this is a startup bound, not a
        # soak-test duration.
        config.init_timeout = 1200
        config.create_timeout = 1200
        config.operation_timeout = 1200
        if mf.initialize() != 0:
            raise RuntimeError("memfabric_hybrid.initialize failed")
        if shm.initialize(store_url, world_size, global_rank, device, config) != 0:
            raise RuntimeError("MemFabric SHM initialize failed")
        self._handle = shm.create(
            shm_id,
            world_size,
            global_rank,
            LOCAL_MEMORY_BYTES,
            shm.ShmDataOpType.MTE,
            0,
        )
        self.gva = int(self._handle.gva)
        # The device kernel queries MemFabric's actual symmetric stride from
        # SHM metadata.  It is 1 GiB on this A3 runtime even though each rank
        # contributes only 4 MiB of physical pages.
        self.symmetric_size = LOCAL_MEMORY_BYTES
        # SHM physical pages are not guaranteed to start at zero.  Doorbells
        # live in this segment, so initialize the local contribution before
        # either peer derives its first sequence number.
        self._extension.initialize_control(self.gva, self.symmetric_size, self.global_rank)
        torch.npu.synchronize()
        self.barrier()

    def barrier(self) -> None:
        if self._handle.barrier() != 0:
            raise RuntimeError("MemFabric SHM barrier failed")

    def decoder_exchange(
        self,
        request: torch.Tensor,
        response: torch.Tensor,
        *,
        profiled: bool = False,
    ) -> None:
        exchange = self._extension.decoder_exchange_profiled if profiled else self._extension.decoder_exchange
        exchange(
            request,
            response,
            self.gva,
            self.symmetric_size,
            self.decoder_rank,
            self.service_rank,
        )

    def service_receive(self, request: torch.Tensor) -> None:
        self._extension.service_receive(request, self.gva, self.symmetric_size, self.service_rank)

    def service_receive_profiled(self, request: torch.Tensor, trace: torch.Tensor, layer_id: int) -> None:
        self._extension.service_receive_profiled(
            request,
            trace,
            layer_id,
            self.gva,
            self.symmetric_size,
            self.service_rank,
        )

    def service_respond(self, response: torch.Tensor) -> None:
        self._extension.service_respond(
            response,
            self.gva,
            self.symmetric_size,
            self.decoder_rank,
            self.service_rank,
        )

    def service_respond_profiled(self, response: torch.Tensor, trace: torch.Tensor, layer_id: int) -> None:
        self._extension.service_respond_profiled(
            response,
            trace,
            layer_id,
            self.gva,
            self.symmetric_size,
            self.decoder_rank,
            self.service_rank,
        )

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is None:
            return
        handle.destroy(0)
        self._handle = None
        shm.uninitialize(0)
        mf.uninitialize()


class ShmRemoteIndexerClient:
    """vLLM-facing rank-local client with no host payload path."""

    def __init__(
        self,
        *,
        store_url: str,
        decoder_world_size: int,
        rank: int,
        device: int,
        topk: int,
        profile_device: bool = False,
    ) -> None:
        self.rank = rank
        self.topk = topk
        self.profile_device = profile_device
        self._transport = IndexerShmTransport(
            store_url=store_url,
            world_size=decoder_world_size * 2,
            global_rank=decoder_world_size + rank,
            device=device,
            decoder_rank=decoder_world_size + rank,
            service_rank=rank,
        )
        self._packed: dict[tuple[object, ...], PackedTensors] = {}
        self._responses: dict[int, torch.Tensor] = {}
        self._warned_control_noop = False
        logger.warning(
            "Remote indexer rank %d uses MemFabric SHM device transport; store=%s service_rank=%d",
            rank,
            store_url,
            rank,
        )

    @staticmethod
    def _signature(tensors: dict[str, torch.Tensor]) -> tuple[object, ...]:
        return tuple((name, tuple(tensor.shape), tensor.dtype) for name, tensor in tensors.items())

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
        del layer_id, capturing
        if q.dtype != torch.int8 or new_k.dtype != torch.int8:
            raise ValueError("SHM remote indexer requires the C8 indexer path")
        tensors = {
            "q": q,
            "q_scale": q_scale,
            "weights": weights.to(torch.float16),
            "new_k": new_k,
            "new_k_scale": new_k_scale,
            "slot_mapping": slot_mapping,
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "actual_seq_lengths_key": actual_seq_lengths_key,
            "block_table": block_table,
        }
        signature = self._signature(tensors)
        packed = self._packed.get(signature)
        if packed is None:
            packed = PackedTensors(tensors)
            self._packed[signature] = packed
        request = packed.pack(tensors)

        tokens = q.shape[0]
        response_bytes = tokens * self.topk * 4
        response = self._responses.get(tokens)
        if response is None:
            response = torch.empty(align32(response_bytes), dtype=torch.uint8, device=q.device)
            self._responses[tokens] = response
        self._transport.decoder_exchange(request, response, profiled=self.profile_device)
        return response[:response_bytes].view(torch.int32).view(tokens, 1, self.topk)

    def set_graph_callback_enqueuer(self, enqueuer: object) -> None:
        del enqueuer

    def _control_noop(self, operation: str) -> None:
        if not self._warned_control_noop:
            logger.warning(
                "SHM remote indexer prototype treats %s as a no-op; cache "
                "starts from the configured fill value and short benchmark "
                "does not reuse blocks",
                operation,
            )
            self._warned_control_noop = True

    def reset_cache(self) -> None:
        # Graph capture records kernels but does not execute them on this stack.
        self._control_noop("reset_cache")

    def fill_blocks(self, block_ids: list[int]) -> None:
        if block_ids:
            self._control_noop("fill_blocks")
