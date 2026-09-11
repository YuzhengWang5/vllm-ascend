"""Single-NPU worker for the trivial remote SFA indexer service."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import time
import traceback

import torch
import torch_npu

from .remote_indexer import (
    CONTROL_MESSAGE,
    PROTOCOL_VERSION,
    SELECT_MESSAGE,
    _recv_exact,
    recv_framed,
    recv_raw_select_request,
    send_framed,
    send_raw_select_error,
    send_raw_select_response,
)


class RemoteIndexerWorker:
    def __init__(
        self,
        *,
        rank: int,
        device: int,
        cache_blocks: int,
        block_size: int,
        head_dim: int,
        topk: int,
        indexer_fill_value: int,
        indexer_scale_value: float,
        profile_dir: str | None,
    ) -> None:
        self.rank = rank
        self.device = device
        self.cache_blocks = cache_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        self.topk = topk
        self.indexer_fill_value = indexer_fill_value
        self.indexer_scale_value = indexer_scale_value
        self.profile_dir = profile_dir
        self.caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.host_buffers: dict[
            tuple[str, tuple[int, ...], torch.dtype], torch.Tensor
        ] = {}
        self._profiler = None
        self._profile_active = False
        self._profile_start_requested = False
        self._profile_stop_requested = False
        configured_breakdown_ranks = {
            item.strip()
            for item in os.environ.get("IAAS_BREAKDOWN_RANKS", "").split(",")
            if item.strip()
        }
        self.breakdown_enabled = str(rank) in configured_breakdown_ranks
        self.last_breakdown: dict[str, float] = {}
        torch.npu.set_device(device)

        if self.profile_dir:
            # Signal handlers only flip flags.  Profiler start/stop is performed
            # at the next request boundary on the worker's main thread.
            signal.signal(signal.SIGUSR1, self._request_profile_start)
            signal.signal(signal.SIGUSR2, self._request_profile_stop)
        elif self.rank == 0:
            # With no torch profiler configured, reuse USR1/USR2 to bracket a
            # lightweight per-call breakdown window after macro benchmarking.
            signal.signal(signal.SIGUSR1, self._enable_breakdown)
            signal.signal(signal.SIGUSR2, self._disable_breakdown)

    def _enable_breakdown(self, _signum, _frame) -> None:
        self.breakdown_enabled = True

    def _disable_breakdown(self, _signum, _frame) -> None:
        self.breakdown_enabled = False

    def _request_profile_start(self, _signum, _frame) -> None:
        self._profile_start_requested = True

    def _request_profile_stop(self, _signum, _frame) -> None:
        self._profile_stop_requested = True

    def apply_profile_control(self) -> None:
        if self._profile_stop_requested:
            self._profile_stop_requested = False
            if self._profile_active:
                torch.npu.synchronize()
                self._profiler.stop()
                self._profile_active = False
                print(
                    json.dumps({"event": "profile_stopped", "rank": self.rank}),
                    flush=True,
                )

        if self._profile_start_requested:
            self._profile_start_requested = False
            if self._profile_active:
                return
            if self._profiler is not None:
                raise RuntimeError("Remote indexer v0 supports one profile window per process")
            assert self.profile_dir is not None
            Path(self.profile_dir).mkdir(parents=True, exist_ok=True)
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=torch_npu.profiler.ExportType.Text,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                msprof_tx=False,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                l2_cache=False,
                op_attr=False,
                data_simplification=True,
                record_op_args=False,
            )
            self._profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                record_shapes=True,
                with_stack=False,
                profile_memory=False,
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    self.profile_dir,
                    worker_name=f"indexer_rank{self.rank}_pid{os.getpid()}",
                ),
            )
            self._profiler.start()
            self._profile_active = True
            print(
                json.dumps(
                    {
                        "event": "profile_started",
                        "rank": self.rank,
                        "profile_dir": self.profile_dir,
                    }
                ),
                flush=True,
            )

    def _cache(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self.caches.get(layer_id)
        if cache is None:
            key = torch.full(
                (self.cache_blocks, self.block_size, 1, self.head_dim),
                self.indexer_fill_value,
                dtype=torch.int8,
                device="npu",
            )
            scale = torch.full(
                (self.cache_blocks, self.block_size, 1, 1),
                self.indexer_scale_value,
                dtype=torch.float16,
                device="npu",
            )
            cache = (key, scale)
            self.caches[layer_id] = cache
        return cache

    def host_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (name, shape, dtype)
        buffer = self.host_buffers.get(key)
        if buffer is None:
            buffer = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            self.host_buffers[key] = buffer
        return buffer

    def reset_cache(self) -> None:
        """Undo cache writes performed by the graph's capture execution."""
        for key, scale in self.caches.values():
            key.fill_(self.indexer_fill_value)
            scale.fill_(self.indexer_scale_value)
        torch.npu.synchronize()

    def fill_blocks(self, block_ids: list[int]) -> None:
        """Clear stale index state when the scheduler reuses physical blocks."""
        if not block_ids:
            return
        if min(block_ids) < 0 or max(block_ids) >= self.cache_blocks:
            raise ValueError(
                f"Remote index cache block ids out of range: {block_ids!r}, "
                f"cache_blocks={self.cache_blocks}"
            )
        indices = torch.tensor(
            sorted(set(block_ids)), dtype=torch.long, device="npu"
        )
        for key, scale in self.caches.values():
            key_values = torch.full(
                (len(indices), *key.shape[1:]),
                self.indexer_fill_value,
                dtype=key.dtype,
                device=key.device,
            )
            scale_values = torch.full(
                (len(indices), *scale.shape[1:]),
                self.indexer_scale_value,
                dtype=scale.dtype,
                device=scale.device,
            )
            key.index_copy_(0, indices, key_values)
            scale.index_copy_(0, indices, scale_values)
        torch.npu.synchronize()

    def select(self, request: dict) -> torch.Tensor:
        select_started_at = time.perf_counter_ns()
        valid_block_ids = request["block_table"][request["block_table"] >= 0]
        if valid_block_ids.numel() > 0:
            largest_block_id = int(valid_block_ids.max().item())
            if largest_block_id >= self.cache_blocks:
                raise ValueError(
                    "Remote index cache is too small: "
                    f"largest block id={largest_block_id}, "
                    f"cache_blocks={self.cache_blocks}"
                )
        validation_done_at = time.perf_counter_ns()
        events = None
        if self.breakdown_enabled:
            events = [torch.npu.Event(enable_timing=True) for _ in range(5)]
            events[0].record()
        q = request["q"].to("npu")
        q_scale = request["q_scale"].to("npu")
        weights = request["weights"].to("npu")
        new_k = request["new_k"].to("npu")
        new_k_scale = request["new_k_scale"].to("npu")
        slot_mapping = request["slot_mapping"].to("npu")
        actual_seq_lengths_query = request["actual_seq_lengths_query"].to("npu")
        actual_seq_lengths_key = request["actual_seq_lengths_key"].to("npu")
        block_table = request["block_table"].to("npu")
        if events is not None:
            events[1].record()
        h2d_enqueued_at = time.perf_counter_ns()

        key_cache, scale_cache = self._cache(int(request["layer_id"]))
        # Match the colocated SFA path exactly.  In particular, the Ascend
        # scatter kernel owns the padding-slot semantics used by graph replay;
        # filtering indices on the host and replacing it with IndexCopy is not
        # guaranteed to be equivalent.
        torch_npu.npu_scatter_nd_update_(
            key_cache.view(-1, self.head_dim),
            slot_mapping.view(-1, 1),
            new_k.view(-1, self.head_dim),
        )
        torch_npu.npu_scatter_nd_update_(
            scale_cache.view(-1, 1),
            slot_mapping.view(-1, 1),
            new_k_scale.view(-1, 1),
        )
        if events is not None:
            events[2].record()
        scatter_enqueued_at = time.perf_counter_ns()

        # A3's colocated custom op lowers to the same public FP16 ABI.  A
        # one-die equivalence probe is archived with this experiment.
        topk = torch_npu.npu_quant_lightning_indexer(
            query=q,
            key=key_cache,
            weights=weights,
            query_dequant_scale=q_scale,
            key_dequant_scale=scale_cache.squeeze(2),
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.topk,
            sparse_mode=3,
        )
        if events is not None:
            events[3].record()
        indexer_enqueued_at = time.perf_counter_ns()
        output = topk.cpu()
        if events is not None:
            events[4].record()
            events[4].synchronize()
        d2h_done_at = time.perf_counter_ns()
        if events is not None:
            self.last_breakdown = {
                "validation_wall_ms": (validation_done_at - select_started_at) / 1e6,
                "h2d_enqueue_wall_ms": (h2d_enqueued_at - validation_done_at) / 1e6,
                "scatter_enqueue_wall_ms": (scatter_enqueued_at - h2d_enqueued_at) / 1e6,
                "indexer_enqueue_wall_ms": (indexer_enqueued_at - scatter_enqueued_at) / 1e6,
                "d2h_wait_wall_ms": (d2h_done_at - indexer_enqueued_at) / 1e6,
                "worker_wall_ms": (d2h_done_at - select_started_at) / 1e6,
                "h2d_device_ms": events[0].elapsed_time(events[1]),
                "scatter_device_ms": events[1].elapsed_time(events[2]),
                "indexer_device_ms": events[2].elapsed_time(events[3]),
                "d2h_device_ms": events[3].elapsed_time(events[4]),
            }
        return output


def serve(args: argparse.Namespace) -> None:
    worker = RemoteIndexerWorker(
        rank=args.rank,
        device=args.device,
        cache_blocks=args.cache_blocks,
        block_size=args.block_size,
        head_dim=args.head_dim,
        topk=args.topk,
        indexer_fill_value=args.indexer_fill_value,
        indexer_scale_value=args.indexer_scale_value,
        profile_dir=args.profile_dir,
    )
    if args.pid_file:
        pid_file = Path(args.pid_file)
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen.bind((args.host, args.port))
    listen.listen(1)
    print(
        json.dumps(
            {
                "event": "listening",
                "rank": args.rank,
                "device": args.device,
                "address": f"{args.host}:{args.port}",
                "cache_blocks": args.cache_blocks,
            }
        ),
        flush=True,
    )
    while True:
        connection, peer = listen.accept()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        message_kind = None
        try:
            hello = recv_framed(connection)
            if hello != {
                "op": "hello",
                "version": PROTOCOL_VERSION,
                "rank": args.rank,
            }:
                raise RuntimeError(f"Invalid handshake: {hello!r}")
            send_framed(
                connection,
                {
                    "ok": True,
                    "version": PROTOCOL_VERSION,
                    "select_transport": "raw",
                },
            )
            print(
                json.dumps(
                    {
                        "event": "connected",
                        "rank": args.rank,
                        "peer": peer[0],
                    }
                ),
                flush=True,
            )
            while True:
                message_kind = _recv_exact(connection, 1)
                worker.apply_profile_control()
                started_at = time.perf_counter_ns()
                if message_kind == SELECT_MESSAGE:
                    # Keep the fallback valid for the unsigned wire field even
                    # when header decoding itself fails.
                    request_id = 0
                    try:
                        request = recv_raw_select_request(
                            connection, worker.host_buffer
                        )
                        received_at = time.perf_counter_ns()
                        request_id = int(request["request_id"])
                        topk = worker.select(request)
                        selected_at = time.perf_counter_ns()
                        send_raw_select_response(connection, request_id, topk)
                        sent_at = time.perf_counter_ns()
                    except Exception as error:
                        send_raw_select_error(connection, request_id, repr(error))
                        raise
                    if worker.breakdown_enabled:
                        print(
                            json.dumps(
                                {
                                    "event": "remote_indexer_service_breakdown",
                                    "rank": args.rank,
                                    "request_id": request_id,
                                    "layer_id": request["layer_id"],
                                    "tokens": request["q"].shape[0],
                                    "request_recv_ms": (received_at - started_at) / 1e6,
                                    "worker_select_ms": (selected_at - received_at) / 1e6,
                                    "response_send_ms": (sent_at - selected_at) / 1e6,
                                    "service_total_ms": (sent_at - started_at) / 1e6,
                                    **worker.last_breakdown,
                                }
                            ),
                            flush=True,
                        )
                    elif request_id % args.log_every == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "select",
                                    "rank": args.rank,
                                    "request_id": request_id,
                                    "layer_id": request["layer_id"],
                                    "tokens": request["q"].shape[0],
                                    "elapsed_ms": (
                                        time.perf_counter_ns() - started_at
                                    )
                                    / 1e6,
                                }
                            ),
                            flush=True,
                        )
                    continue
                if message_kind != CONTROL_MESSAGE:
                    raise RuntimeError(
                        f"Unsupported message kind: {message_kind!r}"
                    )
                request = recv_framed(connection)
                if request.get("op") == "reset_cache":
                    if request.get("rank") != args.rank:
                        raise RuntimeError(
                            "Request rank mismatch: "
                            f"expected={args.rank}, got={request.get('rank')}"
                        )
                    worker.reset_cache()
                    send_framed(connection, {"ok": True, "op": "reset_cache"})
                    print(
                        json.dumps(
                            {
                                "event": "cache_reset",
                                "rank": args.rank,
                                "layers": len(worker.caches),
                            }
                        ),
                        flush=True,
                    )
                    continue
                if request.get("op") == "fill_blocks":
                    if request.get("rank") != args.rank:
                        raise RuntimeError(
                            "Request rank mismatch: "
                            f"expected={args.rank}, got={request.get('rank')}"
                        )
                    block_ids = request.get("block_ids")
                    if not isinstance(block_ids, list):
                        raise TypeError("fill_blocks requires a block_ids list")
                    worker.fill_blocks(block_ids)
                    send_framed(connection, {"ok": True, "op": "fill_blocks"})
                    print(
                        json.dumps(
                            {
                                "event": "blocks_filled",
                                "rank": args.rank,
                                "blocks": len(set(block_ids)),
                                "layers": len(worker.caches),
                            }
                        ),
                        flush=True,
                    )
                    continue
                raise RuntimeError(
                    f"Unsupported control request: {request.get('op')!r}"
                )
        except (ConnectionError, EOFError):
            pass
        except Exception as error:
            print(
                json.dumps(
                    {
                        "event": "error",
                        "rank": args.rank,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                ),
                flush=True,
            )
            if message_kind != SELECT_MESSAGE:
                try:
                    send_framed(connection, {"ok": False, "error": repr(error)})
                except Exception:
                    pass
        finally:
            connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--cache-blocks", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--indexer-fill-value", type=int, default=1)
    parser.add_argument("--indexer-scale-value", type=float, default=0.015)
    parser.add_argument("--profile-dir")
    parser.add_argument("--pid-file")
    parser.add_argument("--log-every", type=int, default=61)
    serve(parser.parse_args())


if __name__ == "__main__":
    main()
