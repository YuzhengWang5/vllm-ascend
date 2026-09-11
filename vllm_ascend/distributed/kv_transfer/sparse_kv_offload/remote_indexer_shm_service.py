"""ACLGraph service loop for the MemFabric SHM remote SFA indexer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu

from .indexer_shm_transport import IndexerShmTransport, PackedTensors, align32

A3_CYCLES_PER_US = 50.0


def empty_request(args: argparse.Namespace, batch: int) -> dict[str, torch.Tensor]:
    return {
        "q": torch.empty((batch, args.heads, args.head_dim), dtype=torch.int8, device="npu"),
        "q_scale": torch.empty((batch, args.heads), dtype=torch.float16, device="npu"),
        "weights": torch.empty((batch, args.heads), dtype=torch.float16, device="npu"),
        "new_k": torch.empty((batch, args.head_dim), dtype=torch.int8, device="npu"),
        "new_k_scale": torch.empty((batch, 1), dtype=torch.float16, device="npu"),
        "slot_mapping": torch.empty((batch,), dtype=torch.int32, device="npu"),
        "actual_seq_lengths_query": torch.empty((batch,), dtype=torch.int32, device="npu"),
        "actual_seq_lengths_key": torch.empty((batch,), dtype=torch.int32, device="npu"),
        "block_table": torch.empty((batch, args.block_table_cols), dtype=torch.int32, device="npu"),
    }


def make_cache(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.full(
        (args.cache_blocks, args.block_size, 1, args.head_dim),
        args.indexer_fill_value,
        dtype=torch.int8,
        device="npu",
    )
    scale = torch.full(
        (args.cache_blocks, args.block_size, 1, 1),
        args.indexer_scale_value,
        dtype=torch.float16,
        device="npu",
    )
    return key, scale


def metadata_summary(tensors: dict[str, torch.Tensor], cache_blocks: int, block_size: int) -> dict[str, int | bool]:
    """Copy only address-like metadata for a one-token startup diagnosis."""
    slot_mapping = tensors["slot_mapping"].cpu()
    query_lengths = tensors["actual_seq_lengths_query"].cpu()
    key_lengths = tensors["actual_seq_lengths_key"].cpu()
    block_table = tensors["block_table"].cpu()
    valid_blocks = block_table[block_table >= 0]
    largest_block = int(valid_blocks.max().item()) if valid_blocks.numel() else -1
    smallest_block = int(valid_blocks.min().item()) if valid_blocks.numel() else -1
    slot_min = int(slot_mapping.min().item())
    slot_max = int(slot_mapping.max().item())
    return {
        "slot_min": slot_min,
        "slot_max": slot_max,
        "query_len_min": int(query_lengths.min().item()),
        "query_len_max": int(query_lengths.max().item()),
        "key_len_min": int(key_lengths.min().item()),
        "key_len_max": int(key_lengths.max().item()),
        "block_min": smallest_block,
        "block_max": largest_block,
        "valid_block_count": int(valid_blocks.numel()),
        "block_ids_in_range": largest_block < cache_blocks,
        # -1 is the graph-padding sentinel accepted by ScatterNdUpdate.
        "slots_in_range": slot_min >= -1 and slot_max < cache_blocks * block_size,
    }


def device_breakdown(trace: torch.Tensor, step: int) -> dict[str, object]:
    """Decode same-die cycle deltas; absolute clocks are never compared."""
    rows = trace.cpu().tolist()
    layers: list[dict[str, float | int]] = []
    for layer_id, row in enumerate(rows):
        service = row[0:3] + row[4:7]
        decoder = row[8:12]
        if service != sorted(service) or decoder != sorted(decoder):
            raise RuntimeError(f"non-monotonic device trace at layer {layer_id}: {row}")
        layers.append(
            {
                "layer_id": layer_id,
                "decoder_send_us": (decoder[1] - decoder[0]) / A3_CYCLES_PER_US,
                "decoder_remote_wait_us": (decoder[2] - decoder[1]) / A3_CYCLES_PER_US,
                "decoder_response_copy_us": (decoder[3] - decoder[2]) / A3_CYCLES_PER_US,
                "decoder_exchange_us": (decoder[3] - decoder[0]) / A3_CYCLES_PER_US,
                "service_wait_decoder_us": (service[1] - service[0]) / A3_CYCLES_PER_US,
                "service_request_copy_us": (service[2] - service[1]) / A3_CYCLES_PER_US,
                "service_compute_pack_us": (service[3] - service[2]) / A3_CYCLES_PER_US,
                "service_response_copy_us": (service[4] - service[3]) / A3_CYCLES_PER_US,
                "service_decoder_ack_us": (service[5] - service[4]) / A3_CYCLES_PER_US,
            }
        )
    return {"event": "device_breakdown", "step": step, "layers": layers}


def serve(args: argparse.Namespace) -> None:
    if args.pid_file:
        pid_file = Path(args.pid_file)
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    torch.npu.set_device(args.device)
    transport = IndexerShmTransport(
        store_url=args.store_url,
        world_size=args.world_size,
        global_rank=args.global_rank,
        device=args.device,
        decoder_rank=args.decoder_rank,
        service_rank=args.global_rank,
    )
    caches = [make_cache(args) for _ in range(args.layers)]
    torch.npu.synchronize()

    requests: dict[int, PackedTensors] = {}
    response_sources: dict[int, list[torch.Tensor]] = {}
    graphs: dict[int, list[torch.npu.NPUGraph]] = {}
    trace = torch.empty((args.layers, 12), dtype=torch.int64, device="npu")
    for batch in args.batches:
        request = PackedTensors(empty_request(args, batch))
        tensors = request.views()
        batch_graphs: list[torch.npu.NPUGraph] = []
        batch_response_sources: list[torch.Tensor] = []
        for layer_id, (key_cache, scale_cache) in enumerate(caches):
            graph = torch.npu.NPUGraph()
            with torch.inference_mode(), torch.npu.graph(graph):
                if args.profile_device_breakdown:
                    transport.service_receive_profiled(request.buffer, trace, layer_id)
                else:
                    transport.service_receive(request.buffer)
                torch_npu.npu_scatter_nd_update_(
                    key_cache.view(-1, args.head_dim),
                    tensors["slot_mapping"].view(-1, 1),
                    tensors["new_k"].view(-1, args.head_dim),
                )
                torch_npu.npu_scatter_nd_update_(
                    scale_cache.view(-1, 1),
                    tensors["slot_mapping"].view(-1, 1),
                    tensors["new_k_scale"].view(-1, 1),
                )
                topk = torch_npu.npu_quant_lightning_indexer(
                    query=tensors["q"],
                    key=key_cache,
                    weights=tensors["weights"],
                    query_dequant_scale=tensors["q_scale"],
                    key_dequant_scale=scale_cache.squeeze(2),
                    actual_seq_lengths_query=tensors["actual_seq_lengths_query"],
                    actual_seq_lengths_key=tensors["actual_seq_lengths_key"],
                    block_table=tensors["block_table"],
                    query_quant_mode=0,
                    key_quant_mode=0,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=args.topk,
                    sparse_mode=3,
                )
                response_source = topk.view(torch.uint8).flatten()
                if args.profile_device_breakdown:
                    transport.service_respond_profiled(
                        response_source, trace, layer_id
                    )
                else:
                    transport.service_respond(response_source)
            batch_graphs.append(graph)
            # Keep the graph output allocation alive: the captured transport
            # kernel reuses this exact address on every replay.
            batch_response_sources.append(topk)
            if layer_id in (0, args.layers - 1):
                print(
                    json.dumps(
                        {
                            "event": "graph_captured",
                            "batch": batch,
                            "layer_id": layer_id,
                        }
                    ),
                    flush=True,
                )
        requests[batch] = request
        response_sources[batch] = batch_response_sources
        graphs[batch] = batch_graphs

    print(
        json.dumps(
            {
                "event": "service_ready",
                "global_rank": args.global_rank,
                "device": args.device,
                "layers": args.layers,
                "batches": args.batches,
                "request_bytes": {batch: requests[batch].buffer.numel() for batch in args.batches},
                "response_bytes": {
                    batch: align32(batch * args.topk * 4)
                    for batch in args.batches
                },
            }
        ),
        flush=True,
    )

    # vLLM's FULL_DECODE_ONLY startup executes one model pass for each capture
    # shape, in descending capture-size order.  Queue the matching service
    # graphs before entering the steady B8 loop.  This keeps dispatch entirely
    # device-side on the hot path; no Python/CPU payload inspection is needed.
    for batch in args.prelude_batches:
        for graph in graphs[batch]:
            graph.replay()
        torch.npu.synchronize()
        print(json.dumps({"event": "prelude_done", "batch": batch}), flush=True)

    steady_batch = args.batches[-1]
    token_step = 0
    if args.debug_first_token_metadata:
        request = requests[steady_batch]
        tensors = request.views()
        for layer_id, (key_cache, scale_cache) in enumerate(caches):
            transport.service_receive(request.buffer)
            torch.npu.synchronize()
            summary = metadata_summary(tensors, args.cache_blocks, args.block_size)
            print(
                json.dumps(
                    {
                        "event": "first_token_metadata",
                        "layer_id": layer_id,
                        **summary,
                    }
                ),
                flush=True,
            )
            if not summary["block_ids_in_range"] or not summary["slots_in_range"]:
                raise RuntimeError(
                    f"received address metadata exceeds remote index cache: layer={layer_id}, summary={summary}"
                )
            torch_npu.npu_scatter_nd_update_(
                key_cache.view(-1, args.head_dim),
                tensors["slot_mapping"].view(-1, 1),
                tensors["new_k"].view(-1, args.head_dim),
            )
            torch_npu.npu_scatter_nd_update_(
                scale_cache.view(-1, 1),
                tensors["slot_mapping"].view(-1, 1),
                tensors["new_k_scale"].view(-1, 1),
            )
            topk = torch_npu.npu_quant_lightning_indexer(
                query=tensors["q"],
                key=key_cache,
                weights=tensors["weights"],
                query_dequant_scale=tensors["q_scale"],
                key_dequant_scale=scale_cache.squeeze(2),
                actual_seq_lengths_query=tensors["actual_seq_lengths_query"],
                actual_seq_lengths_key=tensors["actual_seq_lengths_key"],
                block_table=tensors["block_table"],
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=args.topk,
                sparse_mode=3,
            )
            transport.service_respond(topk.view(torch.uint8).flatten())
            torch.npu.synchronize()
        token_step = 1
        print(json.dumps({"event": "debug_first_token_done"}), flush=True)

    while args.max_token_steps <= 0 or token_step < args.max_token_steps:
        # Queue a whole model pass.  Each receive kernel blocks on its own
        # doorbell, while later layer graphs are already queued on the stream.
        for graph in graphs[steady_batch]:
            graph.replay()
        torch.npu.synchronize()
        token_step += 1
        if args.profile_device_breakdown and args.profile_log_every > 0 and token_step % args.profile_log_every == 0:
            print(json.dumps(device_breakdown(trace, token_step)), flush=True)
        if token_step % args.log_every == 0:
            print(
                json.dumps({"event": "token_step", "step": token_step}),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-url", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--global-rank", type=int, required=True)
    parser.add_argument("--decoder-rank", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layers", type=int, default=61)
    parser.add_argument("--batches", type=int, nargs="+", default=[32, 16, 8])
    parser.add_argument("--prelude-batches", type=int, nargs="*", default=[32, 16])
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--cache-blocks", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--block-table-cols", type=int, default=2054)
    parser.add_argument("--indexer-fill-value", type=int, default=1)
    parser.add_argument("--indexer-scale-value", type=float, default=0.015)
    parser.add_argument("--max-token-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=16)
    parser.add_argument("--debug-first-token-metadata", action="store_true")
    parser.add_argument("--profile-device-breakdown", action="store_true")
    parser.add_argument("--profile-log-every", type=int, default=0)
    parser.add_argument("--pid-file")
    args = parser.parse_args()
    if len(set(args.batches)) != len(args.batches):
        parser.error("--batches must not contain duplicates")
    if any(batch not in args.batches for batch in args.prelude_batches):
        parser.error("every --prelude-batches value must occur in --batches")
    serve(args)


if __name__ == "__main__":
    main()
