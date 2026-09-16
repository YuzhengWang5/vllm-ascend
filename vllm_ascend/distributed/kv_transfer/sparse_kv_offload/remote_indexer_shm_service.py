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
DATA_NAMES = ("q", "q_scale", "weights", "new_k", "new_k_scale")
METADATA_NAMES = (
    "slot_mapping",
    "actual_seq_lengths_query",
    "actual_seq_lengths_key",
    "block_table",
)


def empty_request(args: argparse.Namespace, batch: int) -> dict[str, torch.Tensor]:
    index_dtype = torch.bfloat16 if args.index_dtype == "bf16" else torch.int8
    weight_dtype = torch.bfloat16 if args.index_dtype == "bf16" else torch.float16
    return {
        "q": torch.empty((batch, args.heads, args.head_dim), dtype=index_dtype, device="npu"),
        "q_scale": torch.empty((batch, args.heads), dtype=torch.float16, device="npu"),
        "weights": torch.empty((batch, args.heads), dtype=weight_dtype, device="npu"),
        "new_k": torch.empty((batch, args.head_dim), dtype=index_dtype, device="npu"),
        "new_k_scale": torch.empty((batch, 1), dtype=torch.float16, device="npu"),
        "slot_mapping": torch.empty((batch,), dtype=torch.int32, device="npu"),
        "actual_seq_lengths_query": torch.empty((batch,), dtype=torch.int32, device="npu"),
        "actual_seq_lengths_key": torch.empty((batch,), dtype=torch.int32, device="npu"),
        "block_table": torch.empty((batch, args.block_table_cols), dtype=torch.int32, device="npu"),
    }


def data_request(args: argparse.Namespace, batch: int) -> dict[str, torch.Tensor]:
    full = empty_request(args, batch)
    return {name: full[name] for name in DATA_NAMES}


def make_cache(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor | None]:
    key = torch.full(
        (args.cache_blocks, args.block_size, 1, args.head_dim),
        args.indexer_fill_value,
        dtype=torch.bfloat16 if args.index_dtype == "bf16" else torch.int8,
        device="npu",
    )
    if args.index_dtype == "bf16":
        return key, None
    scale = torch.full(
        (args.cache_blocks, args.block_size, 1, 1),
        args.indexer_scale_value,
        dtype=torch.float16,
        device="npu",
    )
    return key, scale


def compute_topk(
    args: argparse.Namespace,
    tensors: dict[str, torch.Tensor],
    key_cache: torch.Tensor,
    scale_cache: torch.Tensor | None,
) -> torch.Tensor:
    if args.index_dtype == "bf16":
        topk, _ = torch_npu.npu_lightning_indexer(
            query=tensors["q"],
            key=key_cache,
            weights=tensors["weights"],
            actual_seq_lengths_query=tensors["actual_seq_lengths_query"],
            actual_seq_lengths_key=tensors["actual_seq_lengths_key"],
            block_table=tensors["block_table"],
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=args.topk,
            sparse_mode=3,
        )
        return topk
    assert scale_cache is not None
    return torch_npu.npu_quant_lightning_indexer(
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
        "unique_valid_block_count": int(torch.unique(valid_blocks).numel()),
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
    data_requests: dict[int, PackedTensors] = {}
    responses: dict[int, torch.Tensor] = {}
    graphs: dict[int, list[torch.npu.NPUGraph]] = {}
    resident_sources: dict[int, list[torch.Tensor]] = {}
    oracle_topk: dict[int, torch.Tensor] = {}
    trace = torch.empty((args.layers, 12), dtype=torch.int64, device="npu")
    graph_pool = torch.npu.graph_pool_handle()
    for batch in args.batches:
        if args.synthetic_oracle_topk:
            oracle_topk[batch] = torch.arange(
                args.topk, dtype=torch.int32, device="npu"
            ).view(1, args.topk).expand(batch, args.topk).contiguous()
        request = PackedTensors(empty_request(args, batch))
        full_tensors = request.views()
        data_only_request = PackedTensors(data_request(args, batch))
        data_only_tensors = data_only_request.views()
        response_bytes = batch * args.topk * 4
        response = torch.empty(align32(response_bytes), dtype=torch.uint8, device="npu")
        batch_resident_sources = [
            torch.full(
                (batch, args.topk),
                -1,
                dtype=torch.int32,
                device="npu",
            )
            for _ in range(args.layers)
        ]
        batch_graphs: list[torch.npu.NPUGraph] = []
        for layer_id, (key_cache, scale_cache) in enumerate(caches):
            if args.metadata_once_per_step and layer_id != 0:
                active_request = data_only_request
                tensors = {
                    **data_only_tensors,
                    **{name: full_tensors[name] for name in METADATA_NAMES},
                }
            else:
                active_request = request
                tensors = full_tensors
            graph = torch.npu.NPUGraph()
            with torch.inference_mode(), torch.npu.graph(graph, pool=graph_pool):
                if args.profile_device_breakdown:
                    transport.service_receive_profiled(
                        active_request.buffer, trace, layer_id
                    )
                else:
                    transport.service_receive(active_request.buffer)
                torch_npu.npu_scatter_nd_update_(
                    key_cache.view(-1, args.head_dim),
                    tensors["slot_mapping"].view(-1, 1),
                    tensors["new_k"].view(-1, args.head_dim),
                )
                if scale_cache is not None:
                    torch_npu.npu_scatter_nd_update_(
                        scale_cache.view(-1, 1),
                        tensors["slot_mapping"].view(-1, 1),
                        tensors["new_k_scale"].view(-1, 1),
                    )
                topk = compute_topk(args, tensors, key_cache, scale_cache)
                if args.synthetic_oracle_topk:
                    topk = oracle_topk[batch]
                if args.service_managed_resident:
                    if args.force_resident_miss:
                        # Controlled synthetic worst case: retain the service's
                        # logical->physical mapping, but invalidate its resident
                        # tags before every layer replay so the decoder really
                        # reads every selected MLA token from the BM pool.
                        batch_resident_sources[layer_id].fill_(-1)
                    response_values = response[:response_bytes].view(
                        torch.int32
                    ).view(batch, args.topk)
                    transport.service_resident_update(
                        topk,
                        tensors["block_table"],
                        tensors["slot_mapping"],
                        batch_resident_sources[layer_id],
                        response_values,
                        args.block_size,
                        args.forced_resident_miss_count,
                    )
                else:
                    response_values = topk
                    response[:response_bytes].copy_(
                        response_values.view(torch.uint8).flatten()
                    )
                if args.profile_device_breakdown:
                    transport.service_respond_profiled(response, trace, layer_id)
                else:
                    transport.service_respond(response)
            batch_graphs.append(graph)
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
        data_requests[batch] = data_only_request
        responses[batch] = response
        graphs[batch] = batch_graphs
        resident_sources[batch] = batch_resident_sources

    print(
        json.dumps(
            {
                "event": "service_ready",
                "global_rank": args.global_rank,
                "device": args.device,
                "layers": args.layers,
                "batches": args.batches,
                "prelude_batches": args.prelude_batches,
                "request_bytes": {batch: requests[batch].buffer.numel() for batch in args.batches},
                "data_request_bytes": {
                    batch: data_requests[batch].buffer.numel()
                    for batch in args.batches
                },
                "metadata_once_per_step": args.metadata_once_per_step,
                "response_bytes": {batch: responses[batch].numel() for batch in args.batches},
                "service_managed_resident": args.service_managed_resident,
                "index_dtype": args.index_dtype,
                "synthetic_oracle_topk": args.synthetic_oracle_topk,
                "forced_resident_miss_count": args.forced_resident_miss_count,
            }
        ),
        flush=True,
    )

    # If a deployment uses static startup replays, these shapes must match
    # decoder requests exactly.  The eager GLM benchmark passes an empty list
    # because API health precedes any request; its first request must enter the
    # dynamic byte-size dispatch loop below.
    for batch in args.prelude_batches:
        for graph in graphs[batch]:
            graph.replay()
        torch.npu.synchronize()
        print(json.dumps({"event": "prelude_done", "batch": batch}), flush=True)

    steady_batch = args.batches[-1]
    token_step = 0
    debug_steady_done = not args.debug_first_token_metadata
    if args.debug_first_token_metadata:
        request = requests[steady_batch]
        full_tensors = request.views()
        data_only_request = data_requests[steady_batch]
        data_only_tensors = data_only_request.views()
        response = responses[steady_batch]
        response_bytes = steady_batch * args.topk * 4
        for layer_id, (key_cache, scale_cache) in enumerate(caches):
            if args.metadata_once_per_step and layer_id != 0:
                active_request = data_only_request
                tensors = {
                    **data_only_tensors,
                    **{name: full_tensors[name] for name in METADATA_NAMES},
                }
            else:
                active_request = request
                tensors = full_tensors
            transport.service_receive(active_request.buffer)
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
            if scale_cache is not None:
                torch_npu.npu_scatter_nd_update_(
                    scale_cache.view(-1, 1),
                    tensors["slot_mapping"].view(-1, 1),
                    tensors["new_k_scale"].view(-1, 1),
                )
            topk = compute_topk(args, tensors, key_cache, scale_cache)
            if args.synthetic_oracle_topk:
                topk = oracle_topk[steady_batch]
            if args.service_managed_resident:
                if args.force_resident_miss:
                    resident_sources[steady_batch][layer_id].fill_(-1)
                response_values = response[:response_bytes].view(
                    torch.int32
                ).view(steady_batch, args.topk)
                transport.service_resident_update(
                    topk,
                    tensors["block_table"],
                    tensors["slot_mapping"],
                    resident_sources[steady_batch][layer_id],
                    response_values,
                    args.block_size,
                    args.forced_resident_miss_count,
                )
            else:
                response_values = topk
                response[:response_bytes].copy_(
                    response_values.view(torch.uint8).flatten()
                )
            transport.service_respond(response)
            torch.npu.synchronize()
            if layer_id in (0, args.layers - 1):
                response_cpu = response_values.cpu().reshape(-1)
                valid_response = response_cpu[response_cpu >= 0]
                print(
                    json.dumps(
                        {
                            "event": "first_token_response",
                            "layer_id": layer_id,
                            "response_count": int(response_cpu.numel()),
                            "valid_physical_sources": int(valid_response.numel()),
                            "negative_sources": int((response_cpu < 0).sum()),
                            "unique_physical_sources": int(
                                torch.unique(valid_response).numel()
                            ),
                            "physical_source_min": (
                                int(valid_response.min())
                                if valid_response.numel()
                                else -1
                            ),
                            "physical_source_max": (
                                int(valid_response.max())
                                if valid_response.numel()
                                else -1
                            ),
                        }
                    ),
                    flush=True,
                )
        token_step = 1
        print(json.dumps({"event": "debug_first_token_done"}), flush=True)

    request_bytes_to_batch = {
        int(request.buffer.numel()): batch for batch, request in requests.items()
    }
    if len(request_bytes_to_batch) != len(requests):
        raise RuntimeError("full request byte sizes must uniquely identify graph batches")
    request_control = torch.empty(8, dtype=torch.int32, device="npu")
    seen_dynamic_batches: set[int] = set()
    while args.max_token_steps <= 0 or token_step < args.max_token_steps:
        if args.dynamic_batch_by_request_bytes:
            transport.service_peek_request(request_control)
            request_bytes = int(request_control.cpu()[1])
            try:
                steady_batch = request_bytes_to_batch[request_bytes]
            except KeyError as error:
                raise RuntimeError(
                    f"unknown full request size {request_bytes}; expected "
                    f"{request_bytes_to_batch}"
                ) from error
            if steady_batch not in seen_dynamic_batches:
                print(json.dumps({
                    "event": "dynamic_batch_selected",
                    "batch": steady_batch,
                    "request_bytes": request_bytes,
                }), flush=True)
                seen_dynamic_batches.add(steady_batch)
        # Queue a whole model pass.  Each receive kernel blocks on its own
        # doorbell, while later layer graphs are already queued on the stream.
        for graph in graphs[steady_batch]:
            graph.replay()
        torch.npu.synchronize()
        token_step += 1
        # The first manually inspected pass can still be a one-sequence
        # vLLM graph warmup.  Inspect the already-computed layer-60 response
        # after subsequent replays until a genuinely populated batch arrives;
        # this is a warmup-only correctness check and then disables itself.
        if not debug_steady_done:
            response_values = responses[steady_batch][: steady_batch * args.topk * 4].view(
                torch.int32
            ).view(steady_batch, args.topk)
            response_cpu = response_values.cpu().reshape(-1)
            valid_response = response_cpu[response_cpu >= 0]
            if valid_response.numel() >= steady_batch * args.topk // 2:
                summary = metadata_summary(full_tensors, args.cache_blocks, args.block_size)
                print(
                    json.dumps(
                        {
                            "event": "first_populated_batch",
                            "step": token_step,
                            **summary,
                            "response_count": int(response_cpu.numel()),
                            "valid_physical_sources": int(valid_response.numel()),
                            "negative_sources": int((response_cpu < 0).sum()),
                            "unique_physical_sources": int(torch.unique(valid_response).numel()),
                        }
                    ),
                    flush=True,
                )
                debug_steady_done = True
        if args.profile_device_breakdown and args.profile_log_every > 0 and token_step % args.profile_log_every == 0:
            print(json.dumps(device_breakdown(trace, token_step)), flush=True)
        if token_step % args.log_every == 0:
            resident_misses = None
            if args.service_managed_resident:
                last_response = responses[steady_batch][
                    : steady_batch * args.topk * 4
                ].view(torch.int32)
                resident_misses = int((last_response >= 0).sum().item())
            print(
                json.dumps({
                    "event": "token_step",
                    "step": token_step,
                    "batch": steady_batch,
                    "last_layer_resident_misses": resident_misses,
                    "last_layer_topk_entries": steady_batch * args.topk,
                }),
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
    parser.add_argument("--index-dtype", choices=("c8", "bf16"), default="c8")
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
    parser.add_argument("--service-managed-resident", action="store_true")
    parser.add_argument("--synthetic-oracle-topk", action="store_true")
    parser.add_argument("--forced-resident-miss-count", type=int, default=0)
    parser.add_argument(
        "--force-resident-miss",
        action="store_true",
        help="invalidate service resident tags before every layer; synthetic bandwidth stress only",
    )
    parser.add_argument("--metadata-once-per-step", action="store_true")
    parser.add_argument("--dynamic-batch-by-request-bytes", action="store_true")
    parser.add_argument("--pid-file")
    args = parser.parse_args()
    if len(set(args.batches)) != len(args.batches):
        parser.error("--batches must not contain duplicates")
    if any(batch not in args.batches for batch in args.prelude_batches):
        parser.error("every --prelude-batches value must occur in --batches")
    if not 0 <= args.forced_resident_miss_count <= args.topk:
        parser.error("--forced-resident-miss-count must be in [0, topk]")
    if args.forced_resident_miss_count and (
        not args.service_managed_resident or not args.synthetic_oracle_topk
    ):
        parser.error("forced resident misses require service-managed synthetic oracle Top-K")
    if args.force_resident_miss and args.forced_resident_miss_count:
        parser.error("force-all-miss and forced-resident-miss-count are mutually exclusive")
    serve(args)


if __name__ == "__main__":
    main()
