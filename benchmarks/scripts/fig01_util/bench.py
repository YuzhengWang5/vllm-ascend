#!/usr/bin/env python3
"""Sustained, graph-replayed stage workloads for A3 AI Core usage sampling.

One process runs on each of the 16 dies.  The external monitor samples
``npu-smi info`` only inside the marked steady-state window.  This script
does not infer a desired utilization value from latency or operation counts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu

import routed
import stages
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import moe_mlp
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op


BLOCK = 128
INDEX_HEADS = 64
INDEX_DIM = 128
HEADS = 16  # DeepSeek-V3.2: 128 attention heads / TP8 per die.
LATENT = 512
ROPE = 64
BATCHES = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def _mark(path: Path, value: dict | None) -> None:
    if value is None:
        path.unlink(missing_ok=True)
        return
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value))
    os.replace(temp, path)


def make_indexer_graph(batch: int, calls: int = 8):
    prefix = 32768  # C8 index, 32K prefix per request.
    blocks_per_row = prefix // BLOCK
    blocks = batch * blocks_per_row
    block_table = torch.arange(blocks, dtype=torch.int32, device="npu").view(batch, blocks_per_row)
    key = torch.ones((blocks, BLOCK, 1, INDEX_DIM), dtype=torch.int8, device="npu")
    key_scale = torch.full((blocks, BLOCK, 1, 1), 0.015, dtype=torch.float16, device="npu")
    query_scale = torch.full((batch, INDEX_HEADS), 0.015, dtype=torch.float16, device="npu")
    weights = torch.ones((batch, INDEX_HEADS), dtype=torch.float16, device="npu")
    qlens = torch.arange(1, batch + 1, dtype=torch.int32, device="npu")
    klens = torch.full((batch,), prefix, dtype=torch.int32, device="npu")
    queries = [torch.full((batch, INDEX_HEADS, INDEX_DIM), 1 + i % 7,
                          dtype=torch.int8, device="npu") for i in range(calls)]
    graph = torch.npu.NPUGraph()
    outputs = []
    with torch.inference_mode(), torch.npu.graph(graph):
        for query in queries:
            outputs.append(torch_npu.npu_quant_lightning_indexer(
                query=query, key=key, weights=weights,
                query_dequant_scale=query_scale,
                key_dequant_scale=key_scale.squeeze(2),
                actual_seq_lengths_query=qlens,
                actual_seq_lengths_key=klens, block_table=block_table,
                query_quant_mode=0, key_quant_mode=0,
                layout_query="TND", layout_key="PA_BSND",
                sparse_count=2048, sparse_mode=3,
            ))
    return graph, outputs[-1], (key, key_scale, queries, outputs, weights,
                                query_scale, qlens, klens, block_table)


def make_dense_graph(batch: int, prefix: int, calls: int = 8,
                     heads: int = HEADS, comm_group=None):
    """The production MLA decode FIA v2 path, with BF16 paged latent KV."""
    blocks_per_row = prefix // BLOCK
    blocks = batch * blocks_per_row
    block_table = torch.arange(blocks, dtype=torch.int32, device="npu").view(batch, blocks_per_row)
    # FIA v2 page layout is [physical block, KV heads, block size, dim].
    key = torch.randn((blocks, 1, BLOCK, LATENT), dtype=torch.bfloat16, device="npu")
    key_rope = torch.randn((blocks, 1, BLOCK, ROPE), dtype=torch.bfloat16, device="npu")
    queries = [torch.randn((batch, heads, 1, LATENT), dtype=torch.bfloat16, device="npu")
               for _ in range(calls)]
    query_ropes = [torch.randn((batch, heads, 1, ROPE), dtype=torch.bfloat16, device="npu")
                   for _ in range(calls)]
    graph = torch.npu.NPUGraph()
    outputs = []
    with torch.inference_mode(), torch.npu.graph(graph):
        for query, query_rope in zip(queries, query_ropes):
            out, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query, key, key, query_rope=query_rope, key_rope=key_rope,
                num_query_heads=heads, num_key_value_heads=1,
                input_layout="BNSD_NBSD", atten_mask=None, sparse_mode=0,
                softmax_scale=(LATENT + ROPE) ** -0.5,
                block_table=block_table, block_size=BLOCK,
                actual_seq_kvlen=[prefix] * batch,
                actual_seq_qlen=None, return_softmax_lse=False,
            )
            outputs.append(out)
            if comm_group is not None:
                payload = out.reshape(-1)[:batch * routed.HIDDEN]
                dist.all_reduce(payload, group=comm_group)
    return graph, outputs[-1], (key, key_rope, queries, query_ropes, outputs, block_table)


def _init_moe(rank: int):
    tp_group0 = dist.new_group(list(range(0, routed.TP)), backend="hccl")
    tp_group1 = dist.new_group(list(range(routed.TP, 2 * routed.TP)), backend="hccl")
    tp_group = tp_group0 if rank < routed.TP else tp_group1
    probe = torch.ones(1, dtype=torch.float32, device="npu")
    dist.all_reduce(probe)
    backend = dist.group.WORLD._get_backend(torch.device("npu"))
    mc2_group_name = backend.get_hccl_comm_name(rank)
    moe_mlp._EXTRA_CTX = SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)
    return tp_group, mc2_group_name, routed.make_weights(1)


def _make_graph(stage: str, batch: int, rank: int, mode: str, moe_state, tp_group):
    heads = 128 if mode == "single" else 16
    comm_group = tp_group if mode == "tp8_comm" else None
    if stage == "indexer":
        graph, output, refs = make_indexer_graph(batch)
        kind = "indexer"
    elif stage == "sparse":
        graph, output, refs, _ = stages.make_sfa_graph(batch, 8, heads=heads,
                                                       comm_group=comm_group)
        kind = "finite"
    elif stage == "dense32":
        graph, output, refs = make_dense_graph(batch, 32768, heads=heads,
                                               comm_group=comm_group)
        kind = "finite"
    elif stage == "dense64":
        graph, output, refs = make_dense_graph(batch, 65536, heads=heads,
                                               comm_group=comm_group)
        kind = "finite"
    else:
        tp_group, mc2_group_name, weights = moe_state
        graph, output, refs, ids, group_list, mask = routed.make_graph(
            batch, 1, rank, tp_group, mc2_group_name, weights
        )
        refs = (refs, ids, group_list, mask)
        kind = "finite"
    return graph, output, refs, kind


def _verify(stage: str, batch: int, rank: int, output, refs, kind: str):
    if kind == "indexer":
        values = output.cpu().reshape(batch, 2048)
        assert bool(((values >= 0) & (values < 32768)).all()), "invalid indexer top-k"
    else:
        assert bool(torch.isfinite(output).all().cpu()), f"nonfinite {stage} output"
    if stage == "moe":
        _, ids, _, mask = refs
        local_ids = ids[torch.tensor(mask, dtype=torch.bool, device="npu")]
        observed = torch.bincount(local_ids.reshape(-1).to(torch.int64),
                                  minlength=routed.TOTAL_EXPERTS).cpu().tolist()
        expected = [0] * routed.TOTAL_EXPERTS
        rows_per_tp_rank = math.ceil(batch / routed.TP)
        for local_index in range(sum(mask)):
            token = rank // routed.TP * batch + rank % routed.TP * rows_per_tp_rank + local_index
            for expert in routed.routed_ids(token):
                expected[expert] += 1
        assert observed == expected, f"router mismatch at rank {rank}, batch {batch}"


def run_point(args, batch: int, rank: int, moe_state, marker: Path):
    tp_group = args.tp_group
    graph, output, refs, kind = _make_graph(args.stage, batch, rank,
                                            args.mode, moe_state, tp_group)
    for _ in range(args.warmup):
        graph.replay()
    torch.npu.synchronize()
    _verify(args.stage, batch, rank, output, refs, kind)

    start = torch.npu.Event(enable_timing=True)
    stop = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(args.pilot_replays):
        graph.replay()
    stop.record()
    stop.synchronize()
    local_graph_ms = start.elapsed_time(stop) / args.pilot_replays
    # Pilot rank maxima can be inflated by one-time HCCL scheduling. Use the
    # fastest measured rank so the steady-state window is at least the target
    # duration even when another rank is slower.
    min_ms = torch.tensor(local_graph_ms, dtype=torch.float32, device="npu")
    dist.all_reduce(min_ms, op=dist.ReduceOp.MIN)
    target_replays = max(2048, math.ceil(args.seconds * 1.1 * 1000 / float(min_ms)))
    target_replays = min(target_replays, args.max_replays)
    dist.barrier()

    if rank == 0:
        _mark(marker, {"stage": args.stage, "batch": batch,
                       "mode": args.mode, "state": "active"})
    dist.barrier()
    start = torch.npu.Event(enable_timing=True)
    stop = torch.npu.Event(enable_timing=True)
    wall_start = time.monotonic()
    start.record()
    for i in range(target_replays):
        graph.replay()
        if (i + 1) % 128 == 0:
            torch.npu.synchronize()
    stop.record()
    stop.synchronize()
    wall_end = time.monotonic()
    dist.barrier()
    if rank == 0:
        _mark(marker, None)
    device_ms = start.elapsed_time(stop)
    record = {
        "stage": args.stage, "batch": batch, "rank": rank,
        "mode": args.mode,
        "status": "pass", "graph_replays": target_replays,
        "pilot_ms_per_graph": local_graph_ms,
        "device_ms_total": device_ms,
        "wall_seconds": wall_end - wall_start,
        "graph_calls": 1 if args.stage == "moe" else 8,
    }
    print(json.dumps(record), flush=True)
    del graph, output, refs
    torch.npu.empty_cache()
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("indexer", "sparse", "moe", "dense32", "dense64"))
    parser.add_argument("--mode", default="tp8_no_comm",
                        choices=("single", "tp8_no_comm", "tp8_comm"))
    parser.add_argument("--batches", nargs="+", type=int, default=BATCHES)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--pilot-replays", type=int, default=32)
    parser.add_argument("--max-replays", type=int, default=500000)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    assert dist.get_world_size() == (1 if args.mode == "single" else 16)
    assert args.mode == "tp8_no_comm" or args.stage in {"sparse", "dense32", "dense64"}
    torch_npu.npu.config.allow_internal_format = True
    enable_custom_op()
    init_device_properties_triton()
    if args.mode == "tp8_comm":
        tp_group0 = dist.new_group(list(range(0, routed.TP)), backend="hccl")
        tp_group1 = dist.new_group(list(range(routed.TP, 2 * routed.TP)), backend="hccl")
        args.tp_group = tp_group0 if rank < routed.TP else tp_group1
    else:
        args.tp_group = None
    moe_state = _init_moe(rank) if args.stage == "moe" else None
    marker = Path(args.marker)
    for batch in args.batches:
        try:
            run_point(args, batch, rank, moe_state, marker)
        except Exception as exc:
            if rank == 0:
                _mark(marker, None)
            print(json.dumps({"stage": args.stage, "batch": batch, "rank": rank,
                              "status": "error", "error": repr(exc)}), flush=True)
            raise
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
