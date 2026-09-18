#!/usr/bin/env python3
"""A3 DP2xTP8 stage microbenchmarks with K dependent calls per ACL graph."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import moe_mlp
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, enable_custom_op


HEADS = 16  # DeepSeek-V3.2: 128 attention heads / TP8 per die.
LATENT = 512
ROPE = 64
TOPK = 2048
BLOCK = 128
HIDDEN = 7168
INTERMEDIATE = 2048
EXPERTS = 16
LAYERS = {"sfa": 61, "expert_balanced": 58, "expert_per_expert": 58}


def make_sfa_graph(batch: int, k: int):
    pages_per_row = TOPK // BLOCK
    pages = batch * pages_per_row
    query = torch.randn(batch, HEADS, LATENT, dtype=torch.bfloat16, device="npu")
    sparse_indices = torch.arange(TOPK, dtype=torch.int32, device="npu").view(1, 1, TOPK).repeat(batch, 1, 1)
    block_table = torch.arange(pages, dtype=torch.int32, device="npu").view(batch, pages_per_row)
    actual_query = torch.arange(1, batch + 1, dtype=torch.int32, device="npu")
    actual_kv = torch.full((batch,), TOPK, dtype=torch.int32, device="npu")
    keys = [torch.randn(pages, BLOCK, 1, LATENT, dtype=torch.bfloat16, device="npu") for _ in range(k)]
    key_ropes = [torch.randn(pages, BLOCK, 1, ROPE, dtype=torch.bfloat16, device="npu") for _ in range(k)]
    query_ropes = [torch.randn(batch, HEADS, ROPE, dtype=torch.bfloat16, device="npu") for _ in range(k)]
    graph = torch.npu.NPUGraph()
    with torch.inference_mode(), torch.npu.graph(graph):
        for layer in range(k):
            result = torch.ops._C_ascend.npu_sparse_flash_attention(
                query=query,
                key=keys[layer],
                value=keys[layer],
                sparse_indices=sparse_indices,
                scale_value=(LATENT + ROPE) ** -0.5,
                sparse_block_size=1,
                block_table=block_table,
                actual_seq_lengths_query=actual_query,
                actual_seq_lengths_kv=actual_kv,
                query_rope=query_ropes[layer],
                key_rope=key_ropes[layer],
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
                return_softmax_lse=False,
            )
            query = result[0] if isinstance(result, tuple) else result
    keepalive = [sparse_indices, block_table, actual_query, actual_kv, keys, key_ropes, query_ropes]
    return graph, query, keepalive, batch


def make_weights(k: int):
    weights = []
    for _ in range(k):
        w1_nd = torch.randint(-8, 8, (EXPERTS, HIDDEN, 2 * INTERMEDIATE), dtype=torch.int8, device="npu")
        w2_nd = torch.randint(-8, 8, (EXPERTS, INTERMEDIATE, HIDDEN), dtype=torch.int8, device="npu")
        w1 = torch_npu.npu_format_cast(w1_nd, ACL_FORMAT_FRACTAL_NZ)
        w2 = torch_npu.npu_format_cast(w2_nd, ACL_FORMAT_FRACTAL_NZ)
        w1_scale = torch.full((EXPERTS, 2 * INTERMEDIATE), 0.01, dtype=torch.float32, device="npu")
        w2_scale = torch.full((EXPERTS, HIDDEN), 0.01, dtype=torch.bfloat16, device="npu")
        weights.append((w1, w1_scale, w2, w2_scale))
        del w1_nd, w2_nd
    torch.npu.empty_cache()
    return weights


def make_expert_graph(batch: int, k: int, per_expert: bool, weights):
    rows = batch * EXPERTS if per_expert else batch
    allocated_rows = max(128, math.ceil(rows / 128) * 128)
    inputs = []
    scales = []
    for _ in range(k):
        bf16_input = torch.randn((allocated_rows, HIDDEN), dtype=torch.bfloat16, device="npu")
        quantized, dynamic_scale = torch_npu.npu_dynamic_quant(bf16_input, dst_type=torch.int8)
        inputs.append(quantized)
        scales.append(dynamic_scale)
        del bf16_input
    base, extra = divmod(rows, EXPERTS)
    counts = [base + int(i < extra) for i in range(EXPERTS)]
    running = 0
    offsets = []
    for count in counts:
        running += count
        offsets.append(running)
    group_list = torch.tensor(offsets, dtype=torch.int64, device="npu")
    graph = torch.npu.NPUGraph()
    with torch.inference_mode(), torch.npu.graph(graph):
        for input_hidden, dynamic_scale, (w1, w1_scale, w2, w2_scale) in zip(inputs, scales, weights):
            result = moe_mlp.quant_apply_mlp(
                hidden_states=input_hidden,
                w1=[w1], w1_scale=[w1_scale],
                w2=[w2], w2_scale=[w2_scale],
                group_list=group_list,
                group_list_type=0,
                dynamic_scale=dynamic_scale,
                fusion=False,
                act_quant_type=torch.int8,
                weight_quant_type=torch.int8,
                use_bf16=True,
            )
            output = result[0] if isinstance(result, tuple) else result
    return graph, output, [inputs, scales, group_list, weights], rows, allocated_rows, counts


def measure(graph, warmup: int, iterations: int, repeats: int):
    for _ in range(warmup):
        graph.replay()
    torch.npu.synchronize()
    device_ms = []
    wall_ms = []
    for _ in range(repeats):
        dist.barrier()
        start = torch.npu.Event(enable_timing=True)
        stop = torch.npu.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        for _ in range(iterations):
            graph.replay()
        stop.record()
        stop.synchronize()
        device_ms.append(start.elapsed_time(stop) / iterations)
        wall_ms.append((time.perf_counter() - wall_start) * 1000 / iterations)
    return device_ms, wall_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=LAYERS, required=True)
    ap.add_argument("--batches", nargs="+", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    torch_npu.npu.config.allow_internal_format = True
    enable_custom_op()
    init_device_properties_triton()
    moe_mlp._EXTRA_CTX = SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)
    weights = make_weights(args.k) if args.stage.startswith("expert") else None
    for batch in args.batches:
        try:
            if args.stage == "sfa":
                graph, output, keepalive, rows = make_sfa_graph(batch, args.k)
                counts = None
                expected_shape = (batch, HEADS, LATENT)
            else:
                graph, output, keepalive, rows, allocated_rows, counts = make_expert_graph(
                    batch, args.k, args.stage == "expert_per_expert", weights
                )
                expected_shape = (allocated_rows, HIDDEN)
            device, wall = measure(graph, args.warmup, args.iterations, args.repeats)
            active_output = output if args.stage == "sfa" else output[:rows]
            if tuple(output.shape) != expected_shape or not bool(torch.isfinite(active_output).all().cpu()):
                raise AssertionError(f"invalid output {tuple(output.shape)}; expected {expected_shape}")
            ms = statistics.median(device)
            record = {
                "status": "pass", "stage": args.stage, "rank": rank, "device": local_rank,
                "local_batch": batch, "global_batch": 2 * batch,
                "expert_rows_per_rank": rows if counts is not None else None,
                "expert_allocated_rows_per_rank": allocated_rows if counts is not None else None,
                "expert_counts": counts, "k": args.k,
                "mode": "aclgraph_dependent_calls" if args.stage == "sfa"
                    else "aclgraph_serial_padded_int8_distinct_weights",
                "warmup": args.warmup, "iterations": args.iterations, "repeats": args.repeats,
                "device_ms_per_graph_samples": device, "wall_ms_per_graph_samples": wall,
                "device_ms_per_graph_median": ms,
                "device_ms_per_call_median": ms / args.k,
                "model_stage_tokens_per_s": (2 * batch * args.k * 1000 / (LAYERS[args.stage] * ms))
                    if args.stage != "expert_per_expert" else None,
                "expert_rows_per_s_per_rank": rows * args.k * 1000 / ms
                    if counts is not None else None,
            }
            print(json.dumps(record), flush=True)
        except Exception as exc:
            print(json.dumps({"status": "error", "stage": args.stage, "rank": rank,
                              "local_batch": batch, "k": args.k, "error": repr(exc)}), flush=True)
            raise
        finally:
            if "graph" in locals():
                del graph, output, keepalive
            torch.npu.empty_cache()
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
