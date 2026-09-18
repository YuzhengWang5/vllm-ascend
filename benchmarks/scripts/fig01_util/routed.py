#!/usr/bin/env python3
"""A3 routed MoE microbenchmark: gate, MC2 dispatch, W8A8 MLP, combine, TP gather.

Uses the production expert selector, MC2 operators, and quant_apply_mlp.  The
gate input is synthetic but selects a deterministic near-uniform set of experts.
"""

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
import torch.nn.functional as F
import torch_npu

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import moe_mlp
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, enable_custom_op


HIDDEN = 7168
INTERMEDIATE = 2048
TOTAL_EXPERTS = 256
LOCAL_EXPERTS = 16
TOPK = 8
TP = 8
DP = 2
EP = 16
LAYERS = 58


def routed_ids(token: int) -> list[int]:
    ids = []
    for j in range(4):
        group = (token + j) % 8
        local = ((token // 8) * 2 + j) % 16
        ids.extend([group * 32 + local, group * 32 + 16 + (local + 8) % 16])
    assert len(set(ids)) == TOPK
    return ids


def make_gate_and_input(batch: int, rank: int):
    dp_rank = rank // TP
    full = torch.zeros((batch, HIDDEN), dtype=torch.bfloat16, device="npu")
    gate = torch.full((TOTAL_EXPERTS, HIDDEN), -8.0, dtype=torch.float32, device="npu")
    for i in range(batch):
        token = dp_rank * batch + i
        full[i, token] = 1.0
        for expert in routed_ids(token):
            gate[expert, token] = 8.0
    bias = torch.zeros((TOTAL_EXPERTS,), dtype=torch.float32, device="npu")
    return full, gate, bias


def make_weights(k: int):
    weights = []
    for _ in range(k):
        w1_nd = torch.randint(-8, 8, (LOCAL_EXPERTS, HIDDEN, 2 * INTERMEDIATE), dtype=torch.int8, device="npu")
        w2_nd = torch.randint(-8, 8, (LOCAL_EXPERTS, INTERMEDIATE, HIDDEN), dtype=torch.int8, device="npu")
        w1 = torch_npu.npu_format_cast(w1_nd, ACL_FORMAT_FRACTAL_NZ)
        w2 = torch_npu.npu_format_cast(w2_nd, ACL_FORMAT_FRACTAL_NZ)
        w1_scale = torch.full((LOCAL_EXPERTS, 2 * INTERMEDIATE), 0.01, dtype=torch.float32, device="npu")
        w2_scale = torch.full((LOCAL_EXPERTS, HIDDEN), 0.01, dtype=torch.bfloat16, device="npu")
        weights.append((w1, w1_scale, w2, w2_scale))
        del w1_nd, w2_nd
    torch.npu.empty_cache()
    return weights


def one_layer(x_full, batch, tp_rank, rank, mask, gate, bias, weights, mc2_group_name, tp_group):
    padded_batch = math.ceil(batch / TP) * TP
    local_rows = padded_batch // TP
    logits = F.linear(x_full.float(), gate)
    if padded_batch != batch:
        x_full = F.pad(x_full, (0, 0, 0, padded_batch - batch))
        logits = F.pad(logits, (0, 0, 0, padded_batch - batch))
    x_local = torch.tensor_split(x_full, TP, dim=0)[tp_rank].contiguous()
    logits_local = torch.tensor_split(logits, TP, dim=0)[tp_rank].contiguous()
    scores, ids = select_experts(
        hidden_states=x_local, router_logits=logits_local,
        top_k=TOPK, use_grouped_topk=True, renormalize=True,
        topk_group=4, num_expert_group=8, scoring_func="sigmoid",
        routed_scaling_factor=2.5, e_score_correction_bias=bias,
    )
    (expanded, dynamic_scale, assist, group_list, ep_recv, tp_recv, expand_scales) = (
        torch_npu.npu_moe_distribute_dispatch_v2(
            x=x_local, expert_ids=ids,
            expert_shard_type=0, shared_expert_rank_num=0,
            moe_expert_num=TOTAL_EXPERTS, global_bs=0,
            expert_token_nums_type=0, x_active_mask=mask,
            scales=None, quant_mode=2,
            group_ep=mc2_group_name, ep_world_size=EP, ep_rank_id=rank,
            group_tp=mc2_group_name, tp_world_size=1, tp_rank_id=0,
            comm_alg="",
        )[:7]
    )
    w1, w1_scale, w2, w2_scale = weights
    result = moe_mlp.quant_apply_mlp(
        hidden_states=expanded,
        w1=[w1], w1_scale=[w1_scale],
        w2=[w2], w2_scale=[w2_scale],
        group_list=group_list, group_list_type=0,
        dynamic_scale=dynamic_scale,
        fusion=False,
        act_quant_type=torch.int8, weight_quant_type=torch.int8,
        use_bf16=True,
    )
    mlp_out = result[0] if isinstance(result, tuple) else result
    # The production W8A8 path passes an empty TP count tensor at combine.
    tp_recv = torch.empty(1, dtype=torch.int32, device="npu")
    combined = torch_npu.npu_moe_distribute_combine_v2(
        expand_x=mlp_out, expert_ids=ids, expert_scales=scores.to(torch.float32),
        expert_shard_type=0, shared_expert_rank_num=0,
        moe_expert_num=TOTAL_EXPERTS, global_bs=0,
        x_active_mask=mask,
        ep_send_counts=ep_recv, group_ep=mc2_group_name,
        ep_world_size=EP, ep_rank_id=rank,
        expand_scales=expand_scales, comm_quant_mode=0, comm_alg="",
        assist_info_for_combine=assist,
        tp_send_counts=tp_recv, group_tp=mc2_group_name,
        tp_world_size=1, tp_rank_id=0,
    )
    gather_parts = [torch.empty((local_rows, HIDDEN), dtype=combined.dtype, device="npu") for _ in range(TP)]
    dist.all_gather(gather_parts, combined, group=tp_group)
    gathered = torch.cat(gather_parts, dim=0)[:batch]
    return gathered, ids, group_list


def make_graph(batch, k, rank, tp_group, mc2_group_name, weights):
    tp_rank = rank % TP
    padded_batch = math.ceil(batch / TP) * TP
    local_rows = padded_batch // TP
    mask_values = [tp_rank * local_rows + i < batch for i in range(local_rows)]
    mask = torch.tensor(mask_values, dtype=torch.bool, device="npu")
    base, gate, bias = make_gate_and_input(batch, rank)
    inputs = [base.clone() for _ in range(k)]
    graph = torch.npu.NPUGraph()
    outputs = []
    with torch.inference_mode(), torch.npu.graph(graph):
        for layer in range(k):
            y, ids, group_list = one_layer(
                inputs[layer], batch, tp_rank, rank, mask, gate, bias,
                weights[layer], mc2_group_name, tp_group
            )
            outputs.append(y)
    # ACL Graph replays captured calls in stream order. Retain every output so
    # all K routed stages are observable without adding a synthetic pointwise op.
    return graph, outputs[-1], [inputs, outputs, gate, bias, mask, weights], ids, group_list, mask_values


def measure(graph, warmup, iterations, repeats):
    for _ in range(warmup):
        graph.replay()
    torch.npu.synchronize()
    device, wall = [], []
    for _ in range(repeats):
        dist.barrier()
        start = torch.npu.Event(enable_timing=True)
        stop = torch.npu.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        for _ in range(iterations):
            graph.replay()
        stop.record()
        stop.synchronize()
        device.append(start.elapsed_time(stop) / iterations)
        wall.append((time.perf_counter() - t0) * 1000 / iterations)
    return device, wall


def main():
    ap = argparse.ArgumentParser()
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
    assert dist.get_world_size() == EP
    tp_group0 = dist.new_group(list(range(0, TP)), backend="hccl")
    tp_group1 = dist.new_group(list(range(TP, 2 * TP)), backend="hccl")
    tp_group = tp_group0 if rank < TP else tp_group1
    probe = torch.ones(1, dtype=torch.float32, device="npu")
    dist.all_reduce(probe)
    backend = dist.group.WORLD._get_backend(torch.device("npu"))
    mc2_group_name = backend.get_hccl_comm_name(rank)
    torch_npu.npu.config.allow_internal_format = True
    enable_custom_op()
    init_device_properties_triton()
    moe_mlp._EXTRA_CTX = SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)
    weights = make_weights(args.k)
    for batch in args.batches:
        try:
            graph, output, keepalive, ids, group_list, mask = make_graph(
                batch, args.k, rank, tp_group, mc2_group_name, weights
            )
            device, wall = measure(graph, args.warmup, args.iterations, args.repeats)
            if tuple(output.shape) != (batch, HIDDEN) or not bool(torch.isfinite(output).all().cpu()):
                raise AssertionError(f"invalid routed MoE output {tuple(output.shape)}")
            local_ids = ids[torch.tensor(mask, dtype=torch.bool, device="npu")]
            observed = torch.bincount(local_ids.reshape(-1).to(torch.int64), minlength=TOTAL_EXPERTS)
            expected = [0] * TOTAL_EXPERTS
            local_rows = math.ceil(batch / TP)
            for local_index in range(sum(mask)):
                token = rank // TP * batch + rank % TP * local_rows + local_index
                for expert in routed_ids(token):
                    expected[expert] += 1
            observed_cpu = observed.cpu().tolist()
            if observed_cpu != expected:
                raise AssertionError(f"router mismatch, rank {rank}, batch {batch}")
            ms = statistics.median(device)
            print(json.dumps({
                "status": "pass", "stage": "routed_moe", "rank": rank,
                "device": local_rank, "local_batch": batch, "global_batch": DP * batch,
                "k": args.k, "mode": "aclgraph_serial_router_mc2_mlp_combine_tp_gather",
                "local_active_tokens": sum(mask),
                "local_expert_assignments": sum(observed_cpu),
                "local_expert_counts": observed_cpu,
                "group_list_last_layer": group_list.cpu().tolist(),
                "warmup": args.warmup, "iterations": args.iterations, "repeats": args.repeats,
                "device_ms_per_graph_samples": device,
                "wall_ms_per_graph_samples": wall,
                "device_ms_per_graph_median": ms,
                "device_ms_per_call_median": ms / args.k,
                "model_stage_tokens_per_s": DP * batch * args.k * 1000 / (LAYERS * ms),
            }), flush=True)
        except Exception as exc:
            print(json.dumps({"status": "error", "stage": "routed_moe", "rank": rank,
                              "local_batch": batch, "k": args.k, "error": repr(exc)}), flush=True)
            raise
        finally:
            if "graph" in locals():
                del graph, output, keepalive, ids, group_list
            torch.npu.empty_cache()
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
