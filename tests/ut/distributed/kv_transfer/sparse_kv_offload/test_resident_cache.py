import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.resident_cache import (
    SERVICE_RESIDENT_HIT,
    SERVICE_RESIDENT_INVALID,
    direct_mapped_miss_sources,
)


def test_direct_mapped_miss_sources_tracks_hits_and_updates():
    block_table = torch.tensor([[5, 9, -1], [2, 7, -1]], dtype=torch.int32)
    topk = torch.tensor(
        [[[0, 3, 4, -1]], [[1, 6, 7, -1]]],
        dtype=torch.int32,
    )
    slot_mapping = torch.tensor([99, 99], dtype=torch.int32)
    resident = torch.full((2, 4), SERVICE_RESIDENT_HIT, dtype=torch.int32)

    first = direct_mapped_miss_sources(
        topk, block_table, slot_mapping, resident, block_size=4
    )
    assert first.tolist() == [
        [20, 23, 36, SERVICE_RESIDENT_INVALID],
        [9, 30, 31, SERVICE_RESIDENT_INVALID],
    ]

    second = direct_mapped_miss_sources(
        topk, block_table, slot_mapping, resident, block_size=4
    )
    assert second.tolist() == [
        [SERVICE_RESIDENT_HIT] * 3 + [SERVICE_RESIDENT_INVALID],
        [SERVICE_RESIDENT_HIT] * 3 + [SERVICE_RESIDENT_INVALID],
    ]

    reordered = topk.clone()
    reordered[0, 0, :2] = torch.tensor([3, 0], dtype=torch.int32)
    third = direct_mapped_miss_sources(
        reordered, block_table, slot_mapping, resident, block_size=4
    )
    assert third[0].tolist() == [23, 20, SERVICE_RESIDENT_HIT, SERVICE_RESIDENT_INVALID]


def test_direct_mapped_miss_sources_reloads_current_decode_slot():
    block_table = torch.tensor([[5, 9]], dtype=torch.int32)
    topk = torch.tensor([[0, 3, 4]], dtype=torch.int32)
    resident = torch.full((1, 3), SERVICE_RESIDENT_HIT, dtype=torch.int32)
    slot_mapping = torch.tensor([23], dtype=torch.int32)

    direct_mapped_miss_sources(topk, block_table, slot_mapping, resident, 4)
    second = direct_mapped_miss_sources(
        topk, block_table, slot_mapping, resident, 4
    )
    assert second.tolist() == [[SERVICE_RESIDENT_HIT, 23, SERVICE_RESIDENT_HIT]]
