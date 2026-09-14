"""Device-side resident-cache helpers shared by decoder and indexer service."""

from __future__ import annotations

import torch


# The service response reuses the original int32 top-k tensor footprint.
# Non-negative values are physical source slots which must be reloaded.
SERVICE_RESIDENT_HIT = -1
SERVICE_RESIDENT_INVALID = -2


def direct_mapped_miss_sources(
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    resident_sources: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Translate logical top-k tokens to physical MLA slots and encode misses.

    Each top-k position owns one fixed resident-buffer position.  A changed
    physical source is returned as a miss; an unchanged source is a hit.  The
    current decode token is always reloaded because its host KV contents were
    just updated even when its physical slot is unchanged.
    """
    logical_tokens = topk_indices.reshape(resident_sources.shape).to(torch.int64)
    safe_tokens = logical_tokens.clamp_min(0)
    logical_blocks = torch.div(safe_tokens, block_size, rounding_mode="floor")
    block_columns = block_table.shape[1]
    safe_blocks = logical_blocks.clamp_max(block_columns - 1)
    physical_blocks = torch.gather(block_table, 1, safe_blocks)
    offsets = torch.remainder(safe_tokens, block_size)
    physical_sources_i64 = physical_blocks.to(torch.int64) * block_size + offsets
    valid = (
        (logical_tokens >= 0)
        & (logical_blocks < block_columns)
        & (physical_blocks >= 0)
    )
    physical_sources = torch.where(
        valid,
        physical_sources_i64,
        torch.full_like(physical_sources_i64, SERVICE_RESIDENT_HIT),
    ).to(torch.int32)
    changed = physical_sources != resident_sources
    current_token_updated = valid & (
        physical_sources == slot_mapping.reshape(-1, 1).to(torch.int32)
    )
    misses = torch.where(
        valid,
        torch.where(
            changed | current_token_updated,
            physical_sources,
            torch.full_like(physical_sources, SERVICE_RESIDENT_HIT),
        ),
        torch.full_like(physical_sources, SERVICE_RESIDENT_INVALID),
    )
    resident_sources.copy_(physical_sources)
    return misses
