#include "kernel_operator.h"
#include "smem_shm_aicore_base_api.h"

namespace {

constexpr uint64_t kRequestDoorbellOffset = 0;
constexpr uint64_t kServiceSeenOffset = 32;
constexpr uint64_t kProfileOffset = 64;
constexpr uint64_t kTpFanoutDoorbellOffset = 128;
constexpr uint64_t kTpFanoutAckOffset = 160;
constexpr uint64_t kPayloadOffset = 4096;
constexpr uint64_t kMaxPayloadBytes = 1UL << 20;
constexpr uint32_t kUbPayloadBytes = 64U << 10;
constexpr uint64_t kUbControlOffset = kUbPayloadBytes;
constexpr uint32_t kCopyEvent = 0;
constexpr uint32_t kProfileRowBytes = 96;
// A3 exposes two AIV/vector subcores per one of its 24 AI cores.
constexpr uint32_t kResidentBlocks = 48;
constexpr int32_t kResidentHit = -1;
constexpr int32_t kResidentInvalid = -2;
constexpr uint32_t kDescriptorTile = 256;

template <typename T>
__aicore__ inline AscendC::LocalTensor<T> LocalAt(uint64_t byte_offset) {
  AscendC::LocalTensor<T> tensor;
  tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
  tensor.address_.bufferAddr = byte_offset;
  return tensor;
}

template <typename T>
__aicore__ inline void CopyIn(AscendC::LocalTensor<T> dst,
                              AscendC::GlobalTensor<T> src,
                              uint32_t bytes) {
  AscendC::DataCopyExtParams params;
  params.blockCount = 1;
  params.blockLen = bytes;
  params.srcStride = 0;
  params.dstStride = 0;
  AscendC::DataCopyPadExtParams<T> pad;
  pad.isPad = false;
  AscendC::DataCopyPad(dst, src, params, pad);
}

template <typename T>
__aicore__ inline void CopyOut(AscendC::GlobalTensor<T> dst,
                               AscendC::LocalTensor<T> src,
                               uint32_t bytes) {
  AscendC::DataCopyExtParams params;
  params.blockCount = 1;
  params.blockLen = bytes;
  params.srcStride = 0;
  params.dstStride = 0;
  AscendC::DataCopyPad(dst, src, params);
}

__aicore__ inline void ServiceResidentUpdate(
    uint64_t count, GM_ADDR topk_addr, GM_ADDR block_table_addr,
    GM_ADDR slot_mapping_addr, GM_ADDR resident_sources_addr,
    GM_ADDR response_addr, uint32_t topk_size, uint32_t block_table_cols,
    uint32_t block_size) {
  AscendC::GlobalTensor<int32_t> topk;
  AscendC::GlobalTensor<int32_t> block_table;
  AscendC::GlobalTensor<int32_t> slot_mapping;
  AscendC::GlobalTensor<int32_t> resident_sources;
  AscendC::GlobalTensor<int32_t> response;
  topk.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(topk_addr), count);
  block_table.SetGlobalBuffer(
      reinterpret_cast<__gm__ int32_t*>(block_table_addr));
  slot_mapping.SetGlobalBuffer(
      reinterpret_cast<__gm__ int32_t*>(slot_mapping_addr));
  resident_sources.SetGlobalBuffer(
      reinterpret_cast<__gm__ int32_t*>(resident_sources_addr), count);
  response.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(response_addr),
                           count);
  const uint32_t batch = static_cast<uint32_t>(count / topk_size);
  const uint32_t topk_bytes = topk_size * sizeof(int32_t);
  const uint32_t table_bytes = block_table_cols * sizeof(int32_t);
  const uint32_t table_offset = (topk_bytes + 31U) & ~31U;
  const uint32_t resident_offset = table_offset + ((table_bytes + 31U) & ~31U);
  const uint32_t response_offset = resident_offset + ((topk_bytes + 31U) & ~31U);
  const uint32_t slots_offset = response_offset + ((topk_bytes + 31U) & ~31U);
  auto topk_local = LocalAt<int32_t>(0);
  auto table_local = LocalAt<int32_t>(table_offset);
  auto resident_local = LocalAt<int32_t>(resident_offset);
  auto response_local = LocalAt<int32_t>(response_offset);
  auto slots_local = LocalAt<int32_t>(slots_offset);

  CopyIn(slots_local, slot_mapping, batch * sizeof(int32_t));
  AscendC::PipeBarrier<PIPE_ALL>();
  const uint64_t first = AscendC::GetBlockIdx();
  const uint64_t stride = AscendC::GetBlockNum();
  for (uint64_t row = first; row < batch; row += stride) {
    const uint64_t topk_base = row * topk_size;
    const uint64_t table_base = row * block_table_cols;
    CopyIn(topk_local, topk[topk_base], topk_bytes);
    CopyIn(table_local, block_table[table_base], table_bytes);
    CopyIn(resident_local, resident_sources[topk_base], topk_bytes);
    AscendC::PipeBarrier<PIPE_ALL>();
    const int32_t current_slot = slots_local.GetValue(row);
    for (uint32_t position = 0; position < topk_size; ++position) {
      const int32_t logical = topk_local.GetValue(position);
      int32_t physical = kResidentInvalid;
      if (logical >= 0) {
        const uint32_t logical_block =
            static_cast<uint32_t>(logical) / block_size;
        if (logical_block < block_table_cols) {
          const int32_t physical_block = table_local.GetValue(logical_block);
          if (physical_block >= 0) {
            physical = physical_block * static_cast<int32_t>(block_size) +
                       logical % static_cast<int32_t>(block_size);
          }
        }
      }
      if (physical == kResidentInvalid) {
        response_local.SetValue(position, kResidentInvalid);
        resident_local.SetValue(position, kResidentHit);
      } else {
        response_local.SetValue(
            position,
            (resident_local.GetValue(position) != physical ||
             physical == current_slot)
                ? physical
                : kResidentHit);
        resident_local.SetValue(position, physical);
      }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    CopyOut(response[topk_base], response_local, topk_bytes);
    CopyOut(resident_sources[topk_base], resident_local, topk_bytes);
    AscendC::PipeBarrier<PIPE_ALL>();
  }
}

__aicore__ inline void DecoderResidentDescriptors(
    uint64_t count, GM_ADDR encoded_sources_addr, GM_ADDR gvas_addr,
    GM_ADDR addrs_addr, GM_ADDR sizes_addr, GM_ADDR descriptor_count_addr,
    GM_ADDR current_slots_addr,
    uint64_t gva_k_base, uint64_t gva_v_base, uint64_t addr_k_base,
    uint64_t addr_v_base, uint32_t token_bytes_k, uint32_t token_bytes_v,
    uint32_t topk_size, uint32_t resident_capacity) {
  AscendC::GlobalTensor<int32_t> encoded_sources;
  AscendC::GlobalTensor<int64_t> gvas;
  AscendC::GlobalTensor<int64_t> addrs;
  AscendC::GlobalTensor<int32_t> sizes;
  AscendC::GlobalTensor<int32_t> descriptor_count;
  AscendC::GlobalTensor<int32_t> current_slots;
  encoded_sources.SetGlobalBuffer(
      reinterpret_cast<__gm__ int32_t*>(encoded_sources_addr), count);
  gvas.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(gvas_addr),
                       count * 2);
  addrs.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(addrs_addr),
                        count * 2);
  sizes.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sizes_addr),
                        count * 2);
  descriptor_count.SetGlobalBuffer(
      reinterpret_cast<__gm__ int32_t*>(descriptor_count_addr), 1);
  current_slots.SetGlobalBuffer(
      reinterpret_cast<__gm__ int32_t*>(current_slots_addr), count);
  constexpr uint32_t kEncodedOffset = 0;
  constexpr uint32_t kCurrentOffset = 8192;
  constexpr uint32_t kPrefixOffset = 16384;
  constexpr uint32_t kCountOffset = 16640;
  constexpr uint32_t kGvasOffset = 16896;
  constexpr uint32_t kAddrsOffset = 20992;
  constexpr uint32_t kSizesOffset = 25088;
  auto encoded_local = LocalAt<int32_t>(kEncodedOffset);
  auto current_local = LocalAt<int32_t>(kCurrentOffset);
  auto prefix_local = LocalAt<int32_t>(kPrefixOffset);
  auto count_local = LocalAt<int32_t>(kCountOffset);
  auto gvas_local = LocalAt<int64_t>(kGvasOffset);
  auto addrs_local = LocalAt<int64_t>(kAddrsOffset);
  auto sizes_local = LocalAt<int32_t>(kSizesOffset);
  const uint32_t row = AscendC::GetBlockIdx();
  const uint32_t batch = static_cast<uint32_t>(count / topk_size);

  // Phase 1: one AIV per request row counts misses.  The beginning of sizes
  // is temporary scratch; sparse_copy will only observe it after phase 3.
  if (row < batch) {
    CopyIn(encoded_local, encoded_sources[static_cast<uint64_t>(row) * topk_size],
           topk_size * sizeof(int32_t));
    AscendC::PipeBarrier<PIPE_ALL>();
    int32_t row_misses = 0;
    for (uint32_t position = 0; position < topk_size; ++position) {
      if (encoded_local.GetValue(position) >= 0) {
        ++row_misses;
      }
    }
    count_local.SetValue(0, row_misses);
    AscendC::PipeBarrier<PIPE_ALL>();
    CopyOut(sizes[row], count_local, sizeof(int32_t));
  }
  AscendC::SyncAll<true>();

  // Phase 2: a single AIV computes an exclusive prefix over at most 48 rows.
  if (row == 0) {
    CopyIn(prefix_local, sizes, batch * sizeof(int32_t));
    AscendC::PipeBarrier<PIPE_ALL>();
    int32_t total_misses = 0;
    for (uint32_t index = 0; index < batch; ++index) {
      const int32_t row_misses = prefix_local.GetValue(index);
      prefix_local.SetValue(index, total_misses);
      total_misses += row_misses;
    }
    count_local.SetValue(0, total_misses * 2);
    AscendC::PipeBarrier<PIPE_ALL>();
    CopyOut(sizes, prefix_local, batch * sizeof(int32_t));
    CopyOut(descriptor_count, count_local, sizeof(int32_t));
  }
  AscendC::SyncAll<true>();

  // Every active row reads its prefix before anyone overwrites sizes scratch.
  int32_t row_prefix = 0;
  if (row < batch) {
    CopyIn(prefix_local, sizes, batch * sizeof(int32_t));
    AscendC::PipeBarrier<PIPE_ALL>();
    row_prefix = prefix_local.GetValue(row);
  }
  AscendC::SyncAll<true>();

  // Phase 3: compact only misses.  K/V descriptors are interleaved; ordering
  // is irrelevant to sparse_copy as long as the three arrays stay aligned.
  if (row < batch) {
    int32_t emitted = 0;
    for (uint32_t tile_base = 0; tile_base < topk_size;
         tile_base += kDescriptorTile) {
      const uint32_t tile_count =
          topk_size - tile_base > kDescriptorTile
              ? kDescriptorTile
              : topk_size - tile_base;
      uint32_t tile_misses = 0;
      for (uint32_t local = 0; local < tile_count; ++local) {
        const uint32_t position = tile_base + local;
        const int32_t encoded = encoded_local.GetValue(position);
        current_local.SetValue(
            local, encoded == kResidentInvalid
                       ? kResidentHit
                       : static_cast<int32_t>(position));
        if (encoded < 0) {
          continue;
        }
        const uint64_t source = static_cast<uint64_t>(encoded);
        const uint64_t destination =
            static_cast<uint64_t>(row) * resident_capacity + position;
        const uint32_t out = tile_misses * 2;
        gvas_local.SetValue(
            out, static_cast<int64_t>(gva_k_base + source * token_bytes_k));
        gvas_local.SetValue(
            out + 1,
            static_cast<int64_t>(gva_v_base + source * token_bytes_v));
        addrs_local.SetValue(
            out,
            static_cast<int64_t>(addr_k_base + destination * token_bytes_k));
        addrs_local.SetValue(
            out + 1,
            static_cast<int64_t>(addr_v_base + destination * token_bytes_v));
        sizes_local.SetValue(out, static_cast<int32_t>(token_bytes_k));
        sizes_local.SetValue(out + 1, static_cast<int32_t>(token_bytes_v));
        ++tile_misses;
      }
      AscendC::PipeBarrier<PIPE_ALL>();
      CopyOut(current_slots[static_cast<uint64_t>(row) * topk_size + tile_base],
              current_local, tile_count * sizeof(int32_t));
      if (tile_misses > 0) {
        const uint64_t descriptor_base =
            static_cast<uint64_t>(row_prefix + emitted) * 2;
        const uint32_t descriptor_bytes64 =
            tile_misses * 2 * sizeof(int64_t);
        const uint32_t descriptor_bytes32 =
            tile_misses * 2 * sizeof(int32_t);
        CopyOut(gvas[descriptor_base], gvas_local, descriptor_bytes64);
        CopyOut(addrs[descriptor_base], addrs_local, descriptor_bytes64);
        CopyOut(sizes[descriptor_base], sizes_local, descriptor_bytes32);
        emitted += tile_misses;
      }
      AscendC::PipeBarrier<PIPE_ALL>();
    }
  }
}

__aicore__ inline uint32_t Align32(uint32_t bytes) {
  return (bytes + 31U) & ~31U;
}

__aicore__ inline __ubuf__ uint8_t* PayloadUb() {
  return reinterpret_cast<__ubuf__ uint8_t*>(0);
}

__aicore__ inline __ubuf__ uint32_t* ControlUb() {
  return reinterpret_cast<__ubuf__ uint32_t*>(kUbControlOffset);
}

__aicore__ inline uint64_t ReadCycle() {
  pipe_barrier(PIPE_ALL);
  return AscendC::GetSystemCycle();
}

__aicore__ inline void CopyGmToGm(__gm__ uint8_t* dst,
                                  __gm__ uint8_t* src,
                                  uint32_t logical_bytes) {
  if (logical_bytes == 0) {
    return;
  }
  const uint32_t bytes = Align32(logical_bytes);
  auto ub = PayloadUb();
  uint32_t offset = 0;
  AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(kCopyEvent);
  while (offset < bytes) {
    const uint32_t remain = bytes - offset;
    const uint32_t chunk = remain > kUbPayloadBytes ? kUbPayloadBytes : remain;
    smem_shm_copy_gm2ub(ub, src + offset, chunk);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(kCopyEvent);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(kCopyEvent);
    smem_shm_copy_ub2gm(dst + offset, ub, chunk);
    offset += chunk;
    if (offset < bytes) {
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kCopyEvent);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kCopyEvent);
    }
  }
  AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kCopyEvent);
}

__aicore__ inline uint32_t ReadSequence(__gm__ uint32_t* doorbell) {
  auto ub = ControlUb();
  AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(kCopyEvent);
  smem_shm_copy_gm2ub(ub, doorbell, 32U);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(kCopyEvent);
  AscendC::LocalTensor<uint32_t> control;
  control.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
  control.address_.bufferAddr = reinterpret_cast<uint64_t>(ub);
  return control.GetValue(0);
}

__aicore__ inline uint32_t ReadControlWord(__gm__ uint32_t* doorbell,
                                           uint32_t index) {
  auto ub = ControlUb();
  AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(kCopyEvent);
  smem_shm_copy_gm2ub(ub, doorbell, 32U);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(kCopyEvent);
  AscendC::LocalTensor<uint32_t> control;
  control.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
  control.address_.bufferAddr = reinterpret_cast<uint64_t>(ub);
  return control.GetValue(index);
}

__aicore__ inline void WriteSequence(__gm__ uint32_t* doorbell,
                                     uint32_t sequence) {
  auto ub = ControlUb();
  AscendC::LocalTensor<uint32_t> control;
  control.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
  control.address_.bufferAddr = reinterpret_cast<uint64_t>(ub);
  control.SetValue(0, sequence);
  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(kCopyEvent);
  smem_shm_copy_ub2gm(doorbell, ub, 32U);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kCopyEvent);
}

__aicore__ inline void WriteRequestControl(__gm__ uint32_t* doorbell,
                                           uint32_t sequence,
                                           uint32_t request_bytes) {
  auto ub = ControlUb();
  AscendC::LocalTensor<uint32_t> control;
  control.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
  control.address_.bufferAddr = reinterpret_cast<uint64_t>(ub);
  control.SetValue(0, sequence);
  control.SetValue(1, request_bytes);
  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(kCopyEvent);
  smem_shm_copy_ub2gm(doorbell, ub, 32U);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kCopyEvent);
}

__aicore__ inline void WaitSequence(__gm__ uint32_t* doorbell,
                                    uint32_t expected) {
  while (ReadSequence(doorbell) != expected) {
  }
}

__aicore__ inline void WriteDecoderTrace(__gm__ uint8_t* decoder_base,
                                         uint64_t start,
                                         uint64_t request_sent,
                                         uint64_t response_ready,
                                         uint64_t response_copied) {
  auto ub = reinterpret_cast<__ubuf__ uint64_t*>(ControlUb());
  AscendC::LocalTensor<uint64_t> trace;
  trace.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
  trace.address_.bufferAddr = reinterpret_cast<uint64_t>(ub);
  trace.SetValue(0, start);
  trace.SetValue(1, request_sent);
  trace.SetValue(2, response_ready);
  trace.SetValue(3, response_copied);
  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(kCopyEvent);
  smem_shm_copy_ub2gm(
      decoder_base + kProfileOffset,
      reinterpret_cast<__ubuf__ uint8_t*>(ub), 32U);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kCopyEvent);
}

__aicore__ inline void WriteServiceReceiveTrace(
    __gm__ uint8_t* trace_base, uint32_t layer_id, uint64_t receive_start,
    uint64_t request_ready, uint64_t request_copied) {
  auto ub = reinterpret_cast<__ubuf__ uint64_t*>(ControlUb());
  AscendC::LocalTensor<uint64_t> trace;
  trace.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
  trace.address_.bufferAddr = reinterpret_cast<uint64_t>(ub);
  trace.SetValue(0, receive_start);
  trace.SetValue(1, request_ready);
  trace.SetValue(2, request_copied);
  trace.SetValue(3, 0U);
  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(kCopyEvent);
  smem_shm_copy_ub2gm(
      trace_base + layer_id * kProfileRowBytes,
      reinterpret_cast<__ubuf__ uint8_t*>(ub), 32U);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kCopyEvent);
}

__aicore__ inline void WriteServiceResponseTrace(
    __gm__ uint8_t* trace_base, __gm__ uint8_t* decoder_base,
    uint32_t layer_id, uint64_t respond_start, uint64_t response_sent,
    uint64_t decoder_done) {
  auto ub_bytes = reinterpret_cast<__ubuf__ uint8_t*>(ControlUb());
  auto ub = reinterpret_cast<__ubuf__ uint64_t*>(ub_bytes);
  AscendC::LocalTensor<uint64_t> trace;
  trace.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
  trace.address_.bufferAddr = reinterpret_cast<uint64_t>(ub);
  trace.SetValue(0, respond_start);
  trace.SetValue(1, response_sent);
  trace.SetValue(2, decoder_done);
  trace.SetValue(3, 0U);
  AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(kCopyEvent);
  smem_shm_copy_gm2ub(ub_bytes + 32U, decoder_base + kProfileOffset, 32U);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(kCopyEvent);
  smem_shm_copy_ub2gm(
      trace_base + layer_id * kProfileRowBytes + 32U, ub_bytes, 64U);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kCopyEvent);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kCopyEvent);
}

__aicore__ inline __gm__ uint8_t* RankBase(__gm__ uint8_t* gva,
                                           uint64_t symmetric_size,
                                           uint32_t rank) {
  return gva + symmetric_size * rank;
}

__aicore__ inline __gm__ uint8_t* PayloadSlot(__gm__ uint8_t* rank_base,
                                              uint32_t slot) {
  return rank_base + kPayloadOffset + kMaxPayloadBytes * slot;
}

[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
indexer_shm_initialize_control(GM_ADDR gva_addr, uint64_t symmetric_size,
                               uint32_t rank) {
  symmetric_size = smem_shm_get_symmetric_size();
  auto gva = reinterpret_cast<__gm__ uint8_t*>(gva_addr);
  auto rank_base = RankBase(gva, symmetric_size, rank);
  WriteSequence(
      reinterpret_cast<__gm__ uint32_t*>(rank_base + kRequestDoorbellOffset),
      0U);
  WriteSequence(
      reinterpret_cast<__gm__ uint32_t*>(rank_base + kServiceSeenOffset), 0U);
  WriteSequence(
      reinterpret_cast<__gm__ uint32_t*>(rank_base + kProfileOffset), 0U);
  WriteSequence(reinterpret_cast<__gm__ uint32_t*>(
                    rank_base + kTpFanoutDoorbellOffset),
                0U);
  WriteSequence(
      reinterpret_cast<__gm__ uint32_t*>(rank_base + kTpFanoutAckOffset), 0U);
}

[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
indexer_shm_service_resident_update(
    GM_ADDR topk_addr, GM_ADDR block_table_addr, GM_ADDR slot_mapping_addr,
    GM_ADDR resident_sources_addr, GM_ADDR response_addr, uint64_t count,
    uint32_t topk_size, uint32_t block_table_cols, uint32_t block_size) {
  ServiceResidentUpdate(count, topk_addr, block_table_addr, slot_mapping_addr,
                        resident_sources_addr, response_addr, topk_size,
                        block_table_cols, block_size);
}

[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
indexer_shm_decoder_resident_descriptors(
    GM_ADDR encoded_sources_addr, GM_ADDR gvas_addr, GM_ADDR addrs_addr,
    GM_ADDR sizes_addr, GM_ADDR descriptor_count_addr,
    GM_ADDR current_slots_addr, uint64_t count, uint64_t gva_k_base,
    uint64_t gva_v_base, uint64_t addr_k_base, uint64_t addr_v_base,
    uint32_t token_bytes_k, uint32_t token_bytes_v, uint32_t topk_size,
    uint32_t resident_capacity) {
  DecoderResidentDescriptors(
      count, encoded_sources_addr, gvas_addr, addrs_addr, sizes_addr,
      descriptor_count_addr, current_slots_addr, gva_k_base, gva_v_base,
      addr_k_base, addr_v_base, token_bytes_k, token_bytes_v, topk_size,
      resident_capacity);
}

// Fan out one rank's response to the other ranks without launching an HCCL
// collective.  The leader publishes into its symmetric segment and waits for
// one acknowledgement in each follower's segment before reusing the slot.
[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
indexer_shm_tp_fanout_leader(GM_ADDR gva_addr, uint64_t symmetric_size,
                             uint32_t leader_rank, uint32_t group_size,
                             GM_ADDR source_addr, uint32_t source_bytes) {
  symmetric_size = smem_shm_get_symmetric_size();
  auto gva = reinterpret_cast<__gm__ uint8_t*>(gva_addr);
  auto source = reinterpret_cast<__gm__ uint8_t*>(source_addr);
  auto leader_base = RankBase(gva, symmetric_size, leader_rank);
  auto leader_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(leader_base + kTpFanoutDoorbellOffset);
  const uint32_t sequence = ReadSequence(leader_doorbell) + 1U;
  CopyGmToGm(PayloadSlot(leader_base, sequence & 1U), source, source_bytes);
  WriteSequence(leader_doorbell, sequence);
  for (uint32_t rank = leader_rank + 1U;
       rank < leader_rank + group_size; ++rank) {
    auto follower_base = RankBase(gva, symmetric_size, rank);
    auto follower_ack = reinterpret_cast<__gm__ uint32_t*>(
        follower_base + kTpFanoutAckOffset);
    WaitSequence(follower_ack, sequence);
  }
}

[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
indexer_shm_tp_fanout_follower(GM_ADDR gva_addr, uint64_t symmetric_size,
                               uint32_t rank, uint32_t leader_rank,
                               GM_ADDR output_addr, uint32_t output_bytes) {
  symmetric_size = smem_shm_get_symmetric_size();
  auto gva = reinterpret_cast<__gm__ uint8_t*>(gva_addr);
  auto output = reinterpret_cast<__gm__ uint8_t*>(output_addr);
  auto rank_base = RankBase(gva, symmetric_size, rank);
  auto leader_base = RankBase(gva, symmetric_size, leader_rank);
  auto follower_ack =
      reinterpret_cast<__gm__ uint32_t*>(rank_base + kTpFanoutAckOffset);
  auto leader_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(leader_base + kTpFanoutDoorbellOffset);
  const uint32_t sequence = ReadSequence(follower_ack) + 1U;
  WaitSequence(leader_doorbell, sequence);
  CopyGmToGm(output, PayloadSlot(leader_base, sequence & 1U), output_bytes);
  WriteSequence(follower_ack, sequence);
}

[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
indexer_shm_decoder_exchange(GM_ADDR gva_addr, uint64_t symmetric_size,
                             uint32_t decoder_rank, uint32_t service_rank,
                             GM_ADDR request_addr, uint32_t request_bytes,
                             GM_ADDR response_addr, uint32_t response_bytes) {
  symmetric_size = smem_shm_get_symmetric_size();
  auto gva = reinterpret_cast<__gm__ uint8_t*>(gva_addr);
  auto request = reinterpret_cast<__gm__ uint8_t*>(request_addr);
  auto response = reinterpret_cast<__gm__ uint8_t*>(response_addr);
  auto decoder_base = RankBase(gva, symmetric_size, decoder_rank);
  auto service_base = RankBase(gva, symmetric_size, service_rank);
  auto decoder_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(decoder_base + kRequestDoorbellOffset);
  auto service_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(service_base + kRequestDoorbellOffset);
  const uint32_t sequence = ReadSequence(decoder_doorbell) + 1U;
  const uint32_t slot = sequence & 1U;
  CopyGmToGm(PayloadSlot(service_base, slot), request, request_bytes);
  WriteRequestControl(service_doorbell, sequence, request_bytes);
  WaitSequence(decoder_doorbell, sequence);
  CopyGmToGm(response, PayloadSlot(decoder_base, slot), response_bytes);
}

[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
indexer_shm_service_peek_request(GM_ADDR gva_addr, uint64_t symmetric_size,
                                 uint32_t service_rank,
                                 GM_ADDR control_addr) {
  symmetric_size = smem_shm_get_symmetric_size();
  auto gva = reinterpret_cast<__gm__ uint8_t*>(gva_addr);
  auto control = reinterpret_cast<__gm__ uint8_t*>(control_addr);
  auto service_base = RankBase(gva, symmetric_size, service_rank);
  auto request_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(service_base + kRequestDoorbellOffset);
  auto seen_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(service_base + kServiceSeenOffset);
  const uint32_t sequence = ReadSequence(seen_doorbell) + 1U;
  WaitSequence(request_doorbell, sequence);
  CopyGmToGm(control, reinterpret_cast<__gm__ uint8_t*>(request_doorbell), 32U);
}

[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
indexer_shm_service_receive(GM_ADDR gva_addr, uint64_t symmetric_size,
                            uint32_t service_rank, GM_ADDR request_addr,
                            uint32_t request_bytes) {
  symmetric_size = smem_shm_get_symmetric_size();
  auto gva = reinterpret_cast<__gm__ uint8_t*>(gva_addr);
  auto request = reinterpret_cast<__gm__ uint8_t*>(request_addr);
  auto service_base = RankBase(gva, symmetric_size, service_rank);
  auto request_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(service_base + kRequestDoorbellOffset);
  auto seen_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(service_base + kServiceSeenOffset);
  const uint32_t sequence = ReadSequence(seen_doorbell) + 1U;
  WaitSequence(request_doorbell, sequence);
  CopyGmToGm(request, PayloadSlot(service_base, sequence & 1U), request_bytes);
  WriteSequence(seen_doorbell, sequence);
}

[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
indexer_shm_service_respond(GM_ADDR gva_addr, uint64_t symmetric_size,
                            uint32_t decoder_rank, uint32_t service_rank,
                            GM_ADDR response_addr, uint32_t response_bytes) {
  symmetric_size = smem_shm_get_symmetric_size();
  auto gva = reinterpret_cast<__gm__ uint8_t*>(gva_addr);
  auto response = reinterpret_cast<__gm__ uint8_t*>(response_addr);
  auto decoder_base = RankBase(gva, symmetric_size, decoder_rank);
  auto service_base = RankBase(gva, symmetric_size, service_rank);
  auto seen_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(service_base + kServiceSeenOffset);
  auto decoder_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(decoder_base + kRequestDoorbellOffset);
  const uint32_t sequence = ReadSequence(seen_doorbell);
  CopyGmToGm(PayloadSlot(decoder_base, sequence & 1U), response, response_bytes);
  WriteSequence(decoder_doorbell, sequence);
}

[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
indexer_shm_decoder_exchange_profiled(
    GM_ADDR gva_addr, uint64_t symmetric_size, uint32_t decoder_rank,
    uint32_t service_rank, GM_ADDR request_addr, uint32_t request_bytes,
    GM_ADDR response_addr, uint32_t response_bytes) {
  symmetric_size = smem_shm_get_symmetric_size();
  auto gva = reinterpret_cast<__gm__ uint8_t*>(gva_addr);
  auto request = reinterpret_cast<__gm__ uint8_t*>(request_addr);
  auto response = reinterpret_cast<__gm__ uint8_t*>(response_addr);
  auto decoder_base = RankBase(gva, symmetric_size, decoder_rank);
  auto service_base = RankBase(gva, symmetric_size, service_rank);
  auto decoder_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(decoder_base + kRequestDoorbellOffset);
  auto service_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(service_base + kRequestDoorbellOffset);
  auto service_ack =
      reinterpret_cast<__gm__ uint32_t*>(service_base + kProfileOffset);
  const uint32_t sequence = ReadSequence(decoder_doorbell) + 1U;
  const uint32_t slot = sequence & 1U;
  const uint64_t start = ReadCycle();
  CopyGmToGm(PayloadSlot(service_base, slot), request, request_bytes);
  WriteRequestControl(service_doorbell, sequence, request_bytes);
  const uint64_t request_sent = ReadCycle();
  WaitSequence(decoder_doorbell, sequence);
  const uint64_t response_ready = ReadCycle();
  CopyGmToGm(response, PayloadSlot(decoder_base, slot), response_bytes);
  const uint64_t response_copied = ReadCycle();
  WriteDecoderTrace(decoder_base, start, request_sent, response_ready,
                    response_copied);
  WriteSequence(service_ack, sequence);
}

[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
indexer_shm_decoder_exchange_tensors_profiled(
    GM_ADDR gva_addr, uint64_t symmetric_size, uint32_t decoder_rank,
    uint32_t service_rank, GM_ADDR response_addr, uint32_t response_bytes,
    GM_ADDR src0_addr, uint32_t bytes0, GM_ADDR src1_addr, uint32_t bytes1,
    GM_ADDR src2_addr, uint32_t bytes2, GM_ADDR src3_addr, uint32_t bytes3,
    GM_ADDR src4_addr, uint32_t bytes4, GM_ADDR src5_addr, uint32_t bytes5,
    GM_ADDR src6_addr, uint32_t bytes6, GM_ADDR src7_addr, uint32_t bytes7,
    GM_ADDR src8_addr, uint32_t bytes8) {
  symmetric_size = smem_shm_get_symmetric_size();
  auto gva = reinterpret_cast<__gm__ uint8_t*>(gva_addr);
  auto response = reinterpret_cast<__gm__ uint8_t*>(response_addr);
  auto decoder_base = RankBase(gva, symmetric_size, decoder_rank);
  auto service_base = RankBase(gva, symmetric_size, service_rank);
  auto decoder_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(decoder_base + kRequestDoorbellOffset);
  auto service_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(service_base + kRequestDoorbellOffset);
  auto service_ack =
      reinterpret_cast<__gm__ uint32_t*>(service_base + kProfileOffset);
  const uint32_t core_id = AscendC::GetBlockIdx();
  const uint32_t sequence = ReadSequence(decoder_doorbell) + 1U;
  const uint32_t slot = sequence & 1U;
  auto remote_request = PayloadSlot(service_base, slot);
  const uint32_t offset1 = Align32(bytes0);
  const uint32_t offset2 = offset1 + Align32(bytes1);
  const uint32_t offset3 = offset2 + Align32(bytes2);
  const uint32_t offset4 = offset3 + Align32(bytes3);
  const uint32_t offset5 = offset4 + Align32(bytes4);
  const uint32_t offset6 = offset5 + Align32(bytes5);
  const uint32_t offset7 = offset6 + Align32(bytes6);
  const uint32_t offset8 = offset7 + Align32(bytes7);
  const uint32_t request_bytes = offset8 + Align32(bytes8);
  uint64_t start = 0;
  AscendC::SyncAll<true>();
  if (core_id == 0U) {
    start = ReadCycle();
  }
  AscendC::SyncAll<true>();
#define COPY_SOURCE(index, offset)                                      \
  CopyGmToGm(remote_request + offset,                                   \
             reinterpret_cast<__gm__ uint8_t*>(src##index##_addr),      \
             bytes##index)
  if (core_id == 0U) {
    COPY_SOURCE(0, 0U);
  } else if (core_id == 1U) {
    COPY_SOURCE(1, offset1);
  } else if (core_id == 2U) {
    COPY_SOURCE(2, offset2);
  } else if (core_id == 3U) {
    COPY_SOURCE(3, offset3);
  } else if (core_id == 4U) {
    COPY_SOURCE(4, offset4);
  } else if (core_id == 5U) {
    COPY_SOURCE(5, offset5);
  } else if (core_id == 6U) {
    COPY_SOURCE(6, offset6);
  } else if (core_id == 7U) {
    COPY_SOURCE(7, offset7);
  } else {
    COPY_SOURCE(8, offset8);
  }
#undef COPY_SOURCE
  AscendC::SyncAll<true>();
  if (core_id != 0U) {
    return;
  }
  WriteRequestControl(service_doorbell, sequence, request_bytes);
  const uint64_t request_sent = ReadCycle();
  WaitSequence(decoder_doorbell, sequence);
  const uint64_t response_ready = ReadCycle();
  CopyGmToGm(response, PayloadSlot(decoder_base, slot), response_bytes);
  const uint64_t response_copied = ReadCycle();
  WriteDecoderTrace(decoder_base, start, request_sent, response_ready,
                    response_copied);
  WriteSequence(service_ack, sequence);
}

[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
indexer_shm_service_receive_profiled(
    GM_ADDR gva_addr, uint64_t symmetric_size, uint32_t service_rank,
    GM_ADDR request_addr, uint32_t request_bytes, GM_ADDR trace_addr,
    uint32_t layer_id) {
  symmetric_size = smem_shm_get_symmetric_size();
  auto gva = reinterpret_cast<__gm__ uint8_t*>(gva_addr);
  auto request = reinterpret_cast<__gm__ uint8_t*>(request_addr);
  auto trace = reinterpret_cast<__gm__ uint8_t*>(trace_addr);
  auto service_base = RankBase(gva, symmetric_size, service_rank);
  auto request_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(service_base + kRequestDoorbellOffset);
  auto seen_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(service_base + kServiceSeenOffset);
  const uint32_t sequence = ReadSequence(seen_doorbell) + 1U;
  const uint64_t receive_start = ReadCycle();
  WaitSequence(request_doorbell, sequence);
  const uint64_t request_ready = ReadCycle();
  CopyGmToGm(request, PayloadSlot(service_base, sequence & 1U), request_bytes);
  const uint64_t request_copied = ReadCycle();
  WriteSequence(seen_doorbell, sequence);
  WriteServiceReceiveTrace(trace, layer_id, receive_start, request_ready,
                           request_copied);
}

[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
indexer_shm_service_respond_profiled(
    GM_ADDR gva_addr, uint64_t symmetric_size, uint32_t decoder_rank,
    uint32_t service_rank, GM_ADDR response_addr, uint32_t response_bytes,
    GM_ADDR trace_addr, uint32_t layer_id) {
  symmetric_size = smem_shm_get_symmetric_size();
  auto gva = reinterpret_cast<__gm__ uint8_t*>(gva_addr);
  auto response = reinterpret_cast<__gm__ uint8_t*>(response_addr);
  auto trace = reinterpret_cast<__gm__ uint8_t*>(trace_addr);
  auto decoder_base = RankBase(gva, symmetric_size, decoder_rank);
  auto service_base = RankBase(gva, symmetric_size, service_rank);
  auto seen_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(service_base + kServiceSeenOffset);
  auto decoder_doorbell =
      reinterpret_cast<__gm__ uint32_t*>(decoder_base + kRequestDoorbellOffset);
  auto service_ack =
      reinterpret_cast<__gm__ uint32_t*>(service_base + kProfileOffset);
  const uint32_t sequence = ReadSequence(seen_doorbell);
  const uint64_t respond_start = ReadCycle();
  CopyGmToGm(PayloadSlot(decoder_base, sequence & 1U), response,
             response_bytes);
  const uint64_t response_sent = ReadCycle();
  WriteSequence(decoder_doorbell, sequence);
  WaitSequence(service_ack, sequence);
  const uint64_t decoder_done = ReadCycle();
  WriteServiceResponseTrace(trace, decoder_base, layer_id, respond_start,
                            response_sent, decoder_done);
}

}  // namespace

extern "C" void indexer_shm_initialize_control_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t rank) {
  indexer_shm_initialize_control<<<1, nullptr, stream>>>(gva, symmetric_size,
                                                         rank);
}

extern "C" void indexer_shm_service_resident_update_do(
    void* stream, int32_t* topk, int32_t* block_table,
    int32_t* slot_mapping, int32_t* resident_sources, int32_t* response,
    uint64_t count, uint32_t topk_size, uint32_t block_table_cols,
    uint32_t block_size) {
  indexer_shm_service_resident_update<<<kResidentBlocks, nullptr, stream>>>(
      reinterpret_cast<uint8_t*>(topk),
      reinterpret_cast<uint8_t*>(block_table),
      reinterpret_cast<uint8_t*>(slot_mapping),
      reinterpret_cast<uint8_t*>(resident_sources),
      reinterpret_cast<uint8_t*>(response), count, topk_size,
      block_table_cols, block_size);
}

extern "C" void indexer_shm_decoder_resident_descriptors_do(
    void* stream, int32_t* encoded_sources, int64_t* gvas, int64_t* addrs,
    int32_t* sizes, int32_t* descriptor_count, int32_t* current_slots,
    uint64_t count, uint64_t gva_k_base, uint64_t gva_v_base,
    uint64_t addr_k_base, uint64_t addr_v_base, uint32_t token_bytes_k,
    uint32_t token_bytes_v, uint32_t topk_size,
    uint32_t resident_capacity) {
  indexer_shm_decoder_resident_descriptors<<<kResidentBlocks, nullptr,
                                             stream>>>(
      reinterpret_cast<uint8_t*>(encoded_sources),
      reinterpret_cast<uint8_t*>(gvas), reinterpret_cast<uint8_t*>(addrs),
      reinterpret_cast<uint8_t*>(sizes),
      reinterpret_cast<uint8_t*>(descriptor_count),
      reinterpret_cast<uint8_t*>(current_slots), count, gva_k_base,
      gva_v_base, addr_k_base, addr_v_base, token_bytes_k, token_bytes_v,
      topk_size, resident_capacity);
}

extern "C" void indexer_shm_tp_fanout_leader_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t leader_rank,
    uint32_t group_size, uint8_t* source, uint32_t source_bytes) {
  indexer_shm_tp_fanout_leader<<<1, nullptr, stream>>>(
      gva, symmetric_size, leader_rank, group_size, source, source_bytes);
}

extern "C" void indexer_shm_tp_fanout_follower_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t rank,
    uint32_t leader_rank, uint8_t* output, uint32_t output_bytes) {
  indexer_shm_tp_fanout_follower<<<1, nullptr, stream>>>(
      gva, symmetric_size, rank, leader_rank, output, output_bytes);
}

extern "C" void indexer_shm_decoder_exchange_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t decoder_rank,
    uint32_t service_rank, uint8_t* request, uint32_t request_bytes,
    uint8_t* response, uint32_t response_bytes) {
  indexer_shm_decoder_exchange<<<1, nullptr, stream>>>(
      gva, symmetric_size, decoder_rank, service_rank, request, request_bytes,
      response, response_bytes);
}

extern "C" void indexer_shm_service_receive_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t service_rank,
    uint8_t* request, uint32_t request_bytes) {
  indexer_shm_service_receive<<<1, nullptr, stream>>>(
      gva, symmetric_size, service_rank, request, request_bytes);
}

extern "C" void indexer_shm_service_peek_request_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size,
    uint32_t service_rank, int32_t* control) {
  indexer_shm_service_peek_request<<<1, nullptr, stream>>>(
      gva, symmetric_size, service_rank,
      reinterpret_cast<uint8_t*>(control));
}

extern "C" void indexer_shm_service_respond_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t decoder_rank,
    uint32_t service_rank, uint8_t* response, uint32_t response_bytes) {
  indexer_shm_service_respond<<<1, nullptr, stream>>>(
      gva, symmetric_size, decoder_rank, service_rank, response, response_bytes);
}

extern "C" void indexer_shm_decoder_exchange_profiled_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t decoder_rank,
    uint32_t service_rank, uint8_t* request, uint32_t request_bytes,
    uint8_t* response, uint32_t response_bytes) {
  indexer_shm_decoder_exchange_profiled<<<1, nullptr, stream>>>(
      gva, symmetric_size, decoder_rank, service_rank, request, request_bytes,
      response, response_bytes);
}

extern "C" void indexer_shm_decoder_exchange_tensors_profiled_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t decoder_rank,
    uint32_t service_rank, uint8_t* response, uint32_t response_bytes,
    uint8_t* src0, uint32_t bytes0, uint8_t* src1, uint32_t bytes1,
    uint8_t* src2, uint32_t bytes2, uint8_t* src3, uint32_t bytes3,
    uint8_t* src4, uint32_t bytes4, uint8_t* src5, uint32_t bytes5,
    uint8_t* src6, uint32_t bytes6, uint8_t* src7, uint32_t bytes7,
    uint8_t* src8, uint32_t bytes8) {
  indexer_shm_decoder_exchange_tensors_profiled<<<9, nullptr, stream>>>(
      gva, symmetric_size, decoder_rank, service_rank, response,
      response_bytes, src0, bytes0, src1, bytes1, src2, bytes2, src3, bytes3,
      src4, bytes4, src5, bytes5, src6, bytes6, src7, bytes7, src8, bytes8);
}

extern "C" void indexer_shm_service_receive_profiled_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t service_rank,
    uint8_t* request, uint32_t request_bytes, uint64_t* trace,
    uint32_t layer_id) {
  indexer_shm_service_receive_profiled<<<1, nullptr, stream>>>(
      gva, symmetric_size, service_rank, request, request_bytes,
      reinterpret_cast<uint8_t*>(trace), layer_id);
}

extern "C" void indexer_shm_service_respond_profiled_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t decoder_rank,
    uint32_t service_rank, uint8_t* response, uint32_t response_bytes,
    uint64_t* trace, uint32_t layer_id) {
  indexer_shm_service_respond_profiled<<<1, nullptr, stream>>>(
      gva, symmetric_size, decoder_rank, service_rank, response, response_bytes,
      reinterpret_cast<uint8_t*>(trace), layer_id);
}
