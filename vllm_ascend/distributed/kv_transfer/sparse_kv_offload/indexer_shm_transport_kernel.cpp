#include "kernel_operator.h"
#include "smem_shm_aicore_base_api.h"

namespace {

constexpr uint64_t kRequestDoorbellOffset = 0;
constexpr uint64_t kServiceSeenOffset = 32;
constexpr uint64_t kProfileOffset = 64;
constexpr uint64_t kPayloadOffset = 4096;
constexpr uint64_t kMaxPayloadBytes = 1UL << 20;
constexpr uint32_t kUbPayloadBytes = 64U << 10;
constexpr uint64_t kUbControlOffset = kUbPayloadBytes;
constexpr uint32_t kCopyEvent = 0;
constexpr uint32_t kProfileRowBytes = 96;

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

__aicore__ inline void CopyGmToGmChained(__gm__ uint8_t* dst,
                                         __gm__ uint8_t* src,
                                         uint32_t logical_bytes,
                                         bool first_segment,
                                         bool last_segment) {
  if (logical_bytes == 0) {
    return;
  }
  const uint32_t bytes = Align32(logical_bytes);
  auto ub = PayloadUb();
  uint32_t offset = 0;
  if (first_segment) {
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(kCopyEvent);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(kCopyEvent);
  } else {
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kCopyEvent);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kCopyEvent);
  }
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
  if (last_segment) {
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kCopyEvent);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kCopyEvent);
  }
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
  WriteSequence(service_doorbell, sequence);
  WaitSequence(decoder_doorbell, sequence);
  CopyGmToGm(response, PayloadSlot(decoder_base, slot), response_bytes);
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
  WriteSequence(service_doorbell, sequence);
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
  const uint32_t sequence = ReadSequence(decoder_doorbell) + 1U;
  const uint32_t slot = sequence & 1U;
  auto remote_request = PayloadSlot(service_base, slot);
  uint32_t offset = 0;
  const uint64_t start = ReadCycle();
#define COPY_SOURCE(index, first, last)                                  \
  CopyGmToGmChained(remote_request + offset,                              \
                    reinterpret_cast<__gm__ uint8_t*>(src##index##_addr), \
                    bytes##index, first, last);                           \
  offset += Align32(bytes##index)
  COPY_SOURCE(0, true, false);
  COPY_SOURCE(1, false, false);
  COPY_SOURCE(2, false, false);
  COPY_SOURCE(3, false, false);
  COPY_SOURCE(4, false, false);
  COPY_SOURCE(5, false, false);
  COPY_SOURCE(6, false, false);
  COPY_SOURCE(7, false, false);
  COPY_SOURCE(8, false, true);
#undef COPY_SOURCE
  WriteSequence(service_doorbell, sequence);
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
  indexer_shm_decoder_exchange_tensors_profiled<<<1, nullptr, stream>>>(
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
