#include "kernel_operator.h"
#include "smem_shm_aicore_base_api.h"

namespace {

constexpr uint64_t kRequestDoorbellOffset = 0;
constexpr uint64_t kServiceSeenOffset = 32;
constexpr uint64_t kPayloadOffset = 4096;
constexpr uint64_t kMaxPayloadBytes = 1UL << 20;
constexpr uint32_t kUbPayloadBytes = 64U << 10;
constexpr uint64_t kUbControlOffset = kUbPayloadBytes;
constexpr uint32_t kCopyEvent = 0;

__aicore__ inline uint32_t Align32(uint32_t bytes) {
  return (bytes + 31U) & ~31U;
}

__aicore__ inline __ubuf__ uint8_t* PayloadUb() {
  return reinterpret_cast<__ubuf__ uint8_t*>(0);
}

__aicore__ inline __ubuf__ uint32_t* ControlUb() {
  return reinterpret_cast<__ubuf__ uint32_t*>(kUbControlOffset);
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
