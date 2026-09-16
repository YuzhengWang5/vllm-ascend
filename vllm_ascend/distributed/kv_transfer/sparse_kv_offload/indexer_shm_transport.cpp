#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <vector>

extern "C" void indexer_shm_initialize_control_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t rank);
extern "C" void indexer_shm_service_resident_update_do(
    void* stream, int32_t* topk, int32_t* block_table,
    int32_t* slot_mapping, int32_t* resident_sources, int32_t* response,
    uint64_t count, uint32_t topk_size, uint32_t block_table_cols,
    uint32_t block_size, uint32_t forced_miss_count);
extern "C" void indexer_shm_decoder_resident_descriptors_do(
    void* stream, int32_t* encoded_sources, int64_t* gvas, int64_t* addrs,
    int32_t* sizes, int32_t* descriptor_count, int32_t* current_slots,
    uint64_t count, uint64_t gva_k_base, uint64_t gva_v_base,
    uint64_t addr_k_base, uint64_t addr_v_base, uint32_t token_bytes_k,
    uint32_t token_bytes_v, uint32_t topk_size,
    uint32_t resident_capacity);
extern "C" void indexer_shm_tp_fanout_leader_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t leader_rank,
    uint32_t group_size, uint8_t* source, uint32_t source_bytes);
extern "C" void indexer_shm_tp_fanout_follower_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t rank,
    uint32_t leader_rank, uint8_t* output, uint32_t output_bytes);
extern "C" void indexer_shm_decoder_exchange_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t decoder_rank,
    uint32_t service_rank, uint8_t* request, uint32_t request_bytes,
    uint8_t* response, uint32_t response_bytes);
extern "C" void indexer_shm_service_receive_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t service_rank,
    uint8_t* request, uint32_t request_bytes);
extern "C" void indexer_shm_service_peek_request_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size,
    uint32_t service_rank, int32_t* control);
extern "C" void indexer_shm_service_respond_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t decoder_rank,
    uint32_t service_rank, uint8_t* response, uint32_t response_bytes);
extern "C" void indexer_shm_decoder_exchange_profiled_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t decoder_rank,
    uint32_t service_rank, uint8_t* request, uint32_t request_bytes,
    uint8_t* response, uint32_t response_bytes);
extern "C" void indexer_shm_decoder_exchange_tensors_profiled_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t decoder_rank,
    uint32_t service_rank, uint8_t* response, uint32_t response_bytes,
    uint8_t* src0, uint32_t bytes0, uint8_t* src1, uint32_t bytes1,
    uint8_t* src2, uint32_t bytes2, uint8_t* src3, uint32_t bytes3,
    uint8_t* src4, uint32_t bytes4, uint8_t* src5, uint32_t bytes5,
    uint8_t* src6, uint32_t bytes6, uint8_t* src7, uint32_t bytes7,
    uint8_t* src8, uint32_t bytes8);
extern "C" void indexer_shm_service_receive_profiled_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t service_rank,
    uint8_t* request, uint32_t request_bytes, uint64_t* trace,
    uint32_t layer_id);
extern "C" void indexer_shm_service_respond_profiled_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t decoder_rank,
    uint32_t service_rank, uint8_t* response, uint32_t response_bytes,
    uint64_t* trace, uint32_t layer_id);

namespace {

constexpr int64_t kMaxPayloadBytes = 1LL << 20;

int64_t Align32(int64_t bytes) { return (bytes + 31) & ~31; }

void InitializeControl(int64_t gva, int64_t symmetric_size, int64_t rank) {
  TORCH_CHECK(symmetric_size > 0, "symmetric_size must be positive");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  indexer_shm_initialize_control_do(
      stream, reinterpret_cast<uint8_t*>(gva), symmetric_size, rank);
}

void CheckNpuTensor(const at::Tensor& tensor, at::ScalarType dtype,
                    const char* name) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              name, " must be an NPU tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void ServiceResidentUpdate(const at::Tensor& topk,
                           const at::Tensor& block_table,
                           const at::Tensor& slot_mapping,
                           at::Tensor& resident_sources,
                           at::Tensor& response, int64_t block_size,
                           int64_t forced_miss_count) {
  CheckNpuTensor(topk, at::kInt, "topk");
  CheckNpuTensor(block_table, at::kInt, "block_table");
  CheckNpuTensor(slot_mapping, at::kInt, "slot_mapping");
  CheckNpuTensor(resident_sources, at::kInt, "resident_sources");
  CheckNpuTensor(response, at::kInt, "response");
  TORCH_CHECK((topk.dim() == 2 || topk.dim() == 3) &&
                  block_table.dim() == 2,
              "topk must be rank-2/3 and block_table rank-2");
  const int64_t topk_size = topk.size(-1);
  const int64_t rows = topk.numel() / topk_size;
  TORCH_CHECK(slot_mapping.numel() == rows,
              "slot_mapping rows must equal topk rows");
  TORCH_CHECK(resident_sources.numel() == topk.numel() &&
                  response.numel() == topk.numel(),
              "resident_sources and response must match topk elements");
  TORCH_CHECK(block_table.size(0) == rows,
              "block_table rows must equal topk rows");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(forced_miss_count >= 0 && forced_miss_count <= topk_size,
              "forced_miss_count must be in [0, topk_size]");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  indexer_shm_service_resident_update_do(
      stream, static_cast<int32_t*>(topk.data_ptr()),
      static_cast<int32_t*>(block_table.data_ptr()),
      static_cast<int32_t*>(slot_mapping.data_ptr()),
      static_cast<int32_t*>(resident_sources.data_ptr()),
      static_cast<int32_t*>(response.data_ptr()), topk.numel(), topk_size,
      block_table.size(1), block_size, forced_miss_count);
}

void DecoderResidentDescriptors(
    const at::Tensor& encoded_sources, at::Tensor& gvas, at::Tensor& addrs,
    at::Tensor& sizes, at::Tensor& descriptor_count,
    at::Tensor& current_slots, int64_t gva_k_base, int64_t gva_v_base,
    int64_t addr_k_base, int64_t addr_v_base, int64_t token_bytes_k,
    int64_t token_bytes_v, int64_t resident_capacity) {
  CheckNpuTensor(encoded_sources, at::kInt, "encoded_sources");
  CheckNpuTensor(gvas, at::kLong, "gvas");
  CheckNpuTensor(addrs, at::kLong, "addrs");
  CheckNpuTensor(sizes, at::kInt, "sizes");
  CheckNpuTensor(descriptor_count, at::kInt, "descriptor_count");
  CheckNpuTensor(current_slots, at::kInt, "current_slots");
  TORCH_CHECK(encoded_sources.dim() == 2,
              "encoded_sources must be rank-2");
  const int64_t count = encoded_sources.numel();
  TORCH_CHECK(gvas.numel() >= 2 * count && addrs.numel() >= 2 * count &&
                  sizes.numel() >= 2 * count,
              "descriptor buffers are too small");
  TORCH_CHECK(current_slots.numel() >= count,
              "current_slots buffer is too small");
  TORCH_CHECK(descriptor_count.numel() >= 1,
              "descriptor_count must have one element");
  TORCH_CHECK(token_bytes_k > 0 && token_bytes_v > 0 &&
                  resident_capacity >= encoded_sources.size(1),
              "invalid resident descriptor geometry");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  indexer_shm_decoder_resident_descriptors_do(
      stream, static_cast<int32_t*>(encoded_sources.data_ptr()),
      static_cast<int64_t*>(gvas.data_ptr()),
      static_cast<int64_t*>(addrs.data_ptr()),
      static_cast<int32_t*>(sizes.data_ptr()),
      static_cast<int32_t*>(descriptor_count.data_ptr()),
      static_cast<int32_t*>(current_slots.data_ptr()), count, gva_k_base,
      gva_v_base, addr_k_base, addr_v_base, token_bytes_k, token_bytes_v,
      encoded_sources.size(1), resident_capacity);
}

void CheckPayload(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              name, " must be an NPU tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kByte, name, " must be uint8");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.numel() > 0 && tensor.numel() <= kMaxPayloadBytes,
              name, " bytes must be in [1, 1 MiB], got ", tensor.numel());
  TORCH_CHECK(tensor.numel() % 32 == 0, name, " allocation must be 32-byte aligned");
}

void TpFanoutLeader(const at::Tensor& source, int64_t gva,
                    int64_t symmetric_size, int64_t leader_rank,
                    int64_t group_size, int64_t shm_world_size) {
  CheckPayload(source, "source");
  TORCH_CHECK(group_size > 1 && group_size <= 16,
              "group_size must be in [2, 16]");
  TORCH_CHECK(leader_rank >= 0 && leader_rank + group_size <= shm_world_size,
              "TP group is outside SHM world");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  indexer_shm_tp_fanout_leader_do(
      stream, reinterpret_cast<uint8_t*>(gva), symmetric_size, leader_rank,
      group_size, static_cast<uint8_t*>(source.data_ptr()), source.numel());
}

void TpFanoutFollower(at::Tensor& output, int64_t gva,
                      int64_t symmetric_size, int64_t rank,
                      int64_t leader_rank, int64_t shm_world_size) {
  CheckPayload(output, "output");
  TORCH_CHECK(shm_world_size > 0, "shm_world_size must be positive");
  TORCH_CHECK(rank >= 0 && rank < shm_world_size,
              "rank must be in SHM world [0, ", shm_world_size - 1, "]");
  TORCH_CHECK(leader_rank >= 0 && leader_rank < shm_world_size,
              "leader_rank must be in SHM world [0, ", shm_world_size - 1,
              "]");
  TORCH_CHECK(rank != leader_rank, "leader cannot use follower operation");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  indexer_shm_tp_fanout_follower_do(
      stream, reinterpret_cast<uint8_t*>(gva), symmetric_size, rank,
      leader_rank, static_cast<uint8_t*>(output.data_ptr()), output.numel());
}

void DecoderExchange(const at::Tensor& request, at::Tensor& response,
                     int64_t gva, int64_t symmetric_size,
                     int64_t decoder_rank, int64_t service_rank) {
  CheckPayload(request, "request");
  CheckPayload(response, "response");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  indexer_shm_decoder_exchange_do(
      stream, reinterpret_cast<uint8_t*>(gva), symmetric_size, decoder_rank,
      service_rank, static_cast<uint8_t*>(request.data_ptr()), request.numel(),
      static_cast<uint8_t*>(response.data_ptr()), response.numel());
}

void ServiceReceive(at::Tensor& request, int64_t gva, int64_t symmetric_size,
                    int64_t service_rank) {
  CheckPayload(request, "request");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  indexer_shm_service_receive_do(
      stream, reinterpret_cast<uint8_t*>(gva), symmetric_size, service_rank,
      static_cast<uint8_t*>(request.data_ptr()), request.numel());
}

void ServicePeekRequest(at::Tensor& control, int64_t gva,
                        int64_t symmetric_size, int64_t service_rank) {
  CheckNpuTensor(control, at::kInt, "control");
  TORCH_CHECK(control.numel() >= 8,
              "control must hold one 32-byte doorbell cache line");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  indexer_shm_service_peek_request_do(
      stream, reinterpret_cast<uint8_t*>(gva), symmetric_size, service_rank,
      static_cast<int32_t*>(control.data_ptr()));
}

void ServiceRespond(const at::Tensor& response, int64_t gva,
                    int64_t symmetric_size, int64_t decoder_rank,
                    int64_t service_rank) {
  CheckPayload(response, "response");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  indexer_shm_service_respond_do(
      stream, reinterpret_cast<uint8_t*>(gva), symmetric_size, decoder_rank,
      service_rank, static_cast<uint8_t*>(response.data_ptr()), response.numel());
}

void DecoderExchangeProfiled(const at::Tensor& request, at::Tensor& response,
                             int64_t gva, int64_t symmetric_size,
                             int64_t decoder_rank, int64_t service_rank) {
  CheckPayload(request, "request");
  CheckPayload(response, "response");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  indexer_shm_decoder_exchange_profiled_do(
      stream, reinterpret_cast<uint8_t*>(gva), symmetric_size, decoder_rank,
      service_rank, static_cast<uint8_t*>(request.data_ptr()), request.numel(),
      static_cast<uint8_t*>(response.data_ptr()), response.numel());
}

void DecoderExchangeTensorsProfiled(
    const std::vector<at::Tensor>& requests, at::Tensor& response, int64_t gva,
    int64_t symmetric_size, int64_t decoder_rank, int64_t service_rank) {
  TORCH_CHECK(requests.size() == 9,
              "direct request packing requires exactly 9 tensors, got ",
              requests.size());
  CheckPayload(response, "response");
  int64_t request_bytes = 0;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& tensor = requests[i];
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                "request tensor ", i, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), "request tensor ", i,
                " must be contiguous");
    const int64_t bytes = tensor.numel() * tensor.element_size();
    TORCH_CHECK(bytes > 0 && bytes <= kMaxPayloadBytes,
                "request tensor ", i, " has invalid byte size ", bytes);
    request_bytes += Align32(bytes);
  }
  TORCH_CHECK(request_bytes <= kMaxPayloadBytes,
              "direct packed request exceeds 1 MiB: ", request_bytes);
  auto stream = c10_npu::getCurrentNPUStream().stream();
#define DATA(index) static_cast<uint8_t*>(requests[index].data_ptr())
#define BYTES(index)                                                        \
  static_cast<uint32_t>(requests[index].numel() * requests[index].element_size())
  indexer_shm_decoder_exchange_tensors_profiled_do(
      stream, reinterpret_cast<uint8_t*>(gva), symmetric_size, decoder_rank,
      service_rank, static_cast<uint8_t*>(response.data_ptr()),
      response.numel(), DATA(0), BYTES(0), DATA(1), BYTES(1), DATA(2),
      BYTES(2), DATA(3), BYTES(3), DATA(4), BYTES(4), DATA(5), BYTES(5),
      DATA(6), BYTES(6), DATA(7), BYTES(7), DATA(8), BYTES(8));
#undef BYTES
#undef DATA
}

void CheckTrace(const at::Tensor& trace, int64_t layer_id) {
  TORCH_CHECK(trace.device().type() == c10::DeviceType::PrivateUse1,
              "trace must be an NPU tensor");
  TORCH_CHECK(trace.scalar_type() == at::kLong, "trace must be int64");
  TORCH_CHECK(trace.is_contiguous(), "trace must be contiguous");
  TORCH_CHECK(trace.dim() == 2 && trace.size(1) == 12,
              "trace must have shape [layers, 12]");
  TORCH_CHECK(layer_id >= 0 && layer_id < trace.size(0),
              "layer_id is outside trace");
}

void ServiceReceiveProfiled(at::Tensor& request, at::Tensor& trace,
                            int64_t layer_id, int64_t gva,
                            int64_t symmetric_size, int64_t service_rank) {
  CheckPayload(request, "request");
  CheckTrace(trace, layer_id);
  auto stream = c10_npu::getCurrentNPUStream().stream();
  indexer_shm_service_receive_profiled_do(
      stream, reinterpret_cast<uint8_t*>(gva), symmetric_size, service_rank,
      static_cast<uint8_t*>(request.data_ptr()), request.numel(),
      static_cast<uint64_t*>(trace.data_ptr()), layer_id);
}

void ServiceRespondProfiled(const at::Tensor& response, at::Tensor& trace,
                            int64_t layer_id, int64_t gva,
                            int64_t symmetric_size, int64_t decoder_rank,
                            int64_t service_rank) {
  CheckPayload(response, "response");
  CheckTrace(trace, layer_id);
  auto stream = c10_npu::getCurrentNPUStream().stream();
  indexer_shm_service_respond_profiled_do(
      stream, reinterpret_cast<uint8_t*>(gva), symmetric_size, decoder_rank,
      service_rank, static_cast<uint8_t*>(response.data_ptr()), response.numel(),
      static_cast<uint64_t*>(trace.data_ptr()), layer_id);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("initialize_control", &InitializeControl);
  module.def("service_resident_update", &ServiceResidentUpdate);
  module.def("decoder_resident_descriptors", &DecoderResidentDescriptors);
  module.def("tp_fanout_leader", &TpFanoutLeader);
  module.def("tp_fanout_follower", &TpFanoutFollower);
  module.def("decoder_exchange", &DecoderExchange);
  module.def("service_receive", &ServiceReceive);
  module.def("service_peek_request", &ServicePeekRequest);
  module.def("service_respond", &ServiceRespond);
  module.def("decoder_exchange_profiled", &DecoderExchangeProfiled);
  module.def("decoder_exchange_tensors_profiled",
             &DecoderExchangeTensorsProfiled);
  module.def("service_receive_profiled", &ServiceReceiveProfiled);
  module.def("service_respond_profiled", &ServiceRespondProfiled);
}
