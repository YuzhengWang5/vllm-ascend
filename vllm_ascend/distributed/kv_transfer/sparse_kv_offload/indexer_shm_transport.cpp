#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <vector>

extern "C" void indexer_shm_initialize_control_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t rank);
extern "C" void indexer_shm_decoder_exchange_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t decoder_rank,
    uint32_t service_rank, uint8_t* request, uint32_t request_bytes,
    uint8_t* response, uint32_t response_bytes);
extern "C" void indexer_shm_service_receive_do(
    void* stream, uint8_t* gva, uint64_t symmetric_size, uint32_t service_rank,
    uint8_t* request, uint32_t request_bytes);
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

void CheckPayload(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              name, " must be an NPU tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kByte, name, " must be uint8");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.numel() > 0 && tensor.numel() <= kMaxPayloadBytes,
              name, " bytes must be in [1, 1 MiB], got ", tensor.numel());
  TORCH_CHECK(tensor.numel() % 32 == 0, name, " allocation must be 32-byte aligned");
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
  module.def("decoder_exchange", &DecoderExchange);
  module.def("service_receive", &ServiceReceive);
  module.def("service_respond", &ServiceRespond);
  module.def("decoder_exchange_profiled", &DecoderExchangeProfiled);
  module.def("decoder_exchange_tensors_profiled",
             &DecoderExchangeTensorsProfiled);
  module.def("service_receive_profiled", &ServiceReceiveProfiled);
  module.def("service_respond_profiled", &ServiceRespondProfiled);
}
