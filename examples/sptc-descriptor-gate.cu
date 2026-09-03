#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}
void sparse_check(cusparseStatus_t status, const char* what) {
  if (status != CUSPARSE_STATUS_SUCCESS)
    throw std::runtime_error(std::string(what) + ": " + cusparseLtGetErrorString(status));
}

struct Args {
  int batches = 4096;
  int iterations = 100;
  int warmup = 10;
  int parents = 32;
  int deviations = 16;
  std::string pattern = "mixed";
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto value = [&]() {
      if (++i >= argc) throw std::runtime_error("missing option value");
      return std::stoi(argv[i]);
    };
    if (arg == "--batches") args.batches = value();
    else if (arg == "--iterations") args.iterations = value();
    else if (arg == "--warmup") args.warmup = value();
    else if (arg == "--parents") args.parents = value();
    else if (arg == "--deviations") args.deviations = value();
    else if (arg == "--pattern") {
      if (++i >= argc) throw std::runtime_error("missing option value");
      args.pattern = argv[i];
    }
    else throw std::runtime_error("unknown option: " + arg);
  }
  if (args.batches <= 0 || args.iterations <= 0 || args.warmup < 0
      || args.parents <= 0 || args.deviations <= 0
      || args.parents % 16 != 0 || args.deviations % 16 != 0)
    throw std::runtime_error("invalid Gate 4 dimensions");
  if (args.pattern != "mixed" && args.pattern != "all-short"
      && args.pattern != "all-long" && args.pattern != "all-skip")
    throw std::runtime_error(
      "pattern must be mixed, all-short, all-long, or all-skip");
  return args;
}

constexpr int kReduction = 32;
constexpr float kSplit = 8.0f;
constexpr float kFinalSplit = 12.0f;

struct DeferredDescriptor {
  int src = -1;
  int dev_begin = 0;
  unsigned short parent_count = 0;
  unsigned short dev_count = 0;
  int parent_indices[32]{};
  unsigned int promoted[16]{};
};

struct TileOutput {
  unsigned int class_mask = 0;
  unsigned int short_count = 0;
  unsigned int long_count = 0;
  unsigned int skip_count = 0;
  unsigned int descriptor_valid = 0;
};

struct CandidateRecord {
  int level = -1;
  int from = -1;
  int to = -1;
  int parent = -1;
  int num_children = 0;
  float slack = 0.0f;
};

static_assert(sizeof(DeferredDescriptor) == 204,
  "gate descriptor must match production DeferredLpqTile size");
static_assert(sizeof(CandidateRecord) == 24,
  "gate candidate must match production PfxtNode size");

__device__ unsigned char classify(const float value) {
  return value <= kSplit ? 0 : (value <= kFinalSplit ? 1 : 2);
}

__global__ void pack_sptc_operands(
  const __half* parent_slacks,
  const __half* deviation_deltas,
  __half* a,
  __half* b,
  const int batches,
  const int parents,
  const int deviations) {
  const auto tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto parent_count = static_cast<std::size_t>(batches) * parents;
  const auto deviation_count = static_cast<std::size_t>(batches) * deviations;
  if (tid < parent_count) {
    const auto dst = tid * kReduction;
    a[dst] = parent_slacks[tid];
    a[dst + 1] = __float2half(1.0f);
  }
  if (tid < deviation_count) {
    const auto dst = tid * kReduction;
    b[dst] = __float2half(1.0f);
    b[dst + 1] = deviation_deltas[tid];
  }
}

template <bool FromMma>
__global__ void classify_and_emit_descriptor(
  const __half* sums,
  const __half* parent_slacks,
  const __half* deviation_deltas,
  unsigned char* classes,
  CandidateRecord* candidates,
  DeferredDescriptor* descriptors,
  TileOutput* outputs,
  const int batches,
  const int parents,
  const int deviations) {
  const int batch = blockIdx.x;
  if (batch >= batches) return;
  const int products = parents * deviations;
  __shared__ unsigned int class_mask;
  __shared__ unsigned int short_count;
  __shared__ unsigned int long_count;
  __shared__ unsigned int skip_count;
  if (threadIdx.x == 0) {
    class_mask = 0;
    short_count = 0;
    long_count = 0;
    skip_count = 0;
  }
  __syncthreads();
  for (int product = threadIdx.x; product < products; product += blockDim.x) {
    const int parent = product / deviations;
    const int deviation = product - parent * deviations;
    const auto global = static_cast<std::size_t>(batch) * products + product;
    const float sum = FromMma
      ? __half2float(sums[global])
      : __half2float(parent_slacks[batch * parents + parent])
          + __half2float(deviation_deltas[batch * deviations + deviation]);
    const auto candidate_class = classify(sum);
    classes[global] = candidate_class;
    atomicOr(&class_mask, 1u << candidate_class);
    if (candidate_class == 0) atomicAdd(&short_count, 1u);
    else if (candidate_class == 1) atomicAdd(&long_count, 1u);
    else atomicAdd(&skip_count, 1u);
  }
  __syncthreads();
  const bool all_long = class_mask == (1u << 1);
  if (all_long) {
    auto& descriptor = descriptors[batch];
    if (threadIdx.x == 0) {
      descriptor.src = batch;
      descriptor.dev_begin = batch * deviations;
      descriptor.parent_count = static_cast<unsigned short>(parents);
      descriptor.dev_count = static_cast<unsigned short>(deviations);
    }
    for (int parent = threadIdx.x; parent < 32; parent += blockDim.x) {
      descriptor.parent_indices[parent] =
        parent < parents ? batch * parents + parent : -1;
    }
    for (int word = threadIdx.x; word < 16; word += blockDim.x) {
      descriptor.promoted[word] = 0;
    }
  }
  else {
    for (int product = threadIdx.x; product < products; product += blockDim.x) {
      const int parent = product / deviations;
      const int deviation = product - parent * deviations;
      const auto global = static_cast<std::size_t>(batch) * products + product;
      const auto candidate_class = classes[global];
      if (candidate_class == 2) continue;
      const float sum = FromMma
        ? __half2float(sums[global])
        : __half2float(parent_slacks[batch * parents + parent])
            + __half2float(deviation_deltas[batch * deviations + deviation]);
      candidates[global] = CandidateRecord{
        1, batch, batch * deviations + deviation,
        batch * parents + parent, 0, sum};
    }
  }
  if (threadIdx.x == 0) {
    outputs[batch] = TileOutput{
      class_mask, short_count, long_count, skip_count,
      static_cast<unsigned int>(all_long)};
  }
}

float elapsed_ms(cudaEvent_t begin, cudaEvent_t end) {
  float ms = 0.0f;
  cuda_check(cudaEventElapsedTime(&ms, begin, end), "event elapsed");
  return ms;
}

template <typename Launch>
float time_average(
  cudaEvent_t begin, cudaEvent_t end, const int iterations, Launch&& launch) {
  cuda_check(cudaEventRecord(begin), "record timing begin");
  for (int i = 0; i < iterations; ++i) launch();
  cuda_check(cudaEventRecord(end), "record timing end");
  cuda_check(cudaEventSynchronize(end), "wait timing");
  return elapsed_ms(begin, end) / iterations;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse_args(argc, argv);
    const int m = args.parents;
    const int n = args.deviations;
    constexpr int k = kReduction;
    const std::size_t a_batch = static_cast<std::size_t>(m) * k;
    const std::size_t b_batch = static_cast<std::size_t>(n) * k;
    const std::size_t c_batch = static_cast<std::size_t>(m) * n;
    const std::size_t a_count = a_batch * args.batches;
    const std::size_t b_count = b_batch * args.batches;
    const std::size_t c_count = c_batch * args.batches;

    std::vector<__half> h_a(a_count, __float2half(0.0f));
    std::vector<__half> h_b(b_count, __float2half(0.0f));
    std::vector<__half> h_parents(static_cast<std::size_t>(args.batches) * m);
    std::vector<__half> h_deviations(static_cast<std::size_t>(args.batches) * n);
    for (int batch = 0; batch < args.batches; ++batch) {
      for (int row = 0; row < m; ++row) {
        float slack = 1.0f;
        if (args.pattern == "all-long") slack = 8.5f;
        else if (args.pattern == "all-skip") slack = 12.5f;
        else if (args.pattern == "mixed")
          slack = 5.0f + static_cast<float>((batch * 17 + row * 7) % 96) / 16.0f;
        h_parents[static_cast<std::size_t>(batch) * m + row] = __float2half(slack);
        h_a[static_cast<std::size_t>(batch) * a_batch + row * k] = __float2half(slack);
        h_a[static_cast<std::size_t>(batch) * a_batch + row * k + 1] = __float2half(1.0f);
      }
      for (int row = 0; row < n; ++row) {
        const float delta = args.pattern == "mixed"
          ? 0.5f + static_cast<float>((batch * 11 + row * 5) % 48) / 16.0f
          : 0.25f;
        h_deviations[static_cast<std::size_t>(batch) * n + row] = __float2half(delta);
        h_b[static_cast<std::size_t>(batch) * b_batch + row * k] = __float2half(1.0f);
        h_b[static_cast<std::size_t>(batch) * b_batch + row * k + 1] = __float2half(delta);
      }
    }

    __half *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    __half *d_parents = nullptr, *d_deviations = nullptr;
    unsigned char *d_sptc_classes = nullptr, *d_cuda_classes = nullptr;
    CandidateRecord *d_sptc_candidates = nullptr, *d_cuda_candidates = nullptr;
    DeferredDescriptor *d_sptc_descriptors = nullptr, *d_cuda_descriptors = nullptr;
    TileOutput *d_sptc_outputs = nullptr, *d_cuda_outputs = nullptr;
    const auto device_allocation_begin = std::chrono::steady_clock::now();
    cuda_check(cudaMalloc(&d_a, a_count * sizeof(__half)), "allocate A");
    cuda_check(cudaMalloc(&d_b, b_count * sizeof(__half)), "allocate B");
    cuda_check(cudaMalloc(&d_c, c_count * sizeof(__half)), "allocate C");
    cuda_check(cudaMalloc(&d_parents, h_parents.size() * sizeof(__half)), "allocate parents");
    cuda_check(cudaMalloc(&d_deviations, h_deviations.size() * sizeof(__half)), "allocate deviations");
    cuda_check(cudaMalloc(&d_sptc_classes, c_count), "allocate sparse classes");
    cuda_check(cudaMalloc(&d_cuda_classes, c_count), "allocate CUDA classes");
    cuda_check(cudaMalloc(&d_sptc_candidates,
      c_count * sizeof(CandidateRecord)), "allocate sparse candidates");
    cuda_check(cudaMalloc(&d_cuda_candidates,
      c_count * sizeof(CandidateRecord)), "allocate CUDA candidates");
    cuda_check(cudaMalloc(&d_sptc_descriptors,
      args.batches * sizeof(DeferredDescriptor)), "allocate sparse descriptors");
    cuda_check(cudaMalloc(&d_cuda_descriptors,
      args.batches * sizeof(DeferredDescriptor)), "allocate CUDA descriptors");
    cuda_check(cudaMalloc(&d_sptc_outputs,
      args.batches * sizeof(TileOutput)), "allocate sparse outputs");
    cuda_check(cudaMalloc(&d_cuda_outputs,
      args.batches * sizeof(TileOutput)), "allocate CUDA outputs");
    const float gate_device_allocation_ms = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - device_allocation_begin).count();
    cuda_check(cudaMemcpy(d_a, h_a.data(), a_count * sizeof(__half), cudaMemcpyHostToDevice), "copy A");
    cuda_check(cudaMemcpy(d_b, h_b.data(), b_count * sizeof(__half), cudaMemcpyHostToDevice), "copy B");
    cuda_check(cudaMemcpy(d_parents, h_parents.data(), h_parents.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy parents");
    cuda_check(cudaMemcpy(d_deviations, h_deviations.data(), h_deviations.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy deviations");

    const auto library_setup_begin = std::chrono::steady_clock::now();
    cusparseLtHandle_t handle;
    cusparseLtMatDescriptor_t mat_a, mat_b, mat_c;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t selection;
    cusparseLtMatmulPlan_t plan;
    sparse_check(cusparseLtInit(&handle), "initialize cuSPARSELt");
    sparse_check(cusparseLtStructuredDescriptorInit(
      &handle, &mat_a, m, k, k, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW,
      CUSPARSELT_SPARSITY_50_PERCENT), "describe sparse A");
    sparse_check(cusparseLtDenseDescriptorInit(
      &handle, &mat_b, n, k, k, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW), "describe B");
    sparse_check(cusparseLtDenseDescriptorInit(
      &handle, &mat_c, m, n, n, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW), "describe C");
    const int32_t batches = args.batches;
    const int64_t a_stride = a_batch;
    const int64_t b_stride = b_batch;
    const int64_t c_stride = c_batch;
    for (auto* descriptor : {&mat_a, &mat_b, &mat_c})
      sparse_check(cusparseLtMatDescSetAttribute(
        &handle, descriptor, CUSPARSELT_MAT_NUM_BATCHES, &batches, sizeof(batches)),
        "set batch count");
    sparse_check(cusparseLtMatDescSetAttribute(
      &handle, &mat_a, CUSPARSELT_MAT_BATCH_STRIDE, &a_stride, sizeof(a_stride)), "set A stride");
    sparse_check(cusparseLtMatDescSetAttribute(
      &handle, &mat_b, CUSPARSELT_MAT_BATCH_STRIDE, &b_stride, sizeof(b_stride)), "set B stride");
    sparse_check(cusparseLtMatDescSetAttribute(
      &handle, &mat_c, CUSPARSELT_MAT_BATCH_STRIDE, &c_stride, sizeof(c_stride)), "set C stride");
    sparse_check(cusparseLtMatmulDescriptorInit(
      &handle, &matmul, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, &mat_a, &mat_b, &mat_c, &mat_c,
      CUSPARSE_COMPUTE_32F), "describe matmul");
    sparse_check(cusparseLtMatmulAlgSelectionInit(
      &handle, &selection, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT), "select algorithm");
    sparse_check(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &selection), "create plan");
    sparse_check(cusparseLtMatmulDescSetAttribute(
      &handle, &matmul, CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, &d_a, sizeof(d_a)),
      "set sparse pointer");

    size_t compressed_size = 0, compression_workspace_size = 0;
    sparse_check(cusparseLtSpMMACompressedSize(
      &handle, &plan, &compressed_size, &compression_workspace_size), "size compression");
    void* d_compressed = nullptr;
    void* d_compression_workspace = nullptr;
    cuda_check(cudaMalloc(&d_compressed, compressed_size), "allocate compressed A");
    cuda_check(cudaMalloc(&d_compression_workspace, compression_workspace_size), "allocate compression workspace");

    size_t matmul_workspace_size = 0;
    sparse_check(cusparseLtMatmulGetWorkspace(&handle, &plan, &matmul_workspace_size), "size matmul workspace");
    void* d_matmul_workspace = nullptr;
    cuda_check(cudaMalloc(&d_matmul_workspace, matmul_workspace_size), "allocate matmul workspace");
    const float library_setup_ms = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - library_setup_begin).count();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "create start event");
    cuda_check(cudaEventCreate(&stop), "create stop event");

    const auto operand_items = std::max(h_parents.size(), h_deviations.size());
    auto launch_pack = [&]() {
      pack_sptc_operands<<<static_cast<int>((operand_items + 255) / 256), 256>>>(
        d_parents, d_deviations, d_a, d_b, args.batches, m, n);
      cuda_check(cudaGetLastError(), "launch sparse operand pack");
    };
    auto launch_compress = [&]() {
      sparse_check(cusparseLtSpMMACompress(
        &handle, &plan, d_a, d_compressed, d_compression_workspace, nullptr),
        "compress A");
    };
    auto launch_mma = [&]() {
      sparse_check(cusparseLtMatmul(
        &handle, &plan, &alpha, d_compressed, d_b, &beta, d_c, d_c,
        d_matmul_workspace, nullptr, 0), "run sparse matmul");
    };
    auto launch_sptc_emit = [&]() {
      classify_and_emit_descriptor<true><<<args.batches, 256>>>(
        d_c, d_parents, d_deviations, d_sptc_classes,
        d_sptc_candidates, d_sptc_descriptors, d_sptc_outputs,
        args.batches, m, n);
      cuda_check(cudaGetLastError(), "launch sparse descriptor emission");
    };
    auto launch_cuda = [&]() {
      classify_and_emit_descriptor<false><<<args.batches, 256>>>(
        nullptr, d_parents, d_deviations, d_cuda_classes,
        d_cuda_candidates, d_cuda_descriptors, d_cuda_outputs,
        args.batches, m, n);
      cuda_check(cudaGetLastError(), "launch fused CUDA descriptor emission");
    };

    launch_pack();
    launch_compress();
    cuda_check(cudaEventRecord(start), "record search start");
    sparse_check(cusparseLtMatmulSearch(
      &handle, &plan, &alpha, d_compressed, d_b, &beta, d_c, d_c,
      d_matmul_workspace, nullptr, 0), "search sparse matmul");
    cuda_check(cudaEventRecord(stop), "record search stop");
    cuda_check(cudaEventSynchronize(stop), "wait sparse search");
    const float search_ms = elapsed_ms(start, stop);
    for (int i = 0; i < args.warmup; ++i) {
      launch_pack();
      launch_compress();
      launch_mma();
      launch_sptc_emit();
      launch_cuda();
    }
    cuda_check(cudaDeviceSynchronize(), "warmup synchronization");

    const float pack_ms = time_average(start, stop, args.iterations, launch_pack);
    const float compression_ms =
      time_average(start, stop, args.iterations, launch_compress);
    const float mma_ms = time_average(start, stop, args.iterations, launch_mma);
    const float sptc_emit_ms =
      time_average(start, stop, args.iterations, launch_sptc_emit);
    const float cuda_ms = time_average(start, stop, args.iterations, launch_cuda);
    const float sptc_optimistic_ms =
      time_average(start, stop, args.iterations, [&]() {
        launch_mma();
        launch_sptc_emit();
      });
    const float sptc_all_overhead_ms =
      time_average(start, stop, args.iterations, [&]() {
        launch_pack();
        launch_compress();
        launch_mma();
        launch_sptc_emit();
      });

    std::vector<unsigned char> h_sptc(c_count), h_cuda(c_count);
    std::vector<TileOutput> h_sptc_outputs(args.batches), h_cuda_outputs(args.batches);
    std::vector<DeferredDescriptor> h_sptc_descriptors(args.batches);
    std::vector<DeferredDescriptor> h_cuda_descriptors(args.batches);
    std::vector<CandidateRecord> h_sptc_candidates(c_count);
    std::vector<CandidateRecord> h_cuda_candidates(c_count);
    launch_pack();
    launch_compress();
    launch_mma();
    launch_sptc_emit();
    launch_cuda();
    cuda_check(cudaMemcpy(h_sptc.data(), d_sptc_classes, c_count, cudaMemcpyDeviceToHost), "copy sparse classes");
    cuda_check(cudaMemcpy(h_cuda.data(), d_cuda_classes, c_count, cudaMemcpyDeviceToHost), "copy CUDA classes");
    cuda_check(cudaMemcpy(h_sptc_outputs.data(), d_sptc_outputs,
      h_sptc_outputs.size() * sizeof(TileOutput), cudaMemcpyDeviceToHost),
      "copy sparse tile outputs");
    cuda_check(cudaMemcpy(h_cuda_outputs.data(), d_cuda_outputs,
      h_cuda_outputs.size() * sizeof(TileOutput), cudaMemcpyDeviceToHost),
      "copy CUDA tile outputs");
    cuda_check(cudaMemcpy(h_sptc_descriptors.data(), d_sptc_descriptors,
      h_sptc_descriptors.size() * sizeof(DeferredDescriptor), cudaMemcpyDeviceToHost),
      "copy sparse descriptors");
    cuda_check(cudaMemcpy(h_cuda_descriptors.data(), d_cuda_descriptors,
      h_cuda_descriptors.size() * sizeof(DeferredDescriptor), cudaMemcpyDeviceToHost),
      "copy CUDA descriptors");
    cuda_check(cudaMemcpy(h_sptc_candidates.data(), d_sptc_candidates,
      h_sptc_candidates.size() * sizeof(CandidateRecord), cudaMemcpyDeviceToHost),
      "copy sparse candidates");
    cuda_check(cudaMemcpy(h_cuda_candidates.data(), d_cuda_candidates,
      h_cuda_candidates.size() * sizeof(CandidateRecord), cudaMemcpyDeviceToHost),
      "copy CUDA candidates");
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < c_count; ++i) mismatches += h_sptc[i] != h_cuda[i];
    std::size_t output_mismatches = 0;
    std::size_t descriptor_mismatches = 0;
    std::size_t candidate_mismatches = 0;
    unsigned long long descriptors = 0;
    unsigned long long materialized = 0;
    for (int batch = 0; batch < args.batches; ++batch) {
      output_mismatches += std::memcmp(
        &h_sptc_outputs[batch], &h_cuda_outputs[batch], sizeof(TileOutput)) != 0;
      const auto& output = h_sptc_outputs[batch];
      descriptors += output.descriptor_valid;
      materialized += output.descriptor_valid == 0
        ? output.short_count + output.long_count : 0;
      if (output.descriptor_valid != 0) {
        descriptor_mismatches += std::memcmp(
          &h_sptc_descriptors[batch], &h_cuda_descriptors[batch],
          sizeof(DeferredDescriptor)) != 0;
      }
      else {
        const auto begin = static_cast<std::size_t>(batch) * m * n;
        for (int product = 0; product < m * n; ++product) {
          const auto pos = begin + product;
          if (h_sptc[pos] != 2) {
            candidate_mismatches += std::memcmp(
              &h_sptc_candidates[pos], &h_cuda_candidates[pos],
              sizeof(CandidateRecord)) != 0;
          }
        }
      }
    }
    const bool pass = mismatches == 0 && output_mismatches == 0
      && descriptor_mismatches == 0 && candidate_mismatches == 0;

    std::cout << "sptc_descriptor_gate"
      << " pattern=" << args.pattern
      << " batches=" << args.batches
      << " parents=" << m
      << " deviations=" << n
      << " products=" << c_count
      << " iterations=" << args.iterations
      << " pack_ms=" << pack_ms
      << " compression_ms=" << compression_ms
      << " gate_device_allocation_ms=" << gate_device_allocation_ms
      << " library_plan_workspace_setup_ms=" << library_setup_ms
      << " search_ms=" << search_ms
      << " mma_ms=" << mma_ms
      << " sptc_classify_emit_ms=" << sptc_emit_ms
      << " sptc_optimistic_ms=" << sptc_optimistic_ms
      << " sptc_all_overhead_ms=" << sptc_all_overhead_ms
      << " cuda_fused_classify_emit_ms=" << cuda_ms
      << " mma_only_vs_cuda=" << cuda_ms / mma_ms
      << " optimistic_speedup=" << cuda_ms / sptc_optimistic_ms
      << " all_overhead_speedup=" << cuda_ms / sptc_all_overhead_ms
      << " compressed_bytes=" << compressed_size
      << " dense_sparse_operand_bytes=" << a_count * sizeof(__half)
      << " dense_b_bytes=" << b_count * sizeof(__half)
      << " intermediate_c_bytes=" << c_count * sizeof(__half)
      << " sptc_extra_resident_bytes="
      << (a_count + b_count + c_count) * sizeof(__half) + compressed_size
           + compression_workspace_size + matmul_workspace_size
      << " descriptor_bytes=" << descriptors * sizeof(DeferredDescriptor)
      << " materialized_candidate_bytes="
      << materialized * sizeof(CandidateRecord)
      << " descriptors=" << descriptors
      << " materialized_products=" << materialized
      << " class_mismatches=" << mismatches
      << " output_mismatches=" << output_mismatches
      << " descriptor_mismatches=" << descriptor_mismatches
      << " candidate_mismatches=" << candidate_mismatches
      << " pass=" << (pass ? 1 : 0)
      << '\n';

    sparse_check(cusparseLtMatDescriptorDestroy(&mat_a), "destroy A descriptor");
    sparse_check(cusparseLtMatDescriptorDestroy(&mat_b), "destroy B descriptor");
    sparse_check(cusparseLtMatDescriptorDestroy(&mat_c), "destroy C descriptor");
    sparse_check(cusparseLtMatmulAlgSelectionDestroy(&selection), "destroy selection");
    sparse_check(cusparseLtMatmulPlanDestroy(&plan), "destroy plan");
    sparse_check(cusparseLtDestroy(&handle), "destroy handle");
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_parents); cudaFree(d_deviations);
    cudaFree(d_sptc_classes); cudaFree(d_cuda_classes);
    cudaFree(d_sptc_candidates); cudaFree(d_cuda_candidates);
    cudaFree(d_sptc_descriptors); cudaFree(d_cuda_descriptors);
    cudaFree(d_sptc_outputs); cudaFree(d_cuda_outputs);
    cudaFree(d_compressed); cudaFree(d_compression_workspace); cudaFree(d_matmul_workspace);
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "sptc_descriptor_gate_error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
