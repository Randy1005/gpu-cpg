#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
  int parents = 16;
  int deviations = 16;
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
    else throw std::runtime_error("unknown option: " + arg);
  }
  if (args.batches <= 0 || args.iterations <= 0 || args.warmup < 0
      || args.parents <= 0 || args.deviations <= 0
      || args.parents % 16 != 0 || args.deviations % 16 != 0)
    throw std::runtime_error("invalid Gate 4 dimensions");
  return args;
}

constexpr int kReduction = 32;
constexpr float kSplit = 8.0f;
constexpr float kFinalSplit = 12.0f;

__device__ unsigned char classify(const float value) {
  return value <= kSplit ? 0 : (value <= kFinalSplit ? 1 : 2);
}

__global__ void classify_sptc_outputs(
  const __half* sums, unsigned char* classes, const std::size_t count) {
  const auto tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid < count) classes[tid] = classify(__half2float(sums[tid]));
}

__global__ void fused_cuda_classify(
  const __half* parent_slacks,
  const __half* deviation_deltas,
  unsigned char* classes,
  const int batches,
  const int parents,
  const int deviations) {
  const auto tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto count = static_cast<std::size_t>(batches) * parents * deviations;
  if (tid >= count) return;
  const int tile_pos = static_cast<int>(tid % (parents * deviations));
  const int parent = tile_pos / deviations;
  const int deviation = tile_pos % deviations;
  const int batch = static_cast<int>(tid / (parents * deviations));
  const float sum = __half2float(parent_slacks[batch * parents + parent])
    + __half2float(deviation_deltas[batch * deviations + deviation]);
  classes[tid] = classify(sum);
}

float elapsed_ms(cudaEvent_t begin, cudaEvent_t end) {
  float ms = 0.0f;
  cuda_check(cudaEventElapsedTime(&ms, begin, end), "event elapsed");
  return ms;
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
        const float slack = 1.0f + static_cast<float>((batch * 17 + row * 7) % 120) / 16.0f;
        h_parents[static_cast<std::size_t>(batch) * m + row] = __float2half(slack);
        h_a[static_cast<std::size_t>(batch) * a_batch + row * k] = __float2half(slack);
        h_a[static_cast<std::size_t>(batch) * a_batch + row * k + 1] = __float2half(1.0f);
      }
      for (int row = 0; row < n; ++row) {
        const float delta = 0.5f + static_cast<float>((batch * 11 + row * 5) % 96) / 16.0f;
        h_deviations[static_cast<std::size_t>(batch) * n + row] = __float2half(delta);
        h_b[static_cast<std::size_t>(batch) * b_batch + row * k] = __float2half(1.0f);
        h_b[static_cast<std::size_t>(batch) * b_batch + row * k + 1] = __float2half(delta);
      }
    }

    __half *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    __half *d_parents = nullptr, *d_deviations = nullptr;
    unsigned char *d_sptc_classes = nullptr, *d_cuda_classes = nullptr;
    cuda_check(cudaMalloc(&d_a, a_count * sizeof(__half)), "allocate A");
    cuda_check(cudaMalloc(&d_b, b_count * sizeof(__half)), "allocate B");
    cuda_check(cudaMalloc(&d_c, c_count * sizeof(__half)), "allocate C");
    cuda_check(cudaMalloc(&d_parents, h_parents.size() * sizeof(__half)), "allocate parents");
    cuda_check(cudaMalloc(&d_deviations, h_deviations.size() * sizeof(__half)), "allocate deviations");
    cuda_check(cudaMalloc(&d_sptc_classes, c_count), "allocate sparse classes");
    cuda_check(cudaMalloc(&d_cuda_classes, c_count), "allocate CUDA classes");
    cuda_check(cudaMemcpy(d_a, h_a.data(), a_count * sizeof(__half), cudaMemcpyHostToDevice), "copy A");
    cuda_check(cudaMemcpy(d_b, h_b.data(), b_count * sizeof(__half), cudaMemcpyHostToDevice), "copy B");
    cuda_check(cudaMemcpy(d_parents, h_parents.data(), h_parents.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy parents");
    cuda_check(cudaMemcpy(d_deviations, h_deviations.data(), h_deviations.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy deviations");

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

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "create start event");
    cuda_check(cudaEventCreate(&stop), "create stop event");
    cuda_check(cudaEventRecord(start), "record compression start");
    sparse_check(cusparseLtSpMMACompress(
      &handle, &plan, d_a, d_compressed, d_compression_workspace, nullptr), "compress A");
    cuda_check(cudaEventRecord(stop), "record compression stop");
    cuda_check(cudaEventSynchronize(stop), "wait compression");
    const float compression_ms = elapsed_ms(start, stop);

    size_t matmul_workspace_size = 0;
    sparse_check(cusparseLtMatmulGetWorkspace(&handle, &plan, &matmul_workspace_size), "size matmul workspace");
    void* d_matmul_workspace = nullptr;
    cuda_check(cudaMalloc(&d_matmul_workspace, matmul_workspace_size), "allocate matmul workspace");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cuda_check(cudaEventRecord(start), "record search start");
    sparse_check(cusparseLtMatmulSearch(
      &handle, &plan, &alpha, d_compressed, d_b, &beta, d_c, d_c,
      d_matmul_workspace, nullptr, 0), "search sparse matmul");
    cuda_check(cudaEventRecord(stop), "record search stop");
    cuda_check(cudaEventSynchronize(stop), "wait sparse search");
    const float search_ms = elapsed_ms(start, stop);
    auto launch_sptc = [&]() {
      sparse_check(cusparseLtMatmul(
        &handle, &plan, &alpha, d_compressed, d_b, &beta, d_c, d_c,
        d_matmul_workspace, nullptr, 0), "run sparse matmul");
      classify_sptc_outputs<<<static_cast<int>((c_count + 255) / 256), 256>>>(
        d_c, d_sptc_classes, c_count);
      cuda_check(cudaGetLastError(), "launch sparse classification");
    };
    auto launch_cuda = [&]() {
      fused_cuda_classify<<<static_cast<int>((c_count + 255) / 256), 256>>>(
        d_parents, d_deviations, d_cuda_classes, args.batches, m, n);
      cuda_check(cudaGetLastError(), "launch fused CUDA classification");
    };
    for (int i = 0; i < args.warmup; ++i) { launch_sptc(); launch_cuda(); }
    cuda_check(cudaDeviceSynchronize(), "warmup synchronization");

    cuda_check(cudaEventRecord(start), "record sparse start");
    for (int i = 0; i < args.iterations; ++i) launch_sptc();
    cuda_check(cudaEventRecord(stop), "record sparse stop");
    cuda_check(cudaEventSynchronize(stop), "wait sparse replay");
    const float sptc_ms = elapsed_ms(start, stop) / args.iterations;

    cuda_check(cudaEventRecord(start), "record CUDA start");
    for (int i = 0; i < args.iterations; ++i) launch_cuda();
    cuda_check(cudaEventRecord(stop), "record CUDA stop");
    cuda_check(cudaEventSynchronize(stop), "wait CUDA replay");
    const float cuda_ms = elapsed_ms(start, stop) / args.iterations;

    std::vector<unsigned char> h_sptc(c_count), h_cuda(c_count);
    cuda_check(cudaMemcpy(h_sptc.data(), d_sptc_classes, c_count, cudaMemcpyDeviceToHost), "copy sparse classes");
    cuda_check(cudaMemcpy(h_cuda.data(), d_cuda_classes, c_count, cudaMemcpyDeviceToHost), "copy CUDA classes");
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < c_count; ++i) mismatches += h_sptc[i] != h_cuda[i];

    std::cout << "sptc_hardware_gate"
      << " batches=" << args.batches
      << " parents=" << m
      << " deviations=" << n
      << " products=" << c_count
      << " iterations=" << args.iterations
      << " compression_ms=" << compression_ms
      << " search_ms=" << search_ms
      << " sptc_classify_ms=" << sptc_ms
      << " cuda_fused_ms=" << cuda_ms
      << " speedup=" << (sptc_ms == 0.0f ? 0.0f : cuda_ms / sptc_ms)
      << " compressed_bytes=" << compressed_size
      << " dense_sparse_operand_bytes=" << a_count * sizeof(__half)
      << " class_mismatches=" << mismatches
      << " pass=" << (mismatches == 0 ? 1 : 0)
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
    cudaFree(d_compressed); cudaFree(d_compression_workspace); cudaFree(d_matmul_workspace);
    return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "sptc_hardware_gate_error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
