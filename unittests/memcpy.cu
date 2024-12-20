#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <gpucpg/gpucpg.cuh>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CHUNK_SZ 32 
#define BLOCK_SZ 1024
#define WARPS_PER_BLOCK 32
struct warpmem_t {
  int data[CHUNK_SZ];
};

template<typename group_t> __device__
void memcpy_SIMD(group_t g, int N, int* dest, int* src) {
  int lane = g.thread_rank();

  for (int idx = lane; idx < N; idx += g.size()) {
    dest[idx] = src[idx];
  }
  g.sync();
} 

__global__ void test_kernel(int* data, int N) {
  __shared__ warpmem_t smem[WARPS_PER_BLOCK];
  
  int gid = threadIdx.x + blockDim.x*blockIdx.x;
  int tid = threadIdx.x;
  cg::thread_block_tile<32> tile32 = 
    cg::tiled_partition<32>(cg::this_thread_block());

  int warp_id = gid / tile32.size();
  warpmem_t* my_smem = smem + (tid / tile32.size());
 
  // copy data to smem via warps
  int beg = warp_id * CHUNK_SZ;
  memcpy_SIMD(tile32, CHUNK_SZ, my_smem->data, &data[beg]);

  if (tile32.thread_rank() == 0 && warp_id == 28) {
    printf("my data is:\n");
    for (int i = 0; i < CHUNK_SZ; i++) {
      printf("%d ", my_smem->data[i]);
    }
    printf("\n");
  }
}


TEST_CASE("memcpy" * doctest::timeout(300)) {
  int N = 1<<20;
  int num_blocks = (N + BLOCK_SZ - 1) / BLOCK_SZ;

  int* data;
  cudaMallocManaged(&data, N*sizeof(int));
  std::iota(data, data+N, 0);
 
  test_kernel<<<num_blocks, BLOCK_SZ>>>(data, N);
  cudaDeviceSynchronize();
}



