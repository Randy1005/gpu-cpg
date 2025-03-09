#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;

template<int warp_size>
__global__ void bu_count_frontiers(
	int* depths, // this is the status array
	int num_verts,
	int* ftr_counts) {
	auto this_block = cg::this_thread_block();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// determine the range each thread is responsible for
	int verts_per_thread = (num_verts + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
	int my_beg = tid * verts_per_thread;
	int my_end = min((tid + 1) * verts_per_thread, num_verts);

	// get the lane ID of this thread
	auto warp = cg::tiled_partition<warp_size>(this_block);
	int lane = warp.thread_rank();

	while (warp.any(my_beg < my_end)) {
		unsigned ballot_mask;
		for (int i = 0; i < warp.size(); i++) {
			int depth{0};
			// every thread in my warp send me their my_begs
			int wbeg = warp.shfl(my_beg, i) + lane;
			// every thread in my warp send me their my_ends
			int wend = warp.shfl(my_end, i);

			if (wbeg < wend) {
				depth = depths[wbeg];
			}

			unsigned my_mask = warp.ballot(depth == -1);
			// store the counting result (now it's a bitset)
			// to the i-th lane
			if (lane == i) {
				ballot_mask = my_mask;
			}
		}
		// accumulate the final counting result
		ftr_counts[tid] += __popc(ballot_mask);
		my_beg += warp.size();
	}
}