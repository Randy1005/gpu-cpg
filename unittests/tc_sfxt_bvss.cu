#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <gpucpg/tc_sfxt_bvss.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdint>
#include <vector>

TEST_CASE("tc sfxt bvss packs fanout CSR into sigma-8 virtual slice sets") {
  const int n = 10;
  const std::vector<int> row_ptr {0, 2, 3, 5, 5, 6, 6, 6, 6, 6, 6};
  const std::vector<int> col_idx {1, 9, 8, 0, 9, 2};

  const auto bvss = gpucpg::tc_sfxt::build_bvss_from_fanout_csr(
    n, row_ptr, col_idx, 8);

  CHECK(bvss.sigma == 8);
  CHECK(bvss.slices_per_thread == 4);
  CHECK(bvss.slice_capacity == 128);
  CHECK(bvss.n_intervals == 2);
  CHECK(bvss.real_ptrs == std::vector<int>({0, 1, 2}));
  CHECK(bvss.virtual_to_real == std::vector<int>({0, 1}));
  CHECK(bvss.row_ids.size() == 256);
  CHECK(bvss.masks.size() == 64);
  CHECK(bvss.unpadded_slices == 6);
  CHECK(bvss.total_set_bits == 6);
  CHECK(bvss.compression_ratio() == doctest::Approx(6.0 / 48.0));

  CHECK(gpucpg::tc_sfxt::decode_row_neighbors(bvss, 0) == std::vector<int>({1, 9}));
  CHECK(gpucpg::tc_sfxt::decode_row_neighbors(bvss, 1) == std::vector<int>({8}));
  CHECK(gpucpg::tc_sfxt::decode_row_neighbors(bvss, 2) == std::vector<int>({0, 9}));
  CHECK(gpucpg::tc_sfxt::decode_row_neighbors(bvss, 4) == std::vector<int>({2}));
}

TEST_CASE("tc sfxt bvss integrity check rejects mismatched CSR") {
  const int n = 4;
  const std::vector<int> row_ptr {0, 1, 2, 2, 2};
  const std::vector<int> col_idx {1, 2};

  const auto bvss = gpucpg::tc_sfxt::build_bvss_from_fanout_csr(
    n, row_ptr, col_idx, 8);

  CHECK(gpucpg::tc_sfxt::verify_bvss_matches_csr(bvss, n, row_ptr, col_idx));

  const std::vector<int> wrong_col_idx {1, 3};
  CHECK_FALSE(gpucpg::tc_sfxt::verify_bvss_matches_csr(
    bvss, n, row_ptr, wrong_col_idx));
}

TEST_CASE("tc sfxt init kernel marks sinks and queues their VSS ranges") {
  const int n = 10;
  const std::vector<int> row_ptr {0, 2, 3, 5, 5, 6, 6, 6, 6, 6, 6};
  const std::vector<int> col_idx {1, 9, 8, 0, 9, 2};
  const auto bvss = gpucpg::tc_sfxt::build_bvss_from_fanout_csr(
    n, row_ptr, col_idx, 8);

  const std::vector<int> h_sinks {1, 9};
  thrust::device_vector<int> d_sinks(h_sinks);
  thrust::device_vector<unsigned int> frontier_words(1, 0);
  thrust::device_vector<unsigned int> visited_words(1, 0);
  thrust::device_vector<unsigned int> queued_words(1, 0);
  thrust::device_vector<int> levels(n, -1);
  thrust::device_vector<int> queue(bvss.n_vss, -1);
  thrust::device_vector<int> queue_size(1, 0);
  thrust::device_vector<int> real_ptrs(bvss.real_ptrs);

  gpucpg::tc_sfxt::init_tc_bfs_from_sinks<<<1, 32>>>(
    thrust::raw_pointer_cast(d_sinks.data()),
    static_cast<int>(d_sinks.size()),
    thrust::raw_pointer_cast(frontier_words.data()),
    thrust::raw_pointer_cast(visited_words.data()),
    thrust::raw_pointer_cast(queued_words.data()),
    thrust::raw_pointer_cast(queue.data()),
    thrust::raw_pointer_cast(queue_size.data()),
    thrust::raw_pointer_cast(real_ptrs.data()),
    thrust::raw_pointer_cast(levels.data()),
    bvss.n_vss);
  cudaDeviceSynchronize();

  thrust::host_vector<unsigned int> h_frontier(frontier_words);
  thrust::host_vector<unsigned int> h_visited(visited_words);
  thrust::host_vector<int> h_levels(levels);
  thrust::host_vector<int> h_queue(queue);
  thrust::host_vector<int> h_queue_size(queue_size);

  CHECK((h_frontier[0] & ((1u << 1) | (1u << 9))) == ((1u << 1) | (1u << 9)));
  CHECK((h_visited[0] & ((1u << 1) | (1u << 9))) == ((1u << 1) | (1u << 9)));
  CHECK(h_levels[1] == 0);
  CHECK(h_levels[9] == 0);
  CHECK(h_queue_size[0] == 2);
  CHECK(h_queue[0] == 0);
  CHECK(h_queue[1] == 1);
}

TEST_CASE("tc sfxt scalar BVSS step discovers unvisited rows with fanout to frontier") {
  const int n = 10;
  const std::vector<int> row_ptr {0, 2, 3, 5, 5, 6, 6, 6, 6, 6, 6};
  const std::vector<int> col_idx {1, 9, 8, 0, 9, 2};
  const auto bvss = gpucpg::tc_sfxt::build_bvss_from_fanout_csr(
    n, row_ptr, col_idx, 8);

  thrust::device_vector<int> real_ptrs(bvss.real_ptrs);
  thrust::device_vector<int> virtual_to_real(bvss.virtual_to_real);
  thrust::device_vector<int> row_ids(bvss.row_ids);
  thrust::device_vector<unsigned int> masks(bvss.masks);
  thrust::device_vector<unsigned int> frontier_words(1, 1u << 9);
  thrust::device_vector<unsigned int> next_words(1, 0);
  thrust::device_vector<unsigned int> visited_words(1, 1u << 9);
  thrust::device_vector<unsigned int> next_queued_words(1, 0);
  thrust::device_vector<int> curr_queue {1};
  thrust::device_vector<int> next_queue(bvss.n_vss, -1);
  thrust::device_vector<int> curr_size {1};
  thrust::device_vector<int> next_size {0};
  thrust::device_vector<int> levels(n, -1);
  levels[9] = 0;

  gpucpg::tc_sfxt::scalar_bvss_pull_bfs_step<<<1, 32>>>(
    thrust::raw_pointer_cast(real_ptrs.data()),
    thrust::raw_pointer_cast(virtual_to_real.data()),
    thrust::raw_pointer_cast(row_ids.data()),
    thrust::raw_pointer_cast(masks.data()),
    thrust::raw_pointer_cast(frontier_words.data()),
    thrust::raw_pointer_cast(next_words.data()),
    thrust::raw_pointer_cast(visited_words.data()),
    thrust::raw_pointer_cast(next_queued_words.data()),
    thrust::raw_pointer_cast(curr_queue.data()),
    thrust::raw_pointer_cast(next_queue.data()),
    thrust::raw_pointer_cast(curr_size.data()),
    thrust::raw_pointer_cast(next_size.data()),
    thrust::raw_pointer_cast(levels.data()),
    1,
    bvss.n_vss);
  cudaDeviceSynchronize();

  thrust::host_vector<unsigned int> h_next_words(next_words);
  thrust::host_vector<unsigned int> h_visited(visited_words);
  thrust::host_vector<int> h_levels(levels);
  thrust::host_vector<int> h_next_queue(next_queue);
  thrust::host_vector<int> h_next_size(next_size);

  CHECK((h_next_words[0] & ((1u << 0) | (1u << 2))) == ((1u << 0) | (1u << 2)));
  CHECK((h_visited[0] & ((1u << 0) | (1u << 2) | (1u << 9))) ==
        ((1u << 0) | (1u << 2) | (1u << 9)));
  CHECK(h_levels[0] == 1);
  CHECK(h_levels[2] == 1);
  CHECK(h_next_size[0] == 1);
  CHECK(h_next_queue[0] == 0);
}

TEST_CASE("tc sfxt scalar BVSS loop computes backward levels from sinks") {
  const int n = 8;
  const std::vector<int> row_ptr {0, 2, 3, 4, 5, 6, 7, 7, 7};
  const std::vector<int> col_idx {
    2, 4,  // 0 -> 2,4
    2,     // 1 -> 2
    3,     // 2 -> 3
    6,     // 3 -> 6
    5,     // 4 -> 5
    6      // 5 -> 6
  };
  const auto bvss = gpucpg::tc_sfxt::build_bvss_from_fanout_csr(
    n, row_ptr, col_idx, 8);

  const std::vector<int> sinks {6, 7};
  const auto levels = gpucpg::tc_sfxt::run_scalar_bvss_pull_bfs_levels(
    n, bvss, sinks);

  CHECK(levels == std::vector<int>({3, 3, 2, 1, 2, 1, 0, 0}));
}

TEST_CASE("tc sfxt tensor-core BVSS loop matches scalar reference") {
  const int n = 8;
  const std::vector<int> row_ptr {0, 2, 3, 4, 5, 6, 7, 7, 7};
  const std::vector<int> col_idx {2, 4, 2, 3, 6, 5, 6};
  const auto bvss = gpucpg::tc_sfxt::build_bvss_from_fanout_csr(
    n, row_ptr, col_idx, 8);

  const std::vector<int> sinks {6, 7};
  const auto scalar_levels = gpucpg::tc_sfxt::run_scalar_bvss_pull_bfs_levels(
    n, bvss, sinks);
  const auto tc_levels = gpucpg::tc_sfxt::run_tc_bvss_pull_bfs_levels(
    n, bvss, sinks);

  CHECK(tc_levels == scalar_levels);
  CHECK(tc_levels == std::vector<int>({3, 3, 2, 1, 2, 1, 0, 0}));
}
