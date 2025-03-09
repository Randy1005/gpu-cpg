#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <gpucpg/bfs_bu_step_helpers.cuh>

struct printf_functor {
  __host__ __device__
  void operator() (int x) {
      printf("%4d ", x);
  }
};


template <typename Iterator>
void print_range(const std::string& name, Iterator first, Iterator last) {
  std::cout << name << ": ";
  thrust::for_each(first, last, printf_functor());
  std::cout << '\n';
}


TEST_CASE("counting" * doctest::timeout(300)) {
  int num_verts = 10;
  std::vector<int> h_depths = {0, 1, 2, -1, -1, 3, -1, -1, 4, -1};

  thrust::device_vector<int> t_depths(h_depths);
  auto d_depths = thrust::raw_pointer_cast(t_depths.data());
  
  int blk_size = 4;
  int num_blks = (num_verts + blk_size - 1) / blk_size;

  std::vector<int> h_ftr_counts(blk_size*num_blks, 0);
  thrust::device_vector<int> t_ftr_counts(h_ftr_counts);
  auto d_ftr_counts = thrust::raw_pointer_cast(t_ftr_counts.data());
  
  bu_count_frontiers<2>
    <<<num_blks, blk_size>>>
      (d_depths, num_verts, d_ftr_counts);

  thrust::device_vector<int> t_ftr_offsets(blk_size*num_blks, 0);
  // use thrust exclusive scan to get the prefix sum
  thrust::exclusive_scan(t_ftr_counts.begin(), t_ftr_counts.end(), t_ftr_offsets.begin());

  print_range("ftr_counts", t_ftr_counts.begin(), t_ftr_counts.end());
  print_range("ftr_offsets", t_ftr_offsets.begin(), t_ftr_offsets.end());

}



