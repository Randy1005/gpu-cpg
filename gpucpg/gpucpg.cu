#include "gpucpg.h"
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

namespace gpucpg {

void CpGen::do_reduction() {
  int const N = 1000;
  thrust::device_vector<int> data(N);
  thrust::fill(data.begin(), data.end(), 1);
  int const result = thrust::reduce(thrust::device, data.begin(), data.end(), 0);

  std::cout << "reduce sum=" << result << '\n';
}


} // namespace gpucpg
