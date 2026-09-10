#include "adaptive_path_bound.cuh"
#include <thrust/host_vector.h>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

struct Path { float slack; };
static __global__ void check_guard(float upper, float guard, float delta, int depth, int* failed) {
  float value = nextafterf(guard, __int_as_float(0x7f800000));
  for (int i=0; i<depth; ++i) {
    value = __fadd_rn(value, delta);
    if (!(value > upper)) *failed = 1;
  }
}

int main() {
  using namespace gpucpg::adaptive_bound;
  assert(enabled(nullptr)); assert(!enabled(nullptr,false));
  assert(enabled("1")); assert(!enabled("0"));
  bool rejected=false;
  try { enabled("observe"); } catch(const std::invalid_argument&) { rejected=true; }
  assert(rejected);
  const float inf=std::numeric_limits<float>::infinity();
  assert(std::isinf(safe_cutoff(0,0,1)));
  assert(std::isinf(safe_cutoff(-1,0,1)));
  assert(std::isinf(safe_cutoff(1,inf,1)));
  assert(std::isinf(safe_cutoff(1,0,-1)));
  for(float invalid : {inf,-inf,std::numeric_limits<float>::quiet_NaN()}) {
    thrust::host_vector<float> h{0,1,invalid,-0.5f};
    thrust::device_vector<float> d=h;
    assert(!std::isfinite(minimum_delta(d)));
  }
  thrust::host_vector<float> h{-0.5f,0,1};
  thrust::device_vector<float> d=h;
  assert(minimum_delta(d)==-0.5f);
  for(int k : {1,3,16}) {
    Reservoir reservoir;
    thrust::host_vector<Path> hp(1000);
    for(int i=0;i<1000;++i) hp[i].slack=float((i*37)%101)*0.125f;
    thrust::device_vector<Path> dp=hp;
    for(int count : {16,32,33,100,1000,1000}) {
      std::vector<float> exact;
      for(int i=0;i<count;++i) exact.push_back(hp[i].slack);
      std::sort(exact.begin(),exact.end());
      assert(reservoir.update(dp.data().get(),count,k)==exact[k-1]);
      thrust::host_vector<float> retained=reservoir.costs();
      assert(retained.size()==size_t(k)); assert(reservoir.seen()==count);
      for(int i=0;i<k;++i) assert(retained[i]==exact[i]);
    }
    rejected=false;
    try { reservoir.update(dp.data().get(),999,k); } catch(const std::logic_error&) { rejected=true; }
    assert(rejected);
    rejected=false;
    try { reservoir.update(dp.data().get(),1000,k+1); } catch(const std::logic_error&) { rejected=true; }
    assert(rejected);
    reservoir.release(); assert(reservoir.empty() && reservoir.seen()==0);
    assert(reservoir.update(dp.data().get(),16,1)==0);
    reservoir.release();
  }
  for(float u : {0.001f,1.0f,35.1f,1762.26f,20607.62f})
  for(float delta : {0.0f,0.1f,-0.000000476837158f,-0.0000305175781f,-0.001953125f})
  for(int depth : {0,1,10,602,4096}) {
    float guard=safe_cutoff(u,delta,depth);
    assert(guard>=u);
    if(delta>=0 || depth==0) assert(guard==u);
    thrust::device_vector<int> failed(1,0);
    check_guard<<<1,1>>>(u,guard,delta,depth,failed.data().get());
    assert(int(failed[0])==0);
  }
  std::cout << "ADAPTIVE PATH BOUND UNIT PASS\n";
}
