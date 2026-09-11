#include "gpucpg.cuh"
#include "tc_pfxt_adaptive.cuh"
#include "tc_pfxt_candidates.cuh"
#include "strip24.cuh"
#include "descriptor_coverage.hpp"
#include <thrust/host_vector.h>
#include <cassert>
using namespace gpucpg;
template<class V>auto ptr(V& v){return thrust::raw_pointer_cast(v.data());}
void test_producer_and_promotion(){
  static_assert(sizeof(strip24::Record)==24);
  assert(!strip24::pack(3,4));assert(strip24::pack(4,4));
  for (int n = 0; n <= 32; ++n) assert(!strip24::pack(n,33));
  thrust::device_vector<PfxtNode> nodes(1000),longs(1000);
  nodes[0]=PfxtNode(0,-1,0,-1,0,0);nodes[1]=PfxtNode(0,-1,0,-1,0,10);nodes[2]=PfxtNode(0,-1,0,-1,0,20);
  thrust::device_vector<int> current(3,0),succ(1,-1),next(1,-1),offsets(std::vector<int>{0,16}),dsts(32,7);
  std::vector<float> weights(32);for(int i=8;i<32;++i)weights[i]=10;
  thrust::device_vector<float> deltas(weights);
  thrust::device_vector<unsigned long long> adaptive(14,0);
  adaptive[10]=(unsigned long long)tc_pfxt::AdaptiveMode::ORDINARY;
  thrust::device_vector<int> totals(5,0),ts(1,3),tl(1,0),overflow(1,0);
  strip24::Input a{ptr(current),3,ptr(succ),ptr(next),ptr(offsets),ptr(dsts),ptr(deltas),nullptr,
    ptr(nodes),ptr(longs),0,5,100,false,false,ptr(adaptive),4};
  strip24::produce<true><<<1,256>>>(a,ptr(totals));
  thrust::host_vector<int> counts(totals);
  assert(counts[0]==1&&counts[1]==8&&counts[2]==40&&counts[3]==3&&counts[4]==40);
  strip24::Queue q;int base=q.append(3,40);
  strip24::produce<false><<<1,256>>>(a,ptr(totals),ptr(q.records),base,3,ptr(ts),ptr(tl),1000,1000,ptr(overflow));
  assert((int)overflow[0]==0);assert((int)ts[0]==11);assert((int)tl[0]==0);
  assert(q.minimum()==10);assert(q.count(5,ptr(nodes),ptr(deltas))==0);
  assert(q.count(12,ptr(nodes),ptr(deltas))==16);
  assert(q.count(12,ptr(nodes),ptr(deltas))==16); // cached, not consumed
  assert(q.fill(12,ptr(nodes),ptr(dsts),ptr(deltas),11)==16);
  assert(q.remaining==24);assert(q.minimum()==20);
  assert(q.count(12,ptr(nodes),ptr(deltas))==0); // cannot promote twice
  assert(q.fill(25,ptr(nodes),ptr(dsts),ptr(deltas),27)==16);
  assert(q.fill(35,ptr(nodes),ptr(dsts),ptr(deltas),43)==8);
  assert(q.remaining==0);assert(q.promoted_total==40);
  thrust::host_vector<PfxtNode> output(nodes);
  for(int i=11;i<51;++i){assert(output[i].level==1);assert(output[i].to==7);assert(output[i].from==0);assert(output[i].parent>=0&&output[i].parent<3);}
  q.clear();assert(q.records.empty());assert(q.count(100,ptr(nodes),ptr(deltas))==0);
  // Reject packs: all LONG outputs must go to the ordinary node pile.
  thrust::fill(current.begin(),current.end(),0);thrust::fill(totals.begin(),totals.end(),0);
  a.minimum=32;ts[0]=3;tl[0]=0;
  strip24::produce<true><<<1,256>>>(a,ptr(totals));counts=totals;
  assert(counts[3]==0&&counts[4]==0&&counts[2]==40);
  strip24::produce<false><<<1,256>>>(a,ptr(totals),nullptr,0,0,ptr(ts),ptr(tl),1000,1000,ptr(overflow));
  assert((int)tl[0]==40&& (int)ts[0]==11 && (int)overflow[0]==0);
  // Mask bit 31 and final-bound rejection.
  thrust::fill(current.begin(),current.end(),0);thrust::fill(totals.begin(),totals.end(),0);
  offsets[1]=32;a.n=1;a.minimum=4;
  strip24::produce<true><<<1,256>>>(a,ptr(totals));counts=totals;
  assert(counts[2]==24&&counts[3]==1);
  q.append(1,24);ts[0]=3;tl[0]=0;
  strip24::produce<false><<<1,256>>>(a,ptr(totals),ptr(q.records),0,1,ptr(ts),ptr(tl),1000,1000,ptr(overflow));
  thrust::host_vector<strip24::Record> records(q.records);
  assert(records[0].mask==0xffffff00u&&records[0].length==32);q.clear();
  thrust::fill(current.begin(),current.end(),0);thrust::fill(totals.begin(),totals.end(),0);
  a.final=true;a.final_split=8;
  strip24::produce<true><<<1,256>>>(a,ptr(totals));counts=totals;assert(counts[2]==0&&counts[3]==0);
  // K-crossing suppression has identical SHORT count and no deferred output.
  a.final=false;a.skip_long=true;thrust::fill(totals.begin(),totals.end(),0);
  strip24::produce<true><<<1,256>>>(a,ptr(totals));counts=totals;
  assert(counts[1]==8&&counts[2]==0&&counts[3]==0);
  cudaError_t e=cudaDeviceSynchronize();assert(e==cudaSuccess);
}

// Append must invalidate a previously prepared split. Resizing the parent
// vector must not invalidate descriptor IDs or cached promotion masks.
void test_append_and_parent_reallocation() {
  thrust::device_vector<PfxtNode> nodes(8);
  nodes[0] = PfxtNode(0, -1, 0, -1, 0, 1);
  thrust::device_vector<float> deltas(std::vector<float>{0,1,2,3,0,1,2,3});
  thrust::device_vector<int> destinations(8, 7);
  strip24::Queue queue;
  queue.append(1, 4);
  queue.records[0] = strip24::Record{0,0,0,4,4,0xf,1};
  assert(queue.count(2, ptr(nodes), ptr(deltas)) == 2);
  nodes.resize(128);
  const int second = queue.append(1, 4);
  assert(second == 1);
  queue.records[second] = strip24::Record{0,0,4,4,4,0xf,1};
  assert(queue.count(2, ptr(nodes), ptr(deltas)) == 4);
  assert(queue.fill(2, ptr(nodes), ptr(destinations), ptr(deltas), 1) == 4);
  assert(queue.remaining == 4 && queue.minimum() == 3);
  assert(queue.count(2, ptr(nodes), ptr(deltas)) == 0);
  assert(queue.fill(4, ptr(nodes), ptr(destinations), ptr(deltas), 5) == 4);
  assert(queue.remaining == 0 && queue.promoted_total == 8);
  thrust::host_vector<PfxtNode> output(nodes);
  for (int i = 1; i <= 8; ++i) {
    assert(output[i].parent == 0 && output[i].from == 0);
    assert(output[i].to == 7 && output[i].level == 1);
    assert(output[i].slack >= 1 && output[i].slack <= 4);
  }
  assert(cudaDeviceSynchronize() == cudaSuccess);
}

int main() {
  DescriptorCoverage coverage;
  coverage.ordinary(100, 3, 80);
  coverage.grouped(7, 2, 600);
  coverage.check_window(0, 707);
  assert(coverage.strips == 3 && coverage.strip_paths == 80);
  assert(coverage.tiles == 2 && coverage.tile_paths == 600 && coverage.individual == 27);
  auto before = coverage.total();
  coverage.ordinary(0, 0, 0); // K suppression emits nothing.
  coverage.grouped(0, 0, 0);
  coverage.check_window(before, 0);
  coverage.ordinary(9, 0, 0); // Rejected packs remain individual.
  coverage.check_window(before, 9);
  bool rejected = false;
  try { coverage.ordinary(3, 1, 4); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected); rejected = false;
  try { coverage.grouped(0, 1, 513); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected); rejected = false;
  try { coverage.check_window(before, 10); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
  test_producer_and_promotion();
  test_append_and_parent_reallocation();
  std::cout << "STRIP24 UNIT PASS\n";
}
