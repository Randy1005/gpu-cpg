#pragma once
// Included after gpucpg.cuh and tc_pfxt_adaptive/candidates.cuh.
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/scan.h>
#include <limits>
#include <cmath>
#include <cstdint>

namespace gpucpg::strip24 {
struct Record {
  std::int32_t parent,src,begin;
  std::uint16_t length,live;
  std::uint32_t mask;
  float minimum;
};
static_assert(sizeof(Record)==24);
static_assert(sizeof(PfxtNode)==24);
// One parent and one <=32-deviation slice. The mask permits holes: only LONG
// positions are represented. Parent IDs remain valid when SHORT storage moves.
__host__ __device__ inline bool pack(int longs,int minimum) {return longs>=minimum;}
struct Input {
  int* current;
  int n;
  const int *succ,*next,*offsets,*dsts;
  const float* deltas;
  const unsigned char* reachable;
  PfxtNode *shorts,*longs;
  int window;
  float split,final_split;
  bool final,skip_long;
  const unsigned long long* adaptive;
  int minimum;
};
// Same parent/thread mapping as the ordinary producer. Count and fill use the
// same exact predicates, so no output-capacity retry or second classification
// kernel is introduced. SHORT and rejected LONG outputs remain normal nodes.
template<bool Count> __global__ void produce(Input a,int* totals,
  Record* records=nullptr,int record_base=0,int expected_records=0,
  int* tail_short=nullptr,int* tail_long=nullptr,
  int short_capacity=0,int long_capacity=0,int* overflow=nullptr) {
  // Count output: ordinary-mode flag, SHORT count, total LONG count,
  // strip count, represented LONG count. Fill reuses slot 3 as its append
  // counter, starting at the count result; no extra reset launch is needed.
  if(!tc_pfxt::should_run_ordinary_branch(static_cast<tc_pfxt::AdaptiveMode>(a.adaptive[10])))return;
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  if constexpr(Count)if(id==0)totals[0]=1;
  if(id>=a.n)return;
  int parent_id=a.window+id;
  PfxtNode parent=a.shorts[parent_id];
  int shorts=0,longs=0,packs=0,packed=0;
  for(int v=a.current[id];v!=-1;){
    for(int begin=a.offsets[v];begin<a.offsets[v+1];begin+=32){
      int length=min(32,a.offsets[v+1]-begin);
      unsigned sm=0,lm=0;float minimum=INFINITY;
      for(int j=0;j<length;++j){
        if(a.reachable&&!a.reachable[begin+j])continue;
        float cost=parent.slack+a.deltas[begin+j];
        auto c=tc_pfxt::classify_candidate(cost,a.split,a.final_split,a.final,a.skip_long);
        if(c==tc_pfxt::CandidateClass::SHORT)sm|=1u<<j;
        if(c==tc_pfxt::CandidateClass::LONG){lm|=1u<<j;if constexpr(!Count)minimum=fminf(minimum,cost);}
      }
      int nl=__popc(lm);bool defer=pack(nl,a.minimum);
      if constexpr(Count){
        shorts+=__popc(sm);longs+=nl;
        if(defer){++packs;packed+=nl;}
      }else{
        if(defer){
          int slot=atomicAdd(totals+3,1)-expected_records;
          if(slot<0||slot>=expected_records)atomicExch(overflow,1);
          else records[record_base+slot]={parent_id,v,begin,(unsigned short)length,
            (unsigned short)nl,lm,minimum};
          lm=0;
        }
        for(int kind=0;kind<2;++kind){
          unsigned mask=kind?lm:sm;
          while(mask){
            int j=__ffs(mask)-1;mask&=mask-1;
            int pos=atomicAdd(kind?tail_long:tail_short,1);
            PfxtNode* out=kind?a.longs:a.shorts;int cap=kind?long_capacity:short_capacity;
            if(pos>=cap||!out)atomicExch(overflow,1);
            else out[pos]=PfxtNode(parent.level+1,v,a.dsts[begin+j],parent_id,0,parent.slack+a.deltas[begin+j]);
          }
        }
      }
    }
    int successor=a.succ[v];v=successor==-1?-1:a.next[successor];
  }
  if constexpr(Count){
    unsigned active=__activemask();
    int values[4]={shorts,longs,packs,packed};
    for(int j=0;j<4;++j){int sum=__reduce_add_sync(active,values[j]);
      if((threadIdx.x&31)==0&&sum)atomicAdd(totals+1+j,sum);}
  }else a.current[id]=-1;
}
static __global__ void prepare(const Record* records,int n,const PfxtNode* nodes,
  const float* deltas,float split,unsigned* pending,int* counts){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  const auto& r=records[i];unsigned selected=0;
  if(r.live&&r.minimum<=split){
    unsigned mask=r.mask;float parent=nodes[r.parent].slack;
    while(mask){int j=__ffs(mask)-1;mask&=mask-1;
      if(parent+deltas[r.begin+j]<=split)selected|=1u<<j;}
  }
  pending[i]=selected;counts[i]=__popc(selected);
}
static __global__ void promote(Record* records,int n,PfxtNode* nodes,
  const int* dsts,const float* deltas,const unsigned* pending,const int* offsets,int base){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  unsigned selected=pending[i];if(!selected)return;
  auto& r=records[i];PfxtNode parent=nodes[r.parent];int pos=base+offsets[i];
  unsigned emit=selected;
  while(emit){int j=__ffs(emit)-1;emit&=emit-1;
    nodes[pos++]=PfxtNode(parent.level+1,r.src,dsts[r.begin+j],r.parent,0,parent.slack+deltas[r.begin+j]);}
  r.mask&=~selected;r.live=__popc(r.mask);r.minimum=INFINITY;
  unsigned left=r.mask;while(left){int j=__ffs(left)-1;left&=left-1;
    r.minimum=fminf(r.minimum,parent.slack+deltas[r.begin+j]);}
}
struct Minimum {__host__ __device__ float operator()(const Record& r)const{return r.minimum;}};
struct Queue {
  thrust::device_vector<Record> records;
  thrust::device_vector<unsigned> pending;
  thrust::device_vector<int> counts,offsets;
  long long remaining=0;
  bool prepared=false;float prepared_split=0;int promoted=0;
  unsigned long long created=0,represented=0,promoted_total=0;
  template<class V>static auto ptr(V& v){return thrust::raw_pointer_cast(v.data());}
  int append(int n,int products){
    int base=records.size();records.resize(size_t(base)+n);
    remaining+=products;created+=n;represented+=products;prepared=false;return base;
  }
  void clear(){records.clear();remaining=0;prepared=false;}
  float minimum()const{
    if(!remaining)return std::numeric_limits<float>::max();
    return thrust::transform_reduce(records.begin(),records.end(),Minimum{},
      std::numeric_limits<float>::max(),thrust::minimum<float>{});
  }
  int count(float split,const PfxtNode* nodes,const float* deltas){
    if(!remaining)return 0;
    // The split search can ask again before fill. Reuse the GPU masks rather
    // than reading parent/deviation costs and classifying a second time.
    if(prepared&&prepared_split==split)return promoted;
    int n=records.size();pending.resize(n);counts.resize(n);offsets.resize(n);
    prepare<<<(n+255)/256,256>>>(ptr(records),n,nodes,deltas,split,ptr(pending),ptr(counts));
    promoted=thrust::reduce(counts.begin(),counts.end(),0);
    prepared=true;prepared_split=split;return promoted;
  }
  int fill(float split,PfxtNode* nodes,const int* dsts,const float* deltas,int base){
    int total=count(split,nodes,deltas);if(!total)return 0;
    thrust::exclusive_scan(counts.begin(),counts.end(),offsets.begin());
    int n=records.size();promote<<<(n+255)/256,256>>>(ptr(records),n,nodes,dsts,deltas,ptr(pending),ptr(offsets),base);
    remaining-=total;promoted_total+=total;prepared=false;return total;
  }
};
}
