# TC-PathGen Stage 2: TC-Accelerated Pfxt Deviation Discovery

## Status

Sfxt TC is deferred. G-PathGen's existing sfxt is used unchanged.
The sfxt phase already produces the two intermediate arrays that pfxt
TC needs: `succs[]` and `d_dists_cache`. Do not attempt to TC-ify
sfxt before completing and validating this stage.

## Objective

Replace the inner loop of `expand_short_pile` with a TC-based
frontier bitmap BFS that:

1. Discovers deviation edge pairs (spur_node u → deviation_dst v)
   for all active paths simultaneously, using BVSS over the deviation
   adjacency A_dev (fanout minus suffix edges, pre-excluded in BVSS)
2. Computes deviation candidate costs in CUDA using per-path
   parent_slack and the shared dist[] values

Parallelism at the path level is preserved. TC adds a new axis:
all frontier vertices discovered in one batched operation per chain
sub-step instead of one thread per path.

**Terminology used throughout:**
- "delta step": one alpha threshold increment in G-PathGen's outer
  delta-stepping loop (steps 1–9 in Gate P profiling)
- "chain sub-step": one iteration of the suffix chain walk inside
  expand_short_pile (v = succs[v]); many chain sub-steps happen
  within one delta step

---

## ⛔ PRELIMINARY — Locate All Intermediate Data

Before writing any TC code, read the codebase and confirm exact
variable names and locations for every piece of data pfxt TC needs.

### Step P.1 — Locate succs[] (suffix tree successor array)

```bash
grep -n "succs\b\|_succs\|d_succs\|h_succs\|succs\[" \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cu \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cuh | head -30
```

Identify:
- Host array name (e.g. `_h_succs[n_nodes]`)
- Device array name (e.g. `_d_succs[n_nodes]`)
- Where it is populated (during sfxt phase, after BFS)
- Confirmed on device before pfxt begins

`succs[v]` = suffix tree successor of v = next vertex on v's shortest
path to the sink. `succs[sink] = -1`.
Used in `expand_short_pile` as: `v = succs[v]` (chain advance).

### Step P.2 — Locate d_dists_cache (sfxt distances)

```bash
grep -n "dists_cache\|_d_dists\|d_dists\b" \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cu \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cuh | head -20
```

Identify:
- Device array name and element type
- `d_dists_cache[v]` = sfxt distance from v to nearest sink,
  stored as `int` = `float_distance × SCALE_UP` (SCALE_UP = 10000)
- Confirmed populated and valid before pfxt begins
- `d_dists_cache[sink] = 0`

### Step P.3 — Read expand_short_pile in full

```bash
grep -n "expand_short_pile\|SHORT_LONG" \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cu | head -10
```

Read `expand_short_pile` completely. Confirm and document:

```
PfxtNode fields:
  .to     = starting spur node for this path's chain walk
  .slack  = accumulated cost from root (CONSTANT during while loop)
  .level  = used for HPQ/LPQ dispatch
  .parent = parent PfxtNode index

Cost formula per deviation edge (v → neighbor):
  new_slack = node.slack
            + d_dists_cache[neighbor]   ← int, already ×SCALE_UP
            + (int)(wgt(v,neighbor) × SCALE_UP)
            - d_dists_cache[v]          ← int, already ×SCALE_UP

Dispatch: if new_slack <= alpha → HPQ; else → LPQ

Confirm: alpha type (int scaled ×SCALE_UP, or float?)
```

### Step P.4 — Locate short_pile, alpha, and delta-step call site

```bash
grep -n "short_pile\|_d_short_pile\|_alpha\|\balpha\b\|split\b\|n_short" \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cu | head -30
```

Identify:
- Device array for active PfxtNodes: likely `_d_short_pile[n_active]`
- Count of active paths per delta step: `n_short_paths` or similar
- Alpha threshold: variable name and type (must be same units as new_slack)
- Line where `expand_short_pile` is called in the host loop

### ⛔ GATE PRELIM — Summary Table

Show to user before writing any TC code:

```
succs[]:
  host: ?   device: ?   type: int*
  succs[sink] = -1: YES/NO
  on device before pfxt: YES/NO

d_dists_cache[]:
  device: ?   type: int* (×SCALE_UP=10000)
  dists_cache[sink] = 0: YES/NO
  on device before pfxt: YES/NO

short_pile:
  device array: ?   element type: PfxtNode
  count variable: ?

alpha:
  variable name: ?   type: int (scaled) or float?

expand_short_pile called at line: ?
cost formula confirmed matches spec: YES/NO
```

---

## Background: Why TC for pfxt

From Gate P profiling (netcard dense40, K=1M, delta step 9):

```
n_active_paths = 2,127,820
unique spur nodes = 4,479        ← unique chain positions
paths per unique spur node = 1,497 (average)
CUDA baseline = 245.852 ms
```

Currently: one thread per path, each independently reads CSR edges
for its current spur node. With 1,497 paths at the same spur node,
1,497 threads make identical global memory reads — ~1,497× redundant.

TC solution: represent all 4,479 unique spur nodes as a frontier
bitmap. TC discovers all deviation edge pairs in one batched operation
per chain sub-step, reading each spur node's adjacency ONCE. CUDA
then computes per-path costs from the shared (u, v, wgt) data.

---

## Complete Data Flow

```
ALREADY ON DEVICE (from sfxt — do not recompute):
  d_succs          [n_nodes]         suffix tree successor per vertex
  d_dists_cache    [n_nodes]         sfxt distances (int ×SCALE_UP)
  d_fanout_adjp    [n_nodes+1]       fanout CSR row pointers
  d_fanout_adjncy  [n_edges]         fanout CSR column indices
  d_fanout_wgts    [n_edges]         fanout edge weights (float)

ACTIVE PATHS (per delta step, from short_pile):
  d_short_pile     [n_active_paths]  PfxtNode array for this delta step
  n_active_paths                     count of active paths

BVSS A_DEV (CPU build → GPU transfer, ONCE before pfxt loop):
  Input:  d_fanout_adjp, d_fanout_adjncy, d_succs
  A_dev = fanout adjacency with succs[u] edge EXCLUDED per vertex u
          (pre-excluded in the mask bits — NO post-filter needed)
  Output: d_realPtrs      [n_intervals+1]
          d_virtualToReal [n_VSS]
          d_rowIds        [n_VSS × τ]
          d_masks_dev     [n_VSS × τ / slicesPerThread]

PER CHAIN SUB-STEP (inside the delta-step expansion loop):
  d_current_v  [n_active_paths]  current chain position per path
                                 initialized from d_short_pile[].to
                                 updated each chain sub-step via succs[]
                                 -1 when path has reached sink

  d_F_frontier [ceil(n_nodes/8)] frontier bitmap: bit u=1 if any active
                                 path is currently at vertex u
                                 zeroed at start of each chain sub-step

  d_Q_active   [n_VSS]           active VSS IDs: VSSs whose frontier
                                 word is nonzero; built from d_F_frontier

  d_dev_pairs  [MAX_PAIRS × 2]   TC output: (u, v) integer pairs
                                 one entry per discovered deviation edge
                                 u is always an active frontier vertex
                                 v is always a non-suffix neighbor of u
                                 (suffix edge pre-excluded from A_dev)
  d_n_pairs    [1]               count of pairs in d_dev_pairs

  (Option B only, when n_active_paths > HYBRID_THRESHOLD):
  d_sorted_v       [n_active_paths]  d_current_v after sort (keys)
  d_path_indices   [n_active_paths]  original path indices (values)
  d_group_ptr      [n_nodes+1]       CSR: range of paths per vertex u
                                     d_path_indices[d_group_ptr[u]..
                                     d_group_ptr[u+1]] = paths at u

  HPQ / LPQ output buffers (same as G-PathGen's existing buffers)
```

---

## Section 1 — BVSS Construction for A_dev

A_dev = full fanout adjacency with one edge excluded per vertex:
for vertex u, exclude the edge u → succs[u] (the suffix edge).
This exclusion is baked into the mask bits — TC output is already
deviation-edge-only, no post-filter needed at runtime.

### Step 1.1 — Build A_dev BVSS on CPU

```
σ = 8   (BLEST default, matches m8n8k128 TC instruction)
τ = 128

For each vertex u:
  For each fanout edge (u → v):
    if v == succs[u]: continue   ← pre-exclude suffix edge
    interval i = v / σ
    mask_u_i |= (1 << (v % σ))

For each interval i:
  Collect all (u, mask_u_i) where mask_u_i ≠ 0 → one slice set
  Partition into VSSs of ≤ τ slices
  Record in realPtrs, virtualToReal, rowIds, masks_dev
```

Build on CPU (using host fanout CSR and host succs[]), transfer to
device once before the pfxt delta-stepping loop begins.

### Step 1.2 — Measure compression ratio

```python
comp_ratio = total_set_bits / (n_unpadded_slices × σ)
```

Expected: slightly higher than sfxt BVSS (~13%) since suffix edge
removal makes masks marginally sparser.

### ⛔ GATE 1 — A_dev BVSS Integrity

Spot-check 100 random vertices. For each u:

```python
expected = set(fanout_adjncy[fanout_adjp[u]:fanout_adjp[u+1]])
expected.discard(succs[u])    # suffix edge excluded

actual = set()
for i in range(n_intervals):
    mask = cpu_row_masks[i].get(u, 0)   # CPU-side mask before transfer
    for bit in range(8):
        if mask & (1 << bit):
            actual.add(i * 8 + bit)

assert expected == actual, f"Vertex {u}: {expected} != {actual}"
```

**Do not proceed until Gate 1 passes.**
A wrong mask silently omits deviation edges or includes suffix edges,
producing incorrect or incomplete path enumeration.

---

## Section 2 — TC Deviation Discovery Kernel

### Step 2.1 — Build frontier bitmap (per chain sub-step)

```cuda
__global__ void build_frontier(
    int*     d_current_v,    // [n_active_paths]
    uint8_t* d_F_frontier,   // [ceil(n_nodes/8)] zeroed before call
    int      n_active_paths)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_active_paths) return;
    int v = d_current_v[tid];
    if (v != -1)
        atomicOr(&d_F_frontier[v/8], (uint8_t)(1 << (v%8)));
}
```

### Step 2.2 — Build active VSS queue from frontier

```cuda
__global__ void build_active_vss_queue(
    uint8_t* d_F_frontier,   // [ceil(n_nodes/8)]
    int*     d_realPtrs,     // [n_intervals+1]
    int*     d_Q_active,     // [n_VSS] output
    int*     d_Q_active_size,// [1] output count
    int      n_intervals)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_intervals) return;
    if (d_F_frontier[tid] == 0) return;  // interval has no active paths
    // enqueue all VSSs for this interval
    for (int v = d_realPtrs[tid]; v < d_realPtrs[tid+1]; v++) {
        int pos = atomicAdd(d_Q_active_size, 1);
        d_Q_active[pos] = v;
    }
}
```

### Step 2.3 — TC deviation discovery

Reuse the fragA/fragB TC layout from `tc_sfxt_bvss.cuh` — copy the
m8n8k128 MMA call directly. Change only: input BVSS (d_masks_dev
instead of sfxt masks) and output (emit (u,v) pairs instead of
writing d_level[]).

Since A_dev already excludes suffix edges, ALL TC-discovered pairs
(u, v) are valid deviation edges. No post-filter required.

```cuda
__global__ void tc_deviation_discover(
    int*     d_realPtrs,
    int*     d_virtualToReal,
    int*     d_rowIds,
    uint8_t* d_masks_dev,      // A_dev BVSS — suffix edges pre-excluded
    uint8_t* d_F_frontier,
    int*     d_Q_active,
    int      n_active_vss,
    int*     d_dev_pairs,      // output: [MAX_PAIRS × 2] (u, v) pairs
    int*     d_n_pairs,        // output: pair count
    int      max_pairs)        // overflow guard
{
    for (int w = warpID; w < n_active_vss; w += n_warps) {
        int vss   = d_Q_active[w];
        int ss    = d_virtualToReal[vss];
        uint8_t alpha = d_F_frontier[ss];
        if (alpha == 0) continue;

        // ---- copy TC MMA call from tc_sfxt_bvss.cuh ----
        // fragC[c] > 0: vertex rowIds[vss*τ + lane*4 + c] has
        // at least one deviation neighbor (from A_dev) in interval ss

        for (int c = 0; c < 4; c++) {
            if (fragC[c] > 0) {
                int u = d_rowIds[vss * τ + lane * 4 + c];
                // find which bits of alpha u connects to (from mask)
                uint8_t hits = get_mask_for_vss_row(u, vss) & alpha;
                while (hits) {
                    int bit = __ffs(hits) - 1;
                    int v   = ss * 8 + bit;
                    int pos = atomicAdd(d_n_pairs, 1);
                    if (pos < max_pairs) {
                        d_dev_pairs[pos * 2]     = u;
                        d_dev_pairs[pos * 2 + 1] = v;
                    }
                    hits &= hits - 1;
                }
            }
        }
    }
}
```

Allocate `d_dev_pairs` with `MAX_PAIRS = n_VSS × τ × σ`
(= total_slices × σ). For netcard: upper bound ~2M pairs.
Add overflow check: if `atomicAdd` returns ≥ MAX_PAIRS, set an
overflow flag and report to user after the kernel — do not silently
drop pairs.

---

## Section 3 — Cost Computation (Hybrid Option A / Option B)

Use **Option B** (sort by current chain position) when
`n_active_paths > HYBRID_THRESHOLD` (start with 1000, tune later).
Use **Option A** (no sort) otherwise.

**Option A** (small batches, n_active_paths ≤ HYBRID_THRESHOLD):

For each (u, v) pair from TC, iterate all paths in d_short_pile
and check if current_v[path] == u. Simple but O(n_pairs × n_paths)
for small n_paths this is acceptable:

```cuda
__global__ void compute_costs_option_a(
    int*      d_dev_pairs, int n_pairs,
    int*      d_current_v,
    PfxtNode* d_short_pile, int n_active_paths,
    int*      d_dists_cache,
    float*    d_fanout_wgts, int* d_fanout_adjp, int* d_fanout_adjncy,
    int alpha, PfxtNode* d_hpq, int* d_hpq_sz,
                PfxtNode* d_lpq, int* d_lpq_sz)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_pairs) return;
    int u = d_dev_pairs[tid*2], v = d_dev_pairs[tid*2+1];

    float wgt = find_edge_weight(d_fanout_adjp, d_fanout_adjncy,
                                 d_fanout_wgts, u, v);
    int shared_cost = d_dists_cache[v]
                    + (int)(wgt * SCALE_UP)
                    - d_dists_cache[u];

    for (int p = 0; p < n_active_paths; p++) {
        if (d_current_v[p] != u) continue;
        int new_slack = d_short_pile[p].slack + shared_cost;
        PfxtNode child = make_pfxt_node(d_short_pile[p], v, new_slack);
        if (new_slack <= alpha) {
            d_hpq[atomicAdd(d_hpq_sz, 1)] = child;
        } else {
            d_lpq[atomicAdd(d_lpq_sz, 1)] = child;
        }
    }
}
```

**Option B** (large batches, n_active_paths > HYBRID_THRESHOLD):

Sort `d_current_v` (keys) with `d_path_indices` (values = 0..n-1)
by ascending current_v, then build `d_group_ptr`:

```cuda
// Step B.1: sort
thrust::sequence(d_path_indices, d_path_indices + n_active_paths);
thrust::sort_by_key(d_current_v, d_current_v + n_active_paths,
                    d_path_indices);

// Step B.2: build group_ptr (one entry per vertex u)
// d_group_ptr[u] = start index in d_path_indices for paths at u
// d_group_ptr[u+1] = end index
build_group_ptr<<<...>>>(d_current_v, n_active_paths,
                          d_group_ptr, n_nodes);
```

Then cost computation reads paths for each (u,v) pair coalesced:

```cuda
__global__ void compute_costs_option_b(
    int*      d_dev_pairs, int n_pairs,
    int*      d_group_ptr,       // [n_nodes+1]
    int*      d_path_indices,    // [n_active_paths] sorted by current_v
    PfxtNode* d_short_pile,      // original (unsorted) path data
    int*      d_dists_cache,
    float*    d_fanout_wgts, int* d_fanout_adjp, int* d_fanout_adjncy,
    int alpha, PfxtNode* d_hpq, int* d_hpq_sz,
                PfxtNode* d_lpq, int* d_lpq_sz)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_pairs) return;
    int u = d_dev_pairs[tid*2], v = d_dev_pairs[tid*2+1];

    float wgt = find_edge_weight(d_fanout_adjp, d_fanout_adjncy,
                                 d_fanout_wgts, u, v);
    int shared_cost = d_dists_cache[v]
                    + (int)(wgt * SCALE_UP)
                    - d_dists_cache[u];

    // Coalesced: all paths at u are contiguous in d_path_indices
    for (int p = d_group_ptr[u]; p < d_group_ptr[u+1]; p++) {
        int path_idx  = d_path_indices[p];
        int new_slack = d_short_pile[path_idx].slack + shared_cost;
        PfxtNode child = make_pfxt_node(d_short_pile[path_idx],
                                        v, new_slack);
        if (new_slack <= alpha) {
            d_hpq[atomicAdd(d_hpq_sz, 1)] = child;
        } else {
            d_lpq[atomicAdd(d_lpq_sz, 1)] = child;
        }
    }
}
```

**Note on alpha type:** confirm from PRELIMINARY that alpha is in the
same unit as new_slack (both int×SCALE_UP or both float). If alpha is
float and new_slack is int: convert before comparison.

**Note on `find_edge_weight`:** linear scan O(degree) per pair.
For netcard: ~4,479 unique (u,v) pairs × ~40 degree = ~180K scans.
Profile before optimizing to a precomputed lookup table.

---

## Section 4 — Chain Advancement

After processing all (u,v) pairs for one chain sub-step, advance
all paths one step along their suffix chain:

```cuda
__global__ void advance_chain(
    int* d_current_v,  // [n_active_paths] updated in-place
    int* d_succs,
    int  n_active_paths,
    int* d_n_active)   // output: count of paths not yet at sink
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_active_paths) return;
    int v = d_current_v[tid];
    if (v == -1) return;
    int next = d_succs[v];
    d_current_v[tid] = next;
    if (next != -1) atomicAdd(d_n_active, 1);
}
```

Termination check: after `advance_chain`, read `d_n_active` to
host. If 0, all paths have reached the sink — exit the chain
sub-step loop.

---

## Section 5 — Integration with G-PathGen Delta-Stepping

Replace `expand_short_pile` calls in the host delta-stepping loop.
All other G-PathGen logic (alpha management, LPQ promotion,
termination) is unchanged.

```cpp
// One-time BVSS construction (before the delta-stepping loop):
build_adev_bvss_cpu(h_fanout_adjp, h_fanout_adjncy, h_succs,
                     n_nodes, /*out*/ h_realPtrs, h_virtualToReal,
                                      h_rowIds, h_masks_dev);
transfer_bvss_to_device(...);

// Per delta step (replaces expand_short_pile):
init_current_v<<<...>>>(d_short_pile, n_active_paths, d_current_v);

int h_n_active = n_active_paths;
while (h_n_active > 0) {
    // Chain sub-step
    cudaMemset(d_F_frontier, 0, n_frontier_bytes);
    build_frontier<<<...>>>(d_current_v, d_F_frontier, n_active_paths);

    cudaMemset(d_Q_active_size, 0, sizeof(int));
    build_active_vss_queue<<<...>>>(d_F_frontier, d_realPtrs,
                                    d_Q_active, d_Q_active_size,
                                    n_intervals);
    int h_n_vss;
    cudaMemcpy(&h_n_vss, d_Q_active_size, sizeof(int), D2H);
    if (h_n_vss == 0) break;

    cudaMemset(d_n_pairs, 0, sizeof(int));
    tc_deviation_discover<<<...>>>(d_realPtrs, d_virtualToReal,
        d_rowIds, d_masks_dev, d_F_frontier,
        d_Q_active, h_n_vss, d_dev_pairs, d_n_pairs, MAX_PAIRS);

    int h_n_pairs;
    cudaMemcpy(&h_n_pairs, d_n_pairs, sizeof(int), D2H);

    if (n_active_paths > HYBRID_THRESHOLD) {
        // Option B: sort then coalesced cost computation
        thrust::sequence(d_path_indices, d_path_indices+n_active_paths);
        thrust::sort_by_key(d_current_v, d_current_v+n_active_paths,
                            d_path_indices);
        build_group_ptr<<<...>>>(d_current_v, n_active_paths,
                                  d_group_ptr, n_nodes);
        compute_costs_option_b<<<...>>>(..., h_n_pairs, ...);
    } else {
        // Option A: simple per-pair scan
        compute_costs_option_a<<<...>>>(..., h_n_pairs, ...);
    }

    cudaMemset(d_n_active, 0, sizeof(int));
    advance_chain<<<...>>>(d_current_v, d_succs, n_active_paths,
                            d_n_active);
    cudaMemcpy(&h_n_active, d_n_active, sizeof(int), D2H);
}
// d_hpq / d_lpq now hold new candidates for this delta step
// merge back into G-PathGen's HPQ/LPQ structures as before
```

---

## Correctness Gates

### ⛔ GATE 1 — A_dev BVSS Integrity
(Defined in Section 1 above. Run before any kernel implementation.)

### ⛔ GATE 4 — Deviation Candidate Correctness

Run TC pfxt on delta step 1 (862 active paths, smallest batch).
Add a dump to `expand_short_pile` and to the TC cost kernel to
capture all (spur_node, dst, new_slack) candidates.

Compare as SCALED integers (both sides use int×SCALE_UP):

```python
tc_cands  = load_int_triples('tc_candidates_step1.txt')
gpg_cands = load_int_triples('gpg_candidates_step1.txt')

tc_sorted  = sorted(tc_cands,  key=lambda x: (x[0], x[1]))
gpg_sorted = sorted(gpg_cands, key=lambda x: (x[0], x[1]))

assert len(tc_sorted) == len(gpg_sorted), \
    f"Count mismatch: TC={len(tc_sorted)} GPG={len(gpg_sorted)}"

# Top-10 cheapest by new_slack (scaled int comparison)
tc_top10  = sorted(tc_cands,  key=lambda x: x[2])[:10]
gpg_top10 = sorted(gpg_cands, key=lambda x: x[2])[:10]

for i, (tc, gpg) in enumerate(zip(tc_top10, gpg_top10)):
    # Tolerance: ≤ SCALE_UP/2 = 5000 (half a unit in float space)
    match = abs(tc[2] - gpg[2]) <= SCALE_UP // 2
    print(f"  Rank {i+1}: TC={tc[2]} GPG={gpg[2]} "
          f"{'MATCH' if match else 'MISMATCH'}")
```

**Pass condition:** same candidate count AND all top-10 costs match
within SCALE_UP//2 (5000).

Common failure causes:
1. Gate 1 not passed — suffix edge still in A_dev masks
2. SCALE_UP applied inconsistently (wgt×SCALE_UP done twice or not
   at all in one path)
3. d_succs wrong vertex — confirm succs[sink] = -1

**Do not proceed to Gate 5 until Gate 4 passes on step 1.
Then run Gate 4 on delta steps 1–5 before Gate 5.**

### ⛔ GATE MEM — Memory Verification (run before K=1M)

**Step M.1 — Confirm d_dev_pairs allocation:**

```bash
grep -n "dev_pairs\|MAX_PAIRS\|max_pairs" \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cu \
    /home/cchang289/Research/gpu-cpg/gpucpg/tc_pfxt_bvss.cuh | head -20
```

Correct allocation: `MAX_PAIRS = peak_active_VSSs × τ × σ`

From Gate 2: peak_active_VSSs = 1,996 (netcard step 9)
Correct: 1,996 × 128 × 8 = 2,043,904 → ~16MB ✓

If using total_VSS (~1.19M for netcard):
1,190,000 × 128 × 8 ≈ 1.22B entries × 8 bytes ≈ 10GB → OOM ✗

Fix before proceeding if wrong.

**Step M.2 — Confirm short_pile buffer capacity:**

```bash
grep -n "short_pile\|long_pile\|resize\|reserve\|capacity" \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cu | head -20
```

Peak batch at K=1M = ~2.1M PfxtNodes (netcard step 9).
Buffer must be pre-allocated ≥ 2.5M entries.

**Step M.3 — Check sizeof(PfxtNode) and available GPU memory:**

```bash
grep -n "sizeof.*PfxtNode\|struct PfxtNode" \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cuh | head -10
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
```

Estimated peak GPU memory for netcard K=1M:
- BVSS (rowIds + masks + ptrs): ~650MB
- short_pile (2.1M × sizeof(PfxtNode)): ?
- HPQ/LPQ (up to 2.1M × sizeof(PfxtNode) each): ?
- d_dev_pairs: ~16MB
- d_group_ptr (3.9M × 4B): ~16MB
- other small buffers: ~50MB

**Show to user:**
```
d_dev_pairs MAX_PAIRS allocation: ? entries (correct = ~2M)
short_pile capacity:   ? entries (needs ≥ 2.5M)
sizeof(PfxtNode):      ? bytes
estimated peak GPU mem: ? MB
GPU memory free:        ? MB
GATE MEM: PASS / FLAG
```

If peak > 14GB (leaving 2GB headroom): flag to user before proceeding.

### ⛔ GATE 5 — Final Top-K Cost Match (K=1M)

Gates at K=1K, 10K, 50K already passed. Now run K=1M on
**des_perf dense40** as the final correctness check before timing.

```python
tc_costs  = sorted(load_float_costs('tc_hpq.txt'))[:K]   # PfxtNode.slack
gpg_costs = sorted(load_float_costs('gpg_hpq.txt'))[:K]

diffs = [abs(a - b) for a, b in zip(tc_costs, gpg_costs)]
print(f"K=1M: max_diff={max(diffs):.6f}  rank={diffs.index(max(diffs))+1}")
```

**Pass condition:** max diff < 1e-3.

Report:
```
K=1000000:
  baseline_count=?  tc_count=?  compared=?
  max_diff=?  max_diff_rank=?
  delta steps reached: baseline=?  TC=?
  largest batch_size=?
  PASS/FAIL
```

**If FAIL:** report rank, baseline cost, TC cost at first mismatch.
Do not start Gate 6 until Gate 5 K=1M passes.

### ⛔ GATE 6 — Timing Breakdown (K=1M)

Only after Gate MEM and Gate 5 K=1M both pass.

**Step G.1 — Confirm all 8 benchmark files are present:**
```
Circuit dense40:
  des_perf  benchmarks/tc_pfxt_dense40/des_perf/  ✓
  vga_lcd   ?  (generate with densify if missing)
  netcard   benchmarks/tc_pfxt_dense40/netcard/   ✓
  leon2     ?  (generate with densify if missing)

DIMACS (no densification):
  ldoor     ?
  cage15    ?
  nlpkkt120 ?
  nlpkkt160 ?
```

Find or generate all missing files. Report to user before any
timed run. Do not begin timing until all 8 files are confirmed.

**Step G.2 — Timing protocol:**
- K=1M on all 8 benchmarks
- G-PathGen baseline re-measured on same machine, same file
- TC pfxt on same machine, same file
- Mean of 10 timed runs, 3 warmup discarded
- Time four components with CUDA events:
  T_tc / T_sort / T_cost / T_adv

```
benchmark  | G-PathGen pfxt (ms) | TC pfxt (ms)           | speedup
           | (re-measured)       | tc / sort / cost / adv |
-----------|---------------------|------------------------|---------
des_perf   | ?                   | ? / ? / ? / ? = ?      | ?x
vga_lcd    | ?                   | ? / ? / ? / ? = ?      | ?x
netcard    | ?                   | ? / ? / ? / ? = ?      | ?x
leon2      | ?                   | ? / ? / ? / ? = ?      | ?x
ldoor      | ?                   | ? / ? / ? / ? = ?      | ?x
cage15     | ?                   | ? / ? / ? / ? = ?      | ?x
nlpkkt120  | ?                   | ? / ? / ? / ? = ?      | ?x
nlpkkt160  | ?                   | ? / ? / ? / ? = ?      | ?x
```

Per-delta-step breakdown for netcard (pre-filled n_paths and
n_active_vss from Gate 2 profiling — verify match at runtime):

```
delta | n_paths   | n_active_vss | T_tc  | T_sort | T_cost | T_adv | total
step  |           |              | (ms)  | (ms)   | (ms)   | (ms)  | (ms)
------|-----------|--------------|-------|--------|--------|-------|------
1     | 862       | 287          | ?     | ?      | ?      | ?     | ?
2     | 1,311     | 211          | ?     | ?      | ?      | ?     | ?
3     | 2,118     | 281          | ?     | ?      | ?      | ?     | ?
4     | 3,744     | 369          | ?     | ?      | ?      | ?     | ?
5     | 7,479     | 506          | ?     | ?      | ?      | ?     | ?
6     | 16,473    | 621          | ?     | ?      | ?      | ?     | ?
7     | 40,418    | 760          | ?     | ?      | ?      | ?     | ?
8     | 107,894   | 839          | ?     | ?      | ?      | ?     | ?
9     | 2,127,820 | 1,996        | ?     | ?      | ?      | ?     | ?
```

Show all data to user. Report honestly. Analyze which component
dominates per step and overall, and why.

---

## Known Concerns

### CONCERN 1 — Edge weight lookup O(degree) per pair

`find_edge_weight` does a linear scan of u's fanout edges to find
wgt(u→v). For netcard: ~4,479 unique (u,v) pairs × ~40 degree ≈
180K scans total per chain sub-step. Profile before optimizing.
If bottleneck: precompute (pair_index → wgt) table during BVSS build.

### CONCERN 2 — d_dev_pairs buffer overflow

Allocate MAX_PAIRS = n_VSS × τ × σ (= total_slices × σ).
For netcard step 9: ~1996 × 128 × 8 ≈ 2M pairs.
Add overflow guard in TC kernel (check return of atomicAdd < MAX_PAIRS).
Report overflow to user — never silently drop pairs.

### CONCERN 3 — HYBRID_THRESHOLD calibration

Start with 1000. Profile delta steps 6–8 (paths/src = 31–130) to
find actual crossover where Option B sort overhead is recovered by
coalesced reads. Tune before final timing runs.

### CONCERN 4 — alpha units

Alpha must be the same unit as new_slack (both int×SCALE_UP or both
float). Confirm in GATE PRELIM. If alpha is float in G-PathGen and
new_slack is int×SCALE_UP in TC, add explicit conversion.

### CONCERN 5 — PfxtNode constructor

`make_pfxt_node(parent, deviation_v, new_slack)` must produce a
PfxtNode identical in layout to what `expand_short_pile` produces.
Read the PfxtNode struct definition and existing construction code
before implementing this function.

### CONCERN 6 — Option B modifies d_current_v (key array)

`thrust::sort_by_key` sorts d_current_v in-place as the key array.
After the sort, d_current_v is sorted (no longer in original path order).
The subsequent `advance_chain` call reads d_current_v — this is fine
since advance_chain updates each entry independently.
But if d_current_v is needed in original order elsewhere, save a copy
before the sort.
