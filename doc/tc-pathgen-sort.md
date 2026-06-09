# TC-PathGen Sort Optimization: Sort-Once Per Delta Step

## Problem

After kernel fusion, phase profiling shows sort_ms dominates for large
batch sizes. For d10 (K=1M, batch sizes 714K–1.26M), the estimated
sort cost across chain sub-steps accounts for ~1500ms of 2023ms TC pfxt.

Currently: `thrust::sort_by_key` runs every chain sub-step (O(n log n)).
The sort rebuilds path ordering by current_v so `compute_costs` can
access path.slack with coalesced reads via group_ptr.

After advance_chain, current_v changes for all active paths (v→succs[v]).
The sort must repeat because group ordering is stale.

**Fix:** keep one O(n log n) sort at the start of each delta step.
For sub-steps 1+, replace with an O(n) counting sort rebuild that
updates group_ptr without a full re-sort.

---

## Why O(n) Counting Sort Works

After advance_chain, each path p moves from current_v to succs[current_v].
We need to update the group_ptr mapping from vertex → path range. This
is a histogram problem, not a comparison sort:

```
count[v] = number of paths currently at vertex v
group_ptr[v] = starting index in d_path_indices for paths at v
```

Rebuilding this takes 3 O(n) passes:
1. Count paths per vertex (n atomicAdds)
2. Prefix scan over counts → group_ptr (thrust::exclusive_scan, O(n))
3. Scatter path indices into groups (n atomicAdds)

For 1.26M paths: counting sort ≈ 3-5ms vs thrust sort ≈ 40ms.

---

## Implementation

### Step S.0 — Enable profiling for this run

Set `GPUCPG_TC_PFXT_PROFILE_PHASES=1` for all Gate SORT runs so
tc_ms/sort_ms/cost_ms/adv_ms are reported per step. This makes
bottlenecks visible after the optimization.

### Step S.1 — Add counting sort rebuild kernels

Three new kernels. Add to `tc_pfxt_bvss.cuh` or a new file:

```cuda
// Kernel 1: count paths per vertex
__global__ void count_paths_per_vertex(
    int* d_current_v,   // [n_active_paths] current chain positions
    int* d_count,       // [n_nodes] output counts, zeroed before call
    int  n_active_paths)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_active_paths) return;
    int v = d_current_v[tid];
    if (v >= 0) atomicAdd(&d_count[v], 1);
}

// Kernel 2: scatter path indices into groups
// Call AFTER thrust::exclusive_scan(d_count → d_group_ptr)
// d_count is reused as a per-vertex write cursor (zeroed after prefix scan)
__global__ void scatter_path_indices(
    int* d_current_v,     // [n_active_paths]
    int* d_path_indices,  // [n_active_paths] output: sorted by current_v
    int* d_group_ptr,     // [n_nodes+1] from prefix scan
    int* d_cursor,        // [n_nodes] write cursors, initialized to d_group_ptr
    int  n_active_paths)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_active_paths) return;
    int v = d_current_v[tid];
    if (v < 0) return;
    int pos = atomicAdd(&d_cursor[v], 1);
    d_path_indices[pos] = tid;   // tid is the original path index
}
```

Host-side counting sort function (called per sub-step 1+):

```cpp
void rebuild_group_ptr(
    int* d_current_v, int* d_path_indices,
    int* d_count, int* d_group_ptr, int* d_cursor,
    int  n_active_paths, int n_nodes,
    cudaStream_t stream)
{
    // Step 1: count
    cudaMemsetAsync(d_count, 0, n_nodes * sizeof(int), stream);
    count_paths_per_vertex<<<ROUNDUP(n_active_paths, BS), BS, 0, stream>>>(
        d_current_v, d_count, n_active_paths);

    // Step 2: prefix scan → group_ptr
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
        d_count, d_count + n_nodes + 1, d_group_ptr);

    // Step 3: scatter path indices into groups
    // Copy group_ptr to cursor (d_cursor = d_group_ptr[0..n_nodes-1])
    cudaMemcpyAsync(d_cursor, d_group_ptr, n_nodes * sizeof(int),
                    D2D, stream);
    scatter_path_indices<<<ROUNDUP(n_active_paths, BS), BS, 0, stream>>>(
        d_current_v, d_path_indices, d_group_ptr, d_cursor,
        n_active_paths);
}
```

**New device buffers needed:**
- `d_count [n_nodes]`  — per-vertex path count (reused across sub-steps)
- `d_cursor [n_nodes]` — per-vertex write cursor (reused across sub-steps)

Allocate once before the delta step loop. n_nodes × 4 bytes each:
netcard = 3.9M × 4 = ~15.6MB per buffer. Acceptable.

### Step S.2 — Restructure the chain walk loop

```cpp
// ── BEFORE CHAIN WALK LOOP (per delta step) ──────────────────────────

// ONE full O(n log n) sort at delta step start
thrust::sequence(thrust::device, d_path_indices, d_path_indices + n_active);
thrust::sort_by_key(thrust::device,
    d_current_v_copy, d_current_v_copy + n_active,  // sort a COPY
    d_path_indices);
// Build initial group_ptr from the sorted copy
rebuild_group_ptr(d_current_v, d_path_indices,
    d_count, d_group_ptr, d_cursor,
    n_active, n_nodes, stream);
// Note: d_current_v itself is NOT sorted (advance_chain reads it in-place)
// d_current_v_copy holds the sorted keys, used only for initial group_ptr

// ── CHAIN WALK LOOP ───────────────────────────────────────────────────
for (int sub = 0; sub < h_max_chain_len; sub++) {

    cudaMemsetAsync(d_F_frontier, 0, n_frontier_bytes, stream);
    cudaMemsetAsync(d_n_pairs,    0, sizeof(int),       stream);

    build_frontier<<<..., stream>>>();
    build_active_vss_queue<<<..., stream>>>();
    tc_deviation_discover<<<..., stream>>>();

    // Use current group_ptr (exact at sub=0, counting-sort at sub>=1)
    compute_costs_device_npairs<<<..., stream>>>(d_group_ptr, d_path_indices, ...);

    advance_chain<<<..., stream>>>();

    // Periodic active check (unchanged from kernel fusion)
    if ((sub + 1) % ACTIVE_CHECK_INTERVAL == 0) {
        cudaMemcpy(&h_n_active, d_n_active, sizeof(int), D2H);
        if (h_n_active == 0) break;
        cudaMemset(d_n_active, 0, sizeof(int));
    }

    // ── COUNTING SORT REBUILD (replaces thrust::sort_by_key) ─────────
    // Rebuild group_ptr for next sub-step in O(n), not O(n log n)
    // current_v already updated by advance_chain above
    if (sub + 1 < h_max_chain_len) {
        rebuild_group_ptr(d_current_v, d_path_indices,
            d_count, d_group_ptr, d_cursor,
            n_active, n_nodes, stream);
    }
}
```

**Key correctness note:** `d_current_v_copy` (used for the initial sort)
is a copy of d_current_v at the start of the delta step. The original
`d_current_v` is updated in-place by advance_chain each sub-step.
The counting sort rebuild reads from `d_current_v` (the updated values),
so group_ptr is always current.

### Step S.3 — Allocate new buffers

```bash
# Find where TC pfxt scratch buffers are allocated
grep -n "d_path_indices\|d_group_ptr\|thrust::device_vector" \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cu | head -20
```

Add allocation for d_count and d_cursor alongside existing buffers.
Allocate once before the delta-stepping loop, reuse across all delta steps.

---

## Expected Improvement

**d10 (K=1M, 2 delta steps, batch 714K–1.26M):**

```
Old sort cost: ~20 sub-steps × (15ms + 40ms) per delta step = ~1100ms total
New sort cost: 2 full sorts + ~38 counting sorts
            = 2×55ms + 38×5ms = 110ms + 190ms = 300ms total
Savings: ~800ms → d10 TC pfxt: 2023ms → ~1220ms → TC/GPG: 10.6× → ~6.4×
```

**d20 (K=1M, 11 delta steps, moderate batches):**

```
Old sort: dominates less (smaller batches per step)
New sort: further reduced
Expected: d20 TC pfxt 928ms → ~600ms → TC/GPG: 3.26× → ~2.1×
```

**d50 (K=100K, 8 delta steps, small-moderate batches):**

```
Sort was already less dominant
Expected: modest improvement, TC/GPG: 4.58× → ~3.5×
```

---

## Correctness Gates

### ⛔ GATE SORT-1 — After counting sort rebuild added

Build and run correctness check on original netcard K=1M with
GPUCPG_TC_PFXT_FUSION=1 AND counting sort rebuild enabled:

```bash
cmake --build build -j8

NETCARD=$(find benchmarks -name "netcard*.txt" | \
    grep -v "d10\|d20\|d30\|d40\|d50\|dense\|crossover" | head -1)

for K in 1000 10000 50000 1000000; do
    GPUCPG_TC_PFXT_FUSION=1 ./build/examples/tc-pfxt-gate5 $NETCARD $K
    # Required: max_diff=0 at every K
done
```

If any K fails: the counting sort scatter is producing wrong groupings.
Check the scatter_path_indices kernel — ensure `d_cursor` is initialized
from `d_group_ptr[0..n_nodes-1]` before scatter, not zeroed.

### ⛔ GATE SORT-2 — Timing with full phase profiling

Run timing on all five densities with both optimizations enabled AND
phase profiling on:

```bash
for D in 10 20 30 40 50; do
    GRAPH=<netcard_d${D}>

    # 1 warmup
    GPUCPG_TC_PFXT_FUSION=1 GPUCPG_TC_PFXT_PROFILE_PHASES=1 \
        ./build/examples/tc-pfxt-gate5 $GRAPH $K_MAX --mode tc-timing \
        > /dev/null

    # 3 measured
    for T in 1 2 3; do
        GPUCPG_TC_PFXT_FUSION=1 GPUCPG_TC_PFXT_PROFILE_PHASES=1 \
            ./build/examples/tc-pfxt-gate5 $GRAPH $K_MAX --mode tc-timing
    done
done
```

Report the full per-step breakdown (tc_ms / sort_ms / cost_ms / adv_ms /
cuda_ms) for each density and the updated crossover table:

```
density | K_max | GPG_ms | TC_no_opt | TC_fusion | TC_sort_once | TC/GPG
--------|-------|--------|-----------|-----------|--------------|-------
d10     | 1M    | 191ms  | 2563ms    | 2023ms    | ?            | ?
d20     | 1M    | 284ms  | 3524ms    |  928ms    | ?            | ?
d30     | 200K  | 82ms   | 2578ms    |  440ms    | ?            | ?
d40     | 200K  | 102ms  | 2224ms    |  467ms    | ?            | ?
d50     | 100K  | 83ms   | 2133ms    |  380ms    | ?            | ?
```

Also report per-step breakdown showing where remaining time goes after
sort-once. This identifies the next bottleneck (likely cost_ms) for the
proposal's optimization roadmap.

---

---

## Unit Tests — Write and Pass Before Any Integration

The counting sort has subtle failure modes. Test each kernel in
isolation before touching the main pfxt loop. All five tests must pass
before Gate SORT-1.

Add tests to `unittests/tc_pfxt_sort.cu` (new file). Run with:
```bash
cmake --build build --target tc_pfxt_sort -j8
./build/unittests/tc_pfxt_sort
```

### Test 1 — count_paths_per_vertex

```cpp
// Hand-checkable case: 8 paths, 5 nodes, one inactive (-1)
int n_nodes = 5, n_paths = 8;
int current_v[] = {0, 2, 0, 4, 2, 0, -1, 2};
// Expected counts: [3, 0, 3, 0, 1]  (-1 skipped)

count_paths_per_vertex<<<1, 32>>>(d_current_v, d_count, n_paths);
// Verify h_count == [3, 0, 3, 0, 1]
// Invariant: sum of counts == 7 (one path has v=-1)
```

**What to catch:** inactive paths (v==-1) must NOT increment any count.
If they do, group_ptr becomes wrong and scatter writes out of bounds.

### Test 2 — scatter_path_indices

```cpp
// group_ptr = [0, 3, 3, 6, 6, 7] from prefix scan of [3,0,3,0,1]
// cursor initialized to group_ptr[0..4] = [0, 3, 3, 6, 6]  ← NOT zeroed

scatter_path_indices<<<1, 32>>>(...);

// Verify: every active path in the right group
for (int v = 0; v < n_nodes; v++)
    for (int i = h_group_ptr[v]; i < h_group_ptr[v+1]; i++)
        assert(h_current_v[h_path_indices[i]] == v);

// Verify: total active entries = 7 (not 8, one was -1)
```

**Most likely failure:** cursor initialized to zero instead of group_ptr.
All paths scatter to position 0 of each group, silently overwriting
each other. Count checks pass but path indices are wrong.

### Test 3 — Full counting sort round-trip (exhaustive)

```cpp
// Random test: n_active=100K, n_nodes=50K
// Invariant 1: every path in the right group
for (int v = 0; v < n_nodes; v++)
    for (int i = h_group_ptr[v]; i < h_group_ptr[v+1]; i++)
        assert(h_current_v[h_path_indices[i]] == v);

// Invariant 2: no path appears twice, no active path is missing
std::vector<bool> seen(n_active, false);
for (int v = 0; v < n_nodes; v++)
    for (int i = h_group_ptr[v]; i < h_group_ptr[v+1]; i++) {
        int p = h_path_indices[i];
        assert(!seen[p]);   // duplicate detection
        seen[p] = true;
    }
for (int i = 0; i < n_active; i++)
    if (h_current_v[i] >= 0)
        assert(seen[i]);    // missing path detection
```

Run for all three edge cases:
- All paths at same vertex
- All paths at distinct vertices (each group size 1)
- Random distribution (above)

### Test 4 — d_current_v_copy does not alias d_current_v

```cpp
// Sort the COPY only, verify original is unchanged
int original[] = {3, 1, 4, 1, 5, 9, 2, 6};
cudaMemcpy(d_current_v,      original, ...);
cudaMemcpy(d_current_v_copy, original, ...);
thrust::sort(thrust::device, d_current_v_copy, d_current_v_copy+8);
cudaMemcpy(h_result, d_current_v, ...);
for (int i = 0; i < 8; i++) assert(h_result[i] == original[i]);
```

**What to catch:** if d_current_v is accidentally sorted instead of the
copy, advance_chain reads stale values → wrong deviation candidates,
silently incorrect results.

### Test 5 — advance_chain then rebuild (end-to-end sub-step simulation)

```cpp
// 5 paths, succs = [3, 4, 3, -1, -1, -1]
// current_v before: [0, 0, 1, 2, -1]
// current_v after advance: [3, 3, 4, -1, -1]

// Step 1: initial group → verify group[0]={0,1}, group[1]={2}, group[2]={3}
rebuild_group_ptr(current_v_before, ...);
// verify...

// Step 2: advance_chain (apply succs to current_v)

// Step 3: rebuild on updated current_v
rebuild_group_ptr(current_v_after, ...);
// Verify: group[3]={0,1}, group[4]={2}, active=3
// Paths 3 and 4 (which had v=-1) must NOT appear anywhere
```

This validates the core invariant: after advance_chain moves paths to
new vertices, rebuild_group_ptr correctly reflects where every path ended up.

---

## Integration Order

```
Test 1 → count_paths_per_vertex passes
Test 2 → scatter_path_indices passes
Test 3+4 → rebuild_group_ptr passes (exhaustive + alias check)
Test 5 → advance+rebuild round-trip passes

Only then: integrate into chain walk loop (Step S.2)
         → Gate SORT-1 (original netcard K=1K/10K/50K/1M, max_diff=0)
         → Gate SORT-2 (timing with GPUCPG_TC_PFXT_PROFILE_PHASES=1)
```

Do not skip steps. If Test 3 fails after Tests 1+2 pass: bug is in
rebuild_group_ptr assembly (prefix scan or buffer sizes), not in the
individual kernels.

---

## Known Concerns

### CONCERN 1 — Counting sort approximation for sub-steps 1+

The counting sort rebuild produces EXACT grouping (not approximate):
`group_ptr[v]` correctly spans all paths where `d_current_v[p] == v`
at that sub-step. The only approximation is that `d_path_indices` within
each group is unordered (atomic scatter, not stable). This is fine:
coalesced read of `d_short_pile[d_path_indices[p]].slack` is only coalesced
when indices are ascending. Unsorted indices within a group slightly reduce
coalescing benefit vs the initial sort.

Impact: cost_ms may be slightly higher than with a fully sorted arrangement,
but significantly better than Option A (no grouping at all).

### CONCERN 2 — d_current_v_copy allocation

The initial sort sorts a copy of d_current_v to avoid disturbing the
original (which advance_chain reads in-place). Allocate d_current_v_copy
once alongside d_path_indices. Size: n_active_paths × 4 bytes.

At peak (d10 step 2, n_active=1.26M): ~5MB. Fine.

### CONCERN 3 — cost_ms may become the next bottleneck

After sort_ms is reduced, cost_ms may dominate. From the pre-fusion
profiling (when phase data was available):

```
d50 step 8: sort_ms=17ms, cost_ms=43ms
```

cost_ms scales with n_pairs (deviation candidates to process).
If sort_ms drops by 80%, cost_ms will be the largest remaining component.
The warp ballot optimization (reduce HPQ/LPQ atomic contention by 32×)
is the natural next fix for cost_ms. Do not implement now — gather data
from GATE SORT-2 first to confirm cost_ms is the bottleneck.
