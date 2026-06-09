# TC-PathGen Stage 1: TC-Accelerated Sfxt Levelization

## Objective

Replace G-PathGen's existing parallel BFS levelization in the sfxt
phase with a BLEST-style Tensor Core BFS. The per-level distance
relaxation (MinPlus arithmetic) stays unchanged as CUDA.

The sfxt phase is a **backward BFS from sinks** (capture FFs) toward
sources (launch FFs). It has two sub-steps:

  1. Backward BFS levelization — assigns each vertex a topological
     level based on its distance from the sinks ← **TC replaces this**
  2. Per-level distance relaxation — computes dist[v] = min over
     successors w of (dist[w] + weight(v,w)) ← **existing CUDA, unchanged**

## Why fanout adjacency (not fanin)

BLEST is pull-based: for vertex u, check if any of u's incoming
neighbors (in the BFS graph direction) are in the frontier.

For sfxt backward BFS, u enters the next frontier if at least one of
u's **fanout** neighbors v (u→v in the timing graph) is already in
the frontier. So the "incoming neighbors in the BFS direction" are
u's fanout destinations in the timing graph.

Therefore: BVSS row u = u's fanout neighbors.
Build BVSS from `_d_fanout_adjp` / `_d_fanout_adjncy`.

This is the opposite of G-PathGen's current **push-based** BFS, which
iterates fanin edges of each frontier vertex. TC pull replaces that
with fanout-based checking of each candidate vertex.

## Complete Data Flow

Every intermediate variable, its source, and its consumer:

```
ALREADY ON DEVICE (from G-PathGen init):
  _d_fanout_adjp   [n_nodes+1]  fanout CSR row pointers
  _d_fanout_adjncy [n_edges]    fanout CSR column indices
  _d_fanout_wgts   [n_edges]    fanout edge weights

ON HOST (from read_input):
  _sinks           [n_sinks]    zero-out-degree vertex IDs
  → must copy to device: d_sinks[n_sinks]

BVSS CONSTRUCTION (CPU, transferred once to device):
  Input:  _d_fanout_adjp, _d_fanout_adjncy
  Output: d_realPtrs      [n_intervals+1]
          d_virtualToReal [n_VSS]
          d_rowIds        [n_VSS × τ]
          d_masks         [n_VSS × τ / slicesPerThread]

TC BFS KERNEL (per BFS level, in-place):
  Input:  d_realPtrs, d_virtualToReal, d_rowIds, d_masks (BVSS)
          d_sinks, n_sinks (first call only, for initialization)
          d_F_curr   [ceil(n_nodes/8)]  current frontier bitmap
          d_Q_curr   [n_VSS]            active VSS queue
          d_Q_curr_size
  Output: d_F_next   [ceil(n_nodes/8)]  next frontier bitmap
          d_Q_next   [n_VSS]            next VSS queue
          d_Q_next_size
  In/Out: d_visited  [ceil(n_nodes/8)]  visited bitmap
          d_level    [n_nodes]          topological level (written once per vertex)

POST-PROCESSING (after TC BFS loop completes):
  Input:  d_level [n_nodes]
  Step 1 (device): thrust::sort_by_key(d_level, d_level+n_nodes, d_queue)
    Produces: d_queue [n_nodes]  vertices sorted by ascending level
  Step 2 (host): cudaMemcpy d_level → h_level_sorted, then scan to build:
    Produces: _h_verts_lvlp [max_level+2]  CPU-side level pointer array

RELAXATION KERNEL (unchanged, per level d):
  Input:  _d_fanout_adjp, _d_fanout_adjncy, _d_fanout_wgts
          d_queue       [n_nodes]       vertices sorted by level
          _h_verts_lvlp [max_level+2]   level pointers (CPU, drives loop)
          d_dists_cache [n_nodes]       initialized: sinks=0, others=INT_MAX
  Output: d_dists_cache [n_nodes]       final shortest distances (sfxt result)
```

Note: `_h_verts_lvlp` and `d_queue` are the same arrays consumed by
the existing `relax_bu_step` loop. The TC BFS produces them as
replacements for what G-PathGen's existing BFS produced.

## Benchmarks

4 circuit graphs (matching G-PathGen Table 1 for direct comparison):
  des_perf, vga_lcd, netcard, leon2

4 DIMACS graphs (matching G-PathGen Table 1):
  ldoor, cage15, nlpkkt120, nlpkkt160

All circuit runs: K=1M, dense40. Baseline: G-PathGen sfxt runtime
from Table 1. DIMACS graphs: same K=1M.

## Graph reordering

Use G-PathGen's existing levelization-based vertex reordering only.
No additional BVSS-specific reordering. BVSS compression ratio for
the sfxt fanout adjacency is measured as a characterization metric.

---

## ⛔ PRELIMINARY — Verify Fanout CSR and Sinks on Device

**Important — push vs pull direction change:**

G-PathGen's existing BFS (`bfs_td_step_privatized`) is **push-based**
and uses `d_fanin_adjp` / `d_fanin_adjncy`:

```cpp
// existing push BFS: for each frontier vertex v,
// iterate v's fanin predecessors u (u→v) and push u to next frontier
bfs_td_step_privatized<<<...>>>(d_fanin_adjp, d_fanin_adjncy, ...);
```

The TC replacement is **pull-based** (BLEST's model) and uses
`d_fanout_adjp` / `d_fanout_adjncy`:

```
// TC pull BFS: for each candidate vertex u,
// check if any of u's fanout neighbors v (u→v) is in the frontier
// BVSS row u = u's fanout neighbors → uses fanout CSR
```

Both compute the same backward BFS from sinks and produce the same
level[] values. The push→pull switch changes which adjacency is
accessed. Do NOT use fanin for BVSS — that would invert the BFS
direction and produce wrong levels.

The BVSS uses fanout edges. Fanout CSR is already on device from
G-PathGen's normal initialization. Verify this and locate sinks.

```bash
# Confirm fanout device arrays exist and when they're allocated
grep -n "_d_fanout_adjp\|_d_fanout_adjncy\|_d_fanout_wgts\|cudaMemcpy.*fanout" \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cu | head -20

# Locate sink vertex storage
grep -n "_sinks\|sinks\b\|zero.*out\|no.*fanout\|d_sinks" \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cu \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cuh | head -20
```

### ⛔ GATE PRELIM — Confirm Before Writing Any Code

Show to user:

```
Fanout CSR device arrays (used for BVSS):
  _d_fanout_adjp:   allocated/transferred at line ?
  _d_fanout_adjncy: allocated/transferred at line ?
  Both on device before sfxt phase begins: YES/NO

Sink vertices:
  Host array name: ?  (e.g. _sinks)
  Count on netcard: ?
  Device copy exists already: YES/NO
  If NO: must cudaMemcpy _sinks → d_sinks before TC BFS init
```

If fanout CSR is not on device before sfxt: flag to user.
If sinks are not on device: add one cudaMemcpy in the host code
before the TC BFS initialization call. Do not proceed until clear.

---

## Section 3 — BVSS Construction (Fanout Adjacency)

Build BVSS over the fanout adjacency. No edge exclusion — full
fanout CSR used. (pfxt will exclude suffix edges; sfxt does not.)

### Step 3.1 — Adapt BLEST's BVSS construction

Borrow BLEST's BVSS construction code. Only change: input CSR is
`_d_fanout_adjp` / `_d_fanout_adjncy` (not fanin, not transposed).

```
σ = 8   (BLEST default, matches m8n8k128 TC instruction)
τ = 128 (WARP_SIZE × slicesPerThread = 32 × 4)

For each vertex u (row in BVSS):
  For each fanout edge (u → v) where v falls in interval i = v/σ:
    mask_u_i |= (1 << (v % σ))   ← v is a fanout destination of u

For each interval i:
  Collect all (u, mask_u_i) where mask_u_i ≠ 0 → one slice set
  Partition into VSSs of ≤ τ slices each
  Record in d_realPtrs, d_virtualToReal, d_rowIds, d_masks
```

Note: BLEST's standard usage builds row u = u's fanin neighbors
(transposed graph). For sfxt, row u = u's **fanout** neighbors.
This is the one structural difference from BLEST's original use.

Build four static arrays on CPU, transfer to GPU once:
- `d_realPtrs[n_intervals+1]`              — interval → VSS range
- `d_virtualToReal[n_VSS]`                 — VSS → interval ID
- `d_rowIds[n_VSS × τ]`                   — vertex IDs per slice (padded)
- `d_masks[n_VSS × τ / slicesPerThread]`  — packed 8-bit connectivity

### Step 3.2 — Measure compression ratio

```python
comp_ratio = total_set_bits / (n_unpadded_slices × σ)
```

Record for all 8 benchmarks. Compare against pfxt's ~13% from the
BVSS study. Expected: similar since both use fanout adjacency;
small difference because sfxt includes suffix edges that pfxt excludes.

### ⛔ GATE 3.1 — BVSS Integrity Check

Spot-check 100 randomly sampled vertices. For each vertex u, confirm
BVSS row u exactly matches u's fanout neighbors from the CSR:

```python
for u in random.sample(range(n_nodes), 100):
    # fanout neighbors from CSR
    csr_nbrs = set(fanout_adjncy[fanout_adjp[u] : fanout_adjp[u+1]])

    # fanout neighbors from BVSS masks
    bvss_nbrs = set()
    for i in range(n_intervals):
        mask = row_masks[i].get(u, 0)  # mask for vertex u in interval i
        for bit in range(σ):
            if mask & (1 << bit):
                bvss_nbrs.add(i * σ + bit)

    assert csr_nbrs == bvss_nbrs, \
        f"Vertex {u}: CSR fanout={csr_nbrs} BVSS row={bvss_nbrs}"

print("GATE 3.1 PASS: BVSS fanout matches CSR for 100 sampled vertices")
```

**Do not proceed to kernel implementation until Gate 3.1 passes.**
A mismatch silently produces wrong level[] values, which corrupts
the d_queue sort and `_h_verts_lvlp`, breaking the relaxation loop.

---

## Section 4 — TC Backward BFS Levelization Kernel

### Step 4.1 — Kernel structure

Follows BLEST Algorithm 2. Two adaptations from BLEST's standard use:
1. BVSS built over fanout (not transposed/fanin) adjacency
2. Output is level[] only — no distance computation

```cuda
__global__ void tc_backward_bfs_levelization(
    // BVSS (built from fanout adjacency)
    int*     d_realPtrs,       // [n_intervals+1]
    int*     d_virtualToReal,  // [n_VSS]
    int*     d_rowIds,         // [n_VSS × τ]
    uint8_t* d_masks,          // packed 8-bit connectivity masks
    // Frontier bitmaps
    uint8_t* d_F_curr,         // [ceil(n_nodes/8)]
    uint8_t* d_F_next,         // [ceil(n_nodes/8)]
    uint8_t* d_visited,        // [ceil(n_nodes/8)]
    // Queues
    int*     d_Q_curr,         // [n_VSS]
    int*     d_Q_next,         // [n_VSS]
    int*     d_Q_curr_size,
    int*     d_Q_next_size,
    // Output
    int*     d_level,          // [n_nodes]
    int      curr_level)
{
    for (int w = warpID; w < *d_Q_curr_size; w += n_warps) {
        int vss   = d_Q_curr[w];
        int ss    = d_virtualToReal[vss];
        uint8_t alpha = d_F_curr[ss];  // 8-bit frontier word for interval ss

        // ---- COPY TC CALL DIRECTLY FROM BLEST SOURCE ----
        // fragA: packed masks from d_rowIds/d_masks for this VSS
        // fragB: alpha word (layout per BLEST Figure 2, XZ/YZ construction)
        // Two m8n8k128 calls → fragC[0..3]
        // fragC[c] > 0 iff vertex rowIds[...][c] has ≥1 fanout neighbor in frontier

        for (int c = 0; c < 4; c++) {
            if (fragC[c] > 0) {
                int u = d_rowIds[vss * τ + lane * 4 + c];
                // Mark u as visited and assign level
                uint8_t old = atomicOr(&d_visited[u/8], 1 << (u%8));
                if (!(old & (1 << (u%8)))) {  // first visit
                    d_level[u] = curr_level;
                    atomicOr(&d_F_next[u/8], 1 << (u%8));
                    // Enqueue all VSSs for u's slice set into d_Q_next
                    int ss_u = u / 8;
                    int vss_beg = d_realPtrs[ss_u];
                    int vss_end = d_realPtrs[ss_u + 1];
                    for (int v = vss_beg; v < vss_end; v++) {
                        int pos = atomicAdd(d_Q_next_size, 1);
                        d_Q_next[pos] = v;
                    }
                }
            }
        }
    }
}
```

**TC call:** copy fragA/fragB/fragC construction verbatim from BLEST
source (`mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32`). The
2-call-per-VSS packing (Figure 2) is non-trivial — do not reimplement.

### Step 4.2 — Initialization kernel

```cuda
__global__ void init_tc_bfs(
    int*     d_sinks,         // [n_sinks]
    int      n_sinks,
    uint8_t* d_F_curr,        // [ceil(n_nodes/8)] zeroed before call
    uint8_t* d_visited,       // [ceil(n_nodes/8)] zeroed before call
    int*     d_Q_curr,        // [n_VSS]
    int*     d_Q_curr_size,
    int*     d_realPtrs,      // to look up VSS range for each sink
    int*     d_level)         // [n_nodes] set to 0 for sinks
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_sinks) return;
    int s = d_sinks[tid];
    d_level[s] = 0;
    atomicOr(&d_F_curr[s/8], 1 << (s%8));
    atomicOr(&d_visited[s/8], 1 << (s%8));
    // enqueue VSSs for s's slice set
    int ss_s = s / 8;
    for (int v = d_realPtrs[ss_s]; v < d_realPtrs[ss_s+1]; v++) {
        int pos = atomicAdd(d_Q_curr_size, 1);
        d_Q_curr[pos] = v;
    }
}
```

### Step 4.3 — Host-side BFS loop

```cpp
// Allocate frontier/queue/level arrays
cudaMemset(d_F_curr, 0, n_words); cudaMemset(d_F_next, 0, n_words);
cudaMemset(d_visited, 0, n_words);
cudaMemset(d_level, -1, n_nodes * sizeof(int));
cudaMemset(d_Q_curr_size, 0, sizeof(int));

// Initialize from sinks (d_sinks must be on device)
init_tc_bfs<<<ROUNDUPBLOCKS(n_sinks, BLOCKSIZE), BLOCKSIZE>>>(
    d_sinks, n_sinks, d_F_curr, d_visited,
    d_Q_curr, d_Q_curr_size, d_realPtrs, d_level);
cudaDeviceSynchronize();

int curr_level = 1;  // sinks are level 0; first BFS step yields level 1
while (true) {
    cudaMemset(d_Q_next_size, 0, sizeof(int));
    cudaMemset(d_F_next, 0, n_words);

    tc_backward_bfs_levelization<<<grid, BLOCKSIZE>>>(
        d_realPtrs, d_virtualToReal, d_rowIds, d_masks,
        d_F_curr, d_F_next, d_visited,
        d_Q_curr, d_Q_next, d_Q_curr_size, d_Q_next_size,
        d_level, curr_level);
    cudaDeviceSynchronize();

    int h_next_size = 0;
    cudaMemcpy(&h_next_size, d_Q_next_size, sizeof(int), cudaMemcpyD2H);
    if (h_next_size == 0) break;

    swap(d_F_curr, d_F_next);
    swap(d_Q_curr, d_Q_next);
    swap(d_Q_curr_size, d_Q_next_size);
    curr_level++;
}
int max_level = curr_level - 1;
```

### Step 4.4 — Produce d_queue and _h_verts_lvlp for relaxation

The existing relaxation loop consumes:
- `d_queue [n_nodes]` — vertices sorted ascending by level (device)
- `_h_verts_lvlp [max_level+2]` — CPU-side level pointer into d_queue

Produce from d_level[]:

```cpp
// Step 1: initialize d_queue = [0, 1, ..., n_nodes-1], then sort by level
thrust::sequence(thrust::device, d_queue, d_queue + n_nodes);
thrust::sort_by_key(thrust::device,
                    d_level, d_level + n_nodes,   // keys
                    d_queue);                       // values

// Step 2: copy sorted level array to host, build level pointer
int* h_level_sorted = new int[n_nodes];
cudaMemcpy(h_level_sorted, d_level, n_nodes*sizeof(int), cudaMemcpyD2H);

_h_verts_lvlp[0] = 0;
for (int d = 0; d <= max_level; d++) {
    _h_verts_lvlp[d+1] = _h_verts_lvlp[d];
    while (_h_verts_lvlp[d+1] < n_nodes &&
           h_level_sorted[_h_verts_lvlp[d+1]] == d)
        _h_verts_lvlp[d+1]++;
}
delete[] h_level_sorted;
```

After this, the existing relaxation loop runs unchanged:

```cpp
for (int d = 1; d < curr_depth; d++) {
    const auto d_beg  = _h_verts_lvlp[d];
    const auto d_end  = _h_verts_lvlp[d+1];
    const auto d_size = d_end - d_beg;
    int num_blks = ROUNDUPBLOCKS(d_size, BLOCKSIZE);
    relax_bu_step<<<num_blks, BLOCKSIZE>>>(
        d_fanout_adjp, d_fanout_adjncy, d_fanout_wgts,  // unchanged
        d_dists_cache,                                    // unchanged
        d_queue,                                          // from TC BFS
        d_size, d_beg);
}
```

Relaxation uses `d_fanout_adjp/adjncy/wgts` — same forward adjacency
as before. TC is not involved in relaxation.

### ⛔ GATE 4 — Level[] Correctness

Compare d_level[] from TC BFS against G-PathGen's existing output.
Add a dump to G-PathGen's existing BFS path to capture its level[]:

```python
tc_levels  = np.loadtxt('tc_levels.txt',  dtype=int)
gpg_levels = np.loadtxt('gpg_levels.txt', dtype=int)

mismatches = np.where(tc_levels != gpg_levels)[0]
print(f"Level mismatches: {len(mismatches)} / {len(tc_levels)}")
for v in mismatches[:10]:
    print(f"  vertex {v}: TC={tc_levels[v]}  G-PathGen={gpg_levels[v]}")
```

**Pass condition:** zero mismatches on all 8 benchmarks.

Common failure causes in order:
1. Sink initialization incomplete — all zero-out-degree vertices must
   be initialized to level 0 in d_level and set in d_visited/d_F_curr
2. fragB layout wrong — re-read BLEST Figure 2 carefully; fragB
   construction for sfxt is the same as BLEST, no adaptation needed
3. Atomic race on d_visited — ensure atomicOr is used, not plain write

### ⛔ GATE 5 — End-to-End dist[] Correctness

After TC level[] → relaxation kernel, verify d_dists_cache matches
G-PathGen's dist[] output:

```python
tc_dists  = np.loadtxt('tc_dists.txt')
gpg_dists = np.loadtxt('gpg_dists.txt')
diffs = np.abs(tc_dists - gpg_dists)
print(f"Max dist diff:  {diffs.max():.6f}")
print(f"Mean dist diff: {diffs.mean():.8f}")
```

**Pass condition:** max diff < 1e-4 on all 8 benchmarks.

**Show Gate 5 to user before any timing measurement.**
Wrong dist[] values silently invalidate pfxt results downstream.

---

## Section 5 — Timing Measurement and Benchmark Results

Only proceed to this section after Gate 5 passes on all 8 benchmarks.

### Step 5.1 — What to time

Measure four components separately using CUDA events:

```
T_bfs      — TC backward BFS loop (all levels, excluding init)
T_sort     — thrust::sort_by_key to produce d_queue
T_lvlp     — cudaMemcpy D2H + _h_verts_lvlp construction on CPU
T_relax    — existing relax_bu_step loop (unchanged CUDA)

T_sfxt_tc  = T_bfs + T_sort + T_lvlp + T_relax
T_sfxt_gpg = G-PathGen's original sfxt (from Table 1, or re-measured
             on same machine for fair comparison)
```

Report T_sfxt_tc vs T_sfxt_gpg as the headline comparison.
Report the four sub-components to identify where time is spent.

Use the existing Timer class in gpu-cpg if it supports per-phase
breakdown. If not, add CUDA event pairs directly around each phase.

### Step 5.2 — Measurement protocol

```cpp
// Warmup: 3 runs discarded
// Timed: 10 runs, report median
// Each run: full sfxt (TC BFS + sort + _h_verts_lvlp + relaxation)

for (int run = 0; run < 13; run++) {
    reset_all_arrays();   // reset d_level, d_visited, d_F_curr, etc.

    cudaEventRecord(t0);
    // --- TC BFS loop ---
    cudaEventRecord(t1);

    // --- thrust sort ---
    cudaEventRecord(t2);

    // --- _h_verts_lvlp build (CPU, timed with std::chrono) ---

    // --- relaxation loop ---
    cudaEventRecord(t3);
    cudaDeviceSynchronize();

    if (run >= 3) record_times(t0, t1, t2, t3);
}
report_median();
```

### Step 5.3 — Also record characterization metrics

For each benchmark, record alongside timing:

```
n_bfs_levels          — total BFS levels (= max_level)
avg_active_vss        — average active VSSs per BFS level
                        (from Q_curr_size at each level)
peak_active_vss       — max Q_curr_size across all levels
bvss_comp_ratio       — from Section 3.2
```

These characterize WHY the TC speedup is what it is and feed
directly into the advisor presentation narrative.

### Step 5.4 — Benchmark table to produce

Produce this table for all 8 benchmarks. Match G-PathGen Table 1
column format for direct comparison:

```
=== TC-PathGen Sfxt Results ===

Circuit graphs (dense40, K=1M):

benchmark  |  G-PathGen sfxt (ms)  |  TC sfxt total (ms)  |  speedup
           |                       |  BFS / sort / relax  |
-----------|---------------------- |---------------------- |---------
des_perf   |  23.3                 |  ? / ? / ?  = ?      |  ?x
vga_lcd    |  23.9                 |  ? / ? / ?  = ?      |  ?x
netcard    |  130.4                |  ? / ? / ?  = ?      |  ?x
leon2      |  146.8                |  ? / ? / ?  = ?      |  ?x

DIMACS graphs (K=1M):

benchmark  |  G-PathGen sfxt (ms)  |  TC sfxt total (ms)  |  speedup
           |                       |  BFS / sort / relax  |
-----------|---------------------- |---------------------- |---------
ldoor      |  234.6                |  ? / ? / ?  = ?      |  ?x
cage15     |  32.4                 |  ? / ? / ?  = ?      |  ?x
nlpkkt120  |  9.8                  |  ? / ? / ?  = ?      |  ?x
nlpkkt160  |  23.0                 |  ? / ? / ?  = ?      |  ?x

Characterization:
benchmark  |  n_levels  |  avg_active_vss  |  peak_active_vss  |  comp_ratio
-----------|------------|------------------|-------------------|------------
des_perf   |  ?         |  ?               |  ?                |  ?
...
```

G-PathGen baseline numbers (from Table 1 in the paper) are filled in
above. Re-measure on the same A4000 machine for a fair comparison
if the paper numbers were from a different run configuration.

### ⛔ GATE 6 — Results Plausibility Check

Before showing results to advisor, verify:

1. **T_relax is unchanged:** relaxation time should match G-PathGen's
   relaxation-only time (if measured separately). If T_relax is
   significantly different, something changed in the relaxation path.

2. **T_bfs + T_sort << T_sfxt_gpg or close:** if TC BFS + sort is
   much slower than G-PathGen sfxt, investigate whether the BVSS
   active set count is unexpectedly large (check peak_active_vss).

3. **n_bfs_levels matches G-PathGen's depth:** the number of BFS
   levels from TC should equal the max level from G-PathGen's BFS.
   A mismatch means levels are wrong despite Gate 4 passing (possible
   if G-PathGen uses a different level convention — e.g. 0-indexed
   vs 1-indexed sinks).

**Show full table to user before advisor meeting.**
It is acceptable if speedup is modest (< 2×) or even negative on
some benchmarks — the characterization data (active VSSs, comp_ratio)
is the primary deliverable for the advisor presentation. The sfxt
stage demonstrates TC integration and establishes the methodology
for pfxt where the real bottleneck is.

---

## Spec Review — Known Concerns and Resolutions

Read this section before starting implementation.

### CONCERN 1 (Critical) — Topological sort vs simple BFS

G-PathGen's `bfs_td_step_privatized` performs **topological sort**, not
simple BFS shortest distance. Each vertex u enters the frontier only
when ALL its fanout successors are processed (deps[u] reaches 0):

```cpp
// deps[u] initialized to out_degree[u]
if (atomicSub(&deps[neighbor], 1) == 1)   // ALL successors done
    depths[neighbor] = curr_depth + 1;
```

This gives level[u] = max(successors' levels) + 1.

The TC pull BFS gives level[u] = min(successors' levels) + 1 (shortest
path to any sink). For vertices with multiple fanout paths of different
lengths, these differ and produce different level[] arrays.

**Why this matters for relaxation:** if u→v exists and both get the
same BFS level, the relaxation processes them in the same pass, reading
dist[v] before it is computed — incorrect result.

**Why it is safe for dense40 circuit graphs:** the densify utility only
adds edges between ADJACENT topological levels (`v_lvl = u_lvl + 1`).
No multi-level skip edges exist. Therefore simple BFS = topological sort
on these specific graphs.

**For DIMACS graphs:** artificial direction (smaller ID → larger ID)
may produce skip edges. Gate 4 will detect any level mismatch.

**Resolution:** implement simple TC pull BFS. Gate 4 must show zero
mismatches before proceeding to timing. If Gate 4 fails on DIMACS
benchmarks, report the mismatch to user — the fix is a deps-based TC
variant, which is left for future work.

### CONCERN 2 — BVSS must be built AFTER graph reordering

G-PathGen applies vertex ID reordering (`UpdateCSR`) as part of its
sfxt phase. If BVSS is built before reordering, the vertex IDs stored
in `d_rowIds` will be pre-reorder IDs that no longer match the actual
graph — wrong edges discovered by TC.

**Resolution:** confirm the order of operations in the sfxt host code.
Build BVSS only after `UpdateCSR` completes and the reordered CSR
(`_d_fanout_adjp`, `_d_fanout_adjncy`) is on device. Add a comment
in the implementation: "BVSS constructed here — reordering complete."

### CONCERN 3 — `d_dists_cache` initialization

The spec never states what `d_dists_cache` must be initialized to
before the relaxation loop. From `relax_bu_step`:

```cpp
auto min_dist{::cuda::std::numeric_limits<int>::max()};
for (fanout edges of v):
    min_dist = min(min_dist, distances[o_neighbor] + wgt * SCALE_UP);
distances[v] = min_dist;
```

It reads `distances[o_neighbor]` — these must be valid before the
first relaxation call.

**Required initialization before relaxation loop:**
```cpp
// All vertices: INT_MAX (uncomputed)
cudaMemset(d_dists_cache, 0x7f, n_nodes * sizeof(int));

// Sinks: 0 (distance to self = 0)
// Launch a small kernel to set d_dists_cache[sink] = 0 for all sinks
init_sink_dists<<<ROUNDUPBLOCKS(n_sinks,BLOCKSIZE), BLOCKSIZE>>>(
    d_sinks, n_sinks, d_dists_cache);
```

Add this initialization between the TC BFS phase and the relaxation
loop. Without it, uninitialized dist values corrupt relaxation results.

### CONCERN 4 — Timing: use mean, 10 runs, same dense40 file

Randy specified 10-run average (not median). Use mean of 10 timed
runs after 3 warmup runs discarded.

Dense40 circuit graphs are generated with random edge insertion
(non-deterministic). All experiments — TC sfxt, G-PathGen baseline,
and later pfxt — must use the **exact same dense40 graph file** per
benchmark. Generate once, save, reuse.

**Before any timing experiment:**
```bash
# Check if dense40 file already exists from Gate P experiments
ls /tmp/gpucpg-spmm-gatep-*/  # check what was generated

# If not present, generate once per benchmark:
./build/examples/densify 40 benchmarks/netcard.txt \
    > benchmarks/netcard_d40.txt

# Verify edge count matches Gate P:
wc -l benchmarks/netcard_d40.txt
```

Use the same file for G-PathGen baseline re-measurement and TC sfxt
measurement. Never regenerate between runs.

### CONCERN 5 — Gate 4: full level distribution, not just per-vertex

Gate 4 must verify two things:

**Check A — Per-vertex level match:**
```python
# level[v] from TC must equal depths[v] from G-PathGen for every vertex
mismatches = np.where(tc_levels != gpg_depths)[0]
assert len(mismatches) == 0
```

**Check B — Level distribution match:**
```python
# Number of vertices per level must also match entirely
tc_hist  = np.bincount(tc_levels,  minlength=max_level+1)
gpg_hist = np.bincount(gpg_depths, minlength=max_level+1)
dist_mismatches = np.where(tc_hist != gpg_hist)[0]

print("Level distribution comparison:")
print(f"  {'level':>6}  {'TC count':>10}  {'G-PathGen count':>15}  {'match':>5}")
for L in range(max_level+1):
    match = "OK" if tc_hist[L] == gpg_hist[L] else "MISMATCH"
    print(f"  {L:>6}  {tc_hist[L]:>10}  {gpg_hist[L]:>15}  {match:>5}")

assert len(dist_mismatches) == 0, \
    f"Level distribution mismatch at levels: {dist_mismatches}"
```

Both checks must pass. Show the full distribution table to user.

### CONCERN 6 — Which array to dump from G-PathGen for Gate 4

G-PathGen's `bfs_td_step_privatized` writes into the `depths`
parameter. Locate the corresponding member variable in CpGen and dump
it after the BFS completes:

```bash
grep -n "d_depths\|_d_depths\|_h_depths\|depths.*cudaMemcpy" \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cu | head -20
```

Copy `d_depths` (device) → host and write to `gpg_depths.txt` for
Gate 4 comparison. Confirm this is the same array that feeds
`_h_verts_lvlp` construction in G-PathGen's existing sfxt code.

### CONCERN 7 — Q_curr / Q_next allocation size

Allocate both `d_Q_curr` and `d_Q_next` with size = `n_VSS` (total
number of virtual slice sets in the BVSS). This can be large — for
netcard at σ=8, n_VSS was ~1.996K active in step 9 but total n_VSS
across all intervals can be much larger. Use the `n_VSS` value
computed during BVSS construction as the allocation size.

If during the BFS a kernel writes beyond `n_VSS` entries (due to
the same VSS being enqueued multiple times), add a bounds check:

```cuda
int pos = atomicAdd(d_Q_next_size, 1);
if (pos < max_q_size)
    d_Q_next[pos] = vss_id;
// else: overflow — flag to user
```

### CONCERN 8 — Relaxation: `curr_depth` naming

In the relaxation loop `for (int d = 1; d < curr_depth; d++)`, the
variable `curr_depth` must equal `max_level` from the TC BFS
(the highest level assigned to any vertex). Make sure these map
correctly when integrating TC into G-PathGen's host code.

### CONCERN 9 — Gate 6: honest data, not framing

Show the complete timing breakdown and characterization data to the
advisor without pre-framing expectations. If TC sfxt is slower,
report it and explain why (e.g., BVSS overhead for a small frontier,
host-sync per level, sort cost). The data should speak for itself.
