# TC-PathGen Preliminary: BVSS Compression Study

## Objective

Measure whether timing graph pfxt frontiers are amenable to
BLEST-style BVSS (Binarised Virtual Slice Sets) representation.
This determines if TC-based frontier BFS is viable for accelerating
the prefix tree expansion phase of G-PathGen.

This is a **characterization study, not a performance optimization.**
The goal is measurement and understanding, not a speedup claim.
Results inform whether to proceed with TC implementation or to
publish the study as a negative/characterization contribution.

**Working directory:** `/home/cchang289/Research/gpu-cpg`

**Existing data:** `/tmp/gpucpg-spmm-gatep-netcard-d40/`
Step dumps from Gate P are already validated and reusable.
Format: `step_N_hops.txt` — columns: `src suffix_next pfx_cost`

---

## Background: BVSS and What We Are Measuring

BLEST partitions the adjacency matrix columns into σ-bit intervals.
For each interval i, a "slice" is a (row_id, σ-bit_mask) pair where
the mask encodes which columns in that interval the row connects to.
Slices for interval i form a "slice set."

A "virtual slice set" (VSS) is a load-balanced chunk of at most τ =
WARP_SIZE × (σ/σ) slices from one slice set.

**Compression ratio** = (# set bits in all masks) / (total capacity
of all masks) = fraction of edges that fall within active slices.

For the pfxt frontier:
- The "frontier" at chain step k = bitmap of which vertices are
  currently being traversed across all active paths
- The "active slice sets" = those whose σ-bit frontier word is nonzero
  (at least one vertex in the frontier falls in that interval)

**What we measure:**
1. How many unique src vertices appear in each step's HOPS table
2. BVSS compression ratio for the timing graph adjacency at σ = 8, 16, 32
3. Fraction of slice sets that are "active" (have at least one frontier bit set)
4. Average slice density within active sets

**Note:** We do NOT expect to match BLEST's social-network compression
ratios. Timing graphs are DAGs with different connectivity patterns.
The measurement itself is the contribution.

---

## ⛔ PRELIMINARY — Read BLEST Paper and Source Code

### Artifact 1 — BLEST paper (primary reference)

Read the full paper first:
```
https://arxiv.org/pdf/2512.21967
```

From the paper, extract and record before writing any code:

- **Section 3 (BVSS data structure):** exact definition of σ, τ,
  slice, slice set, virtual slice set, realPtrs, virtualToReal,
  rowIds, masks arrays
- **Section 3.2 (graph reordering):** compression ratio formula,
  how it is measured, values in Table 1a
- **Table 4:** compression ratios and update divergence for each
  benchmark at σ=8. These are the comparison baselines:
  - GAP-road: 0.14 compression ratio
  - GAP-twitter: 0.14
  - GAP-web: 0.60
  Use these numbers directly. Do NOT run BLEST to reproduce them.
- **Section 4.1 (multiplication pattern):** the m8n8k128 TC
  instruction layout — understand fragA, fragB, fragC layout
  and why only 2 TC calls are needed per VSS instead of 16

### Artifact 2 — G-PathGen paper (your system reference)

The uploaded G-PathGen paper (ICS 2026) is in context. From it,
note specifically:
- Algorithm 1 (Expand): the suffix chain walk (while u ≠ d,
  u = succs[u]) and per-vertex edge scan (warp-parallel over lanes)
- Algorithm 2 (G-PathGen): HPQ/LPQ dual-queue scheduling, alpha
  threshold, window mechanism
- Section 3.4 (Graph Reordering): levelization-based vertex
  reordering already done in G-PathGen — this is relevant for
  whether BVSS compression can be improved via existing reordering
- Table 1: Sfxt and Pfxt runtime breakdown per benchmark —
  use these as ground-truth performance numbers for context

### Artifact 3 — BLEST GitHub source code

```bash
git clone https://github.com/delbek/blest.git
```

From the source, read and understand:
```bash
# Find BVSS construction code
grep -rn "BVSS\|sliceSet\|realPtr\|virtualToReal\|rowIds\|masks\b" \
    blest/ --include="*.cu" --include="*.cuh" \
    --include="*.cpp" --include="*.h" | head -60

# Find the TC kernel and fragA/fragB layout
grep -rn "fragA\|fragB\|fragC\|mma\|MMA\|m8n8k128\|wmma" \
    blest/ --include="*.cu" --include="*.cuh" | head -40

# Find frontier bitmap and queue management
grep -rn "F_curr\|F_next\|Q_curr\|Q_next\|frontier" \
    blest/ --include="*.cu" --include="*.cuh" | head -40
```

Read the identified files in full. Do not guess at data structure
layouts — the TC tile packing (Figure 2 in the paper) is subtle
and must be understood from source before any adaptation.

### Artifact 4 — gpu-cpg source code (your baseline)

```
https://github.com/Randy1005/gpu-cpg/blob/main/gpucpg/gpucpg.cu
```

Focus on the SHORT_LONG code path:
```bash
# Already cloned at /home/cchang289/Research/gpu-cpg
grep -n "expand_short_pile\|SHORT_LONG\|short_pile\|succs\[v\]" \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cu | head -30
```

Read `expand_short_pile` kernel in full. Understand:
- The while loop (suffix chain walk: `v = succs[v]`)
- The inner for loop (edge scan: deviation candidate generation)
- The slack formula: `new_slack = slack + dist[neighbor] + wgt - dist[v]`
  where `slack` is constant per path (parent pfx node cost)
- How short_pile and long_pile correspond to HPQ and LPQ

### ⛔ GATE PRELIM — Understanding Summary

Before writing any analysis code, write a one-paragraph summary
answering all of the following:

```
1. What is σ in BLEST? What value does BLEST use by default?
2. What is a VSS? How does it differ from a slice set?
3. What does "compression ratio" measure precisely?
   (give the formula from the paper)
4. In BLEST's TC kernel: how many TC calls (m8n8k128) are needed
   per VSS? Why is this optimal?
5. In expand_short_pile: what does `v = succs[v]` do?
   What is the global memory access pattern for this line
   when many threads run concurrently?
6. In expand_short_pile: is warp-based expansion ON or OFF
   by default in gpu-cpg? What does this imply for how many
   threads independently read the same adjacency row?
```

**Show answers to user and wait for confirmation before
writing any code.** All subsequent steps depend on this
understanding being correct.

---

## Step 1 — Build Timing Graph Adjacency in BVSS Format

Use the existing adjacency from gpu-cpg (same graph used for Gate P).
Build BVSS over the DEVIATION ADJACENCY: the full adjacency minus
the suffix edge per vertex (the `suffix_next` column from the dumps).

```bash
# The adjacency files already exist from Gate P experiments
# Use benchmark_edges_tfm.txt (transformed edge weights)
# and benchmark_sfxt.txt and benchmark_levels.txt
ls -lh /tmp/gpucpg-spmm-gatep-netcard-d40/

# Additionally need the successor array (succs[v])
# This is the suffix_next column in the dump files:
# For vertex v: its suffix_next = the suffix tree successor
# Build succs[] array from step dumps or from gpu-cpg directly
```

### Step 1a — Extract Successor Array

```python
# build_succs.py
# For each unique src vertex in the hops dumps,
# record its suffix_next (column 2 of step_N_hops.txt)
# This gives succs[src] = suffix_next for active src vertices

import numpy as np
from collections import defaultdict

succs = {}
for step in range(1, 10):
    fname = f'/tmp/gpucpg-spmm-gatep-netcard-d40/step_{step}_hops.txt'
    data = np.loadtxt(fname, dtype=np.float32)
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    for row in data:
        src, suf_next, _ = int(row[0]), int(row[1]), row[2]
        if src not in succs:
            succs[src] = suf_next

print(f"Unique src vertices with known succs: {len(succs)}")
# Save for later use
np.save('succs_netcard.npy', succs)
```

### Step 1b — Build BVSS for Timing Graph

```python
# build_bvss.py
# Build BVSS over A_dev (adjacency minus suffix edges per vertex)
# for σ = 8, 16, 32

import numpy as np
from collections import defaultdict

def build_bvss(edges_file, succs, n_nodes, sigma):
    """
    Build BVSS data structure for deviation adjacency.

    edges_file: transformed edge weights (src dst wgt per line)
    succs: dict mapping src -> suffix_next (excluded edge per vertex)
    sigma: slice width in bits
    """
    n_intervals = (n_nodes + sigma - 1) // sigma

    # Parse edges, exclude suffix edges
    slice_sets = defaultdict(list)  # interval -> list of (row_id, mask)
    row_masks   = defaultdict(lambda: defaultdict(int))

    print(f"Loading edges from {edges_file}...")
    with open(edges_file) as f:
        for line in f:
            parts = line.split()
            src, dst = int(parts[0]), int(parts[1])
            # Skip suffix edge for this src
            if succs.get(src) == dst:
                continue
            interval = dst // sigma
            bit_pos  = dst % sigma
            row_masks[interval][src] |= (1 << bit_pos)

    # Convert to slice sets
    total_slices = 0
    total_bits   = 0
    active_intervals = 0
    for interval, rows in row_masks.items():
        slices = list(rows.items())  # (row_id, mask) pairs
        slice_sets[interval] = slices
        total_slices += len(slices)
        total_bits   += sum(bin(mask).count('1') for _, mask in slices)
        active_intervals += 1

    # Compute compression ratio
    max_bits = total_slices * sigma
    comp_ratio = total_bits / max_bits if max_bits > 0 else 0

    stats = {
        'sigma':             sigma,
        'n_nodes':           n_nodes,
        'n_intervals':       n_intervals,
        'active_intervals':  active_intervals,
        'total_slices':      total_slices,
        'total_bits':        total_bits,
        'compression_ratio': comp_ratio,
        'avg_slice_density': comp_ratio,
    }
    return slice_sets, stats

# Run for σ = 8, 16, 32
edges_file = 'benchmark_edges_tfm.txt'
succs = np.load('succs_netcard.npy', allow_pickle=True).item()
n_nodes = max(max(succs.keys()), max(succs.values())) + 1

for sigma in [8, 16, 32]:
    _, stats = build_bvss(edges_file, succs, n_nodes, sigma)
    print(f"\n=== σ={sigma} ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
```

### ⛔ GATE 1 — Static BVSS Compression Ratio

**Pass condition for proceeding:**
- `compression_ratio` > 0.05 for at least one σ value
  (at least 5% bit density within active slices — if lower, TC
  tiles waste too many operations on zero bits)

**Record and show to user:**
```
σ   | n_intervals | active_intervals | total_slices | comp_ratio
----|-------------|------------------|--------------|----------
8   | ?           | ?                | ?            | ?
16  | ?           | ?                | ?            | ?
32  | ?           | ?                | ?            | ?
```

**Interpretation:**
- High comp_ratio (> 0.3): adjacency is locally dense → TC very efficient
- Medium comp_ratio (0.05–0.3): TC is viable with some waste
- Low comp_ratio (< 0.05): TC tiles mostly empty → TC not beneficial

**This is a characterization result regardless of outcome.**
Even low compression is a publishable finding: timing graphs have
fundamentally different BVSS structure than social/road networks.
Show to user before proceeding to Step 2.

---

## Step 2 — Frontier Compression Per Step

Measure how many slice sets are ACTIVE at each pfxt chain step.
The frontier at step k = set of unique src vertices in the hops dump
at relative depth k within each path's suffix chain traversal.

```python
# frontier_compression.py
# For each delta-stepping step, reconstruct the frontier at each
# suffix chain depth and compute the active VSS fraction.

import numpy as np
from collections import defaultdict

def frontier_active_ratio(hops_file, n_nodes, sigma, succs):
    """
    Load hops dump, group by suffix chain depth,
    measure active slice set fraction per depth.
    """
    # Each row: (src, suffix_next, pfx_cost)
    # src = current chain position (the "frontier" vertex)
    data = np.loadtxt(hops_file)
    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    src_nodes = data[:, 0].astype(int)
    n_intervals = (n_nodes + sigma - 1) // sigma

    # Count unique active intervals
    active_intervals = set(v // sigma for v in src_nodes)
    n_active = len(active_intervals)
    n_total  = n_intervals

    active_ratio = n_active / n_total
    unique_srcs  = len(set(src_nodes))

    return {
        'n_hops':         len(src_nodes),
        'unique_srcs':    unique_srcs,
        'n_active_sets':  n_active,
        'n_total_sets':   n_total,
        'active_ratio':   active_ratio,
        'avg_paths_per_src': len(src_nodes) / unique_srcs
    }

sigma = 8  # use best σ from Gate 1
n_nodes = ...  # from graph stats

print(f"{'step':>4} | {'n_hops':>10} | {'unique_srcs':>11} | "
      f"{'active_sets':>11} | {'active_ratio':>12} | "
      f"{'paths_per_src':>13}")
print("-" * 75)

for step in range(1, 10):
    fname = f'/tmp/gpucpg-spmm-gatep-netcard-d40/step_{step}_hops.txt'
    stats = frontier_active_ratio(fname, n_nodes, sigma, succs)
    print(f"{step:>4} | {stats['n_hops']:>10,} | "
          f"{stats['unique_srcs']:>11,} | "
          f"{stats['n_active_sets']:>11,} | "
          f"{stats['active_ratio']:>12.4%} | "
          f"{stats['avg_paths_per_src']:>13.1f}")
```

### ⛔ GATE 2 — Frontier Sparsity Characterization

**Record and show to user — this is the key table:**

```
step | n_hops     | unique_srcs | active_sets | active_ratio | paths/src
-----|------------|-------------|-------------|--------------|----------
1    | ?          | ?           | ?           | ?%           | ?
2    | ?          | ?           | ?           | ?%           | ?
...
9    | 6,706,231  | 4,479       | ?           | ?%           | 1,497
```

**Key questions to answer from this table:**

Q1. Does the active_ratio stay low (<5%) across all steps?
  → Confirms local growth property holds throughout, not just step 9

Q2. Does paths_per_src increase monotonically with step number?
  → Confirms exponential path growth is concentrated on fewer unique nodes

Q3. At what step does unique_srcs stabilize?
  → This is the inflection point where TC advantage is highest

**No pass/fail for Gate 2 — this is pure characterization.**
Show table to user and discuss interpretation before Step 3.

---

## Step 3 — Repeat on Other Benchmarks

Run the same frontier compression analysis on leon2 and des_perf
using their Gate P dumps (or generate new dumps following same
procedure from Gate P spec).

```python
# For each benchmark, compute the same table as Gate 2
benchmarks = ['des_perf', 'leon2', 'leon3mp']
for bm in benchmarks:
    dump_dir = f'/tmp/gpucpg-spmm-gatep-{bm}-d40/'
    # ... same analysis as above
```

If Gate P dumps don't exist for other benchmarks, generate them
following the same procedure used for netcard: modify gpu-cpg to
dump per-step HOPS and run on each benchmark with K=1M dense40.

### ⛔ GATE 3 — Cross-Benchmark Consistency

**Comparison table:**

```
benchmark  | σ=8 comp_ratio | avg active_ratio | max paths/src
-----------|----------------|------------------|---------------
netcard    | ?              | ?                | ?
leon2      | ?              | ?                | ?
des_perf   | ?              | ?                | ?
```

**Pass condition:** active_ratio < 10% on at least 2 of 3 benchmarks.
This confirms the local growth property is general, not benchmark-specific.

**Show to user before writing the conclusion.**

---

## Step 4 — Comparison with BLEST Benchmark Graphs

Run the same BVSS compression analysis on one of BLEST's benchmark
graphs (e.g., GAP-road or europe_osm from SuiteSparse) to directly
compare timing graph structure vs general graph structure.

```python
# Download one non-social BLEST benchmark in MTX format
# (e.g., from SuiteSparse Matrix Collection)
# Build BVSS and compute compression ratio
# Compare directly against netcard at same σ
```

This gives the paper's key comparison:
"Timing graphs exhibit X% compression vs Y% for road/social networks
at σ=8, due to Z structural difference (DAG, bounded fan-out, etc.)"

---

## Final Output — Characterization Report

After all gates, produce:

```
=== BVSS Compression Study: Timing Graphs vs General Graphs ===

Hardware: NVIDIA RTX A4000
Benchmarks: netcard, leon2, des_perf (dense40, K=1M)

1. STATIC BVSS COMPRESSION RATIO (Gate 1):
   σ   | timing (netcard) | BLEST (road graph)
   8   | ?%               | ~14% (from BLEST paper Table 4)
   16  | ?%               | ?
   32  | ?%               | ?

2. FRONTIER SPARSITY PER STEP (Gate 2, netcard):
   [Table from Gate 2]

3. CROSS-BENCHMARK CONSISTENCY (Gate 3):
   [Table from Gate 3]

4. KEY FINDINGS:
   [ ] Local growth holds: active_ratio < ?% across all steps
   [ ] Path sharing grows with steps: paths/src peaks at ?× in step 9
   [ ] Timing graph BVSS compression is ?× lower/higher than road graphs
       due to [DAG structure / bounded fan-out / topological ordering]
   [ ] TC is viable/not viable for timing graph pfxt based on these results

5. DECISION:
   If compression > 0.05 and active_ratio < 10%:
     → Proceed to TC kernel implementation
   Else:
     → Publish as characterization: timing graphs are fundamentally
       different from BLEST's target graphs; TC not beneficial for pfxt
       at current graph structure. Recommend future work: graph
       reordering to improve BVSS compression for timing graphs.
```

**Show complete report to user. User makes final decision on
whether to proceed to TC implementation.**

---

## Notes for Agent

**Do not implement any TC kernels in this spec.** This is a
measurement-only study. The only code written is Python analysis
on existing dump files + BVSS construction on CPU.

**Default σ for main analysis:** use σ=8 (matching BLEST's default
for the m8n8k128 TC instruction). Sweep σ=8,16,32 for completeness.

**BLEST paper reference for comparison numbers:**
- Table 4 in the BLEST paper reports compression ratios and update
  divergence for each benchmark. Use these as comparison baseline.
- GAP-road: compression ratio 0.14 (14%) at σ=8 with RCM ordering.
- GAP-twitter: compression ratio 0.14 (14%) at σ=8 with Jaccard ordering.
- Compare against timing graph compression at same σ.

**On the warp vs thread distinction:**
G-PathGen's warp-based expansion (W-Pfxt) is OFF by default. This
means currently one thread per path walks its suffix chain. With TC,
we target replacing this per-thread chain walk with a batched
frontier BFS. The baseline for comparison is the single-thread
per-path expand_short_pile kernel, not the warp variant.
