# Triton Suffix BFS: Minimal Working Experiment Spec

## Objective

Implement the suffix distance backward propagation step of CPG as a
Triton GPU kernel. Validate correctness against gpu-cpg's own sfxt output.
Measure runtime against gpu-cpg's CUDA suffix-distance build for the
same graph.

This is the smallest self-contained experiment that validates whether
Triton is viable for CPG workloads. Do not implement deviation path
enumeration or the full CPG pipeline — suffix BFS only.

**Success definition:**
- Triton produces suffix distances that exactly match gpu-cpg's sfxt output
  after mirroring gpu-cpg's scaled-integer distance semantics
- Primary runtime: Triton end-to-end suffix build is within 2x of gpu-cpg's
  CUDA end-to-end suffix build on the same graph
- Diagnostic runtime: Triton relax-only runtime is reported against CUDA
  relax-only runtime to isolate kernel quality
- Same source code compiles and runs without modification (portability
  validated later on a second GPU)

---

## Background: What is Suffix Distance?

In the pessimism-free CPG formulation, the suffix distance at node V
is the minimum cost path from V to any sink (capture FF endpoint).

```
suffix_dist[sink]    = 0              for all sink nodes
suffix_dist[V]       = min over successors W of:
                           edge_weight(V → W) + suffix_dist[W]
```

This is computed as a backward BFS from all sinks simultaneously,
processing nodes in reverse topological order (level by level from
sinks toward sources). At each level, all nodes at that level are
processed independently — this is the parallel step Triton will own.

---

## Prerequisites

### Environment

```bash
# Python 3.9+
pip install triton          # OpenAI Triton
pip install torch           # needed for Triton tensor management
pip install numpy scipy     # for correctness validation

# Verify Triton sees the GPU
python3 -c "import triton; print(triton.__version__)"
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: something like "NVIDIA RTX A4000"
```

### Dumps needed from gpu-cpg (produced in PRELIMINARY below)

```
benchmark_sfxt.txt          -- sfxt distance per vertex from gpu-cpg
benchmark_edges_tfm.txt     -- transformed edge weights from gpu-cpg
benchmark_levels.txt        -- topological level per vertex from gpu-cpg
benchmark_level_order.txt   -- vertices in gpu-cpg levelized queue order
```

These are produced by the PRELIMINARY step. Do not proceed past
Prerequisites until the PRELIMINARY gate passes.

---

## ⛔ PRELIMINARY — Set Up gpu-cpg as Working Directory

All work — sfxt dumps, correctness reference, and performance baseline —
comes from a single source: `/home/cchang289/Research/gpu-cpg`.
Do not use any other tool for dumps. This keeps the validation
self-contained: Triton sfxt output is compared against gpu-cpg's own
sfxt output values, and runtime is compared against gpu-cpg's CUDA
kernel on the same binary.

### Step P.1 — Read gpu-cpg host drivers to understand the interface

Before running anything, read the host code to understand how gpu-cpg
is invoked, what arguments it expects, and where sfxt values are
computed and stored:

```bash
cat /home/cchang289/Research/gpu-cpg/examples/cpg.cu
cat /home/cchang289/Research/gpu-cpg/examples/gen-big-table.cu
```

From this file, identify:
- The CLI arguments required to run gpu-cpg on a benchmark
- Which function call triggers the sfxt (suffix distance) computation
- What data structures hold sfxt values after computation
- Whether sfxt values are already written to any output file, and whether
  the format is per-vertex or levelized/topological order
- Which timing fields map to levelization, CSR reorder, and relaxation

### Step P.2 — Add/verify gpu-cpg dumps

gpu-cpg already has `dump_succ_dists()`, but verify its semantics before
using it as the correctness reference. The Triton reference file must be
one line per original vertex id:

```
<vertex_id> <sfxt_dist>
```

If the existing dump is not per-vertex by original id, add a new dump next
to the suffix-distance computation.

```bash
# Search for existing sfxt dump in gpu-cpg
grep -rn "sfxt\|suffix_dist\|dump\|write" \
    /home/cchang289/Research/gpu-cpg \
    --include="*.cpp" --include="*.cu" --include="*.hpp" | \
    grep -i "dump\|write\|output" | head -20
```

If no compatible dump exists, add one adjacent to the sfxt computation:

```cpp
// After sfxt computation completes, dump per-vertex sfxt distances
// Format: <vertex_id> <sfxt_dist>  (one line per vertex)
void dump_sfxt(const std::string& path) {
    std::ofstream f(path);
    // Use gpu-cpg's actual field names from CpGen context
    for (size_t i = 0; i < n_vertices; i++) {
        f << i << " " << static_cast<float>(sfxt_dist_scaled[i]) / SCALE_UP << "\n";
    }
    f.close();
}
```

Also dump transformed edge weights, per-node levels, and the levelized
vertex order. Use these filenames consistently throughout:

```
benchmark_sfxt.txt          -- sfxt distance per vertex
benchmark_edges_tfm.txt     -- transformed edge weights
benchmark_levels.txt        -- topological level per vertex
benchmark_level_order.txt   -- vertex id at each position in gpu-cpg queue order
```

Note: `dump_lvls()` currently prints level sizes, not per-node levels.
Either add a per-node level dump from `_h_queue + _h_verts_lvlp`, or
write both `benchmark_levels.txt` and `benchmark_level_order.txt` from the
same levelized queue after `levelize()`.

### Step P.3 — Run gpu-cpg on benchmarks to produce dumps

Using the CLI arguments identified from `examples/cpg.cu`, run gpu-cpg
on `vga_lcd` first, then `leon2` and `netcard`:

```bash
cd /home/cchang289/Research/gpu-cpg

# Run on vga_lcd (smallest benchmark first)
# IMPORTANT: do not run this placeholder literally.
# Read examples/cpg.cu first, then construct the actual command.
# Example structure only — exact flags depend on cpg.cu:
# ./gpu-cpg --graph benchmarks/vga_lcd.edges --k 1000 [other flags]

# Confirm dump files exist and are non-empty
ls -lh benchmark_sfxt.txt benchmark_edges_tfm.txt benchmark_levels.txt benchmark_level_order.txt
wc -l  benchmark_sfxt.txt benchmark_edges_tfm.txt benchmark_levels.txt benchmark_level_order.txt
```

Repeat for leon2 and netcard before proceeding to Gate P.
### ⛔ GATE P — gpu-cpg Dump Correctness

**Verify the dumps are consistent before building anything in Triton.**

```python
# gate_preliminary.py
import numpy as np

# Load dumps
suffix = {}
with open('benchmark_sfxt.txt') as f:
    for line in f:
        node, dist = line.split()
        suffix[int(node)] = float(dist)

edges = []
with open('benchmark_edges_tfm.txt') as f:
    for line in f:
        src, dst, w = line.split()
        edges.append((int(src), int(dst), float(w)))

levels = {}
with open('benchmark_levels.txt') as f:
    for line in f:
        node, level = line.split()
        levels[int(node)] = int(level)

# Check 1: sfxt values are finite (no inf or NaN)
inf_count = sum(1 for v in suffix.values()
                if v == float('inf') or v != v)
if inf_count == 0:
    print(f"Check 1 PASS: all {len(suffix)} sfxt values are finite")
else:
    print(f"Check 1 FAIL: {inf_count} nodes have inf/NaN sfxt values")
    exit(1)

# Check 2: Bellman-Ford condition holds for sampled edges
# For every edge (u → v): sfxt[u] <= weight(u,v) + sfxt[v]
# (sfxt[u] is the minimum cost to sink, cannot exceed any single path)
# Note: do NOT assume sfxt[sink] = 0; sinks may have non-zero values
# in the pessimism-free formulation. Check relative consistency only.
violations = 0
for src, dst, w in edges[:10000]:  # check first 10K edges
    if src not in suffix or dst not in suffix:
        continue
    if suffix[src] > w + suffix[dst] + 1e-4:
        violations += 1
        if violations <= 5:  # print first 5 only
            print(f"  VIOLATION: sfxt[{src}]={suffix[src]:.4f} > "
                  f"w={w:.4f} + sfxt[{dst}]={suffix[dst]:.4f}")
if violations == 0:
    print("Check 2 PASS: sfxt values satisfy Bellman-Ford condition")
else:
    print(f"Check 2 FAIL: {violations} violations in first 10K edges")
    print("Do not proceed. gpu-cpg sfxt dump may be incorrect.")
    print("If violations are borderline (< 1e-3), show to user before exiting.")
    exit(1)

# Check 3: verify level ordering is consistent with edges
# Every edge (u → v) must have level[u] > level[v]
# (u is closer to source, v is closer to sink)
level_violations = 0
for src, dst, w in edges[:10000]:
    if levels[src] <= levels[dst]:
        level_violations += 1
if level_violations == 0:
    print("Check 3 PASS: topological levels are consistent with edges")
else:
    print(f"Check 3 FAIL: {level_violations} edges violate topological order")
    exit(1)

print("\nAll GATE P checks passed. Proceed to CSR construction.")
```

**Show output to user before proceeding.**

---

## Step 1 — Build CSR Representation

Convert the gpu-cpg dumps into CSR format for Triton.

The suffix BFS processes in REVERSE topological order (from sinks at
level 0 upward toward sources). The CSR adjacency matrix is built in
the BACKWARD direction: for each node V, store its SUCCESSORS (nodes
that V points to), since suffix BFS reads successor suffix distances.

```python
# build_csr.py
import numpy as np
import torch

def build_backward_csr(edges_file, levels_file, n_nodes):
    """
    Build CSR for backward suffix BFS.
    For each node V, row V contains V's successors (outgoing edges).
    During suffix BFS: suffix[V] = min over successors W of
                                   (weight[V→W] + suffix[W])
    """
    # Load edges. Mirror gpu-cpg's distance arithmetic: weights are scaled
    # to int by SCALE_UP before relaxation.
    SCALE_UP = 10000
    src_nodes, dst_nodes, weights = [], [], []
    with open(edges_file) as f:
        for line in f:
            s, d, w = line.split()
            src_nodes.append(int(s))
            dst_nodes.append(int(d))
            weights.append(int(float(w) * SCALE_UP))

    # Sort by source node for CSR construction
    order = np.argsort(src_nodes)
    src_nodes = np.array(src_nodes)[order]
    dst_nodes = np.array(dst_nodes)[order]
    weights   = np.array(weights)[order]

    # Build row_ptr
    row_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
    for s in src_nodes:
        row_ptr[s + 1] += 1
    np.cumsum(row_ptr, out=row_ptr)

    # Load levels
    node_levels = np.zeros(n_nodes, dtype=np.int32)
    with open(levels_file) as f:
        for line in f:
            node, level = line.split()
            node_levels[int(node)] = int(level)

    # Group nodes by level for levelized processing
    max_level = node_levels.max()
    level_groups = []
    for lv in range(max_level + 1):
        nodes_at_level = np.where(node_levels == lv)[0].astype(np.int32)
        level_groups.append(nodes_at_level)

    # Move to GPU
    row_ptr_gpu  = torch.tensor(row_ptr,  dtype=torch.int32).cuda()
    col_idx_gpu  = torch.tensor(dst_nodes, dtype=torch.int32).cuda()
    weights_gpu  = torch.tensor(weights,  dtype=torch.int32).cuda()
    node_levels_gpu = torch.tensor(node_levels, dtype=torch.int32).cuda()

    return row_ptr_gpu, col_idx_gpu, weights_gpu, level_groups, max_level

# Usage:
# row_ptr, col_idx, weights, level_groups, max_level = \
#     build_backward_csr('benchmark_edges_tfm.txt',
#                        'benchmark_levels.txt',
#                        n_nodes)
```

### ⛔ GATE 1 — CSR Integrity

```python
# gate1_csr.py

# Check 1: row_ptr is monotone non-decreasing
assert all(row_ptr[i] <= row_ptr[i+1] for i in range(n_nodes)), \
    "row_ptr is not monotone"
print("Check 1 PASS: row_ptr is monotone")

# Check 2: total edges match
n_edges = len(col_idx)
assert row_ptr[-1] == n_edges, \
    f"row_ptr[-1]={row_ptr[-1]} != n_edges={n_edges}"
print(f"Check 2 PASS: edge count consistent ({n_edges} edges)")

# Check 3: col_idx values are in valid range
assert col_idx.min() >= 0 and col_idx.max() < n_nodes, \
    "col_idx out of range"
print("Check 3 PASS: col_idx values in valid range")

# Check 4: all nodes appear in some level group
total_in_groups = sum(len(g) for g in level_groups)
assert total_in_groups == n_nodes, \
    f"Level groups cover {total_in_groups} nodes but n_nodes={n_nodes}"
print(f"Check 4 PASS: all {n_nodes} nodes accounted for in level groups")

# Check 5: level 0 nodes (sinks) have no outgoing edges
for sink in level_groups[0]:
    degree = row_ptr[sink+1] - row_ptr[sink]
    assert degree == 0, f"Sink {sink} has {degree} outgoing edges"
print(f"Check 5 PASS: all {len(level_groups[0])} sinks have degree 0")

# Print summary
print(f"\nCSR summary:")
print(f"  nodes:       {n_nodes}")
print(f"  edges:       {n_edges}")
print(f"  max level:   {max_level}")
print(f"  nodes/level: min={min(len(g) for g in level_groups)} "
      f"max={max(len(g) for g in level_groups)} "
      f"mean={n_nodes/(max_level+1):.1f}")
print(f"  max out-degree: {max(row_ptr[i+1]-row_ptr[i] for i in range(n_nodes))}")
```

**Show output to user. Pay attention to max out-degree — if it's > 64,
the naive Triton kernel may need adjustment. Show to user before proceeding.**

---

## Step 2 — Triton Kernel

The kernel processes one topological level per launch. All nodes at
the same level are processed in parallel — each Triton program instance
handles a contiguous block of nodes.

```python
# triton_suffix_bfs.py
import triton
import triton.language as tl
import torch

@triton.jit
def suffix_bfs_kernel(
    row_ptr_ptr,    # int32[n_nodes+1] CSR row pointers
    col_idx_ptr,    # int32[n_edges]   successor node indices
    weights_ptr,    # int32[n_edges]   transformed edge weights scaled by SCALE_UP
    suffix_ptr,     # int32[n_nodes]   suffix distances scaled by SCALE_UP
    nodes_ptr,      # int32[n_level_nodes] nodes at this level
    n_level_nodes,  # number of nodes at this level
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance handles BLOCK_SIZE nodes at this level
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_level_nodes

    # Load node IDs for this block
    node_ids = tl.load(nodes_ptr + offsets, mask=mask, other=0)

    # Load row pointers for these nodes
    row_start = tl.load(row_ptr_ptr + node_ids,     mask=mask, other=0)
    row_end   = tl.load(row_ptr_ptr + node_ids + 1, mask=mask, other=0)
    degree    = row_end - row_start

    # Initialize min value to INT_MAX, matching gpu-cpg dists_cache.
    INT_MAX: tl.constexpr = 2147483647
    min_val = tl.full([BLOCK_SIZE], INT_MAX, dtype=tl.int32)

    # Find max degree in this block for loop bound
    # (static upper bound needed by Triton — set conservatively)
    MAX_DEGREE: tl.constexpr = 64  # adjust if max out-degree > 64

    for d in range(MAX_DEGREE):
        # Mask out nodes that don't have this many successors
        valid = (d < degree) & mask

        # Load successor index and weight for this successor slot
        edge_idx     = row_start + d
        successor_id = tl.load(col_idx_ptr + edge_idx, mask=valid, other=0)
        edge_weight  = tl.load(weights_ptr  + edge_idx, mask=valid,
                                other=INT_MAX)

        # Load successor's current suffix distance
        succ_dist = tl.load(suffix_ptr + successor_id, mask=valid,
                             other=INT_MAX)

        # Candidate: weight + suffix_dist[successor]
        candidate = edge_weight + succ_dist
        min_val = tl.where(valid & (candidate < min_val), candidate, min_val)

    # Write result back
    tl.store(suffix_ptr + node_ids, min_val, mask=mask)


def run_triton_suffix_bfs(row_ptr, col_idx, weights, level_groups, n_nodes,
                          BLOCK_SIZE=128):
    """
    Run full backward BFS using Triton kernel level by level.
    Returns suffix distance tensor.
    """
    INT_MAX = 2147483647

    # Initialize: sinks get 0, all others get INT_MAX
    suffix_dist = torch.full((n_nodes,), INT_MAX, dtype=torch.int32).cuda()
    for sink in level_groups[0]:
        suffix_dist[sink] = 0

    # Process levels from 1 (one above sinks) upward to sources
    for level in range(1, len(level_groups)):
        nodes_at_level = torch.tensor(level_groups[level],
                                       dtype=torch.int32).cuda()
        n = len(nodes_at_level)
        if n == 0:
            continue

        grid = (triton.cdiv(n, BLOCK_SIZE),)
        suffix_bfs_kernel[grid](
            row_ptr, col_idx, weights,
            suffix_dist,
            nodes_at_level,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return suffix_dist
```

Convert Triton's final scaled integer distances to float only for reporting:

```python
triton_suffix_float = triton_suffix.cpu().numpy().astype(np.float32) / 10000.0
```

**Important note on MAX_DEGREE:** The loop bound in Triton must be a
compile-time constant (`tl.constexpr`). Before compiling the kernel,
check the max out-degree from Gate 1 output. Set MAX_DEGREE to the
smallest power of 2 that is >= max out-degree. If max out-degree is
say 47, set MAX_DEGREE=64. If it is 200, set MAX_DEGREE=256.

**Show Gate 1's max out-degree to user before setting MAX_DEGREE.**

---

## ⛔ GATE 2 — Triton Kernel Correctness

Run on the smallest benchmark (vga_lcd) with K small enough
that gpu-cpg's sfxt dump is fast to generate.

```python
# gate2_correctness.py
import numpy as np
import torch

# Run Triton kernel
triton_suffix = run_triton_suffix_bfs(row_ptr, col_idx, weights,
                                       level_groups, n_nodes)
triton_suffix_scaled = triton_suffix.cpu().numpy()
triton_suffix_np = triton_suffix_scaled.astype(np.float32) / 10000.0

# Load gpu-cpg sfxt reference
gpu_cpg_suffix = np.zeros(n_nodes, dtype=np.float32)
with open('benchmark_sfxt.txt') as f:
    for line in f:
        node, dist = line.split()
        gpu_cpg_suffix[int(node)] = float(dist)

# Compare scaled integers, because gpu-cpg computes distances as int with
# SCALE_UP=10000 and only converts to float during dumps.
gpu_cpg_suffix_scaled = np.rint(gpu_cpg_suffix * 10000.0).astype(np.int32)

# --- Check 1: Exact match on sink nodes ---
sinks = [n for n, l in levels.items() if l == 0]
for s in sinks:
    assert triton_suffix_scaled[s] == 0, \
        f"Sink {s}: Triton={triton_suffix_scaled[s]}, expected 0"
print(f"Check 1 PASS: all {len(sinks)} sinks have suffix_dist = 0")

# --- Check 2: Max absolute difference over all nodes ---
scaled_diff = np.abs(triton_suffix_scaled - gpu_cpg_suffix_scaled)
diff = scaled_diff.astype(np.float32) / 10000.0
max_diff  = diff.max()
mean_diff = diff.mean()
max_scaled_diff = scaled_diff.max()
print(f"Check 2: max_scaled_diff={max_scaled_diff}  "
      f"max_diff={max_diff:.6f}  mean_diff={mean_diff:.8f}")

# Pass threshold: exact scaled-int match
if max_scaled_diff == 0:
    print("Check 2 PASS: exact scaled-integer match")
else:
    print(f"Check 2 FAIL: max_scaled_diff={max_scaled_diff} — logic error in kernel")
    # Find worst nodes
    worst = np.argsort(diff)[-5:]
    for w in worst:
        print(f"  Node {w}: Triton={triton_suffix_np[w]:.6f} "
              f"gpu-cpg={gpu_cpg_suffix[w]:.6f} "
              f"scaled_diff={scaled_diff[w]} level={levels[w]}")
    print("Do not proceed to performance measurement.")
    exit(1)

# --- Check 3: Spot-check specific nodes manually ---
# Pick 5 non-sink nodes and verify the Bellman-Ford condition manually
non_sinks = [n for n, l in levels.items() if l > 0]
sample = np.random.choice(non_sinks, 5, replace=False)
print("\nCheck 3: manual spot-check of 5 nodes")
for node in sample:
    # Get successors from CSR
    start = row_ptr[node].item()
    end   = row_ptr[node+1].item()
    succs = col_idx[start:end].cpu().numpy()
    wts   = weights[start:end].cpu().numpy()

    # What should suffix_dist[node] be?
    expected = min(wts[i] + triton_suffix_scaled[succs[i]]
                   for i in range(len(succs)))
    actual   = triton_suffix_scaled[node]
    match    = actual == expected

    print(f"  Node {node} (level {levels[node]}): "
          f"Triton_scaled={actual}  "
          f"Expected_scaled={expected}  "
          f"{'MATCH' if match else 'MISMATCH'}")

print("\nShow Gate 2 output to user before running performance benchmarks.")
```

**If Check 2 fails with max_diff > 1.0:**

Most likely causes, in order:
1. **MAX_DEGREE too small**: some nodes have more successors than
   MAX_DEGREE, causing those successor slots to be silently skipped.
   Check max out-degree from Gate 1 and increase MAX_DEGREE.
2. **Level ordering wrong**: nodes being processed before their
   successors have correct values. Verify level_groups ordering is
   from level 1 upward (not downward).
3. **Wrong edge direction**: CSR built with predecessor edges instead
   of successor edges. Suffix BFS needs OUTGOING edges from each node.

**Do not proceed to performance measurement until Check 2 passes.**

---

## ⛔ GATE 2 CHECKPOINT — Explicit Stop Before Performance

**Before starting Step 3, confirm ALL of the following are true:**

```
[ ] Gate 2 Check 1 passed: all sinks have suffix_dist = 0
[ ] Gate 2 Check 2 passed: max_diff < 1e-4 on all benchmarks
[ ] Gate 2 Check 3 passed: all 5 spot-check nodes MATCH
[ ] Gate 2 output was shown to user and user confirmed PASS
```

**If any box is unchecked: do not run Step 3 or Step 4.**
Fix the kernel first, re-run Gate 2, confirm with user, then proceed.

---

## Step 3 — Performance Measurement Setup

Prepare both Triton and gpu-cpg for timed runs. Do NOT record final
performance numbers here — that happens after block size tuning in
Step 4. Step 3 sets up the measurement infrastructure; Step 4 sweeps
BLOCK_SIZE and picks the best; Gate 3 records the final numbers.

### Triton runtime

```python
# perf_triton.py
import torch
import time
import numpy as np

def benchmark_triton_relax_only(row_ptr, col_idx, weights, level_groups,
                                n_nodes, BLOCK_SIZE=128,
                                n_warmup=5, n_runs=20):
    """
    Measures only level-by-level Triton relaxation, assuming CSR and
    level_groups are already built.
    """
    # Warmup
    for _ in range(n_warmup):
        _ = run_triton_suffix_bfs(row_ptr, col_idx, weights,
                                   level_groups, n_nodes,
                                   BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = run_triton_suffix_bfs(row_ptr, col_idx, weights,
                                   level_groups, n_nodes,
                                   BLOCK_SIZE=BLOCK_SIZE)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = sorted(times)
    print(f"Triton relax only ({n_runs} runs):")
    print(f"  median: {np.median(times)*1000:.2f} ms")
    print(f"  min:    {min(times)*1000:.2f} ms")
    print(f"  max:    {max(times)*1000:.2f} ms")
    return np.median(times)


def benchmark_triton_e2e(edges_file, levels_file, n_nodes,
                         BLOCK_SIZE=128, n_warmup=5, n_runs=20):
    """
    Primary metric. Measures Triton's end-to-end suffix build:
    dump load + CSR construction + level grouping + GPU tensor creation +
    level-by-level relaxation.
    """
    for _ in range(n_warmup):
        row_ptr, col_idx, weights, level_groups, _ = \
            build_backward_csr(edges_file, levels_file, n_nodes)
        _ = run_triton_suffix_bfs(row_ptr, col_idx, weights,
                                  level_groups, n_nodes,
                                  BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        row_ptr, col_idx, weights, level_groups, _ = \
            build_backward_csr(edges_file, levels_file, n_nodes)
        _ = run_triton_suffix_bfs(row_ptr, col_idx, weights,
                                  level_groups, n_nodes,
                                  BLOCK_SIZE=BLOCK_SIZE)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = sorted(times)
    print(f"Triton end-to-end suffix build ({n_runs} runs):")
    print(f"  median: {np.median(times)*1000:.2f} ms")
    print(f"  min:    {min(times)*1000:.2f} ms")
    print(f"  max:    {max(times)*1000:.2f} ms")
    return np.median(times)
```

### CUDA baseline runtime (gpu-cpg)

The performance baseline is the existing CUDA GPU-CPG application at:

```
/home/cchang289/Research/gpu-cpg
```

**Before measuring:** read the gpu-cpg source to identify which
function/kernel corresponds to the suffix BFS (sfxt) computation.
Search for it:

```bash
grep -rn "sfxt\|suffix\|backward\|bfs" /home/cchang289/Research/gpu-cpg \
    --include="*.cu" --include="*.cuh" --include="*.cpp" | head -30
```

**gpu-cpg already has a `Timer` class with performance measurement
code.** First check if it can be reused to time the suffix BFS step
in isolation:

```bash
grep -rn "Timer\|timer\|elapsed\|cudaEvent" /home/cchang289/Research/gpu-cpg \
    --include="*.cu" --include="*.cuh" --include="*.hpp" | head -20
```

Use gpu-cpg's existing timing fields where possible:

```
lvlize_time       -- topological level construction
prefix_scan_time  -- included only when GPU CSR reorder is enabled
csr_reorder_time  -- included only when GPU CSR reorder is enabled
relax_time        -- bottom-up suffix relaxation over levelized vertices
prop_time         -- whole suffix propagation region; useful sanity check
```

**Primary CUDA baseline:** end-to-end suffix build. For the no-CSR-reorder
path, measure `lvlize_time + relax_time`. For the CSR-reorder path, measure
`lvlize_time + prefix_scan_time + csr_reorder_time + relax_time`.

**Secondary CUDA baseline:** relax only, measured as `relax_time`. This
isolates kernel quality and explains whether any end-to-end gap comes from
setup/levelization/data movement or from the Triton relaxation kernel itself.

Do not include graph loading, path enumeration, slack output, dump writing,
or PFxT expansion in either baseline.

Run both on the same benchmark and same GPU (A4000). Make sure to:
- Use the same graph (same benchmark files)
- Warm up both (5 runs), then measure 20 runs, report median
- Both run on GPU — this is a GPU vs GPU comparison, not CPU vs GPU
- Report both primary end-to-end and secondary relax-only timings

## Step 4 — Block Size Tuning

Run a sweep over BLOCK_SIZE values to find the best configuration
for the A4000:

```python
# tuning.py
for block_size in [32, 64, 128, 256, 512]:
    # Recompile kernel with new block size
    # (Triton recompiles automatically when constexpr changes)
    time_ms = benchmark_triton_relax_only(..., BLOCK_SIZE=block_size)
    print(f"BLOCK_SIZE={block_size}: {time_ms*1000:.2f} ms")
```

Report best BLOCK_SIZE to user. Use this for all subsequent benchmarks.

---

### ⛔ GATE 3 — Performance Acceptability

**Primary pass condition:** Triton median end-to-end suffix-build runtime
is within 2x of gpu-cpg CUDA median end-to-end suffix-build runtime.

**Secondary diagnostic:** report relax-only ratio, but do not use it as the
first-pass go/no-go gate. It explains whether Triton kernel quality is good
even if setup costs dominate end-to-end runtime.

```
Target:   Triton e2e / CUDA e2e <= 2.0x
Ideal:    Triton e2e / CUDA e2e <= 1.2x
Fail:     Triton e2e / CUDA e2e >  2.0x
```

**Record and show to user:**
```
benchmark | nodes | edges | CUDA e2e ms | Triton e2e ms | e2e ratio | CUDA relax ms | Triton relax ms | kernel ratio
----------|-------|-------|-------------|---------------|-----------|---------------|-----------------|-------------
vga_lcd   | ?     | ?     | ?           | ?             | ?         | ?             | ?               | ?
leon2     | ?     | ?     | ?           | ?             | ?         | ?             | ?               | ?
netcard   | ?     | ?     | ?           | ?             | ?         | ?             | ?               | ?
```

**If e2e ratio > 2.0:**

Likely causes and fixes:
1. **Relax-only ratio is also bad:** inspect kernel shape first.
   **MAX_DEGREE padded too large**: if max out-degree is 4 but
   MAX_DEGREE=256, the inner loop wastes 252 iterations. Set
   MAX_DEGREE tightly based on Gate 1 output.
2. **Level groups too small**: if most levels have < 128 nodes, GPU
   occupancy is low. Consider batching multiple small levels into
   one kernel launch. Show level size distribution to user.
3. **BLOCK_SIZE**: Step 4 tuning should already have identified the
   best BLOCK_SIZE. If ratio is still > 2.0 after tuning, report
   to user before concluding.
4. **Relax-only ratio is good but e2e ratio is bad:** setup dominates.
   Inspect Python dump loading, CSR construction, tensor creation, and
   per-level tensor allocation. Keep conclusion separate from kernel quality.

**If e2e ratio <= 2.0:** performance is acceptable. Report to user and
proceed to portability note.

---

## Expected Output Summary

After all gates pass, produce this summary and show to user:

```
=== Triton Suffix BFS Experiment Summary ===

Hardware: NVIDIA RTX A4000

Correctness (Gate 2):
  vga_lcd:  max_diff=X.XXXXXX  PASS/FAIL
  leon2:    max_diff=X.XXXXXX  PASS/FAIL
  netcard:  max_diff=X.XXXXXX  PASS/FAIL

Performance (Gate 3, median over 20 runs):
  benchmark | CUDA e2e ms | Triton e2e ms | e2e ratio | CUDA relax ms | Triton relax ms | kernel ratio | status
  ----------|-------------|---------------|-----------|---------------|-----------------|--------------|-------
  vga_lcd   | ?           | ?             | ?         | ?             | ?               | ?            | PASS/FAIL
  leon2     | ?           | ?             | ?         | ?             | ?               | ?            | PASS/FAIL
  netcard   | ?           | ?             | ?         | ?             | ?               | ?            | PASS/FAIL

Best BLOCK_SIZE: ?
MAX_DEGREE used: ?

Conclusion:
  [ ] Triton produces correct suffix distances (exact scaled-int match with gpu-cpg sfxt dump)
  [ ] Triton end-to-end suffix build runtime within 2x of CUDA on all benchmarks
  [ ] Triton relax-only diagnostic recorded for all benchmarks
  [ ] Direction is validated — proceed to full CPG Triton port
  OR
  [ ] Performance gap too large — see bottleneck notes above
```

---

## Notes on Portability (For Later)

Once the experiment is complete on A4000, portability is demonstrated
by running the IDENTICAL Python file on a second GPU. No code changes.

For cloud access if lab GPU is unavailable:
- AWS: `p4d.24xlarge` has 8× A100 (~$32/hr, use spot)
- Google Cloud: `a3-highgpu-8g` has H100 (~$25/hr spot)
- For a few benchmark runs: ~$5-15 total cost

Triton will auto-tune tile sizes for the new architecture on first run.
Compare median runtimes across architectures to demonstrate portability
and scaling behavior vs gpu-cpg CUDA kernels.
