# TC-PathGen Kernel Fusion: Eliminate D2H Sync Overhead

## Problem Statement

The TC pfxt chain walk loop pays two `cudaMemcpy D2H` calls per chain
sub-step:

1. `cudaMemcpy(&h_n_active, d_n_active, 4, D2H)` — termination check
2. `cudaMemcpy(&h_n_pairs,  d_n_pairs,  4, D2H)` — launch parameter

Each D2H memcpy forces a full GPU pipeline drain before the CPU can read
the result. The actual data transfer (4 bytes) is microseconds. The
pipeline drain is 35–70ms per sync.

With ~3 chain sub-steps per delta step and 2 syncs each:
```
cuda_ms per delta step ≈ 3 × 2 × 50ms = 300ms overhead
```

Measured: d50 cuda_ms = 200–424ms per step, totaling 2098ms of 2133ms
TC_pfxt. Useful work (tc+sort+cost+adv) is only 226ms. The 2098ms
overhead is entirely D2H sync latency.

**Fix: eliminate both D2H syncs from the inner chain sub-step loop.**
Keep exactly one D2H per delta step (HPQ size check for early exit).

---

## Projected Improvement

```
             current TC_pfxt   projected TC_pfxt   TC/GPG current   TC/GPG projected
d10 K=1M      2563ms            ~390ms              13.3×            ~2.0×
d50 K=100K    2133ms            ~306ms              25.9×            ~3.7×
d40 K=200K    2224ms            ~320ms              21.9×            ~3.2×
```

These projections assume cuda_ms drops from 200–424ms to ~10ms per delta
step (kernel launch overhead only, no D2H stalls).

---

## Background: Current Chain Walk Structure

Inside one delta step, the host runs this loop:

```cpp
// CURRENT: D2H sync every chain sub-step
while (true) {
    cudaMemset(d_n_pairs, 0, sizeof(int));
    build_frontier<<<...>>>(d_current_v, d_F_frontier, n_active_paths);
    build_active_vss_queue<<<...>>>(...);
    tc_deviation_discover<<<...>>>(..., d_n_pairs);

    // ← SYNC POINT 1: read h_n_pairs to size the cost kernel launch
    int h_n_pairs;
    cudaMemcpy(&h_n_pairs, d_n_pairs, sizeof(int), D2H);
    if (h_n_pairs == 0) {
        // skip sort and cost
    } else {
        if (n_active_paths > HYBRID_THRESHOLD) {
            thrust::sort_by_key(...);
            build_group_ptr<<<...>>>();
        }
        compute_costs<<<ROUNDUP(h_n_pairs, BS), BS>>>(..., h_n_pairs);
    }

    cudaMemset(d_n_active, 0, sizeof(int));
    advance_chain<<<...>>>(d_current_v, d_succs, n_active_paths, d_n_active);

    // ← SYNC POINT 2: check termination
    int h_n_active;
    cudaMemcpy(&h_n_active, d_n_active, sizeof(int), D2H);
    if (h_n_active == 0) break;
}
// One D2H for early exit (delta step boundary)
cudaMemcpy(&h_hpq_size, d_hpq_size, sizeof(int), D2H);
```

---

## Change 1 — Fixed-Iteration Loop (eliminates SYNC POINT 2)

### Rationale

`h_n_active` is read to check if all paths have reached a sink
(`current_v == -1`). The maximum possible number of chain sub-steps is
bounded by the longest suffix chain in the graph — which equals the
maximum sfxt level among all vertices currently being expanded.

If we iterate exactly `max_chain_len` times, extra iterations are safe:
`advance_chain` checks `current_v != -1` before doing any work, so
over-iterating paths that have already reached a sink is a no-op.

### Implementation

**Step 1.1 — Compute max_chain_len during sfxt (not per delta step)**

`max_chain_len` equals the global maximum sfxt topological level. Compute
it once at sfxt time via a single thrust reduction — zero cost to pfxt.

```bash
# Find where sfxt assigns per-vertex levels and where d_depths lives
grep -n "d_depths\|curr_depth\|h_verts_lvlp\|sfxt.*level" \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cu | head -20
```

After sfxt BFS and relaxation complete, add one thrust reduction and
persist d_depths for pfxt use:

```cpp
// At end of sfxt phase — ONE-TIME, not per delta step
// d_depths[v] = topological level of v (already computed by sfxt BFS)
int h_max_sfxt_level = thrust::reduce(
    thrust::device,
    d_depths, d_depths + n_nodes,
    0, thrust::maximum<int>());
_h_max_sfxt_level = h_max_sfxt_level;  // store as member, available to pfxt
```

**Persist d_depths into pfxt:** ensure d_depths is not freed or reused
between sfxt and pfxt. Overhead: one int32 per vertex (~15.6MB for
netcard). If d_depths is cleared after sfxt, defer that clear until
after pfxt completes, or copy to a separate `_d_sfxt_levels` array.

**Use in Change 1 — no kernel, no D2H in pfxt:**

```cpp
// Per delta step: use precomputed value from sfxt, no extra work
int h_max_chain_len = max(_h_max_sfxt_level, 1);
for (int sub = 0; sub < h_max_chain_len; sub++) {
    ...
}
```

This eliminates both the `compute_max_chain_len` kernel AND its D2H
entirely — max_chain_len is already on the host from sfxt time.

**Step 1.2 — Replace while loop with for loop:**

```cpp
// AFTER: fixed iterations, no per-sub-step D2H
for (int sub = 0; sub < h_max_chain_len; sub++) {
    cudaMemset(d_n_pairs, 0, sizeof(int));
    cudaMemset(d_F_frontier, 0, n_frontier_bytes);

    build_frontier<<<...>>>(d_current_v, d_F_frontier, n_active_paths);
    build_active_vss_queue<<<...>>>(..., d_Q_active, d_Q_active_size, n_intervals);

    cudaMemset(d_Q_active_size, 0, sizeof(int));  // reset before use
    // Note: Q_active_size is read on device in tc_discover — see Change 2

    tc_deviation_discover<<<...>>>(..., d_n_pairs);

    if (n_active_paths > HYBRID_THRESHOLD) {
        thrust::sort_by_key(...);
        build_group_ptr<<<...>>>();
    }

    // compute_costs reads d_n_pairs from device — see Change 2
    compute_costs_device_npairs<<<ROUNDUP(MAX_PAIRS, BS), BS>>>(...);

    cudaMemset(d_n_active, 0, sizeof(int));
    advance_chain<<<...>>>(d_current_v, d_succs, n_active_paths, d_n_active);

    // NO D2H HERE — loop continues unconditionally
}
// Single D2H at delta step boundary:
cudaMemcpy(&h_hpq_size, d_hpq_size, sizeof(int), D2H);
if (h_hpq_size >= K && !stop_lpq) {
    stop_lpq = true;
    // complete current delta step per Lemma 2 — already done (loop is complete)
    break;
}
```

### Correctness Argument for Change 1

When `sub >= actual_chain_len` for a path: its `current_v` is already
-1. `build_frontier` does nothing for that path (guards on `v != -1`).
`advance_chain` does nothing for that path (guards on `v != -1`).
`tc_deviation_discover` never includes -1 in the frontier bitmap.
No spurious deviation candidates are generated. Lemma 2 is preserved:
the entire window expansion completes before termination is checked.

### cudaMemset inside loop — use cudaMemsetAsync

`cudaMemset` is synchronous: the CPU blocks until the memset completes
before launching the next kernel. Replace with `cudaMemsetAsync` on
the same stream so the CPU can immediately queue the next kernel
without waiting:

```cpp
// BEFORE (CPU blocks):
cudaMemset(d_F_frontier, 0, n_frontier_bytes);
cudaMemset(d_n_pairs, 0, sizeof(int));

// AFTER (CPU continues immediately, GPU orders within stream):
cudaMemsetAsync(d_F_frontier, 0, n_frontier_bytes, stream);
cudaMemsetAsync(d_n_pairs,    0, sizeof(int),       stream);
build_frontier<<<..., stream>>>();   // GPU ensures ordering
```

Since all kernels in the loop use the same stream, the GPU still
executes them in order — correctness is maintained. The CPU is free
to queue subsequent kernel launches without stalling.

This is a secondary optimization after D2H elimination. Impact is
smaller (microseconds per memset vs 35–70ms per D2H), but it is
correct and requires minimal code change. Implement alongside Change 1.
Correctness is unaffected — stream ordering guarantees memset completes
before build_frontier reads d_F_frontier.

---

## Change 2 — Device-Side n_pairs (eliminates SYNC POINT 1)

### Rationale

`h_n_pairs` is read to know how many thread blocks to launch for
`compute_costs`. The fix: always launch `MAX_PAIRS` threads and have
the kernel read `d_n_pairs` from device global memory at the start,
returning immediately if `tid >= *d_n_pairs`.

### Implementation

**Step 2.1 — New kernel signature:**

```cuda
// BEFORE:
__global__ void compute_costs(
    int* d_dev_pairs, int n_pairs,   ← host-side count
    ...)

// AFTER:
__global__ void compute_costs_device_npairs(
    int* d_dev_pairs, int* d_n_pairs,  ← device pointer, read on GPU
    ...)
{
    // Read pair count from device memory — no D2H needed
    int n_pairs = *d_n_pairs;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_pairs) return;   // early exit for excess threads

    // ... rest of kernel body unchanged ...
}
```

**Step 2.2 — Launch with fixed grid:**

```cpp
// MAX_PAIRS is the pre-allocated buffer size (set during BVSS construction)
// Launch the same number of threads every sub-step regardless of actual pairs
int launch_threads = MAX_PAIRS;
compute_costs_device_npairs<<<ROUNDUP(launch_threads, BS), BS>>>(
    d_dev_pairs, d_n_pairs,    // ← d_n_pairs is a device pointer
    d_group_ptr, d_path_indices, d_short_pile,
    d_dists_cache, d_fanout_wgts, d_fanout_adjp, d_fanout_adjncy,
    alpha, d_hpq, d_hpq_sz, d_lpq, d_lpq_sz);
// No D2H for h_n_pairs — GPU reads d_n_pairs internally
```

### Correctness Argument for Change 2

When `tid >= *d_n_pairs`: thread exits immediately at the guard, no
incorrect pairs are processed. The cost is launching excess threads that
immediately exit — at ~1 warp per 32 excess threads, this is negligible
for typical MAX_PAIRS = 2M.

### Issue: build_active_vss_queue also uses D2H for h_n_vss

Check whether there is also a D2H read of `h_n_vss` (count of active
VSSs) used to size the TC discover kernel launch:

```bash
grep -n "n_vss\|Q_active_size\|active_vss\|cudaMemcpy.*vss" \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cu | head -20
```

If `h_n_vss` is also read via D2H before the TC discover launch:
apply the same fix — read `d_Q_active_size` from device inside the
kernel, or launch with a fixed upper bound (total_VSS threads, most
exit immediately if their VSS is not in the active queue).

---

## Implementation Order

Implement and validate one change at a time. Never combine unvalidated
changes.

```
Step 1: Add thrust::reduce for h_max_sfxt_level at end of sfxt phase
        Persist d_depths into pfxt (ensure not cleared early)
Step 2: Change while loop to for loop using _h_max_sfxt_level
        Add cudaMemsetAsync for frontier/pairs resets
        → VALIDATE Gate FUSE-1 (original netcard K=1M)
Step 3: Add compute_costs_device_npairs kernel
Step 4: Integrate Change 2 into the for loop
        Check and fix h_n_vss D2H if present (same pattern)
        → VALIDATE Gate FUSE-2 (original netcard K=1K/10K/50K/1M)
Step 5: Confirm cuda_ms drop — Gate FUSE-3
Step 6: Timing measurement on density crossover (Gate 6 re-run)
```

---

## Correctness Gates

### ⛔ GATE FUSE-1 — After Change 1 only

Use **original (non-densified) netcard** for validation — runs in
seconds rather than minutes, and the G-PathGen golden result for
original netcard K=1M is already known from earlier experiments.

```bash
# Rebuild
cmake --build build -j8

# Find original netcard benchmark
NETCARD=$(find benchmarks -name "netcard*.txt" |     grep -v "d10\|d20\|d30\|d40\|d50\|dense\|crossover" | head -1)
echo "Using: $NETCARD"

# Generate G-PathGen golden result if not already available
./build/examples/cpg $NETCARD 1000000 2>&1 |     tee experiments/kernel_fusion/netcard_orig_gpg_k1m.txt

# Validate TC against golden result
./build/examples/tc-pfxt-gate5 $NETCARD 1000000
# Required: max_diff=0, baseline_count=1000000, exit 0
# If fails: do NOT proceed to Change 2. Debug Change 1 first.
```

Also verify loop count: add a temporary debug print of
`h_max_chain_len` for one K=10 run. Confirm it's a reasonable value
(not 0, not absurdly large like 10,000+).

### ⛔ GATE FUSE-2 — After Change 2 (combined with Change 1)

```bash
# Same benchmark, same command
./build/examples/tc-pfxt-gate5 $NETCARD 1000000
# Required: max_diff=0, baseline_count=1000000, exit 0
```

Also run K=1K, 10K, 50K to catch edge cases at small K:

```bash
for K in 1000 10000 50000 1000000; do
    ./build/examples/tc-pfxt-gate5 $NETCARD $K
done
```

### ⛔ GATE FUSE-3 — Confirm cuda_ms drop

After both changes pass correctness, run a single timed pass on d50 and
inspect the per-step breakdown. The cuda_ms column must drop:

```
BEFORE Change 1+2:   cuda_ms = 200–424ms per step
AFTER  Change 1+2:   cuda_ms = <20ms per step (target)
```

If cuda_ms does not drop below 50ms per step: there is a remaining D2H
sync not captured by Change 1+2. Find and eliminate it before timing.

---

## Timing Measurement (Gate 6 Re-run)

Only after GATE FUSE-1, FUSE-2, FUSE-3 all pass.

Re-run the full density crossover experiment at d10, d30, d40, d50.
Mean of 10 runs, 3 warmup. Same K_max per density as before.
No synthesis jobs running concurrently.

Report the updated crossover table:

```
density | K_max | GPG_ms | TC_ms_old | TC_ms_new | TC/GPG_old | TC/GPG_new
--------|-------|--------|-----------|-----------|------------|----------
d10     | 1M    | 192ms  | 2563ms    | ?         | 13.3×      | ?
d30     | 200K  | 82ms   | 2578ms    | ?         | 31.5×      | ?
d40     | 200K  | 101ms  | 2224ms    | ?         | 21.9×      | ?
d50     | 100K  | 82ms   | 2133ms    | ?         | 25.9×      | ?
```

Also report updated per-step breakdown for d10 and d50 showing new
cuda_ms vs old cuda_ms side by side.

---

## Known Concerns

### CONCERN 1 — D2H count after fusion

After both changes, the only remaining D2H per delta step is:
- `h_hpq_size` read for early exit check (1 D2H per delta step)

`h_max_sfxt_level` is computed once at sfxt time — zero D2H in pfxt.
`h_n_pairs` is eliminated by Change 2.
`h_n_active` is eliminated by Change 1.

Net: from ~6 D2H per delta step to 1 D2H per delta step.
Reduction: ~83% fewer pipeline stalls.

### CONCERN 2 — Over-iteration cost

The for loop runs `h_max_chain_len` times. For sub-steps where all paths
have reached sink: build_frontier produces an empty bitmap, TC discovers
zero pairs, compute_costs exits immediately, advance_chain is a no-op.
Cost: ~5–10ms of kernel launches for a fully empty sub-step.

If max_chain_len is large (e.g., 500 levels) but most paths reach sink
in 3 sub-steps: 497 wasteful iterations × 5ms = ~2.5s overhead. This
would make the fusion counter-productive.

**Mitigation:** cap the loop at a practical bound. Add a device-side
"any active" flag that is cheaply polled every N=10 iterations:

```cpp
for (int sub = 0; sub < h_max_chain_len; sub++) {
    ...kernels...
    advance_chain<<<...>>>(..., d_n_active_flag);

    // Check termination every 10 sub-steps only (10× fewer D2H)
    if ((sub + 1) % 10 == 0) {
        cudaMemcpy(&h_n_active, d_n_active_flag, sizeof(int), D2H);
        if (h_n_active == 0) break;
        cudaMemset(d_n_active_flag, 0, sizeof(int));
    }
}
```

This limits over-iteration to at most 9 extra sub-steps (not the full
diameter).

**Check the actual max_chain_len for netcard d50 before implementing:**
```bash
grep -o "max_chain_len=[0-9]*" /tmp/gpucpg-netcard-d50/*.log 2>/dev/null
# or add a debug print in compute_max_chain_len after the D2H
```

If max_chain_len is ≤ 20 for the dense graphs: over-iteration cost is
trivial and no capping is needed. If > 50: use the periodic check.

### CONCERN 3 — d_sfxt_levels lifetime

Per Step 1.1, sfxt's d_depths array holds per-vertex levels and is used
to compute h_max_sfxt_level via thrust::reduce at sfxt time. The concern
is whether d_depths is freed or reused before pfxt completes.

```bash
grep -n "d_depths\|depths.*free\|depths.*clear\|thrust.*free.*depths" \
    /home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cu | head -20
```

If d_depths is cleared after sfxt: store h_max_sfxt_level on the host
immediately after the thrust::reduce call. The host value is all pfxt
needs — no device array access required during the chain walk loop.

### CONCERN 4 — Thrust sort still inside the loop

The thrust::sort_by_key call is NOT eliminated by kernel fusion. It
remains one sort per sub-step. This becomes the next dominant cost after
D2H syncs are removed.

If cuda_ms drops to ~10ms but sort_ms stays at 9–17ms per sub-step:
sort_ms will become the new bottleneck.

Consider as a follow-up optimization (NOT part of this spec):
sort once at the start of each delta step rather than every sub-step.
Implement and validate separately after kernel fusion is confirmed working.

---

## Future TC Opportunity: Batched Suffix-Chain Discovery

The next opportunity to make PFXT more tensor-core dependent has two
connected parts.

### 1. Multi-hop suffix-frontier construction

The current chain loop processes one suffix position at a time:

```text
frontier(current_v) -> TC discover -> current_v = succs[current_v]
```

Instead, construct a tile of frontiers for several exact suffix positions:

```text
F[:, 0] = current_v
F[:, 1] = succs[current_v]
F[:, 2] = succs^2[current_v]
...
F[:, H-1] = succs^(H-1)[current_v]
```

Pointer chasing remains ordinary CUDA work, but it is performed once to
prepare an `H`-column Boolean frontier tile. Every valid suffix position must
be represented; paths that reach `-1` contribute zeros to later columns.

### 2. Batched Boolean TC deviation discovery

Use the frontier tile in a Boolean matrix-matrix operation:

```text
A_dev^T x F
```

This replaces `H` separate Boolean matrix-vector discovery launches with one
batched tensor-core operation. Each emitted result must retain `(path/source,
destination, hop)` identity so candidate parent links and costs remain exact.

Expected benefit:

- more TC work per launch;
- fewer discovery launches and synchronization points;
- better amortization of BVSS mask loads;
- greatest benefit on benchmarks with many suffix-chain sub-steps.

Correctness requirements:

- include every suffix position exactly once;
- do not merge distinct paths that share a source or equal cost;
- preserve the source vertex and parent path associated with each hit;
- deduplicate only duplicate discovery work, never distinct path candidates;
- finish all positions belonging to the current PFXT window before applying a
  termination decision (Lemma 2).

Primary risks are frontier-tile memory, increased intermediate pair volume,
and loss of the current active-VSS sparsity advantage if `H` is too large.
Evaluate small tile widths first (`H=4, 8, 16`) and compare emitted pairs and
top-K costs against the existing one-hop loop before timing.
