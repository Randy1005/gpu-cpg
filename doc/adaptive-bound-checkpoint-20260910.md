# Adaptive path-bound checkpoint — 2026-09-10

## Result

This checkpoint productionizes a mathematically certified, runtime-derived
cutoff for the adaptive deferred-candidate PFXT path. It replaces neither GPG
nor the adaptive-defer design. After `K` materialized SHORT candidates exist,
it continually tightens the existing split using the `K` best evaluated path
costs seen so far. The feature is enabled by default only for the fully
configured adaptive source-local compact-deviation path. Set
`GPUCPG_ADAPTIVE_PFXT_BOUND=0` to recover the previous threshold-growth-only
behavior. `GPUCPG_ADAPTIVE_PFXT_BOUND_PROFILE=1` prints per-update telemetry.

The implementation is deliberately not an arena experiment. It uses normal
CCCL/Thrust device allocations and charges minimum-delta discovery, candidate
cost gathering, sorting, scalar readback, guard calculation, synchronization,
and release to PFXT time.

## Bound and proof

Let `S` be all distinct materialized SHORT paths inspected so far, and let `U`
be the `K`-th smallest evaluated slack in `S`. Since `S` is a subset of all
valid candidates, at least `K` valid paths have evaluated cost no greater than
`U`; consequently the global `K`-th cost is no greater than `U`.

The device reservoir contains only the current `K` cheapest costs. When a cost
is discarded, it already has `K` retained costs no larger than itself. Inserting
new costs cannot make it relevant to the `K`-th rank later. Thus each update
sorts the retained `K` values plus only the new append-only SHORT suffix; it
does not rescan all paths ever generated. The stored logical payload is `4*K`
bytes: 4 MiB at K=1M.

Candidate slacks use a cached float recurrence `child = parent + delta`. The
minimum cached delta `d` is reduced on GPU. For `d >= 0`, `U` itself is safe.
For negative `d`, the implementation derives a conservative guard by repeatedly
inverting one rounded float addition across the graph's topological depth `H`:

```
T[0] = U
T[i+1] = upward_float(next_float(T[i]) - d)
guard = T[H]
```

Any candidate above `guard` cannot reach cost `<= U` in at most `H` remaining
deviations under the implementation's float recurrence. This is not a
benchmark-fitted epsilon. The bound is disabled safely if the compact deviation
cache is unavailable, a delta is non-finite, an unsupported cost domain is
encountered, or its optional allocation fails; the existing threshold policy
continues without retrying the query.

## RTX 5090 checkpoint data

K=1M, median of three standalone pairs, all outputs validated against corrected
GPG costs. The bounded PFXT number includes all bound maintenance.

| Case | Growth-only PFXT ms | Bound PFXT ms | Bound maintenance ms | Speedup |
|---|---:|---:|---:|---:|
| leon2 d30 | 921.682 | 122.267 | 2.299 | 7.54x |
| des_perf x16 | 49.739 | 9.255 | 2.414 | 5.37x |
| leon3mp x16 | 22.338 | 11.877 | 1.700 | 1.88x |
| netcard d50 | 102.682 | 94.766 | 1.915 | 1.08x |
| des_perf | 25.973 | 27.109 | 1.477 | 0.96x |
| M6 | 18.551 | 4.462 | 1.002 | 4.16x |

The original `des_perf` case is a known 4.4% isolated-PFXT regression because
the bound arrives at step 46 and avoids little remaining generation. This is
why results must be reported per case rather than as an unconditional win.
The cold-query benefit can be much smaller when unchanged SFXT/static setup
dominates; graph file loading is not included in those query numbers.

## Verification

`adaptive-path-bound-test` covers:

- exact incremental top-K retention over batch boundaries, ties, repeat updates,
  invalid count/K changes, and release/reuse;
- finite/non-finite delta handling and control parsing;
- the float guard on GPU over positive and negative deltas, multiple magnitudes,
  and depths through 4096.

Integration validation passed exhaustive two-root tie and weighted DAGs for
K=1,16,64,256. K=1M GPG comparisons passed both enabled and disabled modes on
leon2 d30 and original des_perf. The latter is intentionally retained as a
regression-case check, not hidden from the checkpoint.

## Reproduction

Build for the current RTX 5090:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER="$PWD/.local/cuda-13.3.1/bin/nvcc"
cmake --build build --target adaptive-path-bound-test tc-pfxt-inprocess-exactness -j
ctest --test-dir build -R adaptive_path_bound_unit --output-on-failure
```

To run the adaptive implementation, use the normal in-process command. Bound
control is optional because it defaults on for adaptive mode:

```bash
build/examples/tc-pfxt-inprocess-exactness \
  --benchmark /path/to/graph.csrbin --baseline-file /path/to/gpg.costs \
  --ks 1000000 --mode adaptive

GPUCPG_ADAPTIVE_PFXT_BOUND=0 build/examples/tc-pfxt-inprocess-exactness \
  --benchmark /path/to/graph.csrbin --baseline-file /path/to/gpg.costs \
  --ks 1000000 --mode adaptive
```

No benchmark binaries, generated goldens, or raw timing logs are committed in
this checkpoint.
