# TC PFXT Optimization Notes

This note records how to build, test, and run the current TC PFXT prototype.
The implementation uses tensor-core BVSS only for deviation discovery; candidate
materialization, grouping, scans, split updates, and path-pile management remain
CUDA kernels.

## Build

```bash
cd /home/cchang289/Research/gpu-cpg
cmake --build build -j8 --target \
  tc-pfxt-gate5 \
  tc-pfxt-inprocess-timing \
  tc-pfxt-inprocess-exactness \
  tc_pfxt_candidates \
  tc_pfxt_inprocess
```

## Unit Tests

```bash
./build/unittests/tc_pfxt_candidates
./build/unittests/tc_pfxt_inprocess
```

Expected current result:

```text
tc_pfxt_candidates: 26 test cases, 443 assertions, SUCCESS
tc_pfxt_inprocess:  4 test cases, 15 assertions, SUCCESS
```

## Main Drivers

The in-process drivers load the graph once and reset CPG state between runs.
They do not call `cudaDeviceReset()` by default; use `--reset-device` only for
short debugging runs.

The commands below assume the local density-crossover benchmark artifacts are
available under `benchmarks/tc_pfxt_crossover/`. These graph and golden-cost
files are experiment data, not source files tracked by git.

Baseline G-PathGen timing:

```bash
./build/examples/tc-pfxt-inprocess-timing \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d20.txt \
  --k 1000000 \
  --mode gpg \
  --warmup 1 \
  --trials 3
```

TC PFXT source-local timing:

```bash
GPUCPG_ENABLE_TC_PFXT=1 \
GPUCPG_TC_PFXT_SINGLE_PASS=1 \
GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1 \
GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE=1 \
GPUCPG_TC_PFXT_LIGHT_STAGE_PROFILE=1 \
GPUCPG_TC_PFXT_DISABLE_PHASE_PROFILE=1 \
./build/examples/tc-pfxt-inprocess-timing \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d20.txt \
  --k 1000000 \
  --mode tc \
  --warmup 1 \
  --trials 3
```

Exactness against stored GPG costs:

```bash
./build/examples/tc-pfxt-gate5 \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d20.txt \
  --k 1000000 \
  --mode baseline \
  --out experiments/tc_pfxt_source_local_20260614/golden/netcard_d20_k1000000.golden.costs
```

```bash
GPUCPG_ENABLE_TC_PFXT=1 \
GPUCPG_TC_PFXT_SINGLE_PASS=1 \
GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1 \
GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE=1 \
./build/examples/tc-pfxt-inprocess-exactness \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d20.txt \
  --baseline-file experiments/tc_pfxt_source_local_20260614/golden/netcard_d20_k1000000.golden.costs \
  --ks 1000,10000,50000,1000000
```

Expected exactness line:

```text
INPROCESS EXACTNESS PASS
```

## Primary Runtime Flags

| flag | default | effect |
|---|---:|---|
| `GPUCPG_ENABLE_TC_PFXT=1` | off | enables TC PFXT in `report_paths` |
| `GPUCPG_TC_PFXT_SINGLE_PASS=1` | off | uses the single-pass TC PFXT window flow |
| `GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1` | off | uses the single-work candidate materialization path |
| `GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE=1` | off | uses source-local candidate grouping, the current proposal configuration |
| `GPUCPG_TC_PFXT_LIGHT_STAGE_PROFILE=1` | off | prints high-level per-step runtime categories |
| `GPUCPG_TC_PFXT_DISABLE_PHASE_PROFILE=1` | off | disables heavier phase profiling |

Other `GPUCPG_TC_PFXT_*` environment variables are research-only knobs retained
for reproducing old experiments. Do not use them for headline timing unless the
corresponding experiment section below says otherwise.

## Latest Runtime Reference

These numbers are PFXT expansion time only. The comparison uses the same TC
configuration across all densities: source-local candidate generation. GPG and
previous single-work TC come from
`experiments/tc_pfxt_chain_jump_singlework_retime_20260612_1920/`.

| density | K | GPG ms | source-local TC ms | TC/GPG | prev single-work TC ms | source-local vs prev |
|---|---:|---:|---:|---:|---:|---:|
| d10 | 1M | 186.9 | 415.2 | 2.22x | 363.0 | 0.87x |
| d20 | 1M | 264.1 | 468.9 | 1.78x | 513.5 | 1.10x |
| d30 | 200K | 62.7 | 199.4 | 3.18x | 266.2 | 1.33x |
| d40 | 200K | 64.1 | 226.9 | 3.54x | 306.7 | 1.35x |
| d50 | 100K | 59.6 | 172.3 | 2.89x | 248.4 | 1.44x |

Source-local is still slower than GPG on every density point
(`2.64x` geomean slowdown), but it improves the previous single-work TC path on
d20/d30/d40/d50. It regresses d10 because one large materialization step
dominates.

## Per-Step Runtime Reference

Fresh one-pass per-step logs were collected in
`experiments/tc_pfxt_per_step_singlework_20260613/`. The raw summaries are:

- `runtime_summary.csv`: one row per density and mode.
- `per_step_summary.csv`: one row per PFXT step.

That run used the previous single-work TC configuration. The one-pass numbers
are for attribution only; use the source-local-vs-GPG table above for headline
timing.

| density | K | GPG ms | TC ms | TC/GPG | TC dominant step | GPG dominant step | TC candidate | TC queue | TC discovery |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|
| d10 | 1M | 185.8 | 363.1 | 1.95x | 1 (335.6 ms, batch 714,345) | 1 (185.8 ms, batch 2,194,707) | 246.9 | 22.0 | 9.2 |
| d20 | 1M | 263.1 | 500.0 | 1.90x | 11 (157.9 ms, batch 229,208) | 11 (214.3 ms, batch 5,876,793) | 304.8 | 75.2 | 29.0 |
| d30 | 200K | 62.0 | 269.9 | 4.35x | 1 (96.9 ms, batch 77) | 9 (13.6 ms, batch 68,165) | 101.6 | 53.5 | 23.3 |
| d40 | 200K | 64.3 | 307.0 | 4.78x | 1 (115.2 ms, batch 862) | 8 (17.3 ms, batch 107,894) | 130.1 | 46.9 | 19.1 |
| d50 | 100K | 60.1 | 245.7 | 4.09x | 1 (122.6 ms, batch 497) | 7 (17.6 ms, batch 44,666) | 70.3 | 42.7 | 18.3 |

The density comparison supports the same conclusion as the aggregate timing:
TC deviation discovery is not the bottleneck. Candidate materialization is the
largest measured TC stage on d10/d20, while queue/residual costs become a large
fixed penalty at smaller feasible K values.

## Source-Major Candidate Experiment

Logs are in `experiments/tc_pfxt_source_major_20260613/`.

| experiment | result |
|---|---|
| d10 K=100K, single-work | 91.2327 ms mean |
| d10 K=100K, naive source-major | 176.927 ms mean |
| d10 K=100K, tiled source-major | 93.2593 ms mean |
| d20 K=1M, tiled source-major | failed at step 11 with source-major slot limit |
| d20 K=1M, tiled source-major with fallback | 519.842 ms mean |

The source-major path is therefore not a current win. It is useful evidence that
reshaping candidate scheduling alone is insufficient; a useful next design has
to reduce admitted/materialized candidates, not only change which block fills
them.

## Fused Discovery Interface Shadow Gate

This gate checks whether eliminating the TC-to-CUDA pair interface is worth a
larger fused kernel rewrite. It does not change production materialization.
Instead, it mirrors TC deviation discovery, counts candidate slots directly from
the TC hits, and compares the result against the existing `PairMeta` path.

Run:

```bash
GPUCPG_TC_PFXT_SINGLE_PASS=1 \
GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1 \
GPUCPG_TC_PFXT_FUSED_INTERFACE_SHADOW=1 \
GPUCPG_TC_PFXT_FUSED_INTERFACE_SHADOW_STRICT=1 \
GPUCPG_TC_PFXT_LIGHT_STAGE_PROFILE=1 \
GPUCPG_TC_PFXT_DISABLE_PHASE_PROFILE=1 \
./build/examples/tc-pfxt-gate5 benchmarks/tc_pfxt_crossover/netcard_d20.txt 100000
```

Expected gate output:

```text
runtime_summary_tc_fused_interface_shadow shadow_ms=...
  pairs=...
  candidate_slots=...
  pair_bytes_avoided=...
  pair_meta_bytes_avoided=...
  count_bytes_avoided=...
  mismatches=0
```

The shadow run synchronizes to measure its own cost and should not be used as a
timing result. Stop before retiming unless `mismatches=0` and the measured
shadow/setup cost is clearly smaller than the pair/meta/count interface it would
replace.

Initial d20 gates:

| K | exactness | shadow_ms | pair_meta_ms | candidate_slots | pair/meta/count bytes | interpretation |
|---:|---|---:|---:|---:|---:|---|
| 10K | PASS, max_diff=0 | 115.9 | 7.5 | 10.36M | 37.65 MB | identity passes, second traversal is too expensive |
| 100K | PASS, max_diff=0 | 242.5 | 10.9 | 134.21M | 90.15 MB | identity passes, naive shadow/fused path is not viable |

The result does not reject fusion outright. It rejects any design that performs
a second TC/BVSS traversal or reconstructs a pair-like interface after
discovery. The only plausible next fusion is inside the original discovery
kernel: discover a hit and immediately reserve/fill candidate work without
writing a global `int2` or `PairMeta` array.

## Direct PairMeta Emission Gate

`GPUCPG_TC_PFXT_DIRECT_PAIR_META=1` emits `PairMeta {src,dst,edge_id,wgt}`
directly from TC deviation discovery. This avoids writing the intermediate
global raw `(u,v)` pair array and skips the raw-pair-to-`PairMeta` conversion
kernel. It is fully gated and the known-good single-work path is unchanged when
the flag is not set.

Initial d20 gates:

| K | exactness | TC PFXT ms | discovery ms | candidate ms | pair_meta ms | raw pair bytes avoided | overflow fallbacks |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10K | PASS, `max_diff=0` | 265.3 | 125.0 | 25.3 | 0.24 | 9.4 MB | 0 |
| 100K | PASS, `max_diff=0` | 476.7 | 267.3 | 77.2 | 0.33 | 22.5 MB | 0 |

This is a negative performance result. The removed conversion cost was small:
the no-direct K=10K regression check measured `pair_meta_ms=7.49` ms and
`total_pfxt_ms=157.8` ms. Direct emission removes most of that conversion, but
it moves heavier `PairMeta` writes plus edge metadata reads into the discovery
kernel, increasing discovery time by an order of magnitude. Keep this path as a
guarded research knob, not a timing configuration.

## In-Discovery Short-Only Fusion Gate

`GPUCPG_TC_PFXT_IN_DISCOVERY_SHORT_ONLY=1` tests a true in-discovery path for
short-only substeps. LPQ-active substeps still use the known-good single-work
candidate path, because fusing LPQ-active windows requires capacity pre-counting
and would reread BVSS/graph data.

The implemented gate uses one TC discovery traversal for eligible short-only
substeps and warp-cooperatively scans each discovered pair's parent group. It
does not write global `int2` pairs or `PairMeta` for those substeps, and it
does not emit long candidates.

Initial d20 K=10K gate:

| result | value |
|---|---:|
| exactness | PASS, `max_diff=0` |
| fused substeps | 24 |
| LPQ-active substeps skipped | 141 |
| fused pairs | 456,994 |
| parent visits | 6,897,984 |
| short outputs | 2,821 |
| overflows | 0 |
| in-discovery time | 515.3 ms |
| total TC PFXT time | 654.0 ms |

This fails the hidden-cost gate. The path emits no extra candidates, but moving
parent scanning into the discovery warp collapses the efficient one-block-per-
pair materialization shape and becomes much slower than the pair interface it
removes. Do not use this path for timing claims.

## Source-Local Candidate Experiment

`GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE=1` groups discovered deviation work by
active source and expands the source-local parent/deviation product. This keeps
the exact `(parent, u, v)` identity; the deviation edge alone is not treated as
a unique path. The implementation was validated on d20 against a generated
1M-cost golden file using the in-process exactness runner:

| K | exactness | max diff | TC PFXT ms |
|---:|---|---:|---:|
| 1K | PASS | `4.77e-06` | 50.1 |
| 10K | PASS | `7.63e-06` | 77.6 |
| 50K | PASS | `7.63e-06` | 102.5 |
| 1M | PASS | `7.63e-06` | 468.0 |

Retiming used the in-process timing runner. The comparison baseline is the
previous best three-trial single-work TC run from
`tc_pfxt_chain_jump_singlework_retime_20260612_1920`.

| density | K | previous TC ms | source-local TC ms | change | speedup vs previous | status |
|---|---:|---:|---:|---:|---:|---|
| d10 | 1M | 363.0 | 415.2 | +52.2 ms | 0.87x | regress |
| d20 | 1M | 513.5 | 468.9 | -44.6 ms | 1.10x | improve |
| d30 | 200K | 266.2 | 199.4 | -66.8 ms | 1.33x | improve |
| d40 | 200K | 306.7 | 226.9 | -79.8 ms | 1.35x | improve |
| d50 | 100K | 248.4 | 172.3 | -76.1 ms | 1.44x | improve |

The positive cases show that source-local grouping can be a better optimization
shape than raw pair materialization. Across all five density points,
source-local gives a `1.20x` geomean improvement over previous single-work TC.
Excluding d10, source-local gives a `1.30x` geomean improvement over previous
single-work TC.

Measured source-local stage summaries for representative densities:

| density | K | candidate | queue | discovery | resize | fill | class_skip | materialized products |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| d20 | 1M | 336.4 | 72.1 | 36.7 | 46.0 | 216.2 | 122.9M | 124.1M |
| d30 | 200K | 98.2 | 51.0 | 13.3 | 29.1 | 50.2 | 123.3M | 124.4M |
| d50 | 100K | 79.5 | 40.8 | 14.2 | 23.4 | 41.0 | 63.3M | 64.6M |

Next hotspot ranking from this data:

1. **Reduce large-step materialization traffic.** d10 still regresses and
   spends most of its time filling 57.2M long outputs in one step. This is now
   the main blocker for making source-local a default.
2. **Avoid visiting skipped source-local products.** `class_skip` dominates the
   product space in every passing density. Any useful design must avoid that
   work without adding a separate compaction pass or second TC traversal.
3. **Reduce fill/materialization traffic.** Candidate fill is still the largest
   measured component in the passing cases.
4. **Reuse/reserve queue storage more intelligently.** Resize costs are still
   visible in the passing source-local cases.
5. **Queue/sort reduction.** Queue time remains significant, but this is less
   source-local-specific than the materialization and LPQ pressure issues.

## Current Interpretation

TC deviation discovery is cheap and exact, but end-to-end TC PFXT is dominated
by CUDA candidate materialization, queue work, and residual bookkeeping.
Source-local candidate grouping improves over previous single-work TC on
d20/d30/d40/d50, with a geomean speedup of `1.20x` over all five density points
and `1.30x` excluding d10. Against GPG, source-local remains slower on every
density point (`2.64x` geomean slowdown). d10 regresses versus single-work
because one large materialization step dominates.

## Rejected Experiments

Chunked single-work candidate materialization was tested on June 11, 2026 and
reverted. It improved only d20 slightly, while regressing most other densities:
d10 +6.5%, d20 -2.3%, d30 +6.0%, d40 +0.2%, d50 +7.9% PFXT time versus the
single-work baseline. The implementation and temporary logs were removed; do
not re-enable this path without a different dispatch rule or lower chunk
overhead.
