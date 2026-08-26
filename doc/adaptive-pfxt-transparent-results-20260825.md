# Adaptive PFXT: transparent full-suite results (2026-08-25)

## Result

The GPU-resident adaptive implementation is correct on all 29 graph/density
cases and beats the current GPG PFXT baseline on every case at `K=1,000,000`.
Across the suite, its geometric-mean PFXT speedup is **1.659x**. A graph-wide
fixed-defer policy wins only 12 of 29 cases and has a **0.843x** geometric mean
(that is, it is slower overall). Per-step adaptation is therefore essential.

All performance values below are the mean of three measured, one-shot K=1M
queries after one warm-up query. Every query produced the validated golden
top-K result. There were no candidate-capacity retries, nonzero overflows, or
fallback measurements.

## What each reported cost means

- `GPG PFXT`: uninstrumented PFXT time only. Common file parsing and graph
  construction are intentionally excluded.
- `adaptive production PFXT`: uninstrumented, cached-query time including the
  production GPU-resident per-step oracle and the selected ordinary/deferred
  PFXT work.
- `cold setup`: adaptive-specific, one-time construction of reusable BVSS/tile,
  compact deviation, topology-bound, and oracle metadata. It excludes input
  parsing and common graph construction.
- `cold adaptive`: `cold setup + first profiled PFXT query`.
- `decision`: GPU-resident per-step ordinary/deferred decision time.
- `core PFXT`: instrumented PFXT work excluding `decision`.
- `profile overhead`: instrumented total minus uninstrumented production PFXT.
  Production PFXT remains authoritative for speedup claims.

Cold setup is not competitive for a single query: none of the 29 cases
amortizes setup in one query. Break-even ranges from approximately 22 reused
queries (`leon2_d30`) to 10,066 queries (`nlpkkt120`). This design targets
iterative circuit optimization where topology/order metadata remains cached;
weight-only updates do not require rebuilding topology metadata.

The complete transparent cost table is in
[`transparent_costs.csv`](../experiments/transparent_adaptive_20260825/transparent_costs.csv).
The raw production table is in
[`comparison.csv`](../experiments/transparent_adaptive_20260825/comparison.csv).

## Full production PFXT matrix

| Case | GPG (ms) | Fixed defer (ms) | Adaptive (ms) | Adaptive speedup | Ordinary steps | Deferred steps | Switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| netcard_base | 7.217 | 24.488 | 6.919 | 1.043x | 35 | 0 | 0 |
| netcard_d10 | 39.157 | 138.520 | 30.653 | 1.277x | 39 | 0 | 0 |
| netcard_d20 | 87.854 | 160.081 | 66.154 | 1.328x | 50 | 0 | 0 |
| netcard_d30 | 111.270 | 104.835 | 88.078 | 1.263x | 15 | 452 | 1 |
| netcard_d40 | 158.148 | 110.315 | 101.681 | 1.555x | 1 | 476 | 1 |
| netcard_d50 | 373.816 | 115.259 | 103.810 | 3.601x | 1 | 497 | 1 |
| leon2_d10 | 54.715 | 87.034 | 41.285 | 1.325x | 26 | 84 | 40 |
| leon2_d20 | 259.643 | 106.820 | 59.723 | 4.348x | 32 | 138 | 62 |
| leon2_d30 | 4794.940 | 1938.810 | 1068.200 | 4.489x | 51 | 60 | 26 |
| leon2_d40 | 131.451 | 123.422 | 75.287 | 1.746x | 46 | 62 | 23 |
| leon2_d50 | 147.379 | 129.109 | 113.849 | 1.295x | 11 | 480 | 9 |
| leon3mp_d10 | 84.257 | 199.184 | 75.053 | 1.123x | 41 | 0 | 0 |
| leon3mp_d20 | 90.986 | 185.879 | 61.168 | 1.488x | 48 | 0 | 0 |
| leon3mp_d30 | 122.440 | 130.240 | 83.806 | 1.461x | 55 | 0 | 0 |
| leon3mp_d40 | 212.949 | 127.894 | 71.760 | 2.968x | 45 | 112 | 46 |
| leon3mp_d50 | 151.637 | 78.034 | 71.574 | 2.119x | 3 | 389 | 3 |
| vga_lcd_d10 | 48.453 | 96.673 | 37.886 | 1.279x | 48 | 0 | 0 |
| vga_lcd_d20 | 109.070 | 138.305 | 73.800 | 1.478x | 56 | 1 | 2 |
| vga_lcd_d30 | 75.523 | 99.457 | 46.211 | 1.634x | 46 | 1 | 2 |
| vga_lcd_d40 | 98.201 | 124.854 | 93.331 | 1.052x | 53 | 3 | 4 |
| vga_lcd_d50 | 125.245 | 89.014 | 58.980 | 2.124x | 44 | 9 | 16 |
| des_perf_d10 | 80.387 | 142.345 | 42.837 | 1.877x | 49 | 1 | 1 |
| des_perf_d20 | 32.243 | 85.093 | 21.853 | 1.476x | 51 | 1 | 1 |
| des_perf_d30 | 74.121 | 77.917 | 52.464 | 1.413x | 47 | 2 | 3 |
| des_perf_d40 | 96.646 | 74.486 | 72.253 | 1.338x | 4 | 386 | 7 |
| des_perf_d50 | 90.990 | 74.377 | 71.965 | 1.264x | 5 | 330 | 8 |
| cage15 | 22.703 | 110.878 | 18.982 | 1.196x | 49 | 0 | 0 |
| M6 | 67.541 | 75.053 | 18.344 | 3.682x | 14 | 0 | 0 |
| nlpkkt120 | 5.943 | 10.900 | 5.622 | 1.057x | 2 | 0 | 0 |

## What the transparent profile establishes

- Adaptive wins all 29 cases; the smallest wins are `netcard_base` (1.043x),
  `vga_lcd_d40` (1.052x), and `nlpkkt120` (1.057x).
- The largest wins are `leon2_d30` (4.489x), `leon2_d20` (4.348x),
  `M6` (3.682x), and `netcard_d50` (3.601x).
- Fixed deferral is frequently destructive on sparse/low-intensity workloads.
  The adaptive oracle correctly keeps those cases ordinary.
- Exact-oracle decision share ranges from about 0.136% (`netcard_d50`) to
  11.416% (`leon2_d10`) of production PFXT. The costly region is a short core
  query with many exact decisions; expanding safe O(1) classifications is the
  next clear optimization.
- Fine-grained profiling itself can perturb many-step cases by roughly 10-13%.
  The report exposes this delta and does not use profiled totals for speedup.
- Cold adaptive setup ranges from 714 ms to 148.22 s. It must be cached and
  amortized; it is not a viable one-off-query path in its present form.

## Correctness and implementation evidence

- Unit tests: **102/102 passed**.
- Full validation: **87/87 mode-cases passed** (29 cases times GPG,
  fixed-defer, and adaptive), with zero top-K differences.
- Performance logs: 87/87 production arms and 29/29 transparent profiles,
  each with exactly three measured trials.
- Integrity scan: zero retry markers, zero nonzero overflow fallbacks, and zero
  exactness failures.
- GPU-resident design: parallel descriptor expansion, block-reduced exact
  oracle, safe O(1) gates, cached static metadata, per-step decisions, and
  telemetry remain on device; no per-step device-to-host control round trip is
  used.

Supporting artifacts:

- [`validation.csv`](../experiments/transparent_adaptive_20260825/validation.csv)
- [`timing.csv`](../experiments/transparent_adaptive_20260825/timing.csv)
- [`comparison.csv`](../experiments/transparent_adaptive_20260825/comparison.csv)
- [`transparent_costs.csv`](../experiments/transparent_adaptive_20260825/transparent_costs.csv)
- [`adaptive_steps.csv`](../experiments/transparent_adaptive_20260825/adaptive_steps.csv)
- [`adaptive_switches.csv`](../experiments/transparent_adaptive_20260825/adaptive_switches.csv)

