# Isolating the two descriptor formats

## Read this first: what the controlled timing establishes

**The two descriptor formats have complementary benefits; tight-bound is a
separate work-reduction benefit. None accounts for the entire historical
GPG-to-adaptive speedup.** The `Neither` column below is the same optimized
adaptive pipeline with both packing formats disabled, **not GPG**.

Across all 43 cases, adding 204-byte descriptors to `Neither` gives a 1.044x
geometric-mean speedup with tight-bound off, or 1.052x with it on. Adding
Strip24 to the 204-only pipeline gives 1.313x and 1.357x, respectively.
Enabling both formats versus neither gives 1.372x and 1.427x. Enabling
tight-bound with both formats held fixed gives a separate 1.208x geomean.
These include every measured case, including regressions and cases with
little or no packing opportunity.

In simple terms: 204-byte records bundle a rectangle of waiting children
across parents; Strip24 bundles selected waiting children within one parent;
tight-bound prevents unnecessary later work. Their benefits vary with which
kind of work the graph generates.

Representative controlled comparisons (cold setup + PFXT, median of three):

| Isolated change | Case | Before (ms) | After (ms) | Speedup |
|---|---|---:|---:|---:|
| Enable 204 only; bound off | netcard d40 | 157.733 | 109.404 | 1.442x |
| Enable 204 only; bound off | netcard d50 | 142.471 | 116.298 | 1.225x |
| Add Strip24 to 204; bound off | netcard d20 | 70.292 | 25.648 | 2.741x |
| Add Strip24 to 204; bound off | des_perf d30 | 48.525 | 14.262 | 3.402x |
| Enable tight-bound; both formats | leon2 d30 | 856.568 | 103.129 | 8.306x |
| Enable tight-bound; both formats | des_perf x16 | 51.452 | 9.625 | 5.346x |

The des_perf x16 question is now directly testable: without tight-bound,
`Neither` is 51.592 ms and `Both` is 51.452 ms, effectively unchanged within
the trial ranges. With tight-bound they are 9.672 and 9.625 ms, also close.
The tiny 204-byte coverage does **not** explain its historical GPG-to-adaptive
gain. That comparison included other pipeline differences; this experiment
does not isolate those other differences. Here the large incremental gain is
tight-bound, which reduces final generated SHORT candidates from 175,649,447
to 5,503,945. On leon2 d30 the corresponding count drops from 81,543,398 to
5,765,395. These are final generated SHORT candidate counts, not the number
of all products examined or descriptor creations.

There are real limitations. Tight-bound is not universally faster: with both
formats, netcard x8 rises from 10.098 to 11.395 ms and nlpkkt120 from 5.465 to
6.102 ms. Bound maintenance can cost more than the work it removes; the
timings establish the regressions but do not alone attribute every component.
Likewise, Strip24 adds little on netcard d50 when 204-byte descriptors are
already enabled, and modestly regresses some original/scaled cases. Tiny
differences should not be called wins: the CSV retains each configuration's
minimum and maximum cold runtime alongside its median.

## Measurement boundary and controls

Controlled diagnostic, K = 1M, RTX 5090. Runtime is median cold setup + PFXT
over three standalone trials. Graph-file loading and SFXT are excluded.

Think of one assembly line with two optional ways to bundle waiting paths.
We disable either bundle without replacing the assembly line. A rejected
bundle emits ordinary LONG nodes instead. All variants retain the same
ordinary producer, adaptive policy, source-local representation, warp
aggregation, and arena-disabled allocation policy.

The diagnostic disables the materialized-queue-only final-window shortcut
in every variant. Consequently these are controlled ablation timings,
**not replacements for production headline timings**. Removing a format
also removes its allocation, replay and promotion work: that net effect
is what the experiment is intended to measure.

It also disables the safe-ordinary statistics shortcut in all variants
(`GPUCPG_ADAPTIVE_PFXT_ADAPTIVE_SAFE_MIN_PATHS=2147483647`). Its production
precheck uses a capped grid without a grid-stride loop and can inspect
only a prefix on large frontiers. Queue order changed whether full stats
were skipped on leon2 x8, despite equal modes and expansion counts. The
first attempt stopped at this gate. Full statistics are used consistently
in this corrected experiment; production defaults are not changed. This
is a preparation/selection confound, not evidence of incorrect top-K:
both ordinary and deferred branches still use exact candidate predicates.

All 1,032 timed queries passed their correctness and workload checks.
Correctness is checked during timing; no separate validation pass is
required. Within each case and bound setting, window output
counts, split values, recorded adaptive decisions, and final candidate
counts match. Equal-cost output ordering is not required to be identical.

## Tight-bound off

| Benchmark | Neither (ms) | 204 only (ms) | 24 only (ms) | Both (ms) | 204 alone | 24 added to 204 |
|---|---:|---:|---:|---:|---:|---:|
| netcard | 6.981 | 6.971 | 6.494 | 6.513 | 1.001x | 1.070x |
| netcard d10 | 33.123 | 33.405 | 16.609 | 16.773 | 0.992x | 1.992x |
| netcard d20 | 70.356 | 70.292 | 25.710 | 25.648 | 1.001x | 2.741x |
| netcard d30 | 118.362 | 96.048 | 115.078 | 93.382 | 1.232x | 1.029x |
| netcard d40 | 157.733 | 109.404 | 147.033 | 109.374 | 1.442x | 1.000x |
| netcard d50 | 142.471 | 116.298 | 152.884 | 116.689 | 1.225x | 0.997x |
| netcard x8 | 10.168 | 10.136 | 10.184 | 10.098 | 1.003x | 1.004x |
| netcard x16 | 21.914 | 22.071 | 22.075 | 22.223 | 0.993x | 0.993x |
| leon2 | 9.573 | 9.508 | 9.025 | 9.106 | 1.007x | 1.044x |
| leon2 d10 | 42.534 | 39.646 | 38.552 | 34.951 | 1.073x | 1.134x |
| leon2 d20 | 86.802 | 60.415 | 69.925 | 52.341 | 1.437x | 1.154x |
| leon2 d30 | 874.791 | 871.372 | 857.966 | 856.568 | 1.004x | 1.017x |
| leon2 d40 | 77.326 | 76.463 | 52.184 | 51.617 | 1.011x | 1.481x |
| leon2 d50 | 126.538 | 121.358 | 129.867 | 119.178 | 1.043x | 1.018x |
| leon2 x8 | 16.281 | 16.610 | 16.449 | 16.624 | 0.980x | 0.999x |
| leon2 x16 | 15.956 | 16.219 | 16.241 | 16.385 | 0.984x | 0.990x |
| leon3mp | 11.727 | 11.714 | 11.343 | 11.184 | 1.001x | 1.047x |
| leon3mp d10 | 72.093 | 72.107 | 34.971 | 35.208 | 1.000x | 2.048x |
| leon3mp d20 | 88.335 | 88.138 | 33.059 | 32.990 | 1.002x | 2.672x |
| leon3mp d30 | 59.405 | 59.509 | 32.172 | 32.071 | 0.998x | 1.856x |
| leon3mp d40 | 86.315 | 72.871 | 67.863 | 58.580 | 1.184x | 1.244x |
| leon3mp d50 | 96.094 | 85.099 | 100.053 | 84.253 | 1.129x | 1.010x |
| leon3mp x8 | 11.025 | 11.159 | 11.285 | 11.431 | 0.988x | 0.976x |
| leon3mp x16 | 26.858 | 27.278 | 27.051 | 27.326 | 0.985x | 0.998x |
| des_perf | 26.866 | 26.745 | 27.732 | 27.503 | 1.004x | 0.972x |
| des_perf d10 | 40.549 | 40.230 | 27.141 | 27.670 | 1.008x | 1.454x |
| des_perf d20 | 20.363 | 20.255 | 11.950 | 11.941 | 1.005x | 1.696x |
| des_perf d30 | 48.564 | 48.525 | 14.065 | 14.262 | 1.001x | 3.402x |
| des_perf d40 | 96.098 | 71.839 | 97.499 | 71.802 | 1.338x | 1.001x |
| des_perf d50 | 74.363 | 69.293 | 71.924 | 69.082 | 1.073x | 1.003x |
| des_perf x8 | 7.653 | 7.576 | 7.533 | 7.540 | 1.010x | 1.005x |
| des_perf x16 | 51.592 | 51.701 | 51.547 | 51.452 | 0.998x | 1.005x |
| vga_lcd | 7.946 | 7.994 | 8.152 | 8.395 | 0.994x | 0.952x |
| vga_lcd d10 | 30.916 | 30.995 | 16.759 | 16.684 | 0.997x | 1.858x |
| vga_lcd d20 | 63.374 | 63.094 | 35.053 | 35.144 | 1.004x | 1.795x |
| vga_lcd d30 | 38.335 | 38.356 | 20.788 | 20.931 | 0.999x | 1.833x |
| vga_lcd d40 | 81.290 | 81.031 | 36.064 | 35.640 | 1.003x | 2.274x |
| vga_lcd d50 | 52.536 | 52.654 | 24.389 | 24.687 | 0.998x | 2.133x |
| vga_lcd x8 | 14.460 | 14.528 | 12.789 | 13.170 | 0.995x | 1.103x |
| vga_lcd x16 | 14.690 | 14.861 | 12.356 | 12.434 | 0.988x | 1.195x |
| cage15 | 22.554 | 22.510 | 13.268 | 13.406 | 1.002x | 1.679x |
| M6 | 18.708 | 18.632 | 18.683 | 18.591 | 1.004x | 1.002x |
| nlpkkt120 | 8.084 | 8.094 | 5.458 | 5.465 | 0.999x | 1.481x |

Geometric means (speedup > 1 means faster):

- 204_alone: 1.044x
- 24_alone: 1.316x
- 24_added: 1.313x
- 204_added: 1.042x
- both: 1.372x

## Tight-bound on

| Benchmark | Neither (ms) | 204 only (ms) | 24 only (ms) | Both (ms) | 204 alone | 24 added to 204 |
|---|---:|---:|---:|---:|---:|---:|
| netcard | 7.432 | 7.448 | 7.112 | 7.059 | 0.998x | 1.055x |
| netcard d10 | 32.045 | 32.147 | 15.749 | 15.786 | 0.997x | 2.036x |
| netcard d20 | 69.625 | 69.400 | 24.465 | 24.550 | 1.003x | 2.827x |
| netcard d30 | 124.847 | 96.713 | 121.514 | 94.053 | 1.291x | 1.028x |
| netcard d40 | 148.574 | 99.097 | 137.949 | 98.925 | 1.499x | 1.002x |
| netcard d50 | 134.097 | 108.955 | 147.203 | 109.193 | 1.231x | 0.998x |
| netcard x8 | 11.402 | 11.382 | 11.502 | 11.395 | 1.002x | 0.999x |
| netcard x16 | 18.169 | 18.324 | 18.202 | 18.349 | 0.992x | 0.999x |
| leon2 | 9.637 | 9.818 | 9.187 | 9.177 | 0.982x | 1.070x |
| leon2 d10 | 42.526 | 39.300 | 38.583 | 34.803 | 1.082x | 1.129x |
| leon2 d20 | 86.914 | 61.259 | 70.290 | 53.358 | 1.419x | 1.148x |
| leon2 d30 | 122.887 | 118.664 | 105.344 | 103.129 | 1.036x | 1.151x |
| leon2 d40 | 73.574 | 72.950 | 49.227 | 48.219 | 1.009x | 1.513x |
| leon2 d50 | 118.648 | 114.862 | 122.750 | 112.010 | 1.033x | 1.025x |
| leon2 x8 | 13.016 | 13.099 | 13.150 | 13.409 | 0.994x | 0.977x |
| leon2 x16 | 17.541 | 17.623 | 17.828 | 17.832 | 0.995x | 0.988x |
| leon3mp | 11.461 | 11.506 | 11.044 | 10.979 | 0.996x | 1.048x |
| leon3mp d10 | 59.854 | 59.780 | 23.692 | 23.748 | 1.001x | 2.517x |
| leon3mp d20 | 82.620 | 82.665 | 27.422 | 27.469 | 0.999x | 3.009x |
| leon3mp d30 | 55.896 | 55.761 | 28.943 | 28.948 | 1.002x | 1.926x |
| leon3mp d40 | 88.256 | 71.684 | 68.594 | 57.894 | 1.231x | 1.238x |
| leon3mp d50 | 101.307 | 86.721 | 109.731 | 86.149 | 1.168x | 1.007x |
| leon3mp x8 | 11.697 | 11.617 | 11.844 | 11.887 | 1.007x | 0.977x |
| leon3mp x16 | 17.340 | 17.265 | 17.409 | 17.338 | 1.004x | 0.996x |
| des_perf | 28.600 | 28.106 | 28.883 | 29.022 | 1.018x | 0.968x |
| des_perf d10 | 27.003 | 27.035 | 14.496 | 14.992 | 0.999x | 1.803x |
| des_perf d20 | 22.473 | 22.489 | 12.962 | 13.271 | 0.999x | 1.695x |
| des_perf d30 | 58.288 | 58.169 | 14.982 | 15.128 | 1.002x | 3.845x |
| des_perf d40 | 102.846 | 72.652 | 101.634 | 71.899 | 1.416x | 1.010x |
| des_perf d50 | 69.963 | 65.654 | 67.558 | 64.525 | 1.066x | 1.018x |
| des_perf x8 | 8.119 | 8.022 | 8.072 | 8.064 | 1.012x | 0.995x |
| des_perf x16 | 9.672 | 9.506 | 9.584 | 9.625 | 1.017x | 0.988x |
| vga_lcd | 9.135 | 9.112 | 9.354 | 9.360 | 1.003x | 0.974x |
| vga_lcd d10 | 29.537 | 29.315 | 15.357 | 15.476 | 1.008x | 1.894x |
| vga_lcd d20 | 51.279 | 51.223 | 23.526 | 23.440 | 1.001x | 2.185x |
| vga_lcd d30 | 37.835 | 37.948 | 20.488 | 20.560 | 0.997x | 1.846x |
| vga_lcd d40 | 70.325 | 70.297 | 25.848 | 25.867 | 1.000x | 2.718x |
| vga_lcd d50 | 51.406 | 51.833 | 23.808 | 24.132 | 0.992x | 2.148x |
| vga_lcd x8 | 11.561 | 11.392 | 10.000 | 10.205 | 1.015x | 1.116x |
| vga_lcd x16 | 12.565 | 12.616 | 10.246 | 10.389 | 0.996x | 1.214x |
| cage15 | 23.731 | 23.795 | 14.143 | 14.255 | 0.997x | 1.669x |
| M6 | 4.901 | 4.863 | 4.956 | 4.852 | 1.008x | 1.002x |
| nlpkkt120 | 9.656 | 9.714 | 6.156 | 6.102 | 0.994x | 1.592x |

Geometric means (speedup > 1 means faster):

- 204_alone: 1.052x
- 24_alone: 1.360x
- 24_added: 1.357x
- 204_added: 1.050x
- both: 1.427x

## Isolating tight-bound with each representation held fixed

Each cell is bound-off runtime divided by bound-on runtime, using the
same descriptor configuration. Unlike the packing comparisons, this
comparison permits fewer generated candidates: avoiding that work is
the purpose of the bound. Both sides still validate against GPG.

| Benchmark | Neither: bound speedup | 204 only: bound speedup | 24 only: bound speedup | Both: bound speedup |
|---|---:|---:|---:|---:|
| netcard | 0.939x | 0.936x | 0.913x | 0.923x |
| netcard d10 | 1.034x | 1.039x | 1.055x | 1.063x |
| netcard d20 | 1.011x | 1.013x | 1.051x | 1.045x |
| netcard d30 | 0.948x | 0.993x | 0.947x | 0.993x |
| netcard d40 | 1.062x | 1.104x | 1.066x | 1.106x |
| netcard d50 | 1.062x | 1.067x | 1.039x | 1.069x |
| netcard x8 | 0.892x | 0.891x | 0.885x | 0.886x |
| netcard x16 | 1.206x | 1.205x | 1.213x | 1.211x |
| leon2 | 0.993x | 0.969x | 0.982x | 0.992x |
| leon2 d10 | 1.000x | 1.009x | 0.999x | 1.004x |
| leon2 d20 | 0.999x | 0.986x | 0.995x | 0.981x |
| leon2 d30 | 7.119x | 7.343x | 8.144x | 8.306x |
| leon2 d40 | 1.051x | 1.048x | 1.060x | 1.070x |
| leon2 d50 | 1.067x | 1.057x | 1.058x | 1.064x |
| leon2 x8 | 1.251x | 1.268x | 1.251x | 1.240x |
| leon2 x16 | 0.910x | 0.920x | 0.911x | 0.919x |
| leon3mp | 1.023x | 1.018x | 1.027x | 1.019x |
| leon3mp d10 | 1.204x | 1.206x | 1.476x | 1.483x |
| leon3mp d20 | 1.069x | 1.066x | 1.206x | 1.201x |
| leon3mp d30 | 1.063x | 1.067x | 1.112x | 1.108x |
| leon3mp d40 | 0.978x | 1.017x | 0.989x | 1.012x |
| leon3mp d50 | 0.949x | 0.981x | 0.912x | 0.978x |
| leon3mp x8 | 0.943x | 0.961x | 0.953x | 0.962x |
| leon3mp x16 | 1.549x | 1.580x | 1.554x | 1.576x |
| des_perf | 0.939x | 0.952x | 0.960x | 0.948x |
| des_perf d10 | 1.502x | 1.488x | 1.872x | 1.846x |
| des_perf d20 | 0.906x | 0.901x | 0.922x | 0.900x |
| des_perf d30 | 0.833x | 0.834x | 0.939x | 0.943x |
| des_perf d40 | 0.934x | 0.989x | 0.959x | 0.999x |
| des_perf d50 | 1.063x | 1.055x | 1.065x | 1.071x |
| des_perf x8 | 0.943x | 0.944x | 0.933x | 0.935x |
| des_perf x16 | 5.334x | 5.439x | 5.378x | 5.346x |
| vga_lcd | 0.870x | 0.877x | 0.871x | 0.897x |
| vga_lcd d10 | 1.047x | 1.057x | 1.091x | 1.078x |
| vga_lcd d20 | 1.236x | 1.232x | 1.490x | 1.499x |
| vga_lcd d30 | 1.013x | 1.011x | 1.015x | 1.018x |
| vga_lcd d40 | 1.156x | 1.153x | 1.395x | 1.378x |
| vga_lcd d50 | 1.022x | 1.016x | 1.024x | 1.023x |
| vga_lcd x8 | 1.251x | 1.275x | 1.279x | 1.291x |
| vga_lcd x16 | 1.169x | 1.178x | 1.206x | 1.197x |
| cage15 | 0.950x | 0.946x | 0.938x | 0.940x |
| M6 | 3.818x | 3.831x | 3.770x | 3.832x |
| nlpkkt120 | 0.837x | 0.833x | 0.887x | 0.896x |

All four marginal comparisons and creation counts are retained in the CSV.
The conservation checks count initial LONG storage, excluding SHORT/SKIP
and later promotions. Counters reuse existing producer totals without
new GPU scans or transfers. Timing includes the common diagnostic trace
and accounting overhead; profiling is performed separately.

## Measured memory traffic: not just a record-size estimate

All 32 profiles completed and passed an independent audit of their raw CSV
metrics, GPG exactness results, window and adaptive-decision signatures,
candidate totals and absence of retries. They match the corresponding timed
workloads. The timing audit was repeated afterward and all saved checksums
still matched. No GPU compute process was present at the final check.

The following is **read + write PFXT kernel DRAM traffic in decimal GB**,
measured separately from the standalone cold timings. Each cell is one
profiled query, not a three-trial runtime median. Read/write components and
kernel counts are in [the traffic CSV](descriptor-ablation-traffic-20260910.csv).

| Benchmark | Tight-bound | Neither (GB) | 204 only (GB) | 24 only (GB) | Both (GB) |
|---|---|---:|---:|---:|---:|
| netcard d10 | Off | 8.580 | 8.578 | 0.892 | 0.894 |
| netcard d10 | On | 8.606 | 8.608 | 0.917 | 0.920 |
| netcard d50 | Off | 30.086 | 3.003 | 30.092 | 3.010 |
| netcard d50 | On | 30.837 | 3.605 | 30.845 | 3.627 |
| leon2 d30 | Off | 76.701 | 74.862 | 73.894 | 72.034 |
| leon2 d30 | On | 9.182 | 7.332 | 6.460 | 4.619 |
| des_perf x16 | Off | 27.338 | 27.323 | 27.310 | 27.262 |
| des_perf x16 | On | 0.480 | 0.479 | 0.477 | 0.478 |

The clean attribution examples are:

- **Strip24 on netcard d10, bound off:** adding it to 204-only reduces traffic
  by 89.58% and cold runtime from 33.405 to 16.773 ms (1.992x).
- **204-byte descriptors on netcard d50, bound off:** enabling 204-only versus
  neither reduces traffic by 90.02%, but runtime improves only from 142.471
  to 116.298 ms (1.225x). Less traffic is real, but it is not a 10x runtime
  claim. Profiled kernel count rises from 9,769 to 10,405; substantial other
  work remains. These counters do not by themselves identify which remaining
  overhead dominates wall time or establish achieved bandwidth.
- **Complementarity on leon2 d30, bound on:** adding Strip24 to 204-only
  reduces traffic by 37.01% and cold runtime from 118.664 to 103.129 ms
  (1.151x). Holding both formats fixed and enabling tight-bound reduces
  traffic by 93.59% and runtime by 8.306x. With the redundant later work
  removed, packing has a larger relative effect: adding Strip24 to 204-only
  is 1.017x without tight-bound versus 1.151x with it.
- **The negative control, des_perf x16:** both formats versus neither reduce
  traffic by only 0.28% without tight-bound; runtime is effectively unchanged.
  Holding both formats fixed, tight-bound reduces traffic by 98.25% and
  improves runtime by 5.346x. This supports work elimination, not descriptor
  packing, as the source of that incremental gain.

In plain language: bundling waiting paths saves moving their individual
records, while a tighter bound can stop whole stretches of unnecessary work.
They solve different problems. A 90% reduction in measured traffic does not
mean a 90% reduction in total time, and tiny traffic differences from one
profile should not be overinterpreted. In particular, bound-on traffic is
not always lower: bound maintenance adds work and memory accesses too.

## Reproduction and evidence

Use the `strip24-production` worktree and the `build-strip` build containing
the ablation controls. Existing corrected CSR binaries and the existing GPG
K=1M golden files are reused, not regenerated. The timing run is:

```bash
python3 scripts/run-descriptor-ablation.py --timing-only \
  --data /tmp/gpu-cpg-tutorial.VAYiRz/repo/benchmarks/reproduction/csrbin \
  --reference experiments/strip24-fourway-20260910 \
  --out experiments/descriptor-ablation-timing-20260910
python3 scripts/audit-descriptor-ablation.py \
  experiments/descriptor-ablation-timing-20260910
```

The data path above records this machine's existing tutorial checkout; use
your local corrected CSR directory on another machine. Output directories
must be new: the runners deliberately refuse to overwrite an existing run.
The reference directory must contain `cases.txt` and the `goldens/` files
from the corrected four-way campaign. See `--help` for the complete options.

The independent audit passed all 1,032 raw logs, window/decision signatures,
candidate-count and LONG-representation conservation checks, and the saved
binary/source/runner checksums. The cost comparison uses the existing GPG
absolute 1e-3 plus relative 1e-6 tolerance, not bitwise float equality. It
also verifies K=1,000,000 returned costs and no capacity retries. The CSV
contains cold runtime medians and min/max ranges; it is not a profile export.

After timing, profile the sampled cases separately:

```bash
python3 scripts/profile-descriptor-ablation.py \
  --data /tmp/gpu-cpg-tutorial.VAYiRz/repo/benchmarks/reproduction/csrbin \
  --reference experiments/strip24-fourway-20260910 \
  --ablation experiments/descriptor-ablation-timing-20260910 \
  --out experiments/descriptor-ablation-timing-traffic-20260910
```

The profiling helper uses Nsight Compute 2025.4.0 from CUDA 13.1, application
replay, no cache flush or clock locking, and the `descriptor_ablation_pfxt`
NVTX range. It measures `dram__bytes_op_read.sum` and
`dram__bytes_op_write.sum`, then checks the answer and window workload against
the timed query. These metrics cover **PFXT kernel-attributed DRAM traffic**,
not static setup, copy-engine transfers, allocated capacity, or peak memory.
The timing runner checks for other GPU compute processes before each query;
this is not continuous proof of zero interference during every query.
