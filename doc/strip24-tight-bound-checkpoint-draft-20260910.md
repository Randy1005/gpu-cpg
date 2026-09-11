# Adaptive deferral, Strip24 and tight-bound: isolating where the gains come from

**Updated review draft — RTX 5090, K=1,000,000.** The primary evidence is now
the completed controlled study: **1,032 timed queries and 32 traffic profiles,
all passing correctness and workload checks**. Every runtime includes cold
static setup + PFXT; graph-file loading and SFXT are excluded. Candidate arenas
are disabled throughout.

The presentation story is: **bundle waiting paths across parents (204 bytes),
cover the ordinary branch's waiting children too (Strip24), then avoid
unnecessary later generation (tight-bound).** The first two save storage and
traffic; the last reduces work. Their contributions are measured separately,
not inferred from the total speedup over GPG.

Across all 43 graphs, the sequential geometric-mean speedups are **1.044x
for adding 204-byte deferral, 1.313x for adding Strip24, and 1.208x for enabling
tight-bound**. Combined: **1.657x over the same optimized pipeline with neither
format and no tight-bound**. This is **not a 1.657x comparison against GPG**,
and it includes regressions and cases with little packing opportunity.

## 1. Why another descriptor?

We want the best one million paths, but we encounter many more possibilities
while searching. A LONG path is a valid candidate whose cost is above the
current working cutoff. It might become useful later, but often never does.
Writing a full record for every such path spends memory bandwidth before we
know whether we will need it.

Our existing 204-byte descriptor already handles a useful situation: several
parents share a deviation range, and the entire product is LONG. But when the
adaptive algorithm chooses its ordinary path-generation branch, it still
creates individual LONG nodes. Strip24 fills that gap.

**Strip24 keeps one small recipe for several LONG children of one parent,
instead of writing each child as a separate node.**

It does not turn every 204-byte descriptor into 24 bytes. The large descriptor
can represent up to 32 parents times 16 deviations, or 512 products; Strip24
represents one parent and at most 32 deviation positions. The formats coexist.

## 2. Continue the existing toy example: 15x smaller records

We reuse [the original organization example](adaptive-checkpoint-20260831.md#the-organization-change-with-a-toy-example):

```text
tree:       A --1--> B --1--> T
            C --1------------> T
            D --1------------> T

deviations: A --2--> C       delta = 1
            A --3--> D       delta = 2
            B --2--> C       delta = 2

                     A->C, +1     A->D, +2
parent 102, slack .6     1.6          2.6
parent 100, slack .1     1.1          2.1
parent 105, slack 1.1    2.1          3.1
```

The parent IDs still reference actual PfxtNodes, and the compact deviation CSR
still holds A's outgoing deviation entries. The existing multi-parent branch
keeps its source groups. Strip24's ordinary branch does **not** need to build
such a group: it follows one parent's source cursor and packs that parent's
LONG children from the source's existing deviation range.
However, **A has only two deviations in this exact graph**. Strip24 requires
at least four LONG positions per slice in the tested configuration, so the
ordinary branch would not create a strip for one parent here.

To show a real accepted strip, make one explicit extension: give A fourteen
more deviation edges to E through R. Each new vertex has a tree edge of weight
1 to T. Set the new A-edge weights to 4 through 17, giving added slacks 3
through 16. The original A/B/C/D/T edges and parent IDs remain unchanged.

Now A has sixteen source-contiguous deviations with deltas 1 through 16.
Focus on parent **100**, whose slack is 0.1, and set the working cutoff to 1.5:

```text
parent 100 + A->C:  1.1    SHORT: emit immediately
parent 100 + A->D:  2.1    LONG
parent 100 + A->E:  3.1    LONG
...
parent 100 + A->R: 16.1    LONG
```

Without Strip24, the ordinary branch writes fifteen separate LONG PfxtNodes.
Each is 24 bytes: **15 × 24 = 360 bytes**, in addition to the one SHORT node.

Strip24 writes one 24-byte recipe instead:

| Field | Meaning in this example | Bytes |
|---|---|---:|
| parent | 100: stable index in SHORT storage | 4 |
| src | A | 4 |
| begin | First deviation's index in compact deviation CSR | 4 |
| length | 16 deviation positions in this slice | 2 |
| live | 15 children still represented | 2 |
| mask | Bits 1–15 set; bit 0 is already SHORT (`0xfffe`) | 4 |
| minimum | Cheapest remaining LONG cost: 2.1 | 4 |
| **Total** | | **24** |

The parent and deviation arrays already exist. The recipe references them; it
does not copy their contents. LONG output-record storage is **360 / 24 =
15x smaller**, a 93.33% reduction. The SHORT node is unchanged and excluded
from both sides of that ratio. A full 32-child strip is 32x smaller than
32 individual nodes. The current minimum pack is four LONG children, giving
4x smaller initial records for that smallest accepted pack.

These are **record-storage ratios**, not guaranteed whole-GPU-memory or runtime
ratios. The consumer also uses a pending mask, a count and an output offset:
12 logical scratch bytes per record when needed. Counting that scratch, the
15-LONG-child example is 360 / (24 + 12) = **10x smaller**. Allocation capacity,
temporary scan/reduction storage, and parent/deviation storage are additional.

If we count the unchanged SHORT node too, the entire sixteen-child example
uses 384 bytes of candidate records before versus 48 bytes afterward: **8x**.
Including the strip's 12-byte consumer scratch makes that 384 / 60 = **6.4x**.
The 15x/10x figures above specifically describe the fifteen LONG candidates.

The mask may contain holes. If only deviations 0, 3, 5 and 9 are LONG, the
record sets only those four bits. SHORT positions are emitted immediately;
SKIP positions are not stored. A source with more than 32 deviations is covered
by multiple slices. Each slice must independently meet the packing threshold.

The range is contiguous in the **deviation CSR**, not necessarily in its LONG
subset. For example, `begin=200, length=8` plus bits 1, 3, 4 and 6 represents
children using CSR entries 201, 203, 204 and 206. No reordering is required.
The `minimum` is the cheapest still-waiting child: if it exceeds the working
cutoff, the consumer need not inspect the strip's individual children. It is
updated after promotion. The 32-position limit permits one 32-bit bitmap;
it is a practical design choice, not a demonstrated optimum over 64/128.

### Recovering exact paths later

If the cutoff rises from 1.5 to 3.2, the LONG children with costs 2.1 and 3.1
become SHORT. The consumer reads parent 100, reads deviations A->D and A->E,
and creates those exact two PfxtNodes. It clears bits 1–2, changes `live` to
13, and updates the minimum to 4.1. Bit 0 stays clear: A->C was already
materialized and must not be emitted again. The other 13 children remain
recipes, not materialized nodes.

If the query finishes before those children become useful, their full nodes
are never written. This deferred materialization is the intended saving.

## 3. Preparation, extra passes, and downstream costs

There is no graph reorder, sort, new source grouping, or separate global
packing scan for Strip24. It uses the existing compact deviation CSR and the
ordinary producer's existing count-then-fill traversal.

During **count**, the GPU classifies candidates and counts SHORT nodes, all
LONG candidates, strips, and represented LONG candidates. During **fill**, it
repeats the producer traversal, emits SHORT and rejected LONG nodes normally,
and writes one recipe for each accepted pack. Count/fill is two traversals,
not one; Strip24 is integrated into those existing traversals.

The host still needs aggregate counts to size output buffers. The ordinary
control record grows from three integers to five when this feature is enabled.
No parent or deviation list is copied to the CPU. Buffer allocation/growth and
these scalar synchronizations are included in PFXT time; no new memory pool
is used to improve the comparison.

At a later split update, the consumer:

1. Checks cached minima; a strip whose minimum exceeds the cutoff needs no
   per-child cost inspection.
2. For eligible strips, computes a pending promotion mask and count on GPU.
   Repeated requests at the same cutoff reuse that result until append/promotion.
3. Reduces the counts and prefix-scans them to assign output ranges.
4. Materializes selected children into SHORT storage and updates live masks
   and minima for survivors.

These operations do introduce consumer kernels and scratch storage. They are
included in the measured PFXT time. A strip is worthwhile when avoided node
writes and LONG-queue handling repay its construction and replay costs.
The original production integration also changed producer count aggregation.
The new controlled study below holds that producer and aggregation fixed in
all variants, so toggling a format measures its net storage/consumer benefit,
including its allocation, replay and bookkeeping costs.

## 4. Tight-bound: stop exploring paths we can already prove are too expensive

Strip24 changes how we **store** candidates. Tight-bound reduces how much
unnecessary path generation we **do** after enough valid candidates are known.

Use K=3 and the costs in the original toy product above. To illustrate the
certificate, suppose the evaluated set currently contains costs 1.6, 2.6 and
3.1. We already have three valid paths costing at most 3.1. Therefore the final
third-best cost cannot be greater than 3.1, even before all paths are known.

Later we discover costs 1.1 and 2.1. The best three known costs are now 1.1,
1.6 and 2.1, so the upper bound tightens from 3.1 to 2.1. The path costing 2.6
cannot be among the best three anymore. This is an illustrative discovery order,
not a claim that the GPU processes the toy product in that exact order.

This bound is based on actual valid paths, not on knowing the answer in advance
or fitting a threshold to a benchmark. It is an upper bound on the unknown
K-th cost; it need not equal the final answer.

### Before K, at K, and afterward

* **Before K:** fewer than K evaluated SHORT paths cannot supply this certificate.
  The existing split-growth strategy continues. No imaginary K-th cost is used.
* **After at least K paths exist:** gather their costs on GPU, retain the best K,
  and use the K-th cost as the certified upper bound.
* **At later expansion windows:** add only the new append-only SHORT suffix to
  the retained K costs, sort on GPU, and retain the best K again. Already
  discarded costs cannot become useful: K known costs were already no larger.

The persistent logical cost payload is `4 × K` bytes: **4 MB at K=1,000,000**.
During an update, the buffer also contains newly appended costs, plus sorting
workspace. This is not a claim of a strict 4 MB peak allocation.

The implementation reads aggregate scalar information back to the host: the
minimum cached deviation when checking the domain, and the current K-th cost
when updating the cutoff. Cost arrays remain on GPU. Sorting, allocation,
scalar readback, the safety calculation and synchronization are all charged
to PFXT. This is not a zero-round-trip feature.

### Why pruning descendants is safe

With nonnegative deviation costs, a child's cost cannot be below its parent's
cost. A path above 2.1 cannot lead to a descendant at or below 2.1, so it can be
pruned safely in the example.

Cached floating-point deviations can have small negative values. The code does
not blindly assume exact nonnegativity. It finds the minimum cached deviation
and uses a graph-depth limit to allow for the most a descendant could decrease,
including conservative floating-point rounding. The actual pruning cutoff can
therefore be above the K-th witness cost. Unsupported numerical cases
fall back to the previous policy. Equal-cost ties are not discarded merely for
being equal to the bound.

Tight-bound does not erase previously materialized parent nodes: descendants
may still reference them. It also becomes available only after a window has
produced at least K valid witnesses, possibly overshooting K substantially.
This is why the final stored-candidate count need not approach exactly K even
when later pruning is effective.

## 5. Controlled progression: which addition bought the speedup?

### Four stages of the same pipeline

| Stage | Name used below | What changes from the previous stage? |
|---|---|---|
| A | No descriptors | Shared optimized adaptive machinery, but LONG outputs become individual nodes; bound off |
| B | Adaptive-defer (204) | Enable multi-parent 204-byte descriptors |
| C | +Strip24 | Also pack selected LONG children in the ordinary branch |
| D | +Tight-bound | Keep both formats and enable certified bound tightening |

The A→B comparison isolates the **net benefit of the 204-byte representation**
within this framework; it does not claim to isolate the invention of adaptive
grouping itself. Source-local layout, grouping machinery, adaptive decisions,
ordinary producer/count aggregation and warp aggregation are held fixed.
Rejected packs become individual LONG nodes. B→C isolates Strip24, and C→D
isolates tightening with representation held fixed.

Two controls are essential. The experiment disables the materialized-LONG-only
final-window shortcut in **every** variant, so toggling packing does not change
its search policy. It also forces full adaptive statistics in every variant.
The production safe-ordinary precheck uses a capped grid without a grid-stride
loop; its queue-order-dependent prefix inspection changed whether stats were
skipped on leon2 x8. The first attempt stopped at that comparison gate. The
corrected study bypasses the shortcut uniformly rather than changing production
defaults. Both generation branches still use exact candidate predicates.

Consequently these are **controlled diagnostic timings, not replacement
production-default headlines**. All four packing configurations, including
Strip24-only, were tested with bound off/on. Within each bound setting, every
window's split/output counts, recorded adaptive decisions, final candidate
count and total initial LONG storage agree. Bound off/on intentionally allows
different work: pruning is the intended benefit.

### Full-suite progression

Times are median **cold setup + PFXT in ms** over three standalone trials.
**Each parenthesized speedup is cumulative relative to Adaptive-defer**, the
first column: Adaptive-defer / current runtime. Below 1 means a slowdown.
Counts and unrounded times,
including trial ranges, are retained in the [full ablation CSV](descriptor-ablation-20260910.csv).
The [presentation progression CSV](descriptor-progression-20260910.csv)
contains these four stages and their incremental/cumulative speedups.

#### Original circuits

| Benchmark | A: No descriptors | B: Adaptive-defer (204) | C: +Strip24 | D: +Tight-bound |
|---|---:|---:|---:|---:|
| netcard | 6.981 | 6.971 (1.000x) | 6.513 (1.070x) | 7.059 (0.988x) |
| leon2 | 9.573 | 9.508 (1.000x) | 9.106 (1.044x) | 9.177 (1.036x) |
| leon3mp | 11.727 | 11.714 (1.000x) | 11.184 (1.047x) | 10.979 (1.067x) |
| des_perf | 26.866 | 26.745 (1.000x) | 27.503 (0.972x) | 29.022 (0.922x) |
| vga_lcd | 7.946 | 7.994 (1.000x) | 8.395 (0.952x) | 9.360 (0.854x) |

#### Densified circuits

| Benchmark | A: No descriptors | B: Adaptive-defer (204) | C: +Strip24 | D: +Tight-bound |
|---|---:|---:|---:|---:|
| netcard d10 | 33.123 | 33.405 (1.000x) | 16.773 (1.992x) | 15.786 (2.116x) |
| netcard d20 | 70.356 | 70.292 (1.000x) | 25.648 (2.741x) | 24.550 (2.863x) |
| netcard d30 | 118.362 | 96.048 (1.000x) | 93.382 (1.029x) | 94.053 (1.021x) |
| netcard d40 | 157.733 | 109.404 (1.000x) | 109.374 (1.000x) | 98.925 (1.106x) |
| netcard d50 | 142.471 | 116.298 (1.000x) | 116.689 (0.997x) | 109.193 (1.065x) |
| leon2 d10 | 42.534 | 39.646 (1.000x) | 34.951 (1.134x) | 34.803 (1.139x) |
| leon2 d20 | 86.802 | 60.415 (1.000x) | 52.341 (1.154x) | 53.358 (1.132x) |
| leon2 d30 | 874.791 | 871.372 (1.000x) | 856.568 (1.017x) | 103.129 (8.449x) |
| leon2 d40 | 77.326 | 76.463 (1.000x) | 51.617 (1.481x) | 48.219 (1.586x) |
| leon2 d50 | 126.538 | 121.358 (1.000x) | 119.178 (1.018x) | 112.010 (1.083x) |
| leon3mp d10 | 72.093 | 72.107 (1.000x) | 35.208 (2.048x) | 23.748 (3.036x) |
| leon3mp d20 | 88.335 | 88.138 (1.000x) | 32.990 (2.672x) | 27.469 (3.209x) |
| leon3mp d30 | 59.405 | 59.509 (1.000x) | 32.071 (1.856x) | 28.948 (2.056x) |
| leon3mp d40 | 86.315 | 72.871 (1.000x) | 58.580 (1.244x) | 57.894 (1.259x) |
| leon3mp d50 | 96.094 | 85.099 (1.000x) | 84.253 (1.010x) | 86.149 (0.988x) |
| des_perf d10 | 40.549 | 40.230 (1.000x) | 27.670 (1.454x) | 14.992 (2.683x) |
| des_perf d20 | 20.363 | 20.255 (1.000x) | 11.941 (1.696x) | 13.271 (1.526x) |
| des_perf d30 | 48.564 | 48.525 (1.000x) | 14.262 (3.402x) | 15.128 (3.208x) |
| des_perf d40 | 96.098 | 71.839 (1.000x) | 71.802 (1.001x) | 71.899 (0.999x) |
| des_perf d50 | 74.363 | 69.293 (1.000x) | 69.082 (1.003x) | 64.525 (1.074x) |
| vga_lcd d10 | 30.916 | 30.995 (1.000x) | 16.684 (1.858x) | 15.476 (2.003x) |
| vga_lcd d20 | 63.374 | 63.094 (1.000x) | 35.144 (1.795x) | 23.440 (2.692x) |
| vga_lcd d30 | 38.335 | 38.356 (1.000x) | 20.931 (1.832x) | 20.560 (1.866x) |
| vga_lcd d40 | 81.290 | 81.031 (1.000x) | 35.640 (2.274x) | 25.867 (3.133x) |
| vga_lcd d50 | 52.536 | 52.654 (1.000x) | 24.687 (2.133x) | 24.132 (2.182x) |

#### Scaled circuits

| Benchmark | A: No descriptors | B: Adaptive-defer (204) | C: +Strip24 | D: +Tight-bound |
|---|---:|---:|---:|---:|
| netcard x8 | 10.168 | 10.136 (1.000x) | 10.098 (1.004x) | 11.395 (0.890x) |
| netcard x16 | 21.914 | 22.071 (1.000x) | 22.223 (0.993x) | 18.349 (1.203x) |
| leon2 x8 | 16.281 | 16.610 (1.000x) | 16.624 (0.999x) | 13.409 (1.239x) |
| leon2 x16 | 15.956 | 16.219 (1.000x) | 16.385 (0.990x) | 17.832 (0.910x) |
| leon3mp x8 | 11.025 | 11.159 (1.000x) | 11.431 (0.976x) | 11.887 (0.939x) |
| leon3mp x16 | 26.858 | 27.278 (1.000x) | 27.326 (0.998x) | 17.338 (1.573x) |
| des_perf x8 | 7.653 | 7.576 (1.000x) | 7.540 (1.005x) | 8.064 (0.939x) |
| des_perf x16 | 51.592 | 51.701 (1.000x) | 51.452 (1.005x) | 9.625 (5.372x) |
| vga_lcd x8 | 14.460 | 14.528 (1.000x) | 13.170 (1.103x) | 10.205 (1.424x) |
| vga_lcd x16 | 14.690 | 14.861 (1.000x) | 12.434 (1.195x) | 10.389 (1.430x) |

#### Non-circuit graphs

| Benchmark | A: No descriptors | B: Adaptive-defer (204) | C: +Strip24 | D: +Tight-bound |
|---|---:|---:|---:|---:|
| cage15 | 22.554 | 22.510 (1.000x) | 13.406 (1.679x) | 14.255 (1.579x) |
| M6 | 18.708 | 18.632 (1.000x) | 18.591 (1.002x) | 4.852 (3.840x) |
| nlpkkt120 | 8.084 | 8.094 (1.000x) | 5.465 (1.481x) | 6.102 (1.326x) |

### What to emphasize to the audience

- **204-byte win:** netcard d40 improves 157.733→109.404 ms (1.442x).
  On netcard d50 it improves 142.471→116.298 ms (1.225x); adding Strip24
  afterward changes little, because almost all stored LONG paths already use 204.
- **Strip24 win:** netcard d20 improves 70.292→25.648 ms (2.741x);
  des_perf d30 improves 48.525→14.262 ms (3.402x). These are B→C
  comparisons with identical expansion workloads, not gains from a changed cutoff.
- **Both formats help:** leon2 d20 progresses 86.802→60.415→52.341 ms.
  Neither representation alone captures every useful packing opportunity.
- **Tight-bound win:** leon2 d30 improves 856.568→103.129 ms (8.306x),
  des_perf x16 51.452→9.625 ms (5.346x), and M6 18.591→4.852 ms
  (3.832x), with both descriptor formats held fixed.
- **Do not hide losses:** original des_perf slows 26.745→27.503 ms when
  adding Strip24. Tight-bound slows netcard x8 10.098→11.395 ms and
  nlpkkt120 5.465→6.102 ms. Saved work can be too small to repay maintenance.
  Tiny changes near 1x are not convincing wins; inspect the CSV's trial ranges.

The parenthesized values in these tables are cumulative from Adaptive-defer.
For incremental attribution, use the separate `descriptor-progression-20260910.csv`
columns `strip24_incremental_speedup` and `tight_bound_incremental_speedup`.
The incremental attribution depends on the chosen order. As a cross-check,
the full 2×2 experiment also tests Strip24 without 204: with bound off,
Strip24 alone gives a 1.316x geomean, versus 1.313x when added to 204;
adding 204 on top of Strip24 gives 1.042x, versus 1.044x without Strip24.
With bound on, adding Strip24 to 204 gives 1.357x. Do not add percentage
improvements together or assume every graph benefits equally.

### Where the older GPG comparison belongs

The earlier production-style four-way study remains available in
[its original CSV](strip24-fourway-20260910.csv): geomean speedups over GPG
were 1.465x (adaptive 204), 1.975x (+Strip24), and 2.406x (+bound).
Those are whole-pipeline comparisons, not pure descriptor attribution.
The original integration also changed count aggregation and guarded a
queue-only final-window shortcut; four unbounded cases changed final
candidate counts (netcard d20 and leon3mp d10/d20/d30).

The old tables are therefore replaced here by the controlled progression.
Do not splice old GPG times into the new A→B→C→D chain or attribute the whole
GPG-to-adaptive gain to descriptors. On des_perf x16, for example, historical
GPG was 245.742 ms and adaptive 204 was 50.718 ms, but the controlled experiment
below shows almost no incremental packing benefit. Other pipeline changes
account for that historical gap; this study does not isolate which ones.

## 6. Traffic evidence: does compression actually save memory movement?

These are separate Nsight Compute measurements of **PFXT kernel DRAM reads +
writes**, in decimal GB. They follow the same A→B→C→D stages as the runtime
table. They exclude static setup and copy-engine traffic; they are not peak
memory, allocation capacity, or theoretical record bytes. Each cell is one
profiled query; standalone timings above are not taken from the profiler.

| Benchmark | A: No descriptors | B: 204 | C: +Strip24 | D: +Tight-bound |
|---|---:|---:|---:|---:|
| netcard d10 | 8.580 | 8.578 | 0.894 | 0.920 |
| netcard d50 | 30.086 | 3.003 | 3.010 | 3.627 |
| leon2 d30 | 76.701 | 74.862 | 72.034 | 4.619 |
| des_perf x16 | 27.338 | 27.323 | 27.262 | 0.478 |

**Three useful studies for slides:**

1. **Compression works, but traffic is not runtime.** On netcard d10, B→C
   cuts traffic 89.58% and improves runtime 1.992x. On netcard d50, A→B
   cuts traffic 90.02%, yet improves runtime only 1.225x. The latter still
   executes substantial other work: profiled kernel count rises from 9,769
   to 10,405. We have established traffic savings, not a proportional
   bandwidth-bound runtime model or the exact cause of every remaining stall.
2. **Tightening exposes packing's contribution.** On leon2 d30, C→D cuts
   traffic 93.59% and improves runtime 8.306x. Separately, with bound already
   on, adding Strip24 to 204 cuts traffic 7.332→4.619 GB (37.01%) and
   time 118.664→103.129 ms (1.151x). Without bound, the same addition
   is only 1.017x: much larger later work hides the packing benefit.
3. **A negative control prevents a misleading story.** On des_perf x16,
   A→C reduces traffic only 0.28% and time is effectively unchanged
   (51.592→51.452 ms). C→D reduces traffic 98.25% and time by 5.346x.
   Its big incremental win is work elimination, not descriptor compression.

Even traffic does not always fall with tighter bounds: netcard d10 rises
0.894→0.920 GB from C→D despite fewer final SHORT candidates. Bound
maintenance adds accesses too. Aggregate counters establish the net effect;
they do not alone assign every byte to a particular cause.

[All 32 traffic measurements, separate reads/writes and kernel counts](descriptor-ablation-traffic-20260910.csv)
and [the complete controlled analysis](descriptor-ablation-20260910.md)
include the omitted Strip24-only and bound-on marginal comparisons.

## 7. Candidate counts: tightening removes work, not just records

Every correct query returns K=1M costs. The following counts are **final
materialized SHORT-pile candidates before top-K extraction**, not all products
examined, LONG paths stored, or descriptor records. In this controlled campaign,
A, B and C have the same count in every case; only D changes the pruning policy.

| Benchmark | A/B/C: bound off | D: bound on | Reduction in excess above K |
|---|---:|---:|---:|
| netcard d10 | 4,353,439 | 2,396,640 | 58.35% |
| netcard d50 | 2,864,158 | 1,906,498 | 51.37% |
| leon2 d30 | 81,543,398 | 5,765,395 | 94.08% |
| des_perf x16 | 175,649,447 | 5,503,945 | 97.42% |
| M6 | 44,347,948 | 2,438,158 | 96.68% |
| des_perf d30 | 1,034,196 | 1,032,704 | 4.36% |
| nlpkkt120 | 1,039,987 | 1,038,691 | 3.24% |

The last column is `1 - (count_after-K)/(count_before-K)`, not the percentage
reduction in all generated candidates. All 43 counts remain in the full CSV.
For leon2 d30, 81.54M→5.77M and the matching large traffic reduction support
the pruning explanation. For des_perf d30, tightening removes only 1,492
final candidates and slows C→D from 14.262 to 15.128 ms. Fewer candidates
are not automatically faster, nor do these counts capture all traffic saved
by not writing LONG nodes.

## 8. Descriptor distribution: where each format applies

A 204-byte descriptor and a 24-byte strip are not two sizes of the same
object being converted back and forth. The 204 format shares multiple parents
in the deferred branch; Strip24 packs selected LONG children of one active
SHORT parent in the ordinary branch. Some children still become individual
24-byte PfxtNodes. Later promotion materializes a node, not the other format.

### Count represented candidates, not just descriptor records

For each query, count each LONG candidate-generation event once, at its
initial storage decision: in Strip24, in a 204-byte descriptor, or individually.
Divide by the sum of those three counts. Later replay/promotion is not counted
again. SHORT/SKIP outputs and LONG outputs suppressed after K are outside
this denominator. These are cumulative storage events, not simultaneous live
occupancy, distinct costs, or a percentage of all path-generation work.

The following counts now come from **stage D, repetition 1 of the same new
controlled timing campaign**. They replace the older standalone coverage
table. [Exact counts and packing metrics](descriptor-ablation-coverage-20260910.csv)
are saved separately. A single real run is used so counts remain integers and conserve the
total; independently taking medians of categories could hide small packing
differences. Three timing repetitions and all other variants remain in raw logs.
Physical grouping can vary slightly while all audited workload totals agree.

| Benchmark | LONG paths in Strip24 | LONG paths in 204 | Individual LONG paths |
|---|---:|---:|---:|
| netcard | 1,216,011 (96.118%) | 0 (0.000%) | 49,109 (3.882%) |
| netcard d10 | 72,062,747 (99.963%) | 0 (0.000%) | 27,013 (0.037%) |
| netcard d20 | 187,069,083 (99.714%) | 0 (0.000%) | 537,311 (0.286%) |
| netcard d30 | 86,938 (0.025%) | 336,585,456 (97.598%) | 8,195,197 (2.376%) |
| netcard d40 | 1,036 (<0.001%) | 306,986,244 (98.223%) | 5,553,899 (1.777%) |
| netcard d50 | 1,232 (<0.001%) | 248,783,510 (98.638%) | 3,432,709 (1.361%) |
| netcard x8 | 8,871 (0.981%) | 1,835 (0.203%) | 893,358 (98.816%) |
| netcard x16 | 3,850 (0.141%) | 1,835 (0.067%) | 2,731,356 (99.792%) |
| leon2 | 757,478 (94.884%) | 0 (0.000%) | 40,840 (5.116%) |
| leon2 d10 | 9,460,390 (12.157%) | 53,247,896 (68.426%) | 15,109,738 (19.417%) |
| leon2 d20 | 20,397,316 (9.434%) | 177,580,273 (82.129%) | 18,244,229 (8.438%) |
| leon2 d30 | 18,221,529 (21.208%) | 47,091,951 (54.811%) | 20,603,829 (23.981%) |
| leon2 d40 | 64,181,509 (63.527%) | 30,253,881 (29.945%) | 6,594,984 (6.528%) |
| leon2 d50 | 13,803 (0.008%) | 149,329,095 (89.080%) | 18,292,218 (10.912%) |
| leon2 x8 | 268,366 (25.372%) | 614 (0.058%) | 788,746 (74.570%) |
| leon2 x16 | 376,077 (26.193%) | 614 (0.043%) | 1,059,118 (73.765%) |
| leon3mp | 671,465 (89.363%) | 0 (0.000%) | 79,927 (10.637%) |
| leon3mp d10 | 157,973,388 (99.751%) | 0 (0.000%) | 393,704 (0.249%) |
| leon3mp d20 | 268,800,102 (99.320%) | 0 (0.000%) | 1,839,642 (0.680%) |
| leon3mp d30 | 100,302,593 (99.247%) | 0 (0.000%) | 761,499 (0.753%) |
| leon3mp d40 | 12,569,594 (8.084%) | 130,862,723 (84.167%) | 12,047,981 (7.749%) |
| leon3mp d50 | 2,283 (0.001%) | 181,779,602 (95.306%) | 8,950,214 (4.693%) |
| leon3mp x8 | 48,039 (8.450%) | 253 (0.045%) | 520,246 (91.506%) |
| leon3mp x16 | 607,407 (12.381%) | 253 (0.005%) | 4,298,172 (87.614%) |
| des_perf | 25,890 (23.607%) | 0 (0.000%) | 83,779 (76.393%) |
| des_perf d10 | 51,625,075 (99.927%) | 345 (<0.001%) | 37,448 (0.072%) |
| des_perf d20 | 26,736,993 (99.420%) | 349 (0.001%) | 155,700 (0.579%) |
| des_perf d30 | 242,565,066 (99.811%) | 443 (<0.001%) | 459,881 (0.189%) |
| des_perf d40 | 3,313 (0.001%) | 316,478,772 (95.945%) | 13,373,766 (4.054%) |
| des_perf d50 | 4,672 (0.006%) | 69,830,995 (96.768%) | 2,327,601 (3.225%) |
| des_perf x8 | 21,396 (3.048%) | 234 (0.033%) | 680,413 (96.919%) |
| des_perf x16 | 12,176 (4.196%) | 234 (0.081%) | 277,784 (95.724%) |
| vga_lcd | 20,851 (76.004%) | 328 (1.196%) | 6,255 (22.800%) |
| vga_lcd d10 | 52,957,717 (99.422%) | 0 (0.000%) | 307,925 (0.578%) |
| vga_lcd d20 | 132,322,277 (99.488%) | 78 (<0.001%) | 681,242 (0.512%) |
| vga_lcd d30 | 52,603,442 (99.786%) | 77 (<0.001%) | 112,749 (0.214%) |
| vga_lcd d40 | 206,781,706 (99.639%) | 77 (<0.001%) | 748,212 (0.361%) |
| vga_lcd d50 | 106,059,287 (99.538%) | 1,482 (0.001%) | 491,235 (0.461%) |
| vga_lcd x8 | 1,324,736 (92.117%) | 86 (0.006%) | 113,273 (7.877%) |
| vga_lcd x16 | 2,181,833 (92.123%) | 414 (0.017%) | 186,148 (7.860%) |
| cage15 | 19,744,042 (97.449%) | 0 (0.000%) | 516,817 (2.551%) |
| M6 | 7,559 (99.802%) | 0 (0.000%) | 15 (0.198%) |
| nlpkkt120 | 18,227,437 (100.000%) | 0 (0.000%) | 12 (<0.001%) |

### How full is each recipe?

Coverage tells us how many waiting paths use a format. **Paths per descriptor**
tells us how efficiently it is packed. Neither quantity alone proves a runtime
gain: the consumer must avoid enough work to repay its bookkeeping.

| Benchmark | Strip24 records | Paths/strip | 204-byte records | Paths/204 |
|---|---:|---:|---:|---:|
| netcard | 70,573 | 17.23 | 0 | — |
| netcard d10 | 3,478,895 | 20.71 | 0 | — |
| netcard d50 | 46 | 26.78 | 1,987,153 | 125.20 |
| leon2 d30 | 720,304 | 25.30 | 524,493 | 89.79 |
| des_perf x16 | 3,044 | 4.00 | 15 | 15.60 |
| M6 | 1,399 | 5.40 | 0 | — |

The logical initial record bytes per represented LONG candidate are
`(24*strips + 204*tiles + 24*individual_nodes)/total_LONG`.
This excludes scratch, allocation capacity, shared arrays, replay reads and
later materialization writes. Use measured traffic in section 6 for bandwidth
claims, not this theoretical output-record ratio.

### Distribution + traffic + timing: the explanation

- **netcard d10:** 99.963% of stored LONG paths use Strip24, with 20.71
  paths per strip on average. This matches the large B→C traffic/time benefit.
- **netcard d50:** 98.639% use 204-byte descriptors; Strip24 covers less
  than 0.001%. This matches the large A→B traffic reduction and negligible B→C gain.
- **leon2 d30:** both formats cover substantial fractions. Their savings
  remain useful after tight-bound removes much of the later work.
- **des_perf x16:** 95.724% stay individual, so neither format covers much
  of its stored LONG population. Its 0.081% 204 coverage cannot explain
  the historical GPG-to-adaptive gap; the controlled timing/traffic comparison
  confirms negligible packing gain and a large tightening gain.
- **M6 is a denominator caution:** Strip24 covers 99.802%, but that means
  only 7,559 LONG paths packed out of 7,574 stored. Adding Strip24 is essentially
  neutral; tightening is the big gain. A high percentage over a tiny population
  is not a large optimization opportunity.

### Accounting overhead

Counters reuse producer totals already available to the host for queue sizing:
no new GPU scan, kernel launch or device-to-host transfer is added. Host-side
accumulation/checking and diagnostic traces are enabled for every timed variant.
Each window enforces conservation of LONG storage; independent strip telemetry
also agrees. This is common instrumentation overhead, not proof of zero cost.

The earlier three-pair counter-off/on check observed -1.48%, -0.86% and +1.18%
cold-time changes on netcard d50, leon3mp d10 and des_perf respectively.
These include noise and do not prove exact zero overhead. Its
[instrumentation CSV](descriptor-coverage-overhead-20260910.csv) and
[old coverage CSV](descriptor-coverage-20260910.csv) remain historical artifacts;
they are not mixed into the new timing or distribution tables.

## 9. Measurement audit and reproduction

The new timing study is 43 graphs × 4 packing configurations × 2 bound
settings × 3 repetitions = **1,032 queries**. Correctness is checked during
timing, without an additional full validation pass. Every query returns
K=1M costs matching the corrected GPG golden within the existing absolute
1e-3 plus relative 1e-6 tolerance. This is cost validation, not bitwise
path-identity equality. No retries, overflow or conservation failures were
accepted. Raw-log audit verifies workload signatures, output counts, setup
timers, bound updates without fallback, and saved executable/source checksums.

All allocation/growth, descriptor production, replay, promotion, bookkeeping,
bound cost gathering/sorting, scalar readback and synchronization are charged.
Cold static setup is added **per trial before taking the median**. No graph-file
loading or SFXT time is included. Unit tests cover Strip24 layout/packing,
holes and bit 31, exact promotion, cutoff suppression, queue invalidation,
parent-storage movement, coverage conservation and tight-bound numerical guards.
The existing unit tests passed with assertions enabled.

The 32 separate profiles cover four representative cases × eight settings.
Raw metric totals and exactness/window/decision signatures were independently
checked against timing logs. Nsight Compute uses application replay and the
`descriptor_ablation_pfxt` NVTX range, without cache flushing or clock locking.
Its runtime is not used in the progression table. Pre-query GPU-idle checks
are not continuous proof of no interference; the GPU was idle after completion.

From the `strip24-production` checkout with the current ablation-capable
`build-strip/examples/tc-pfxt-inprocess-exactness` binary:

```bash
python3 scripts/run-descriptor-ablation.py --timing-only \
  --data <corrected-csrbin-directory> \
  --reference <fourway-results-directory> \
  --out <new-timing-directory>
python3 scripts/audit-descriptor-ablation.py <new-timing-directory>
python3 scripts/report-descriptor-ablation.py <new-timing-directory> \
  --csv <new-summary.csv>
python3 scripts/profile-descriptor-ablation.py \
  --data <corrected-csrbin-directory> \
  --reference <fourway-results-directory> \
  --ablation <new-timing-directory> \
  --out <new-profile-directory>
```

Replace angle-bracket placeholders before running. The reference directory
contains `cases.txt` and corrected `goldens/*_k1000000.gpg.costs`; reuse existing
inputs rather than regenerate them. Output directories must be new. The
runner clears inherited GPUCPG environment overrides, maps ablation modes
1/2/3/4 to neither/204/24/both, and applies the uniform controls described in
section 5. The profile helper defaults to Nsight Compute 2025.4.0 installed
with CUDA 13.1 on this machine; override `--ncu` if necessary.

This study's local raw logs are in
`experiments/descriptor-ablation-timing-20260910/` and
`experiments/descriptor-ablation-timing-traffic-20260910/`.
See [the detailed ablation report](descriptor-ablation-20260910.md) for exact
controls and machine-specific reproduction paths. The earlier production
study's 688 passed checks and GPG runtime CSV are preserved separately; they
are no longer used to claim an isolated descriptor benefit here.
