# Strip24 + tight-bound: less storage, less unnecessary path generation

**Review draft — published for review, not finalized.** The code checkpoint
is pushed on `strip24-production` at `156fe0f`. This document uses the completed
43-case four-way run: all 688 correctness/timing checks passed. Its tables
include cold setup + PFXT and are not assembled from older, unmatched studies.

The main story is complementary: **Strip24 avoids writing many not-yet-needed
LONG nodes; tight-bound reduces further generation once K valid cost witnesses
exist.** Neither promises a win on every graph. Across this suite, the successive
geometric-mean speedups over GPG are 1.465x, 1.975x and 2.406x.

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
The implementation also changes producer count aggregation; the measured win
is the combined pipeline change, not a pure byte-count-only experiment.

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

## 5. Four-way performance

All times are **cold static setup + PFXT**, in milliseconds; RTX 5090, K=1,000,000, candidate arena disabled. Each entry is the median of three standalone runs. Parentheses show speedup over GPG in the same row. Graph-file loading and SFXT are excluded from every column.

The two intermediate adaptive columns have tight-bound explicitly **disabled**. Only the final column enables both Strip24 and tight-bound. Each variant uses the same current graph and GPG golden.

### Original circuits

| Benchmark | GPG | Adaptive 204 | Adaptive 204 + Strip24 | Adaptive 204 + Strip24 + bound |
|---|---:|---:|---:|---:|
| netcard | 7.523 | 7.674 (0.98x) | 6.982 (1.08x) | 7.409 (1.02x) |
| leon2 | 12.149 | 10.254 (1.18x) | 9.940 (1.22x) | 9.561 (1.27x) |
| leon3mp | 14.160 | 12.596 (1.12x) | 11.872 (1.19x) | 11.384 (1.24x) |
| des_perf | 25.024 | 26.155 (0.96x) | 27.365 (0.91x) | 28.330 (0.88x) |
| vga_lcd | 7.770 | 8.442 (0.92x) | 8.492 (0.91x) | 9.477 (0.82x) |

### Densified circuits

| Benchmark | GPG | Adaptive 204 | Adaptive 204 + Strip24 | Adaptive 204 + Strip24 + bound |
|---|---:|---:|---:|---:|
| netcard d10 | 39.969 | 34.648 (1.15x) | 17.677 (2.26x) | 16.199 (2.47x) |
| netcard d20 | 88.660 | 72.545 (1.22x) | 26.672 (3.32x) | 25.001 (3.55x) |
| netcard d30 | 113.032 | 96.738 (1.17x) | 93.451 (1.21x) | 93.826 (1.20x) |
| netcard d40 | 161.380 | 109.471 (1.47x) | 109.296 (1.48x) | 98.885 (1.63x) |
| netcard d50 | 378.871 | 116.351 (3.26x) | 116.585 (3.25x) | 108.354 (3.50x) |
| leon2 d10 | 55.567 | 43.664 (1.27x) | 38.659 (1.44x) | 37.941 (1.46x) |
| leon2 d20 | 261.080 | 64.507 (4.05x) | 55.721 (4.69x) | 56.281 (4.64x) |
| leon2 d30 | 4795.310 | 885.785 (5.41x) | 867.227 (5.53x) | 111.302 (43.08x) |
| leon2 d40 | 134.679 | 85.578 (1.57x) | 56.944 (2.37x) | 51.806 (2.60x) |
| leon2 d50 | 149.453 | 122.136 (1.22x) | 119.509 (1.25x) | 112.116 (1.33x) |
| leon3mp d10 | 84.086 | 77.942 (1.08x) | 36.717 (2.29x) | 24.550 (3.43x) |
| leon3mp d20 | 91.748 | 66.567 (1.38x) | 34.550 (2.66x) | 27.978 (3.28x) |
| leon3mp d30 | 125.008 | 91.476 (1.37x) | 33.214 (3.76x) | 29.478 (4.24x) |
| leon3mp d40 | 214.129 | 80.523 (2.66x) | 63.803 (3.36x) | 61.001 (3.51x) |
| leon3mp d50 | 153.543 | 84.928 (1.81x) | 84.468 (1.82x) | 85.678 (1.79x) |
| des_perf d10 | 78.621 | 42.799 (1.84x) | 28.500 (2.76x) | 15.296 (5.14x) |
| des_perf d20 | 31.481 | 22.660 (1.39x) | 12.689 (2.48x) | 13.692 (2.30x) |
| des_perf d30 | 74.017 | 53.115 (1.39x) | 14.767 (5.01x) | 15.550 (4.76x) |
| des_perf d40 | 94.910 | 72.040 (1.32x) | 71.520 (1.33x) | 72.250 (1.31x) |
| des_perf d50 | 90.485 | 69.689 (1.30x) | 68.602 (1.32x) | 64.581 (1.40x) |
| vga_lcd d10 | 48.953 | 38.920 (1.26x) | 17.512 (2.80x) | 16.026 (3.05x) |
| vga_lcd d20 | 110.221 | 75.962 (1.45x) | 36.446 (3.02x) | 24.284 (4.54x) |
| vga_lcd d30 | 76.536 | 47.618 (1.61x) | 21.687 (3.53x) | 21.226 (3.61x) |
| vga_lcd d40 | 99.510 | 94.824 (1.05x) | 37.420 (2.66x) | 26.411 (3.77x) |
| vga_lcd d50 | 126.615 | 60.966 (2.08x) | 25.470 (4.97x) | 24.724 (5.12x) |

### Scaled circuits

| Benchmark | GPG | Adaptive 204 | Adaptive 204 + Strip24 | Adaptive 204 + Strip24 + bound |
|---|---:|---:|---:|---:|
| netcard x8 | 8.750 | 10.470 (0.84x) | 10.721 (0.82x) | 11.958 (0.73x) |
| netcard x16 | 33.735 | 22.712 (1.49x) | 22.972 (1.47x) | 18.732 (1.80x) |
| leon2 x8 | 22.950 | 17.385 (1.32x) | 17.682 (1.30x) | 13.729 (1.67x) |
| leon2 x16 | 8.568 | 16.433 (0.52x) | 16.769 (0.51x) | 18.399 (0.47x) |
| leon3mp x8 | 10.247 | 11.601 (0.88x) | 11.960 (0.86x) | 12.227 (0.84x) |
| leon3mp x16 | 56.488 | 28.028 (2.02x) | 28.264 (2.00x) | 17.793 (3.17x) |
| des_perf x8 | 12.025 | 8.377 (1.44x) | 8.497 (1.42x) | 8.737 (1.38x) |
| des_perf x16 | 245.742 | 50.718 (4.85x) | 50.751 (4.84x) | 10.231 (24.02x) |
| vga_lcd x8 | 28.612 | 18.313 (1.56x) | 14.437 (1.98x) | 11.034 (2.59x) |
| vga_lcd x16 | 28.295 | 17.918 (1.58x) | 13.479 (2.10x) | 11.079 (2.55x) |

### Non-circuit graphs

| Benchmark | GPG | Adaptive 204 | Adaptive 204 + Strip24 | Adaptive 204 + Strip24 + bound |
|---|---:|---:|---:|---:|
| cage15 | 22.814 | 21.597 (1.06x) | 13.711 (1.66x) | 14.535 (1.57x) |
| M6 | 68.023 | 19.089 (3.56x) | 19.108 (3.56x) | 5.238 (12.99x) |
| nlpkkt120 | 6.065 | 8.168 (0.74x) | 5.689 (1.07x) | 6.249 (0.97x) |

Geometric-mean speedup over GPG, across all 43 cases: adaptive: 1.465x, strip24: 1.975x, bound: 2.406x.

### Reading the progression honestly

* Strip24 alone makes netcard d20 fall from 72.545 to 26.672 ms and leon3mp
  d30 from 91.476 to 33.214 ms. Those intermediate columns both have bound off.
* Tight-bound then makes leon2 d30 fall from 867.227 to 111.302 ms, des_perf
  x16 from 50.751 to 10.231 ms, and M6 from 19.108 to 5.238 ms.
* Fewer candidates are not automatically faster. On des_perf d30, tightening
  removes only 1,492 additional stored candidates and increases time from
  14.767 to 15.550 ms. Original vga_lcd also gains little pruning and pays
  extra maintenance. These regressions remain visible.
* The complete method is not universally faster than GPG: for example,
  leon2 x16 is 18.399 versus 8.568 ms, including cold setup. This is separate
  from the much smaller incremental overhead of adding Strip24 to that case.

## 6. How close do we get to K?

Every correct variant returns exactly K results. The counts below are instead the final materialized SHORT-pile candidates **before** final top-K extraction. They are not all symbolic LONG products ever represented. The CSV also records min/max counts across repetitions.

To isolate tightening, compare the Strip24 column with the Strip24 + bound column: the representation is held fixed. The 204-only count provides additional context.

```text
excess_before = count_with_Strip24 - K
excess_after  = count_with_Strip24_and_bound - K
excess reduction = 100% * (1 - excess_after / excess_before)
```

For example, 5,000,000 → 1,400,000 candidates at K=1,000,000 is 4,000,000 → 400,000 excess candidates: **90% fewer excess candidates**. This measures how much closer the stored candidate count is to K, not distance between cost thresholds. K is a reference point, not a proven attainable minimum for internal search work.

| Benchmark | Adaptive 204 count | +Strip24 count | +Strip24 + bound count | Excess reduction: Strip24 → +bound |
|---|---:|---:|---:|---:|
| netcard | 1,257,046 | 1,257,046 | 1,225,474 | 12.28% |
| netcard d10 | 4,353,439 | 4,353,439 | 2,396,640 | 58.35% |
| netcard d20 | 2,084,474 | 3,757,593 | 2,470,661 | 46.67% |
| netcard d30 | 1,120,011 | 1,120,011 | 1,115,803 | 3.51% |
| netcard d40 | 3,672,596 | 3,672,596 | 2,471,298 | 44.95% |
| netcard d50 | 2,864,158 | 2,864,158 | 1,906,498 | 51.37% |
| netcard x8 | 2,235,631 | 2,235,631 | 1,685,484 | 44.52% |
| netcard x16 | 17,632,314 | 17,632,314 | 3,586,124 | 84.45% |
| leon2 | 3,897,584 | 3,897,584 | 1,852,937 | 70.56% |
| leon2 d10 | 1,995,500 | 1,995,500 | 1,670,325 | 32.66% |
| leon2 d20 | 1,551,252 | 1,551,252 | 1,460,343 | 16.49% |
| leon2 d30 | 81,543,398 | 81,543,398 | 5,765,395 | 94.08% |
| leon2 d40 | 2,029,631 | 2,029,631 | 1,583,854 | 43.29% |
| leon2 d50 | 2,787,358 | 2,787,358 | 2,045,352 | 41.51% |
| leon2 x8 | 10,808,122 | 10,808,122 | 1,771,342 | 92.14% |
| leon2 x16 | 1,182,354 | 1,182,354 | 1,178,994 | 1.84% |
| leon3mp | 3,898,207 | 3,898,207 | 1,946,414 | 67.34% |
| leon3mp d10 | 1,100,733 | 18,335,880 | 6,503,347 | 68.25% |
| leon3mp d20 | 2,413,444 | 9,073,472 | 4,262,236 | 59.59% |
| leon3mp d30 | 2,497,881 | 3,491,491 | 1,950,593 | 61.85% |
| leon3mp d40 | 1,913,577 | 1,913,577 | 1,663,113 | 27.42% |
| leon3mp d50 | 1,078,487 | 1,078,487 | 1,078,031 | 0.58% |
| leon3mp x8 | 3,416,102 | 3,416,102 | 2,108,690 | 54.11% |
| leon3mp x16 | 32,215,553 | 32,215,553 | 3,397,905 | 92.32% |
| des_perf | 1,045,966 | 1,045,966 | 1,016,087 | 65.00% |
| des_perf d10 | 13,820,327 | 13,820,327 | 3,780,932 | 78.31% |
| des_perf d20 | 1,222,410 | 1,222,410 | 1,193,177 | 13.14% |
| des_perf d30 | 1,034,196 | 1,034,196 | 1,032,704 | 4.36% |
| des_perf d40 | 1,092,966 | 1,092,966 | 1,088,998 | 4.27% |
| des_perf d50 | 2,995,248 | 2,995,248 | 2,107,297 | 44.50% |
| des_perf x8 | 7,159,537 | 7,159,537 | 2,893,668 | 69.26% |
| des_perf x16 | 175,649,447 | 175,649,447 | 5,503,945 | 97.42% |
| vga_lcd | 1,058,527 | 1,058,527 | 1,057,393 | 1.94% |
| vga_lcd d10 | 3,128,204 | 3,128,204 | 1,974,273 | 54.22% |
| vga_lcd d20 | 9,462,409 | 9,462,409 | 3,998,621 | 64.57% |
| vga_lcd d30 | 2,313,101 | 2,313,101 | 1,921,794 | 29.80% |
| vga_lcd d40 | 6,873,063 | 6,873,063 | 3,393,669 | 59.24% |
| vga_lcd d50 | 2,272,742 | 2,272,742 | 1,726,141 | 42.95% |
| vga_lcd x8 | 11,329,564 | 11,329,564 | 3,195,669 | 78.74% |
| vga_lcd x16 | 9,226,121 | 9,226,121 | 2,923,658 | 76.62% |
| cage15 | 1,546,814 | 1,546,814 | 1,377,707 | 30.93% |
| M6 | 44,347,948 | 44,347,948 | 2,438,158 | 96.68% |
| nlpkkt120 | 1,039,987 | 1,039,987 | 1,038,691 | 3.24% |

N/A means the unbounded Strip24 run already had no excess candidates. Negative values mean the bounded run had more final stored candidates; they must not be hidden or interpreted as a pruning benefit.

### Why some Strip24 counts differ from the 204-only counts

The original ordinary path can sort its materialized LONG pile to choose a
tighter final-window cutoff when a capacity threshold is reached. That shortcut
cannot simply sort the remaining materialized nodes when some other candidates
live inside strips: it would ignore those candidates. The Strip24 integration
therefore guards that queue-only shortcut while strips are present or being
created, and continues the ordinary split-growth flow instead.

This is a search-policy difference as well as a representation change. The
logs show the old final-window shortcut on leon3mp d10 only in the 204-only
variant. Four unbounded cases have different final counts: netcard d20 and
leon3mp d10/d20/d30. On leon3mp d10, 204-only stores 1.10M final candidates,
Strip24 alone stores 18.34M, and Strip24 + bound stores 6.50M. The 68.25% entry
means tightening removes excess relative to **18.34M**, not relative to 1.10M.
Similarly, combined counts remain above the 204-only counts on netcard d20
and leon3mp d20. All variants nevertheless pass the same top-K cost checks.

Thus the table does not claim that the combined method always stores fewer
final nodes than 204-only. Nor are final SHORT counts a complete memory-traffic
measure: they omit the often much larger population of individually stored
LONG nodes that Strip24 avoids. Candidate counts were identical across the
three repetitions of each case/variant in this run.

The large same-representation tightening examples are less ambiguous:
leon2 d30 eliminates 94.08% of excess candidates, des_perf x16 97.42%, and
M6 96.68%. On these cases, the final runtime also falls substantially.

## Measurement audit

All 172 initial correctness checks and 516 timed checks passed (43 cases × four
variants × three repetitions for timing). The summarizer rejects missing/failed
runs, count/overflow/retry errors, missing setup timers or candidate counts,
and bound runs that did not update successfully without fallback. Cost
comparison uses the existing 1e-3 absolute + 1e-6 relative tolerance, not bitwise
equality. The runner exited successfully with `FULL SUITE COMPLETE`.

Descriptor creation, allocation/growth, replay, materialization and bookkeeping are charged. Tight-bound cost gathering/sorting, safety checks and scalar synchronization are also charged. One-time static setup is added per trial before computing medians.

[Full numerical data](strip24-fourway-20260910.csv). The CSV retains separate setup/PFXT/full-query diagnostics, descriptor usage and candidate-count ranges. The headline tables above do not omit cold setup.

The GPU unit tests cover layout, packing/fallback, exact promotion, bit 31,
K/final-cutoff suppression, appended-strip cache invalidation, parent-storage
reallocation, incremental K-cost retention and floating-point guards. Both
passed with assertions enabled. Checksums taken during the suite still match
the benchmark binary and descriptor/bound implementation at completion. The
GPU was idle afterward; pre-query checks found no competing process requiring
a wait, but these checks are not continuous utilization monitoring.

Reproduction helpers are `scripts/run-strip24-full.py --four-way`,
`scripts/summarize-strip24-fourway.py` and `scripts/render-strip24-fourway.py`.
The run uses existing corrected CSR binaries and goldens; it does not regenerate
available inputs. Raw logs remain local under
`experiments/strip24-fourway-20260910/`.
