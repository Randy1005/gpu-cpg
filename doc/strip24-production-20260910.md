# 24-byte strips: producer and consumer prototype

**Update:** the complete 43-case benchmark and current selector recommendation
are in [the full-suite report](strip24-full-20260910.md). Its headline boundary
is **cold setup + PFXT**. The 13-case table below is the earlier explicitly
PFXT-only prototype screen; it is not the full-suite headline.

## Result

The complete prototype passes the >=3% PFXT improvement gate on netcard d10,
leon2 d30, cage15, nlpkkt120 and original leon2. Keep it **opt-in**, not the
default: low-volume cases can regress, and a universal selection policy has not
been validated. Existing 204-byte descriptors and the baseline remain intact.

RTX 5090, K=1M, maintained bound enabled, arena disabled, three standalone
repetitions per mode with alternating order. Entries are medians. Speedup is
baseline/new; below 1 means slower. Baseline means the same checkpoint with
`GPUCPG_STRIP24=0`, not original GPG.

| Case | 204-byte-only PFXT ms | +Strip24 PFXT ms | PFXT speedup | Cold setup + PFXT speedup | Full query speedup |
|---|---:|---:|---:|---:|---:|
| netcard d10 | 29.795 | 12.609 | 2.363x | 2.054x | 1.037x |
| leon2 d30 | 120.023 | 102.172 | 1.175x | 1.160x | 1.016x |
| cage15 | 19.870 | 12.173 | 1.632x | 1.525x | 1.014x |
| nlpkkt120 | 7.384 | 3.938 | 1.875x | 1.539x | 1.011x |
| leon2 | 9.516 | 8.937 | 1.065x | 1.061x | 0.993x |
| netcard d50 | 94.299 | 93.813 | 1.005x | 1.004x | 0.999x |
| leon3mp d50 | 73.951 | 73.520 | 1.006x | 1.004x | 1.007x |
| des_perf d40 | 70.918 | 70.094 | 1.012x | 1.011x | 0.996x |
| des_perf | 27.080 | 28.016 | 0.967x | 0.967x | 0.997x |
| des_perf x16 | 9.126 | 9.252 | 0.986x | 0.988x | 1.002x |
| leon3mp x16 | 11.763 | 12.114 | 0.971x | 0.979x | 1.009x |
| netcard x16 | 11.117 | 11.393 | 0.976x | 0.985x | 0.990x |
| M6 | 4.400 | 4.600 | 0.957x | 0.963x | 0.980x |

All 104 queries passed the existing K-cost comparison: 26 initial correctness
queries, then 78 timed queries. The comparator uses absolute tolerance 1e-3
plus relative tolerance 1e-6, not bitwise equality. No retry or output-overflow
gate fired in the completed suite. The focused GPU unit test passed with
assertions enabled. Pre-query GPU checks found no other compute process; these
checks are not continuous proof of isolation. The GPU was idle after completion.

Full precision, cold setup/PFXT/query milliseconds and usage counters are in
[the CSV](strip24-production-20260910.csv). Cold setup + PFXT is the median of
each trial's sum, not the sum of two independently reported medians. Full query
includes SFXT and is often much larger; millisecond-level PFXT gains must not be
presented as comparable percentage gains of the whole query. Three repetitions
do not establish that small full-query differences are significant.

### Why it helps, and where it does not

* Netcard d10 stores 72,062,747 LONG products in 3,478,895 strips; only 2.81%
  subsequently become SHORT. Initial logical node storage falls from 1,729.51
  MB to 83.49 MB of strips. Avoiding the large ordinary LONG queue pays off.
* Leon2 d30 stores 18,221,529 products in 720,304 strips; only 1.69% are promoted.
  Initial logical storage falls from 437.32 MB to 17.29 MB.
* Cage15 stores 19,744,042 products in 2,126,300 strips; nlpkkt120 stores
  18,227,437 in 720,812. Their lack of old multi-parent descriptors does not
  imply lack of single-parent strip opportunities.
* Original des_perf packs only 25,890 products and later promotes 25,701
  (99.27%). M6 packs 7,559 and promotes all of them. There is little avoided
  materialization to repay the producer and queue overhead: PFXT regresses
  3.46% and 4.53%, respectively.
* Scaled cases pack few products or very short strips; their PFXT regressions
  range from 1.39% to 2.98%. The other dense cases already use the existing
  large descriptor path and offer almost no new strip volume.

These observations support a selective extension, not a universal compression
replacement. Original leon2 also demonstrates why the production gate matters:
it lost the conservative isolated oracle screen but wins PFXT once actual
queue allocation/maintenance are included. That does not prove the cause in
isolation; the producer-count ablation described below is still needed for
precise attribution. No benchmark-name dispatcher has been introduced.

Branch: `strip24-production`, based on the validated tight-bound checkpoint
`614b573`. The oracle branch remains separate. Default behavior is unchanged;
set `GPUCPG_STRIP24=1` to add strips alongside existing 204-byte descriptors.

## Representation and producer

A record stores parent/source/deviation-begin (12 bytes), slice length/live
count (4), remaining LONG mask (4), and minimum live cost (4). A slice contains
at most 32 consecutive deviations of one source for one parent. Parent IDs are
stable indices into the query's SHORT storage; pointers are reacquired after
storage growth.

The ordinary count pass records exact SHORT and LONG counts plus descriptor and
represented-product counts. The same traversal/class predicates in fill emit
SHORT nodes, rejected LONG nodes, or one strip. Counts are warp aggregated.
There is no separate grouping, sorting, oracle scan, or candidate retry.
`GPUCPG_STRIP24_MIN_LONG` defaults to four, clamped to [2,32]. This is a packing
threshold, not yet a validated universal performance policy.

SHORT count reaching K suppresses all new LONG/strip output in that invocation.
The same Thrust-backed count-then-resize allocation policy is retained; no new
GPU memory pool is introduced. Descriptor bytes replace the corresponding
LONG-node allocation. This is charged in PFXT timing.

## Consumer

The first split minimum includes each strip's cached minimum. At each split,
one GPU pass rejects records above the split and computes an eligibility mask
and count for the remaining records. The exact pending mask is retained for
fill, avoiding another classification pass. A prefix scan assigns each strip
its SHORT output range. Fill reconstructs only selected paths, clears their
bits, and updates the minimum of surviving products. Empty strips have an
infinite minimum and no live bits.

The strip queue participates in LPQ emptiness, remaining-product accounting,
split promotion and K retirement. Host readback contains aggregate counts, not
parent or deviation lists. Promotion masks, offsets, minima and path state stay
on the GPU. Count cache invalidates on append/promotion; new SHORT allocation
cannot invalidate stable parent IDs or the already computed masks.

## Gates and reporting

Focused GPU tests cover count/fill conservation, rejected packs, cached
eligibility, repeated split calls, actual reconstructed node fields, bit 31,
final-cutoff SKIP and K-crossing suppression. Full-query correctness precedes
benchmarking; every timed query also checks the existing GPG K-cost golden.

Compare the same binary with `GPUCPG_STRIP24=0` versus `=1`, maintained bound
enabled, candidate arena disabled. Report three boundaries:

* PFXT: includes production descriptor allocation/count/emission/promotion.
* Cold setup + PFXT: adds the existing static setup timer.
* In-memory query: elapsed `report_paths`, including SFXT and static setup;
  graph-file reading and the validator's subsequent `get_slacks` are outside.

The 13-case matched run uses fresh processes and alternating order, with three
measured repetitions after the full correctness gate. No benefit is claimed
until the complete production flow has been measured against 204-byte-only.

## What this comparison means

The baseline is tight-bound adaptive deferral, whose only descriptor format is
204 bytes. It still emits individual LONG nodes in ordinary-mode invocations.
Strip24 adds a format for those individual LONG paths; it does **not** turn every
204-byte descriptor into 24 bytes. Existing multi-parent descriptors remain.

For example, one parent with 24 eligible LONG deviations needs 24 individual
24-byte nodes (576 bytes) in the ordinary path, versus one 24-byte strip here.
The consumer reconstructs only the paths that actually become SHORT, from the
parent ID, deviation range and live bitmap. That is where the storage saving
comes from; merely shrinking existing descriptor headers is not this experiment.

These timings measure the combined producer/queue/consumer change. The new count
kernel also uses warp-aggregated counts instead of the original count structure.
Without a matched new-producer/no-packing ablation, the entire runtime gain must
not be attributed solely to descriptor bytes saved. Node bytes avoided are
logical emitted storage, not measured DRAM traffic; replay scratch, allocation,
queue growth and repeated reads also exist and are included in runtime.

The other oracle, retaining fully LONG rows from MIXED tiles, remains parked:
its conservative creation-plus-eligibility saving was only 0.115–0.696 ms on
the four eligible dense cases. Both oracle formats passed identity/promotion
checks and final K-cost validation. See the separate oracle report in the
`bound-descriptor-reshape` worktree, `doc/specialized-descriptor-gates-20260910.md`.

## Reproduction

Build on the target GPU architecture (the measured machine uses `120`):

```sh
cmake -S . -B build-strip -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-strip --target strip24-test tc-pfxt-inprocess-exactness -j2
build-strip/examples/strip24-test
```

Use the same graph and K=1M GPG golden for both modes, in separate idle-GPU
processes. Unset `GPUCPG_PFXT_CANDIDATE_ARENA` and
`GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA` (these are presence-based switches, so
setting them to zero is not equivalent to unsetting them).

```sh
env -u GPUCPG_PFXT_CANDIDATE_ARENA -u GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA \
  GPUCPG_ADAPTIVE_PFXT_BOUND=1 GPUCPG_STRIP24=0 \
  build-strip/examples/tc-pfxt-inprocess-exactness \
  --benchmark /path/to/graph.csrbin --baseline-file /path/to/graph_k1000000.gpg.costs \
  --ks 1000000 --mode adaptive
```

Repeat with `GPUCPG_STRIP24=1`; `strip24_summary` reports actual records created,
paths represented, and paths subsequently promoted. Repeat three times in
alternating order. The local suite runner records all input paths explicitly;
change its `data` and `goldens` paths on another machine. Its output directory
must be new. Summarize a complete suite with:

```sh
bash scripts/run-strip24-suite.sh experiments/my-strip24-run
python3 scripts/summarize-strip24-suite.py experiments/my-strip24-run
```

Measured build: CUDA 13.3.1, CCCL 3.3.3.0, RTX 5090. Unit assertions are explicitly
enabled in the Release test target. Raw production results are under
`experiments/strip24-production-20260910-v2/`; the earlier directory contains
an aborted runner attempt whose error pattern incorrectly matched zero-valued
overflow telemetry, not a candidate correctness failure.
