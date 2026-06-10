# TC PFXT Optimization Notes

This note records how to build, test, and run the current TC PFXT prototype.
The implementation uses tensor-core BVSS only for deviation discovery; candidate
materialization, grouping, scans, split updates, and path-pile management remain
CUDA kernels.

## Build

```bash
cd /home/cchang289/Research/gpu-cpg
cmake --build build -j8 --target tc-pfxt-gate5 tc_pfxt_candidates
```

## Unit Tests

```bash
./build/unittests/tc_pfxt_candidates
```

Expected current result:

```text
11 test cases, 351 assertions, SUCCESS
```

## Main Gate 5 Driver

Baseline G-PathGen timing:

```bash
./build/examples/tc-pfxt-gate5 \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d20.txt \
  --k 1000000 \
  --mode baseline-timing
```

TC PFXT timing:

```bash
GPUCPG_TC_PFXT_FUSION=1 \
GPUCPG_TC_PFXT_SINGLE_PASS=1 \
./build/examples/tc-pfxt-gate5 \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d20.txt \
  --k 1000000 \
  --mode tc-timing
```

Exactness against stored GPG costs:

```bash
GPUCPG_TC_PFXT_FUSION=1 \
GPUCPG_TC_PFXT_SINGLE_PASS=1 \
./build/examples/tc-pfxt-gate5 \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d20.txt \
  --k 1000000 \
  --mode tc \
  --baseline-file experiments/netcard_density_crossover_20260605/netcard_d20_k1000000_costs.txt
```

Expected exactness line:

```text
max_diff=0
GATE 5 PASS
```

## Optimization Flags

| flag | default | effect |
|---|---:|---|
| `GPUCPG_TC_PFXT_FUSION=1` | off | enables the TC PFXT integrated path |
| `GPUCPG_TC_PFXT_SINGLE_PASS=1` | off | uses the single-pass TC PFXT window flow |
| `GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1` | off | uses the single-work candidate materialization path |
| `GPUCPG_TC_PFXT_USE_ATOMIC_FALLBACK=1` | off | force the atomic fallback path |

The single-work candidate path is intentionally opt-in. It preserves exactness
and reduces duplicate candidate classification work, but the default remains the
legacy pair-meta count/scan/fill path while larger validation continues.

## Latest Runtime Reference

These numbers are PFXT expansion time only. GPG and default TC values come from
`experiments/tc_pfxt_packed_scan_profile_20260609/*_trials.log`; single-work
values come from `experiments/tc_pfxt_single_work_candidate_20260610`.

| density | K | GPG ms | TC default ms | TC single-work ms | default TC/GPG | single-work TC/GPG |
|---|---:|---:|---:|---:|---:|---:|
| d10 | 1M | 191.5 | 456.4 | 375.6 | 2.38x | 1.96x |
| d20 | 1M | 284.9 | 636.6 | 531.6 | 2.23x | 1.87x |
| d30 | 200K | 81.1 | 326.2 | 286.9 | 4.02x | 3.54x |
| d40 | 200K | 95.0 | 386.7 | 328.9 | 4.07x | 3.46x |
| d50 | 100K | 78.0 | 295.5 | 269.0 | 3.79x | 3.45x |

## Current Interpretation

TC deviation discovery is cheap and exact, but end-to-end TC PFXT is dominated
by CUDA candidate materialization and grouping. The single-work candidate path
confirms that candidate fill is optimizable, but it remains opt-in until the
larger benchmark matrix is fully revalidated.
