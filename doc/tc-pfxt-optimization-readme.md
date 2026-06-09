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
| `GPUCPG_TC_PFXT_USE_BLOCK_PAIR_FILL=1` | off | opt-in block-per-pair candidate count/fill |
| `GPUCPG_TC_PFXT_USE_ATOMIC_FALLBACK=1` | off | force the atomic fallback path |

The block-per-pair candidate fill is intentionally opt-in. It improves d20 but
regresses d50, so the default remains the legacy one-thread-per-pair fill.

## Latest Runtime Reference

These numbers are PFXT expansion time only. GPG and default TC values come from
`experiments/tc_pfxt_packed_scan_profile_20260609/*_trials.log`; block-fill
values come from `experiments/tc_pfxt_block_candidate_fill_20260609`.

| density | K | GPG ms | TC default ms | TC block-fill opt-in ms | default TC/GPG | block TC/GPG |
|---|---:|---:|---:|---:|---:|---:|
| d10 | 1M | 191.5 | 456.4 | not run | 2.38x | n/a |
| d20 | 1M | 284.9 | 636.6 | 551.5 | 2.23x | 1.94x |
| d30 | 200K | 81.1 | 326.2 | 327.6 | 4.02x | 4.04x |
| d40 | 200K | 95.0 | 386.7 | 388.8 | 4.07x | 4.09x |
| d50 | 100K | 78.0 | 295.5 | 320.2 | 3.79x | 4.10x |

## Current Interpretation

TC deviation discovery is cheap and exact, but end-to-end TC PFXT is dominated
by CUDA candidate materialization and grouping. The block-per-pair fill confirms
that candidate fill is optimizable, but a global default needs adaptive dispatch
based on source-group size to avoid d50-style regressions.
