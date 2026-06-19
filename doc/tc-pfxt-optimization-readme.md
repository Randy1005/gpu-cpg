# TC PFXT Optimization Runbook

This note records how to build, validate, and retime the current TC PFXT
prototype.

The current proposal configuration uses:

- tensor-core BVSS deviation discovery;
- single-pass PFXT window expansion;
- single-work candidate materialization;
- spur-source grouped candidate generation;
- compact static deviation storage;
- tile-native short-only candidate emission;
- compact active-source grouping.

Candidate materialization, split/window management, and HPQ/LPQ state remain
CUDA work.

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

CUDA tests must run outside the sandbox on this machine.

## Current Runtime Flags

Use this exact flag set for headline TC runs:

```bash
GPUCPG_ENABLE_TC_PFXT=1
GPUCPG_TC_PFXT_SINGLE_PASS=1
GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1
GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE=1
GPUCPG_TC_PFXT_COMPACT_STATIC_DEVS=1
GPUCPG_TC_PFXT_TILE_NATIVE_CANDIDATE=1
GPUCPG_TC_PFXT_COMPACT_SOURCE_GROUPS=1
GPUCPG_TC_PFXT_DISABLE_PHASE_PROFILE=1
```

`GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE` is the legacy environment variable name
for spur-source grouped candidate generation. It is kept for script
compatibility.

Other `GPUCPG_TC_PFXT_*` flags are research knobs. Do not use them for headline
timing unless a note explicitly says so.

## In-Process Timing

The in-process driver loads the graph once and calls `CpGen::reset()` between
runs. It does not reset the CUDA device by default.

Baseline G-PathGen:

```bash
./build/examples/tc-pfxt-inprocess-timing \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d20.txt \
  --k 1000000 \
  --mode gpg \
  --warmup 1 \
  --trials 3
```

Current-best TC:

```bash
GPUCPG_ENABLE_TC_PFXT=1 \
GPUCPG_TC_PFXT_SINGLE_PASS=1 \
GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1 \
GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE=1 \
GPUCPG_TC_PFXT_COMPACT_STATIC_DEVS=1 \
GPUCPG_TC_PFXT_TILE_NATIVE_CANDIDATE=1 \
GPUCPG_TC_PFXT_COMPACT_SOURCE_GROUPS=1 \
GPUCPG_TC_PFXT_DISABLE_PHASE_PROFILE=1 \
./build/examples/tc-pfxt-inprocess-timing \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d20.txt \
  --k 1000000 \
  --mode tc \
  --warmup 1 \
  --trials 3
```

To rerun the full current-best TC density sweep:

```bash
scripts/run_tc_pfxt_current_best_retime.sh
```

The script writes `progress.log` and `summary.csv` under a timestamped
`experiments/tc_pfxt_current_best_inprocess_*` directory.

## Exactness

Generate or reuse a GPG golden cost file:

```bash
./build/examples/tc-pfxt-gate5 \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d20.txt \
  --k 1000000 \
  --mode baseline \
  --out experiments/tc_pfxt_source_local_20260614/golden/netcard_d20_k1000000.golden.costs
```

Compare TC prefixes against that file:

```bash
GPUCPG_ENABLE_TC_PFXT=1 \
GPUCPG_TC_PFXT_SINGLE_PASS=1 \
GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1 \
GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE=1 \
GPUCPG_TC_PFXT_COMPACT_STATIC_DEVS=1 \
GPUCPG_TC_PFXT_TILE_NATIVE_CANDIDATE=1 \
GPUCPG_TC_PFXT_COMPACT_SOURCE_GROUPS=1 \
./build/examples/tc-pfxt-inprocess-exactness \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d20.txt \
  --baseline-file experiments/tc_pfxt_source_local_20260614/golden/netcard_d20_k1000000.golden.costs \
  --ks 1000,10000,50000,100000,1000000
```

Expected final line:

```text
INPROCESS EXACTNESS PASS
```

## Latest Runtime Reference

Latest in-process retime:
`experiments/tc_pfxt_current_best_inprocess_20260618_2115/summary.csv`.

| density | K | GPG ms | TC ms | TC/GPG |
|---|---:|---:|---:|---:|
| d10 | 1M | 186.9 | 323.3 | 1.73x |
| d20 | 1M | 264.1 | 266.7 | 1.01x |
| d30 | 200K | 62.7 | 68.7 | 1.09x |
| d40 | 200K | 64.1 | 104.3 | 1.63x |
| d50 | 100K | 59.6 | 60.2 | 1.01x |

These are PFXT expansion times only. Graph input, SFXT construction, and static
deviation-matrix setup are cached outside the reported PFXT time.

## Current Interpretation

Compact active-source grouping removed a major setup cost from spur-source
grouped materialization. d20 and d50 are now near parity with cached GPG
baselines, and d30 is close.

The remaining bottleneck is product handling:

```text
parents_at_u * deviations_from_u
```

On d20 K=1M the current path still touches about `233M` parent/deviation
products, including `122.9M` products that are skipped by the current window.

Failed optional paths and their lessons are recorded in
[tc-pfxt-lessons-learned.md](tc-pfxt-lessons-learned.md).
