# gpu-cpg

GPU critical path generation experiments, including the current TC-powered PFXT
prototype.

## Build

```bash
cd /home/cchang289/Research/gpu-cpg
cmake --build build -j8 --target \
  cpg \
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

## Main Modes

The examples below use locally generated netcard density-crossover data:
`benchmarks/tc_pfxt_crossover/netcard_d20.txt`. These benchmark and golden-cost
files are large experiment artifacts and are not tracked in git. Generate or
copy them into the shown paths before running the commands.

### G-PathGen Baseline

```bash
./build/examples/tc-pfxt-inprocess-timing \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d20.txt \
  --k 1000000 \
  --mode gpg \
  --warmup 1 \
  --trials 3
```

### TC PFXT, Single-Work Candidate Path

```bash
GPUCPG_ENABLE_TC_PFXT=1 \
GPUCPG_TC_PFXT_SINGLE_PASS=1 \
GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1 \
GPUCPG_TC_PFXT_LIGHT_STAGE_PROFILE=1 \
GPUCPG_TC_PFXT_DISABLE_PHASE_PROFILE=1 \
./build/examples/tc-pfxt-inprocess-timing \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d20.txt \
  --k 1000000 \
  --mode tc \
  --warmup 1 \
  --trials 3
```

### TC PFXT, Source-Local Candidate Path

This is the current proposal configuration.

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

### Exactness Against Golden Costs

Generate or reuse a GPG cost file, then compare multiple K prefixes:

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

The in-process drivers load the graph once and call `CpGen::reset()` between
runs. They do not reset the CUDA device by default. Use `--reset-device` only
for debugging a single short run.

## Current Results

The latest source-local TC comparison is documented in
[doc/tc-pfxt-optimization-readme.md](doc/tc-pfxt-optimization-readme.md).

| density | K | GPG ms | source-local TC ms | TC/GPG |
|---|---:|---:|---:|---:|
| d10 | 1M | 186.9 | 415.2 | 2.22x |
| d20 | 1M | 264.1 | 468.9 | 1.78x |
| d30 | 200K | 62.7 | 199.4 | 3.18x |
| d40 | 200K | 64.1 | 226.9 | 3.54x |
| d50 | 100K | 59.6 | 172.3 | 2.89x |

TC is not faster than GPG yet. The current value is architectural: deviation
discovery has been reformulated as a tensor-core-friendly operation while
preserving exact path ordering. The next work is reducing candidate
materialization overhead so more of PFXT is TC-shaped.
