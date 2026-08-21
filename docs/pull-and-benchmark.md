# Pull and benchmark the persistent fused PFXT branch

This procedure benchmarks the exact revision containing this file on an RTX 5090
against production GPG. Graphs are intentionally not stored in Git. Use the graph
files already present on the benchmark machine and regenerate all GPG golden files
on that machine after pulling.

Do not run `git clean`, delete local graph files, or reuse goldens from another GPU
or revision.

## 1. Record the machine and pull the revision

Start from a quiet GPU. Record any process already using it before running tests.

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv

git switch main
git pull --ff-only origin main
git rev-parse HEAD
git status --short
```

The checkout may contain untracked local graphs. That is expected. Record the SHA
and use that same SHA for every result in the comparison.

## 2. Configure a fresh native RTX 5090 build

Do not reuse an older CMake directory. The expected local toolkit from the original
RTX 5090 setup is CUDA 13.3.1; adjust `CUDA_HOME` only if its location changed.

```bash
export CUDA_HOME="${CUDA_HOME:-$PWD/.local/cuda-13.3.1}"
export BUILD="$PWD/build-rtx5090-persistent"

"$CUDA_HOME/bin/nvcc" --version
cmake --version

cmake -S . -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=120

cmake --build "$BUILD" --target \
  tc-pfxt-gate5 \
  tc-pfxt-inprocess-exactness \
  tc-pfxt-inprocess-timing \
  tc_pfxt_candidates \
  tc_pfxt_inprocess \
  -j"$(nproc)"
```

The configure step fetches CCCL if it is not already cached, so it may require
network access.

Run the focused tests before using the benchmark harnesses:

```bash
ctest --test-dir "$BUILD" \
  -R 'tc_pfxt_(candidates|inprocess)' \
  --output-on-failure
```

## 3. Select local graphs and an output directory

Set `GRAPH_DIR` to the directory on the RTX 5090 machine containing its existing
converted graph inputs. The expected density sweep names are shown below; change
only the paths if the local names differ.

```bash
export GRAPH_DIR=/path/to/local/tc_pfxt_crossover
export OUT="$PWD/experiments/rtx5090_persistent_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"/{golden,exactness,timing,profiles}

GRAPHS=(
  netcard_d10
  netcard_d20
  netcard_d30
  netcard_d40
  netcard_d50
)

for name in "${GRAPHS[@]}"; do
  test -s "$GRAPH_DIR/$name.txt" || {
    echo "missing graph: $GRAPH_DIR/$name.txt" >&2
    exit 1
  }
done
```

Do not copy these graphs into the repository or add them to the commit.

## 4. Regenerate GPG goldens

Generate one million production-GPG costs for every graph. Run this without any
`GPUCPG_*` environment gates. Do not reuse the A4000 golden files.

```bash
for name in "${GRAPHS[@]}"; do
  "$BUILD/examples/tc-pfxt-gate5" \
    --benchmark "$GRAPH_DIR/$name.txt" \
    --k 1000000 \
    --mode baseline \
    --out "$OUT/golden/$name.k1000000.gpg.txt" \
    2>&1 | tee "$OUT/golden/$name.generate.log"
done
```

Each command must exit successfully and write exactly 1,000,000 ranked costs.

## 5. Validate the persistent candidates

The common gates select the fused, authoritative, device-resident persistent
pipeline. The scalar arm is the current A4000 winner. The B1 arm runs the same
pipeline but consumes MMA discovery results, so it is the tensor-core comparison.

```bash
COMMON=(
  GPUCPG_TC_PFXT_SINGLE_PASS=1
  GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1
  GPUCPG_TC_PFXT_BVSS_ALL_SKIP_SUPPRESS=1
  GPUCPG_TC_PFXT_BVSS_TILE_NATIVE_LPQ=1
  GPUCPG_TC_PFXT_ALIGNED_BVSS_AUTHORITATIVE=1
  GPUCPG_TC_PFXT_ALIGNED_BVSS_DIRECT_FUSED=1
  GPUCPG_TC_PFXT_ALIGNED_BVSS_PERSISTENT_LOOP=1
  GPUCPG_TC_PFXT_DISABLE_PHASE_PROFILE=1
)
SCALAR=(GPUCPG_TC_PFXT_ALIGNED_PERSISTENT_SCALAR=1)

for name in "${GRAPHS[@]}"; do
  env "${COMMON[@]}" "${SCALAR[@]}" \
    "$BUILD/examples/tc-pfxt-inprocess-exactness" \
    --benchmark "$GRAPH_DIR/$name.txt" \
    --baseline-file "$OUT/golden/$name.k1000000.gpg.txt" \
    --ks 10000,100000,1000000 \
    2>&1 | tee "$OUT/exactness/$name.scalar.log"

  env "${COMMON[@]}" \
    "$BUILD/examples/tc-pfxt-inprocess-exactness" \
    --benchmark "$GRAPH_DIR/$name.txt" \
    --baseline-file "$OUT/golden/$name.k1000000.gpg.txt" \
    --ks 10000,100000,1000000 \
    2>&1 | tee "$OUT/exactness/$name.b1.log"
done
```

Every K must report `pass=1`, and each log must end with
`INPROCESS EXACTNESS PASS`. The configuration diagnostics must show
`aligned_persistent_loop=1`. Scalar logs must show
`aligned_persistent_scalar=1` and `discovery=scalar`; B1 logs must show
`aligned_persistent_scalar=0`, `discovery=b1_mma`, and `bvss_mma_executed=1`.

Stop the performance comparison for any arm that fails exactness.

## 6. Benchmark GPG, persistent scalar, and persistent B1

Use separate processes so an arm cannot inherit CUDA or static-cache state from
another arm. Run one warmup and five measured trials at K=1,000,000.

```bash
TIMING="$BUILD/examples/tc-pfxt-inprocess-timing"

for name in "${GRAPHS[@]}"; do
  "$TIMING" \
    --benchmark "$GRAPH_DIR/$name.txt" \
    --k 1000000 --mode gpg --warmup 1 --trials 5 \
    2>&1 | tee "$OUT/timing/$name.gpg.log"

  env "${COMMON[@]}" "${SCALAR[@]}" "$TIMING" \
    --benchmark "$GRAPH_DIR/$name.txt" \
    --k 1000000 --mode tc --warmup 1 --trials 5 \
    2>&1 | tee "$OUT/timing/$name.scalar.log"

  env "${COMMON[@]}" "$TIMING" \
    --benchmark "$GRAPH_DIR/$name.txt" \
    --k 1000000 --mode tc --warmup 1 --trials 5 \
    2>&1 | tee "$OUT/timing/$name.b1.log"
done
```

Use `timing_summary` values, not wall-clock time. Report mean, minimum, and maximum
for every arm. Compute speedup as:

```text
GPG-over-candidate speedup = gpg_mean_pfxt_ms / candidate_mean_pfxt_ms
```

Produce a summary table with these columns:

```text
graph,k,gpg_mean_ms,gpg_min_ms,gpg_max_ms,scalar_mean_ms,scalar_min_ms,
scalar_max_ms,b1_mean_ms,b1_min_ms,b1_max_ms,gpg_over_scalar,
gpg_over_b1,scalar_exact,b1_exact,status
```

The production recommendation must be based on exactness and measured speedup,
not nominal tensor-core coverage. Report scalar and B1 independently. B1 is only
a win if its end-to-end performance beats both GPG and the scalar arm.

## 7. Capacity failures

K=1,000,000 is the headline workload. If a dense graph runs out of memory or an
arena reports exhausted retries, record that graph/arm as a capacity failure and
continue with the next graph. Do not silently lower K in the headline table and
do not engineer a memory fallback as part of this benchmark.

An optional K=100,000 run may diagnose the failed graph, but label it clearly as
diagnostic and keep it separate from the K=1,000,000 comparison.

## 8. Profile representative cases after timing

Profile at least `netcard_d10` and `netcard_d50`, or the densest graph that passes
if d50 has a capacity failure. Use one trial and no warmup. Profile the scalar
winner and B1 separately.

Example Nsight Systems command for scalar:

```bash
env "${COMMON[@]}" "${SCALAR[@]}" \
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  -o "$OUT/profiles/netcard_d50.scalar" \
  "$TIMING" --benchmark "$GRAPH_DIR/netcard_d50.txt" \
  --k 1000000 --mode tc --warmup 0 --trials 1

nsys stats --report cuda_gpu_kern_sum,nvtx_sum \
  "$OUT/profiles/netcard_d50.scalar.nsys-rep" \
  > "$OUT/profiles/netcard_d50.scalar.stats.txt"
```

Repeat without `"${SCALAR[@]}"` for B1. From the per-launch CUDA trace, identify
the longest `tc_pfxt_persistent_aligned_fused_loop` launch in each arm. Profile
that launch with Nsight Compute, recording the exact kernel filter and launch-skip
used. Capture at least:

- tensor instruction counts and tensor-pipe utilization,
- SM and achieved occupancy,
- registers per thread and waves per SM,
- ALU, FMA, and LSU instruction counts,
- long-scoreboard stall rate,
- DRAM and L2 throughput.

The scalar arm is expected to have zero tensor instructions. The B1 arm must show
real tensor instructions; however, tensor utilization alone is not a promotion
criterion. Compare end-to-end time and the persistent kernel's time against the
scalar arm.

## 9. Return the results

Return the following to the implementation agent:

1. Commit SHA, GPU/driver/toolkit versions, and build command.
2. Focused CTest output.
3. Exactness logs for scalar and B1 at all densities and K values.
4. The K=1,000,000 timing summary table and raw timing logs.
5. NSYS reports/stats and NCU reports for representative sparse and dense cases.
6. Every OOM, retry overflow, exactness failure, or anomalous outlier verbatim.

Keep local graphs and regenerated golden files on the RTX 5090 machine; they do
not belong in the Git branch.
