# Sparse Tensor Core descriptor-pipeline gates (2026-09-03)

## Verdict

The measured `slack + deviation -> classify -> descriptor/candidate` formulation
is a **no-go for production SpTC offload** on the RTX 5090. This is a workload
mismatch, not a failure to invoke sparse Tensor Cores.

At the production 32-parent x 16-deviation shape, the raw sparse MMA is 1.27x
faster than the complete fused CUDA kernel for the mixed case. That apparent
win disappears when both paths are required to produce the same output:

- fused CUDA: 0.223937 ms;
- sparse MMA alone, incomplete output: 0.176327 ms;
- sparse MMA plus classification and exact candidate emission: 0.431812 ms,
  1.93x slower than CUDA;
- pack, 2:4 compression, MMA, classification, and emission: 0.585673 ms,
  2.62x slower than CUDA.

The decisive fact is that CUDA calculates each `parent_slack + deviation_delta`,
classifies it, and emits the final descriptor or 24-byte candidate in one
kernel. cuSPARSELt must first materialize an FP16 matrix and a second CUDA
kernel must read it to perform the data-dependent classification and emission.
SpTC therefore accelerates an intermediate result, not the end product.

## What is compared

Each logical work unit is the production descriptor shape:

- 32 parent paths;
- 16 source-local compact deviations;
- 512 candidate products;
- a 204-byte `DeferredDescriptor` for an all-long group;
- a 24-byte `CandidateRecord`, matching `PfxtNode`, for every materialized
  non-skip product.

The encoded sparse operation is exact algebraically:

```text
parent row    = [parent_slack, 1, 0, ...]
deviation row = [1, deviation_delta, 0, ...]
dot product   = parent_slack + deviation_delta
```

This satisfies 2:4 structurally. The gate uses FP16 operands and validates its
chosen values exactly against the CUDA path. That does not prove arbitrary
production values near a split boundary are safe under FP16 rounding; a
production proposal would also need a precision/fallback policy.

Both implementations produce and compare:

1. every product class byte;
2. per-group class mask and short/long/skip counts;
3. exact all-long descriptor contents;
4. exact materialized candidate records for non-homogeneous groups.

Any mismatch makes the executable return failure and invalidates the timing.

## Gate results

### Gate A: exact end-product correctness -- pass

Four controlled distributions cover the important output paths:

- `mixed`: candidate materialization;
- `all-short`: maximal candidate materialization;
- `all-long`: compact descriptor emission;
- `all-skip`: no descriptor and no candidate output.

All 16 pattern/scale cases passed with zero class, group-output, descriptor,
and candidate mismatches. CTest also registers one GPU test per distribution.

### Gate B: equal-output steady-state performance -- fail

The following rows use 16,384 production-shaped groups, or 8,388,608 logical
products. Values are means of 100 unprofiled CUDA-event iterations on an idle
RTX 5090.

| Distribution | CUDA fused (ms) | SpTC MMA only (ms) | SpTC MMA + final output (ms) | Full SpTC (ms) | Final-output SpTC slowdown | Full SpTC slowdown |
|---|---:|---:|---:|---:|---:|---:|
| mixed | 0.223937 | 0.176327 | 0.431812 | 0.585673 | 1.93x | 2.62x |
| all-short | 0.147476 | 0.176421 | 0.382995 | 0.529752 | 2.60x | 3.59x |
| all-long | 0.020538 | 0.176224 | 0.194721 | 0.331907 | 9.48x | 16.16x |
| all-skip | 0.020558 | 0.176289 | 0.194736 | 0.328793 | 9.47x | 15.99x |

`MMA only` is deliberately reported as an optimistic, incomplete lower bound.
It does not classify or emit anything. `MMA + final output` assumes operands
are already packed and compressed. `Full SpTC` includes GPU packing and
cuSPARSELt compression as well.

The mixed case is the best argument against a superficial conclusion. MMA by
itself looks favorable because the CUDA kernel writes 168,944,088 bytes of
candidate records. The SpTC consumer must ultimately perform those same writes,
while additionally reading the MMA result, so its end-product path loses.

### Gate C: hidden overhead and resident-state accounting -- fail

For the 16,384-group mixed case:

| Cost | Value |
|---|---:|
| GPU operand pack | 0.022962 ms per step |
| cuSPARSELt 2:4 compression | 0.108683 ms per step |
| sparse MMA | 0.176327 ms per step |
| SpTC classify and emit | 0.236017 ms per step |
| cuSPARSELt plan/workspace setup | 33.2812 ms once |
| cuSPARSELt algorithm search | 3.85488 ms once |
| combined gate allocation | 111.759 ms once |

The combined allocation number initializes both comparison paths and includes
CUDA context/allocation effects; it is disclosed but must not be attributed to
SpTC alone. Plan/workspace setup and algorithm search are SpTC-specific and can
be cached across steps. Excluding all three one-time costs still leaves the
steady-state end-product path 1.93x slower.

SpTC-only matrix/compression state for this case is 96,468,992 bytes. Its main
components are:

- padded sparse A before compression: 33,554,432 bytes;
- dense B: 16,777,216 bytes;
- intermediate C: 16,777,216 bytes;
- compressed A: 20,971,520 bytes;
- remaining bytes: cuSPARSELt workspaces.

This state is in addition to the parent/deviation arrays and final output that
both paths need.

### Gate D: genuine sparse-Tensor-Core execution -- pass

Targeted Nsight Compute profiling of the selected cuSPARSELt kernel reports:

- 4,194,304 Tensor-pipe instructions;
- 34,359,738,368 FP16-to-FP32 sparse-HMMA operations;
- zero sparse-HMMA operations in the CUDA and consumer kernels;
- 8.33% active-warps occupancy for the sparse MMA kernel;
- 24.76% SM throughput and 35.32% memory throughput.

Therefore the result is not caused by a dense or scalar fallback. The library
kernel name identifies a 128x128x64 sparse GEMM even though each independent
logical group is only 32x16 with two meaningful K entries padded into K=32.
The reported 34.36 billion HMMA operations are 4,096 hardware-counted
operations per useful candidate sum. The sparse engine is doing real work, but
most of its preferred matrix shape is unrelated padding.

The final-output profile further exposes the handoff penalty:

| Kernel | Unprofiled time (ms) | DRAM bytes | Active warps | Memory throughput |
|---|---:|---:|---:|---:|
| fused CUDA final output | 0.223937 | 163,579,136 | 92.23% | 75.77% |
| sparse MMA | 0.176327 | 37,823,488 | 8.33% | 35.32% |
| SpTC final-output consumer | 0.236017 | 179,834,112 | 92.30% | 82.80% |

The SpTC consumer moves about 16.3 MB more DRAM traffic than fused CUDA,
consistent with consuming the 16.8 MB intermediate C matrix. Nsight Compute
replay perturbs kernel duration, so the report uses unprofiled CUDA-event times
for performance and Nsight only for hardware/traffic evidence.

### Gate E: representativeness and dispatch opportunity -- fail

This controlled replay is favorable to SpTC: every logical 32x16 group is
completely filled. Prior real K=10,000 d40 profiles average only 48.926,
36.261, and 47.002 useful products per generated 512-slot group on leon2,
leon3mp, and netcard. That is roughly 7.1--9.6% useful fill. Direct source-local
numeric rows also showed only about 0.6--1.1% natural exact-2:4 eligibility.

An adaptive dispatcher therefore cannot rescue this formulation:

- homogeneous groups are much cheaper to classify with scalar min/max bounds;
- mixed groups require final candidate writes, where the SpTC handoff loses;
- real groups are substantially less filled than this already-losing oracle;
- maintaining a second packed/compressed representation adds state and update
  work even when plan/search costs are amortized.

## Architectural conclusion

The mismatch has three independent causes:

1. **Too little arithmetic per product.** The useful numeric operation is one
   addition followed by comparisons, while sparse MMA requires padded matrix
   machinery.
2. **The useful endpoint is data-dependent.** Classification decides whether
   to emit a descriptor, a candidate, or nothing. cuSPARSELt cannot fuse this
   custom control flow into its library MMA.
3. **The existing CUDA path already fuses through the endpoint.** It avoids an
   intermediate matrix and naturally specializes homogeneous groups.

A future SpTC direction should be reopened only if it changes the algorithmic
unit of work: multiple reusable dense frontiers, substantially more arithmetic
per MMA result, and a way to consume accumulators without materializing a full
matrix. Moving the current add/classify operation to SpTC is not enough.

## Artifacts and reproduction

- Gate: `examples/sptc-descriptor-gate.cu`
- Guarded runner: `scripts/run_sptc_descriptor_gate.sh`
- Full matrix: `experiments/sptc_descriptor_gate_20260903/final/sptc_descriptor_gate_matrix.csv`
- Profile summary: `experiments/sptc_descriptor_gate_20260903/final/profile_summary.csv`
- Raw targeted profiles: `cuda_mixed_16384_ncu.csv`,
  `sptc_emit_mixed_16384_ncu.csv`, and `sptc_mma_mixed_16384_ncu.csv` in the
  same experiment directory.

Configure and run:

```bash
cmake -S . -B build-sptc-descriptor \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCUSPARSELT_ROOT=/home/cchang289/gpu-cpg/.local/cusparselt-0.9.1
cmake --build build-sptc-descriptor --target sptc-descriptor-gate -j 8
ctest --test-dir build-sptc-descriptor \
  -R ^sptc_descriptor_gate_ --output-on-failure
scripts/run_sptc_descriptor_gate.sh build-sptc-descriptor
```

The runner refuses to benchmark while `nvidia-smi` reports another compute
process. It also stops immediately if any output comparison fails.

NVIDIA references: [cuSPARSELt documentation](https://docs.nvidia.com/cuda/cusparselt/)
and [PTX ISA Tensor Core instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html).
