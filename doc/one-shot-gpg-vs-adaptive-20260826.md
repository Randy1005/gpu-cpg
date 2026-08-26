# RTX 5090 one-shot adaptive PFXT versus GPG — 2026-08-26

## Scope and timing definition

This run compares K=1,000,000 path-generation queries on the RTX 5090.

The strict one-shot comparison is:

```
GPG first-query PFXT time
vs.
adaptive cold static-setup time + adaptive first-query PFXT time
```

Graph loading and common pre-PFXT graph construction are deliberately excluded from
both sides. Each method was also run three times in-process; those means are retained
in the CSV as secondary stability data, but the headline uses trial 1.

## Correctness gate

All 29 graphs were checked in both GPG and adaptive modes against the existing GPG
golden-cost files at K=1,000,000:

- 58/58 validation arms passed.
- Every arm returned exactly 1,000,000 costs.
- No candidate-capacity retry occurred.
- GPG matched the goldens exactly. Adaptive differences stayed within the accepted
  FP32 tolerance.

The repository test suite also passed 107/107 tests.

## Overall result

- Adaptive wins 16 of 29 strict one-shot comparisons.
- Adaptive loses 13 of 29.
- Geometric-mean one-shot speedup over all 29 graphs: **1.1645x**.
- Best result: **leon2 d30, 4.2574x**.
- Worst result: **nlpkkt120, 0.2530x**.

The complete per-case matrix is in
`experiments/oneshot_full_20260826/one_shot_comparison.csv`.

Representative wins:

| Case | GPG first query (ms) | Adaptive setup (ms) | Adaptive first query (ms) | Adaptive one-shot (ms) | Speedup |
|---|---:|---:|---:|---:|---:|
| netcard d50 | 377.458 | 86.957 | 103.752 | 190.709 | 1.9792x |
| leon2 d20 | 259.910 | 35.447 | 60.060 | 95.507 | 2.7214x |
| leon2 d30 | 4777.570 | 53.685 | 1068.500 | 1122.185 | 4.2574x |
| leon3mp d40 | 215.524 | 58.943 | 71.914 | 130.858 | 1.6470x |
| vga_lcd d50 | 126.932 | 10.984 | 58.644 | 69.628 | 1.8230x |
| M6 | 67.625 | 4.870 | 18.308 | 23.177 | 2.9177x |

Representative losses:

| Case | GPG first query (ms) | Adaptive setup (ms) | Adaptive first query (ms) | Adaptive one-shot (ms) | Speedup |
|---|---:|---:|---:|---:|---:|
| netcard base | 7.529 | 2.977 | 6.895 | 9.872 | 0.7627x |
| leon2 d50 | 150.676 | 92.682 | 111.530 | 204.212 | 0.7378x |
| cage15 | 22.821 | 18.198 | 18.876 | 37.074 | 0.6156x |
| nlpkkt120 | 6.041 | 18.259 | 5.624 | 23.883 | 0.2530x |

The adaptive query path is often faster by itself, but strict one-shot success requires
that saved query work repay static metadata setup immediately.

## Loading acceleration

A versioned binary CSR format and direct loader avoid the legacy text parser's two
endpoint hash-map copies. A streaming converter uses a zero-scatter path for
source-major files and an order-preserving CSR scatter for unordered inputs.

On `des_perf_d50` (303,690 vertices, 15,184,500 edges):

- Legacy text load: 9,479.68 ms.
- Binary CSR load: 430.172 ms.
- Loading improvement: **22.04x**.

All 29 cached graphs occupy 15 GB. Loading diagnostics may be printed by the timing
driver, but are not included in PFXT summaries or speedups.

## Artifacts

- Full matrix: `experiments/oneshot_full_20260826/one_shot_comparison.csv`
- Validation logs: `experiments/oneshot_full_20260826/validation/`
- Timing logs: `experiments/oneshot_full_20260826/timing/`
- Binary caches: `experiments/binary_graph_cache_20260826/`

The large logs and binary caches are local benchmark artifacts and are not intended
for source control.
