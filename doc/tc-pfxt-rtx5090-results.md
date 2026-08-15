# TC-PFXT RTX 5090 baseline (2026-08-12)

This note records the regenerated NetCard benchmark and the first paired GPG/TC-PFXT
measurements on the RTX 5090 machine. Detailed logs and golden output files are in
`experiments/tc_pfxt_rtx5090_20260812/`.

## Machine and build

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0
- Driver: 590.44.01
- Toolkit: local CUDA 13.3.1 (`nvcc` 13.3.73), native `sm_120` build
- CCCL: v3.3.3
- Build directory: `build-cuda13.3`

CUDA 13.3.1 was the current production toolkit at the time of the run; CUDA 13.4 was
still a developer preview.

## Input regeneration

`benchmarks/netcard.edges` contains records of the form
`insert_edge SRC DST T0 ... T7`. The converter assigns vertex IDs in first-seen order,
preserves parallel edges, and uses the minimum numeric timing among the eight values as
the edge weight while ignoring `n/a` entries.

The converted base graph has 3,999,174 vertices and 7,208,344 edges. Each density was
generated independently from that base using `examples/densify.cu` with seed 1:

| Graph | Edges | SHA-256 |
|---|---:|---|
| base | 7,208,344 | `c19d920cc8a45cb15f05f1be3e7f315e6d22887650a732fe0b276914c15e3214` |
| d10 | 39,991,740 | `fc200f370a26ed7222d5df29a0f141c390378371df1b0a821f8725e6c24915f3` |
| d20 | 79,983,480 | `2f9977f743c6c8cd8a4c7976ff9470319002e6ab8bee827c7bf209e0d5228e73` |
| d30 | 119,975,220 | `29bb85981840f96c0a717304dd5b26b6b06fb6627738a61254a831c68a163eac` |
| d40 | 159,966,960 | `924dc451e03434a5bf87151fc55ce1c62f744903c99fcc4e3c55f10ea3ab0e79` |
| d50 | 199,958,700 | `52a990b431327460014b4807277f6a8c927edf65e8bb14732b3c21bfea8ba023` |

## Correctness and TC-PFXT capacity

A fresh GPG K=1,000,000 golden cost sequence was generated for every density. TC-PFXT
was then checked in-process at K=1K, 10K, 50K, 100K, 200K, and 1M against prefixes of
that density's golden sequence. All checks passed through K=1M at d10, d20, d30, d40,
and d50.

The stock TC candidate capacities were insufficient at d20: the source-local output
limit failed at K=200K, and the short-pile limit failed at K=1M. These were capacity
overflows rather than device OOM. Successful K=1M runs used:

```text
GPUCPG_TC_PFXT_SOURCE_LOCAL_MAX_SLOTS=300000000
GPUCPG_TC_PFXT_MIN_SHORT_CAPACITY=5000000
```

Thus this 32 GB GPU can run TC-PFXT at K=1M for all five regenerated densities with
those capacities. This is deliberately separate from using GPG K=1M as the golden
reference.

## Headline timing

The updated headline schedule uses K=1M at every density. Each row is one paired process
that loads and builds the graph once, establishes static state, then runs one warmup plus three
measured GPG trials followed by one warmup plus three measured TC trials. Each trial
clears dynamic algorithm state. Reported time is PFXT expansion only; parsing, graph
construction, SFXT, and static BVSS setup are excluded.

| Density | K | GPG mean ms | TC-PFXT mean ms | TC/GPG |
|---:|---:|---:|---:|---:|
| 10 | 1,000,000 | 40.245 | 155.036 | 3.8523 |
| 20 | 1,000,000 | 108.025 | 363.912 | 3.3688 |
| 30 | 1,000,000 | 166.575 | 183.977 | 1.1045 |
| 40 | 1,000,000 | 289.896 | 451.319 | 1.5568 |
| 50 | 1,000,000 | 1,873.980 | 431.808 | 0.2304 |

On these regenerated inputs and this build, TC-PFXT is slower through d40. At d50/K=1M
it crosses over decisively: 431.808 ms versus 1,873.980 ms for GPG, making TC-PFXT
4.34x faster (or 23.04% of GPG time). These results should not be compared as a pure
GPU-generation uplift against the historical table because the graph was regenerated
and therefore is not byte-identical to the old benchmark artifact.

Re-run the paired schedule with:

```bash
scripts/run_tc_pfxt_rtx5090_headline.sh
```

## Deferred tile LPQ result (2026-08-15)

The source-local candidate path now has an opt-in deferred LPQ representation:

```bash
GPUCPG_TC_PFXT_DEFERRED_TILE_LPQ=1
```

Uniformly long 32x16 tiles snapshot parent indices once, reuse the compact static
deviation CSR, and use a product bitmap to materialize candidates only when they
cross a later split. The tile-native short-only output path grows and retries only
after a genuine capacity overflow.

Correctness was the first gate. Full K=1M cost vectors passed against fresh GPG
goldens for netcard, leon2, and leon3mp at d10, d20, d30, d40, and d50: 15/15
cases passed. The largest absolute cost difference was `3.8147e-05`.

Standalone measurements used one warmup and three measured trials per arm:

| Graph | Density | GPG mean ms | Deferred TC mean ms | TC speedup |
|---|---:|---:|---:|---:|
| netcard | 10 | 40.0198 | 158.232 | 0.253x |
| netcard | 20 | 108.570 | 168.794 | 0.643x |
| netcard | 30 | 166.303 | 128.028 | 1.299x |
| netcard | 40 | 289.027 | 125.223 | 2.308x |
| netcard | 50 | 1865.850 | 126.204 | 14.784x |
| leon2 | 10 | 65.0636 | 157.283 | 0.414x |
| leon2 | 20 | 2174.670 | 320.196 | 6.792x |
| leon2 | 30 | 4412.450 | 3637.260 | 1.213x |
| leon2 | 40 | 319.877 | 328.909 | 0.973x |
| leon2 | 50 | 195.039 | 164.267 | 1.187x |
| leon3mp | 10 | 79.0864 | 290.194 | 0.273x |
| leon3mp | 20 | 71.4289 | 163.378 | 0.437x |
| leon3mp | 30 | 106.642 | 150.737 | 0.707x |
| leon3mp | 40 | 144.088 | 156.377 | 0.921x |
| leon3mp | 50 | 369.906 | 107.842 | 3.430x |

Compact results are recorded in:

- `experiments/tc_pfxt_deferred_full_20260815/validation.csv`
- `experiments/tc_pfxt_deferred_full_20260815/timing.csv`
- `experiments/gpg_vs_deferred_tc_full_20260815/summary.csv`

Reproduce the correctness-first suite and standalone GPG comparison with:

```bash
scripts/run_tc_pfxt_deferred_full_suite.sh
scripts/run_gpg_vs_deferred_tc_full_suite.sh
```
