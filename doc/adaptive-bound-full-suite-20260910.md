# Adaptive-bound full-suite benchmark — 2026-09-10

## Method

RTX 5090, K=1M. Each case ran three standalone repetitions for each variant in
alternating order. Every one of the 156 executions validated its K-cost output
against the corrected current-GPG reference: 39 validation queries followed by
117 timed queries. The table uses median isolated PFXT time; it is not a
cold-query speedup table because static setup/SFXT can dominate some cases.

- **GPG**: current CUDA GPG path.
- **Adaptive-defer**: adaptive source-local deferred candidates, with
  `GPUCPG_ADAPTIVE_PFXT_BOUND=0`.
- **Tight-bound**: the production adaptive path bound enabled.

`Adaptive/GPG` is the original deferred-path improvement. `Bound/Adaptive`
isolates the added effect of the maintained K-witness cutoff. `Bound/GPG` is
the cumulative progression.

| Case | GPG ms | Adaptive-defer ms (Adaptive/GPG) | Tight-bound ms (Bound/Adaptive) | Bound/GPG |
|---|---:|---:|---:|---:|
| des_perf x16 | 246.000 | 49.823 (4.94x) | 9.164 (5.44x) | 26.84x |
| leon3mp x16 | 54.259 | 22.190 (2.45x) | 11.769 (1.89x) | 4.61x |
| netcard x16 | 35.988 | 15.267 (2.36x) | 10.959 (1.39x) | 3.28x |
| leon2 d30 | 4769.120 | 876.094 (5.44x) | 120.055 (7.30x) | 39.72x |
| netcard d10 | 40.447 | 30.969 (1.31x) | 29.587 (1.05x) | 1.37x |
| netcard d50 | 377.330 | 101.730 (3.71x) | 94.153 (1.08x) | 4.01x |
| leon3mp d50 | 153.431 | 71.881 (2.13x) | 73.958 (0.97x) | 2.07x |
| des_perf d40 | 94.976 | 70.542 (1.35x) | 70.781 (1.00x) | 1.34x |
| leon2 | 12.134 | 9.580 (1.27x) | 9.529 (1.01x) | 1.27x |
| des_perf | 24.933 | 26.175 (0.95x) | 27.004 (0.97x) | 0.92x |
| cage15 | 22.789 | 19.110 (1.19x) | 19.963 (0.96x) | 1.14x |
| M6 | 67.413 | 18.508 (3.64x) | 4.409 (4.20x) | 15.29x |
| nlpkkt120 | 6.023 | 5.824 (1.03x) | 7.365 (0.79x) | 0.82x |

| Aggregate | Adaptive/GPG | Bound/Adaptive | Bound/GPG |
|---|---:|---:|---:|
| Geometric mean, 13 cases | 2.06x | 1.57x | 3.23x |
| Wins, 13 cases | 12 | 8 | 11 |
| Geometric mean excluding x16 synthetic scales | 1.83x | — | 2.52x |

## Interpretation

Tight-bound produces substantial additional PFXT wins where adaptive-defer
otherwise keeps generating large numbers of above-cutoff candidates: leon2 d30,
des_perf x16, and M6. It is neutral to mildly harmful when the candidate set
already reaches K with little excess work or when the bound arrives too late;
original des_perf and nlpkkt120 remain important counterexamples. This is why
the table reports the transition explicitly rather than claiming a universal
bound benefit.

The runner and summarizer are
`scripts/run-adaptive-bound-full-suite.sh` and
`scripts/summarize-adaptive-bound-full-suite.py`. Raw logs and generated CSV
remain local under `experiments/adaptive-bound-full-suite-20260910/` and are
not committed.
