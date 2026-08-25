# Deferred-candidate checkpoint (2026-08-24)

This checkpoint makes deferred candidate materialization an adaptive,
device-resident production path.  It is the baseline for subsequent Sparse
Tensor Core feasibility gates; those experiments belong on a separate branch.

## Implementation

- Classify candidate work on the GPU as ordinary or deferred from product
  intensity and sampled skip evidence.
- Keep long and skipped products in compact tile descriptors instead of
  reserving one `PfxtNode` per product.
- Preserve exact counted capacity for materialized short candidates, and use
  overflow-safe shared arena reservations.
- Keep adaptive decisions and telemetry on the device.  A two-window probation
  period and periodic audit avoid paying the oracle on every stable deferred
  substep.
- Preserve deferred backlog ownership across ordinary/deferred transitions.
- Support standalone, one-shot K=1M correctness and timing modes without
  capacity retries.

The focused policy and device-state behavior is covered by
`unittests/tc_pfxt_adaptive.cu`; in-process mode/capacity behavior remains in
`unittests/tc_pfxt_inprocess.cu`.

## Validation

The final RTX 5090 artifact is
`experiments/adaptive_fastlane_full_20260824`.  It contains 29 benchmark cases
and three modes per case: original GPG, fixed GPG deferral, and adaptive tile
deferral.

- 87/87 K=1M exactness arms passed.
- Maximum reported floating-point cost difference: `0.000488281`.
- No short-capacity retry was accepted by the validation driver.
- Adaptive beat original GPG on 25/29 cases.
- Adaptive versus GPG geometric-mean speedup: `1.2941x`.
- Netcard d50: `375.954 ms` GPG versus `125.024 ms` adaptive (`3.0071x`).

The compact authoritative tables are:

- `validation.csv`: exactness and PFXT time for every validation arm.
- `timing.csv`: standalone mean/min/max PFXT timing.
- `comparison.csv`: three-way timing and speedup.
- `adaptive_steps.csv`: ordinary/deferred step counts.
- `adaptive_switches.csv`: transition locations.

Raw logs remain local because they are reproducible and approximately 18 MB;
the checkpoint commits the compact result tables and the benchmark driver.

## Repository cleanup boundary

The checkpoint intentionally excludes intermediate tuning runs, local absolute
benchmark symlinks, build trees, launch logs, and backup/reject files.  Existing
optional profiling and replay hooks are retained: they are referenced, gated
diagnostics and form the evidence infrastructure for the upcoming SpTC gates,
not dead production paths.
