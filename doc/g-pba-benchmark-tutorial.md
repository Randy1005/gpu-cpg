# G-PBA Level-by-Level PFXT Benchmark Tutorial

G-PBA is the repository's baseline breadth-wise PFXT construction. It grows
the path tree one deviation level at a time using
`PfxtExpMethod::BASIC`: count the children of every path in the current level,
scan those counts to assign output positions, and materialize the entire next
level.

The standalone benchmark and resilient suite driver are:

- `examples/pfxt-level-baseline.cu`
- `scripts/run_pfxt_level_baseline.sh`

## Build

Configure and build with the local CUDA 13.3 toolchain and SM120 target:

```bash
cmake -S . -B build-cuda13.3
cmake --build build-cuda13.3 --target \
  pfxt-level-baseline tc-pfxt-inprocess-exactness \
  convert-timing-edges dump-csr-bin -j2
```

The resulting executable is:

```text
build-cuda13.3/examples/pfxt-level-baseline
```

## Supported graph inputs

The G-PBA executable accepts the repository's weighted `.txt` format or its
binary `.csrbin` equivalent. It does **not** read a raw timing `.edges` file
directly.

Convert a raw timing graph to the weighted text representation first:

```bash
build-cuda13.3/examples/convert-timing-edges \
  benchmarks/leon2.edges \
  benchmarks/leon2.txt
```

For each `(source,destination)` pair, this converter takes the minimum valid
value among the eight timing fields, ignores `n/a` fields, and collapses
parallel edges by retaining their minimum weight.

Convert the text graph to a faster-loading CSR binary with:

```bash
build-cuda13.3/examples/dump-csr-bin \
  benchmarks/leon2.txt \
  benchmarks/leon2.csrbin
```

Use `.csrbin` for repeated benchmarking. Conversion changes only the storage
format, not the graph.

## Run one graph

```bash
build-cuda13.3/examples/pfxt-level-baseline \
  --benchmark benchmarks/leon2.csrbin \
  --k 1000000 \
  --max-deviation-levels 12 \
  --golden path/to/leon2_k1000000.gpg.costs
```

Arguments:

- `--benchmark`: weighted `.txt` or `.csrbin` input.
- `--k`: requested number of paths.
- `--max-deviation-levels`: maximum number of PFXT levels G-PBA may grow.
- `--golden`: optional reference path-cost file used for correctness checking.

The final machine-readable line resembles:

```text
level_baseline_summary status=ok count=1000000 expand_ms=... wall_ms=... validation=pass max_difference=0 first_mismatch_rank=0
```

`expand_ms` is the primary G-PBA PFXT measurement. It is the internal
level-by-level expansion interval. `wall_ms` covers the complete
`report_paths()` call and therefore also includes the common stages preceding
PFXT plus final result handling.

Only report performance when `validation=pass`.

## Generate a same-K correctness reference

Golden costs must be generated for the same graph and preferably the same
`K`. Some bounded/approximate path-generation policies can produce a result
whose smaller-`K` run is not identical to the prefix of a larger-`K` run.

Use the current GPG implementation to generate a same-`K` reference:

```bash
build-cuda13.3/examples/tc-pfxt-inprocess-exactness \
  --benchmark benchmarks/leon2.csrbin \
  --current-gpg-baseline \
  --baseline-output leon2_k1000000.gpg.costs \
  --ks 1000000 \
  --mode gpg
```

Then pass that file to G-PBA with `--golden`.

## Deviation-level correctness gate

G-PBA is explicitly depth-bounded. A run can return `K` paths yet still miss
valid top-K paths if `--max-deviation-levels` is too small. This is reported as
`validation_failed`; it must not be treated as benchmark data.

Increase the cap and rerun until the result matches the same-K golden. For
example, the RTX 5090 smoke test found that `des_perf` at `K=10000` failed with
a cap of 10 but matched exactly with a cap of 12. A higher cap can increase
memory use because G-PBA materializes another complete tree level.

## Run a suite safely

Create a headerless CSV manifest with exactly three columns:

```csv
netcard_d10,/absolute/path/netcard_d10.csrbin,/absolute/path/netcard_d10_k1000000.gpg.costs
leon2_d30,/absolute/path/leon2_d30.csrbin,/absolute/path/leon2_d30_k1000000.gpg.costs
des_perf,/absolute/path/des_perf.csrbin,/absolute/path/des_perf_k1000000.gpg.costs
```

The golden-cost field may be empty, although validated runs are strongly
preferred. Run the manifest with:

```bash
GPUCPG_BUILD_DIR=$PWD/build-cuda13.3 \
GPUCPG_LEVEL_BASELINE_K=1000000 \
GPUCPG_LEVEL_BASELINE_MAX_DEV=12 \
GPUCPG_LEVEL_BASELINE_TIMEOUT=1800 \
scripts/run_pfxt_level_baseline.sh manifest.csv experiments/g_pba_run
```

Outputs are:

- `experiments/g_pba_run/results.csv`: one summary row per case.
- `experiments/g_pba_run/logs/<case>.log`: complete output for each case.

Each graph runs in a fresh process. This matters because a CUDA OOM or fatal
kernel failure can leave the current CUDA context unusable. The suite driver
records the failure and continues with a clean process for the next graph.

Possible `status` values include `ok`, `validation_failed`, `host_oom`,
`device_oom`, `timeout`, `signal_N`, and `error`. The default timeout is 1,800
seconds; set `GPUCPG_LEVEL_BASELINE_TIMEOUT=0` to disable it.

## Smoke-test result

The initial `K=10000` original-circuit check produced exact same-K matches for
`netcard`, `leon2`, `leon3mp`, and `vga_lcd` with a deviation cap of 10.
`des_perf` required a cap of 12. No OOM, CUDA exception, or crash occurred in
these smoke tests.
