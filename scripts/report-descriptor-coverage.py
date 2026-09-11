"""Render coverage tables; input is the correctness-checked diagnostic CSV."""
import csv
from pathlib import Path
import statistics
import sys

folder = Path(sys.argv[1])
rows = list(csv.DictReader((folder/'coverage.csv').open()))
assert len(rows) == 43

def label(case):
    return case.replace('_base_x', ' x').replace('_base', '').replace('_d', ' d')

print('### Measured distribution: which format stores the waiting paths?\n')
print('All 43 cases below use Strip24 + tight-bound, K = 1M. Each cell shows\n'
      '**LONG candidates initially stored (percentage of that row)**. These are\n'
      'counts of paths, not counts of descriptor records. Later promotions are\n'
      'not counted again. Percentages may not sum to exactly 100% after rounding.\n')
print('| Benchmark | In 24-byte strips | In 204-byte descriptors | Individual LONG nodes |')
print('|---|---:|---:|---:|')
for r in rows:
    cells = []
    for k in ('strip_paths', 'tile_paths', 'individual'):
        pct = float(r[k+'_pct'])
        shown = '<0.001' if 0 < pct < 0.001 else f'{pct:.3f}'
        cells.append(f'{int(r[k]):,} ({shown}%)')
    print('| '+label(r['case'])+' | '+' | '.join(cells)+' |')

print('\n### How full are the envelopes?\n')
print('A strip and a multi-parent descriptor are like two sizes of envelope.\n'
      'Coverage above counts how many waiting paths go into each size. Packing\n'
      'below counts how many paths each envelope carries on average when created.\n'
      'An individual node stores just one path; a descriptor stores a recipe for\n'
      'recovering several paths, not a copy of each full node.\n')
print('| Benchmark | 24-byte records created | Paths per 24-byte record | 204-byte records created | Paths per 204-byte record |')
print('|---|---:|---:|---:|---:|')
for r in rows:
    if r['case'] not in ('netcard_base','netcard_d10','netcard_d30','netcard_d50',
                         'leon2_d30','leon3mp_d10','des_perf_base_x16'):
        continue
    strip = f'{float(r["paths_per_strip"]):.2f}' if int(r['strips']) else '—'
    tile = f'{float(r["paths_per_tile"]):.2f}' if int(r['tiles']) else '—'
    print(f'| {label(r["case"])} | {int(r["strips"]):,} | {strip} | {int(r["tiles"]):,} | {tile} |')

print('\n### Measurement checks and overhead\n')
print('The counters reuse totals already computed by the GPU producers and already\n'
      'available to the host for queue sizing. The instrumentation adds host-side\n'
      'integer accumulation and checks, but **no new GPU scan, kernel launch, or\n'
      'device-to-host transfer**. Counters are cumulative and survive queue retirement.\n'
      'For every window, the three categories must add up to the existing LONG\n'
      'output count. Strip totals must also match the independent Strip24 telemetry.\n')
print('All 43 coverage runs passed the existing GPG golden-cost check and matched\n'
      'the previous four-way run’s final candidate counts and Strip24 telemetry.\n'
      'The diagnostic is opt-in (`GPUCPG_DESCRIPTOR_COVERAGE=1`). Its accounting\n'
      'covers the measured source-local adaptive pipeline; an unaccounted producer\n'
      'with LONG outputs fails the conservation check instead of silently omitting work.\n')
timings = list(csv.DictReader((folder/'overhead.csv').open()))
assert len(timings) == 18
print('A separate three-pair instrumentation check used the same rebuilt binary\n'
      'with counters off/on, alternating order. Times below are medians of\n'
      '**cold setup + PFXT**, not replacements for the four-way headline timings.\n')
print('| Benchmark | Counters off (ms) | Counters on (ms) | Change |')
print('|---|---:|---:|---:|')
for case in ('netcard_d50','leon3mp_d10','des_perf_base'):
    off, on = [statistics.median(float(r['cold_ms']) for r in timings
                               if r['case'] == case and int(r['enabled']) == flag)
               for flag in (0,1)]
    print(f'| {label(case)} | {off:.3f} | {on:.3f} | {100*(on/off-1):+.2f}% |')
print('\nThese short paired checks include run-to-run timing variation; they do not\n'
      'prove zero overhead. All 18 paired-check queries also passed correctness.\n'
      'Raw distribution counts and packing metrics for every case are in\n'
      '[the coverage CSV](descriptor-coverage-20260910.csv); the paired timings are in\n'
      '[the instrumentation CSV](descriptor-coverage-overhead-20260910.csv).\n')
print('Reproduce with `scripts/measure-descriptor-coverage.py --data <csrbin-directory>`\n'
      '`--reference <fourway-results-directory> --out <new-output-directory>`, then\n'
      '`scripts/report-descriptor-coverage.py <new-output-directory>` via Python 3.\n'
      'The reference directory supplies `cases.txt`, existing goldens, and the\n'
      'previous `*_bound_r1.log` files. Available inputs are reused, not regenerated.\n')
