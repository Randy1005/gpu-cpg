"""Render review-draft tables from a complete four-way summary CSV."""
import csv
import math
from pathlib import Path
import re
import statistics
import sys

rows = list(csv.DictReader(Path(sys.argv[1]).open()))
assert len(rows) == 43 and len({r['case'] for r in rows}) == 43
for r in rows:
    for k in r:
        if k != 'case':
            r[k] = float(r[k]) if r[k] else None

def label(case):
    return re.sub(r'_(d\d+|x\d+)$', r' \1', case.replace('_base',''))

groups = [
    ('Original circuits', lambda c:c.endswith('_base')),
    ('Densified circuits', lambda c:re.search(r'_d\d+$',c)),
    ('Scaled circuits', lambda c:re.search(r'_base_x\d+$',c)),
    ('Non-circuit graphs', lambda c:c in ('cage15','M6','nlpkkt120')),
]
assert [sum(bool(test(r['case'])) for r in rows) for _,test in groups] == [5,25,10,3]

print('## 5. Four-way performance\n')
print('All times are **cold static setup + PFXT**, in milliseconds; RTX 5090, '
      'K=1,000,000, candidate arena disabled. Each entry is the median of three '
      'standalone runs. Parentheses show speedup over GPG in the same row. '
      'Graph-file loading and SFXT are excluded from every column.\n')
print('The two intermediate adaptive columns have tight-bound explicitly '
      '**disabled**. Only the final column enables both Strip24 and tight-bound. '
      'Each variant uses the same current graph and GPG golden.\n')
for title,test in groups:
    print(f'### {title}\n')
    print('| Benchmark | GPG | Adaptive 204 | Adaptive 204 + Strip24 | Adaptive 204 + Strip24 + bound |')
    print('|---|---:|---:|---:|---:|')
    for r in rows:
        if not test(r['case']):
            continue
        values = [f"{r['gpg_cold_ms']:.3f}"]
        for v in ('adaptive','strip24','bound'):
            values.append(f"{r[v+'_cold_ms']:.3f} ({r[v+'_speedup_vs_gpg']:.2f}x)")
        print(f"| {label(r['case'])} | "+' | '.join(values)+' |')
    print()
print('Geometric-mean speedup over GPG, across all 43 cases: '+', '.join(
    f"{v}: {math.exp(statistics.mean(math.log(r[v+'_speedup_vs_gpg']) for r in rows)):.3f}x"
    for v in ('adaptive','strip24','bound'))+'.\n')

print('## 6. How close do we get to K?\n')
print('Every correct variant returns exactly K results. The counts below are '
      'instead the final materialized SHORT-pile candidates **before** final '
      'top-K extraction. They are not all symbolic LONG products ever represented. '
      'The CSV also records min/max counts across repetitions.\n')
print('To isolate tightening, compare the Strip24 column with the Strip24 + bound '
      'column: the representation is held fixed. The 204-only count provides '
      'additional context.\n')
print('```text\nexcess_before = count_with_Strip24 - K\n'
      'excess_after  = count_with_Strip24_and_bound - K\n'
      'excess reduction = 100% * (1 - excess_after / excess_before)\n```\n')
print('For example, 5,000,000 → 1,400,000 candidates at K=1,000,000 is '
      '4,000,000 → 400,000 excess candidates: **90% fewer excess candidates**. '
      'This measures how much closer the stored candidate count is to K, not '
      'distance between cost thresholds. K is a reference point, not a proven '
      'attainable minimum for internal search work.\n')
print('| Benchmark | Adaptive 204 count | +Strip24 count | +Strip24 + bound count | Excess reduction |')
print('|---|---:|---:|---:|---:|')
for r in rows:
    values = [f"{int(r[v+'_candidates']):,}" for v in ('adaptive','strip24','bound')]
    reduction = r['excess_reduction_pct']
    values.append('N/A' if reduction is None else f'{reduction:.2f}%')
    print(f"| {label(r['case'])} | "+' | '.join(values)+' |')
print('\nN/A means the unbounded Strip24 run already had no excess candidates. '
      'Negative values mean the bounded run had more final stored candidates; '
      'they must not be hidden or interpreted as a pruning benefit.\n')
print('## Measurement audit\n')
print('The matrix requires 172 initial correctness checks and 516 timed checks '
      '(43 cases × four variants × three repetitions). The summarizer rejects '
      'missing/failed runs, count/overflow/retry errors, missing setup timers, '
      'or missing candidate-count telemetry. Cost comparison uses the existing '
      '1e-3 absolute + 1e-6 relative tolerance, not bitwise equality.\n')
print('Descriptor creation, allocation/growth, replay, materialization and '
      'bookkeeping are charged. Tight-bound cost gathering/sorting, safety checks '
      'and scalar synchronization are also charged. One-time static setup is '
      'added per trial before computing medians.\n')
print('[Full numerical data](strip24-fourway-20260910.csv). '
      'The CSV retains separate setup/PFXT/full-query diagnostics, descriptor '
      'usage and candidate-count ranges. The headline tables above do not omit '
      'cold setup.\n')
