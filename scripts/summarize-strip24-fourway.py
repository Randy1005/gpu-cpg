"""Emit complete four-way cold-inclusive results and pre-extraction counts."""
import csv
import json
from pathlib import Path
import re
import statistics
import sys

folder = Path(sys.argv[1])
cases = (folder/'cases.txt').read_text().splitlines()
variants = json.loads((folder/'variants.json').read_text())
assert len(cases) == 43 and len(set(cases)) == 43
assert list(variants) == ['gpg', 'adaptive', 'strip24', 'bound']

def read(case, variant, trial):
    path = folder/f'{case}_{variant}_{trial}.log'
    text = path.read_text()
    if 'INPROCESS EXACTNESS PASS' not in text:
        raise ValueError(f'Failed or incomplete: {path}')
    if re.search(r'capacity_retry\s|output overflow|count mismatch|counted/promoted mismatch', text, re.I):
        raise ValueError(f'Capacity/count error: {path}')
    match = re.search(r'^exactness_summary (.*)$', text, re.M)
    if not match:
        raise ValueError(f'Missing exactness summary: {path}')
    fields = dict(re.findall(r'(\w+)=([^\s]+)', match[1]))
    assert fields['K'] == '1000000' and fields['pass'] == '1', path
    assert fields['result_count'] == '1000000', path
    assert fields['mode'] == ('gpg' if variant == 'gpg' else 'adaptive'), path
    setup = re.search(r'runtime_summary_adaptive_breakdown oracle_setup_ms=([\d.eE+-]+)', text)
    if variant != 'gpg' and not setup:
        raise ValueError(f'Missing setup timing: {path}')
    setup_ms = float(setup[1]) if setup else 0.0
    pfxt_ms = float(fields['pfxt_ms'])
    count = int(fields['generated_paths'])
    assert count >= 1000000, path
    strip = re.search(r'strip24_summary created=(\d+) represented=(\d+) promoted=(\d+)', text)
    if variant in ('strip24','bound') and not strip:
        raise ValueError(f'Missing strip telemetry: {path}')
    if variant == 'bound':
        bound = re.search(r'adaptive_bound_summary updates=(\d+) .*disabled_reason=(\S+)', text)
        if not bound or int(bound[1]) == 0 or bound[2] != 'none':
            raise ValueError(f'Bound did not run without fallback: {path}')
    return dict(setup_ms=setup_ms, pfxt_ms=pfxt_ms, cold_ms=setup_ms+pfxt_ms,
                query_ms=float(fields['query_ms']), candidates=count,
                strips=int(strip[1]) if strip else 0,
                represented=int(strip[2]) if strip else 0,
                promoted=int(strip[3]) if strip else 0)

rows = []
for case in cases:
    row = {'case':case}
    for variant in variants:
        read(case,variant,'validation')
        runs = [read(case,variant,f'r{t}') for t in (1,2,3)]
        for key in runs[0]:
            row[f'{variant}_{key}'] = statistics.median(r[key] for r in runs)
        row[f'{variant}_cold_min_ms'] = min(r['cold_ms'] for r in runs)
        row[f'{variant}_cold_max_ms'] = max(r['cold_ms'] for r in runs)
        row[f'{variant}_candidates_min'] = min(r['candidates'] for r in runs)
        row[f'{variant}_candidates_max'] = max(r['candidates'] for r in runs)
    for variant in ('adaptive','strip24','bound'):
        row[f'{variant}_speedup_vs_gpg'] = row['gpg_cold_ms']/row[f'{variant}_cold_ms']
    # Hold Strip24 fixed: this percentage isolates the addition of tight-bound.
    before = row['strip24_candidates']-1000000
    after = row['bound_candidates']-1000000
    row['excess_reduction_pct'] = 100*(1-after/before) if before else ''
    row['bound_speedup_vs_strip24'] = row['strip24_cold_ms']/row['bound_cold_ms']
    rows.append(row)

writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator='\n')
writer.writeheader()
writer.writerows(rows)
