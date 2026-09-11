"""Count initial LONG storage, validate against GPG, and check instrumentation cost."""
import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
binary = root/'build-strip/examples/tc-pfxt-inprocess-exactness'
cases = (a.reference/'cases.txt').read_text().splitlines()
a.out.mkdir(parents=True, exist_ok=False)
env = {k: v for k, v in os.environ.items() if not k.startswith('GPUCPG_')}
env.update(GPUCPG_STRIP24='1', GPUCPG_ADAPTIVE_PFXT_BOUND='1')

def fields(text, prefix):
    match = re.search(r'^'+prefix+r' (.*)$', text, re.M)
    assert match, prefix
    return dict(re.findall(r'(\w+)=([^\s]+)', match[1]))

def run(case, enabled, suffix):
    while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                   '--format=csv,noheader'], text=True).strip():
        print('WAIT GPU BUSY', flush=True)
        time.sleep(10)
    print('START', case, suffix, flush=True)
    with (a.out/f'{case}_{suffix}.log').open('x') as log:
        subprocess.run([str(binary), '--benchmark', str(a.data/f'{case}.csrbin'),
                        '--ks', '1000000', '--baseline-file',
                        str(a.reference/'goldens'/f'{case}_k1000000.gpg.costs'),
                        '--mode', 'adaptive'], stdout=log, stderr=subprocess.STDOUT,
                       env=env | {'GPUCPG_DESCRIPTOR_COVERAGE': str(enabled)}, check=True)
    text = (a.out/f'{case}_{suffix}.log').read_text()
    assert 'INPROCESS EXACTNESS PASS' in text
    assert not re.search(r'capacity_retry |output overflow|count mismatch|conservation mismatch', text)
    exact = fields(text, 'exactness_summary')
    assert exact['pass'] == '1' and exact['result_count'] == '1000000'
    bound = fields(text, 'adaptive_bound_summary')
    assert int(bound['updates']) > 0 and bound['disabled_reason'] == 'none'
    old = (a.reference/f'{case}_bound_r1.log').read_text()
    assert exact['generated_paths'] == fields(old, 'exactness_summary')['generated_paths']
    strip = fields(text, 'strip24_summary')
    assert strip == fields(old, 'strip24_summary')
    setup = float(fields(text, 'runtime_summary_adaptive_breakdown')['oracle_setup_ms'])
    row = dict(case=case, enabled=enabled, trial=suffix,
               setup_ms=setup, pfxt_ms=float(exact['pfxt_ms']),
               cold_ms=setup+float(exact['pfxt_ms']))
    if enabled:
        c = {k: int(v) for k,v in fields(text, 'descriptor_coverage').items()}
        assert c['total'] == c['strip_paths']+c['tile_paths']+c['individual']
        assert c['strips'] == int(strip['created']) and c['strip_paths'] == int(strip['represented'])
        row.update(c)
        for key in ('strip_paths', 'tile_paths', 'individual'):
            row[key+'_pct'] = 100*c[key]/c['total'] if c['total'] else 0
        row['paths_per_strip'] = c['strip_paths']/c['strips'] if c['strips'] else 0
        row['paths_per_tile'] = c['tile_paths']/c['tiles'] if c['tiles'] else 0
        row['record_bytes_per_path'] = ((24*c['strips']+204*c['tiles']+24*c['individual'])
                                      /c['total'] if c['total'] else 0)
    print('PASS', case, suffix, flush=True)
    return row

rows = []
for case in cases:
    rows.append(run(case, 1, 'coverage'))
    with (a.out/'coverage.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)

timings = []
for trial in range(3):
    for case in ('netcard_d50', 'leon3mp_d10', 'des_perf_base'):
        for enabled in ((0,1) if trial%2 == 0 else (1,0)):
            row = run(case, enabled, f'overhead_{trial}_{enabled}')
            timings.append({k: row[k] for k in ('case','enabled','trial','setup_ms','pfxt_ms','cold_ms')})
with (a.out/'overhead.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(timings[0]), lineterminator='\n')
    w.writeheader(); w.writerows(timings)
print('COVERAGE SUITE COMPLETE', flush=True)
