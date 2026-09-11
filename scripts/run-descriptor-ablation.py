"""Controlled 2x2 packing ablation, separately with bound off/on."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--cases', nargs='+')
p.add_argument('--repetitions', type=int, default=3)
p.add_argument('--timing-only', action='store_true', help='Check correctness during timing; omit a separate validation pass')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
binary = root/'build-strip/examples/tc-pfxt-inprocess-exactness'
cases = a.cases or (a.reference/'cases.txt').read_text().splitlines()
a.out.mkdir(parents=True, exist_ok=False)
(a.out/'cases.json').write_text(json.dumps(cases))
(a.out/'run-plan.json').write_text(json.dumps({'timing_only':a.timing_only,'repetitions':a.repetitions})+'\n')
env = {k:v for k,v in os.environ.items() if not k.startswith('GPUCPG_')}
env.update(GPUCPG_STRIP24='1', GPUCPG_DESCRIPTOR_COVERAGE='1')
# The production safe-ordinary probe examines only a capped-grid prefix.
# Always compute full stats here so packing-dependent queue order cannot
# change whether this preparation is skipped (without changing production).
env['GPUCPG_ADAPTIVE_PFXT_ADAPTIVE_SAFE_MIN_PATHS'] = '2147483647'
(a.out/'controls.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith('GPUCPG_')},indent=2)+'\n')
check_paths=[binary,root/'gpucpg/gpucpg.cu',root/'gpucpg/strip24.cuh',root/'gpucpg/descriptor_coverage.hpp',Path(__file__).resolve()]
(a.out/'checksums.txt').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f)+'\n' for f in check_paths))
signatures = {}
rows = []

def fields(text, name):
    m = re.search(r'^'+name+r' (.*)$', text, re.M)
    assert m, name
    return dict(re.findall(r'(\w+)=([^\s]+)', m[1]))

def run(case, bound, mode, trial):
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                   '--format=csv,noheader'], text=True).strip():
        print('WAIT GPU_BUSY', flush=True); time.sleep(10)
    name = f'{case}_b{bound}_m{mode}_{trial}'
    print('START '+name, flush=True)
    log = a.out/(name+'.log')
    with log.open('x') as f:
        subprocess.run([str(binary),'--benchmark',str(a.data/f'{case}.csrbin'),
                        '--ks','1000000','--baseline-file',
                        str(a.reference/'goldens'/f'{case}_k1000000.gpg.costs'),
                        '--mode','adaptive'], stdout=f, stderr=subprocess.STDOUT,
                       env=env | {'GPUCPG_DESCRIPTOR_ABLATION':str(mode+1),
                                  'GPUCPG_ADAPTIVE_PFXT_BOUND':str(bound)}, check=True)
    text = log.read_text()
    assert 'INPROCESS EXACTNESS PASS' in text, name
    assert not re.search('capacity_retry |output overflow|count mismatch|conservation mismatch', text), name
    exact = fields(text,'exactness_summary')
    assert exact['pass'] == '1' and exact['result_count'] == '1000000', name
    windows = re.findall(r'^descriptor_ablation_window .*$', text, re.M)
    assert windows, name
    # Equality is required within each bound setting, not between bound off/on.
    signature = (windows, exact['generated_paths'],
                 re.findall(r'^adaptive_mode_step .*$', text, re.M))
    key = case, bound
    if key not in signatures:
        signatures[key] = signature
    assert signature == signatures[key], 'WORKLOAD MISMATCH '+name
    c = {k:int(v) for k,v in fields(text,'descriptor_coverage').items()}
    assert c['total'] == c['strip_paths'] + c['tile_paths'] + c['individual']
    if not mode & 1: assert c['tiles'] == c['tile_paths'] == 0
    if not mode & 2: assert c['strips'] == c['strip_paths'] == 0
    if bound:
        b = fields(text,'adaptive_bound_summary')
        assert b['disabled_reason'] == 'none' and int(b['updates']) > 0
    setup = float(fields(text,'runtime_summary_adaptive_breakdown')['oracle_setup_ms'])
    row = dict(case=case,bound=bound,mode=mode,trial=trial,
               setup_ms=setup,pfxt_ms=float(exact['pfxt_ms']),
               cold_ms=setup+float(exact['pfxt_ms']),
               generated_paths=int(exact['generated_paths']), **c)
    rows.append(row)
    with (a.out/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)
    print('PASS '+name, flush=True)

if not a.timing_only:
    for case in cases:
        for bound in (0,1):
            for mode in range(4): run(case,bound,mode,'validation')
    print('ALL VALIDATION AND WORKLOAD GATES PASSED',flush=True)
else:
    print('TIMING DIRECTLY; EVERY QUERY CHECKS CORRECTNESS AND WORKLOAD EQUALITY',flush=True)
for trial in range(a.repetitions):
    for case in cases:
        for bound in (0,1):
            for mode in (range(4) if trial%2 == 0 else reversed(range(4))):
                run(case,bound,mode,f'r{trial+1}')
print('ABLATION COMPLETE',flush=True)
