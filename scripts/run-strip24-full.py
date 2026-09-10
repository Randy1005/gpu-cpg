"""Standalone 43-case correctness gate, then paired three-trial timings."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, required=True)
p.add_argument('--goldens', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--four-way', action='store_true',
               help='GPG, adaptive 204, adaptive 204+strip, and strip+tight-bound')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
binary = root / 'build-strip/examples/tc-pfxt-inprocess-exactness'
cases = [f'{c}_{s}' for c in ('netcard','leon2','leon3mp','des_perf','vga_lcd')
         for s in ('base','d10','d20','d30','d40','d50','base_x8','base_x16')]
cases += ['cage15','M6','nlpkkt120']
for c in cases:
    if not (a.data / f'{c}.csrbin').is_file():
        raise RuntimeError(f'Missing input {c}')
a.out.mkdir(parents=True, exist_ok=False)
(a.out / 'goldens').mkdir()
(a.out / 'cases.txt').write_text('\n'.join(cases)+'\n')
env = {k:v for k,v in os.environ.items() if not k.startswith('GPUCPG_')}
variants = {
    '0': ('adaptive', {'GPUCPG_ADAPTIVE_PFXT_BOUND':'1','GPUCPG_STRIP24':'0'}),
    '1': ('adaptive', {'GPUCPG_ADAPTIVE_PFXT_BOUND':'1','GPUCPG_STRIP24':'1'}),
}
if a.four_way:
    variants = {
        'gpg': ('gpg', {'GPUCPG_ADAPTIVE_PFXT_BOUND':'0','GPUCPG_STRIP24':'0'}),
        'adaptive': ('adaptive', {'GPUCPG_ADAPTIVE_PFXT_BOUND':'0','GPUCPG_STRIP24':'0'}),
        'strip24': ('adaptive', {'GPUCPG_ADAPTIVE_PFXT_BOUND':'0','GPUCPG_STRIP24':'1'}),
        'bound': ('adaptive', {'GPUCPG_ADAPTIVE_PFXT_BOUND':'1','GPUCPG_STRIP24':'1'}),
    }
(a.out/'variants.json').write_text(json.dumps(variants, indent=2)+'\n')

def idle():
    while True:
        result = subprocess.run(['nvidia-smi','--query-compute-apps=pid',
                                 '--format=csv,noheader'], capture_output=True,
                                text=True, check=True)
        if not result.stdout.strip():
            return
        print('WAIT GPU_BUSY '+result.stdout.strip().replace('\n',','), flush=True)
        time.sleep(10)

def run(name, command, settings):
    idle()
    log = a.out / f'{name}.log'
    print(f'START {name}', flush=True)
    with log.open('x') as f:
        subprocess.run(command, env=env | settings, stdout=f,
                       stderr=subprocess.STDOUT, check=True)
    text = log.read_text()
    if 'INPROCESS EXACTNESS PASS' not in text:
        raise RuntimeError(f'Correctness failure: {log}')
    if any(x in text for x in ('capacity_retry ', 'output overflow',
                              'count mismatch','counted/promoted mismatch')):
        raise RuntimeError(f'Capacity/count gate: {log}')
    print(f'PASS {name}', flush=True)

idle()
with (a.out / 'unit.log').open('x') as f:
    subprocess.run([str(root/'build-strip/examples/strip24-test')],
                   env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
assert 'STRIP24 UNIT PASS' in (a.out/'unit.log').read_text()

# Existing corrected goldens are reused; generate only missing cases with GPG.
for c in cases:
    golden = a.out/'goldens'/f'{c}_k1000000.gpg.costs'
    existing = a.goldens/golden.name
    common = [str(binary),'--benchmark',str(a.data/f'{c}.csrbin'),'--ks','1000000']
    if existing.is_file():
        shutil.copyfile(existing, golden)
    else:
        run(f'{c}_golden', common+['--current-gpg-baseline','--baseline-output',
                                 str(golden),'--mode','gpg'], {})
    for v, (mode, settings) in variants.items():
        run(f'{c}_{v}_validation',common+['--baseline-file',str(golden),'--mode',mode], settings)

print(f'ALL {len(cases)*len(variants)} VALIDATION QUERIES PASSED; BEGIN TIMING',flush=True)
for trial in (1,2,3):
    for c in cases:
        order = list(variants) if trial%2 else list(reversed(variants))
        for v in order:
            mode, settings = variants[v]
            run(f'{c}_{v}_r{trial}',[str(binary),'--benchmark',str(a.data/f'{c}.csrbin'),
                '--ks','1000000','--baseline-file',
                str(a.out/'goldens'/f'{c}_k1000000.gpg.costs'),'--mode',mode], settings)
print('FULL SUITE COMPLETE',flush=True)
