"""Independent raw-log completion audit; does not trust the summary CSV alone."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

folder=Path(sys.argv[1])
cases=json.loads((folder/'cases.json').read_text())
assert len(cases)==43 and len(set(cases))==43
rows=list(csv.DictReader((folder/'results.csv').open()))
plan=json.loads((folder/'run-plan.json').read_text())
trials=([f'r{i+1}' for i in range(plan['repetitions'])] if plan['timing_only']
        else ['validation']+[f'r{i+1}' for i in range(plan['repetitions'])])
indexed={(r['case'],int(r['bound']),int(r['mode']),r['trial']):r for r in rows}
assert len(rows)==len(indexed)==43*8*len(trials)
checked=0
for case in cases:
    for bound in (0,1):
        signature=None
        for mode in range(4):
            for trial in trials:
                name=f'{case}_b{bound}_m{mode}_{trial}'
                text=(folder/(name+'.log')).read_text()
                assert 'INPROCESS EXACTNESS PASS' in text,name
                assert not re.search(r'WORKLOAD MISMATCH|capacity_retry |output overflow|conservation mismatch|count mismatch',text),name
                def fields(prefix):
                    m=re.search(r'^'+prefix+r' (.*)$',text,re.M)
                    assert m,(name,prefix)
                    return dict(re.findall(r'(\w+)=([^\s]+)',m[1]))
                exact=fields('exactness_summary')
                assert exact['pass']=='1' and exact['K']=='1000000'
                assert exact['result_count']==exact['baseline_count']==exact['compared']=='1000000'
                assert exact['first_mismatch_rank']=='0'
                row=indexed[case,bound,mode,trial]
                assert float(row['pfxt_ms'])==float(exact['pfxt_ms'])
                assert int(row['generated_paths'])==int(exact['generated_paths'])
                setup=float(fields('runtime_summary_adaptive_breakdown')['oracle_setup_ms'])
                assert float(row['setup_ms'])==setup
                assert abs(float(row['cold_ms'])-setup-float(exact['pfxt_ms']))<1e-8
                windows=re.findall(r'^descriptor_ablation_window .*$',text,re.M)
                decisions=re.findall(r'^adaptive_mode_step .*$',text,re.M)
                assert windows and decisions,name
                assert fields('adaptive_mode_summary')['safe_fast_decisions']=='0',name
                c=fields('descriptor_coverage')
                for key,value in c.items():assert int(row[key])==int(value),(name,key)
                assert int(c['total'])==sum(int(c[k]) for k in ('strip_paths','tile_paths','individual'))
                if not mode&1:assert int(c['tiles'])==int(c['tile_paths'])==0
                if not mode&2:assert int(c['strips'])==int(c['strip_paths'])==0
                if bound:
                    b=fields('adaptive_bound_summary')
                    assert int(b['updates'])>0 and b['disabled_reason']=='none'
                else:
                    assert 'adaptive_bound_summary' not in text,name
                current=(windows,decisions,exact['generated_paths'],c['total'])
                if signature is None:signature=current
                assert current==signature,name
                checked+=1
for line in (folder/'checksums.txt').read_text().splitlines():
    digest,filename=line.split(maxsplit=1)
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
print(f'AUDIT PASS: {checked} raw logs, all 43 cases x 8 variants x {len(trials)} runs; workload gates and checksums agree.')
