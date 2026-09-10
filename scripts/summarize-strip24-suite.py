"""Summarize only complete, correctness-validated standalone trial matrices."""
import csv
import re
import statistics
import sys
from pathlib import Path

folder=Path(sys.argv[1]);rows=[]
for path in sorted(folder.glob('*_r[123].log')):
    case,variant,trial=path.stem.rsplit('_',2)
    text=path.read_text()
    if 'INPROCESS EXACTNESS PASS' not in text:raise ValueError(f'{path}: failed or incomplete')
    match=re.search(r'exactness_summary .*?pfxt_ms=([\d.eE+-]+) query_ms=([\d.eE+-]+) pass=1',text)
    setup=re.search(r'runtime_summary_adaptive_breakdown oracle_setup_ms=([\d.eE+-]+)',text)
    if not match or not setup:raise ValueError(f'{path}: missing timing boundary')
    pfxt,query=map(float,match.groups());cold=pfxt+float(setup.group(1))
    usage=re.search(r'strip24_summary created=(\d+) represented=(\d+) promoted=(\d+)',text)
    if variant=='1' and not usage:raise ValueError(f'{path}: missing strip path telemetry')
    rows.append(dict(case=case,variant=int(variant),trial=trial,pfxt_ms=pfxt,
        setup_ms=float(setup.group(1)),cold_setup_pfxt_ms=cold,query_ms=query,
        descriptors=int(usage[1]) if usage else 0,represented=int(usage[2]) if usage else 0,
        promoted=int(usage[3]) if usage else 0))
cases=sorted(set(r['case'] for r in rows));summary=[]
manifest=folder/'cases.txt'
if manifest.exists():
    expected=sorted(manifest.read_text().splitlines())
    if cases!=expected:raise ValueError('case matrix differs from suite manifest')
elif len(cases)!=13:raise ValueError(f'expected 13 cases; found {len(cases)}')
for case in cases:
    out={'case':case}
    for variant in (0,1):
        group=[r for r in rows if r['case']==case and r['variant']==variant]
        if {r['trial'] for r in group}!={'r1','r2','r3'}:raise ValueError(f'{case}/{variant}: missing trials')
        validation=folder/f'{case}_{variant}_validation.log'
        if 'INPROCESS EXACTNESS PASS' not in validation.read_text():raise ValueError(f'{validation}: validation gate failed')
        for key in ('pfxt_ms','setup_ms','cold_setup_pfxt_ms','query_ms','descriptors','represented','promoted'):
            out[f'{key}_{variant}']=statistics.median(r[key] for r in group)
    for key in ('pfxt_ms','cold_setup_pfxt_ms','query_ms'):
        out[f'{key}_speedup']=out[f'{key}_0']/out[f'{key}_1']
    summary.append(out)
writer=csv.DictWriter(sys.stdout,fieldnames=list(summary[0]),lineterminator='\n');writer.writeheader();writer.writerows(summary)
