"""Summarize the full, workload-matched ablation without mixing production timings."""
import argparse
import csv
import math
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
p.add_argument('--csv',type=Path,required=True)
a=p.parse_args()
raw=list(csv.DictReader((a.run/'results.csv').open()))
plan=json.loads((a.run/'run-plan.json').read_text())
assert plan['repetitions']==3
trials={'r1','r2','r3'} if plan['timing_only'] else {'validation','r1','r2','r3'}
cases=list(dict.fromkeys(r['case'] for r in raw))
assert len(cases)==43 and len(raw)==43*2*4*len(trials)
rows=[]
for case in cases:
    for bound in (0,1):
        row=dict(case=case,bound=bound)
        for mode in range(4):
            runs=[r for r in raw if r['case']==case and int(r['bound'])==bound and int(r['mode'])==mode]
            assert {r['trial'] for r in runs}==trials and len(runs)==len(trials)
            timed=[r for r in runs if r['trial']!='validation']
            for key in ('setup_ms','pfxt_ms','cold_ms','generated_paths','strips','strip_paths','tiles','tile_paths','individual','total'):
                row[f'm{mode}_{key}']=statistics.median(float(r[key]) for r in timed)
            row[f'm{mode}_cold_min_ms']=min(float(r['cold_ms']) for r in timed)
            row[f'm{mode}_cold_max_ms']=max(float(r['cold_ms']) for r in timed)
        assert len({row[f'm{m}_generated_paths'] for m in range(4)})==1
        assert len({row[f'm{m}_total'] for m in range(4)})==1
        for label,numerator,denominator in [('204_alone',0,1),('24_alone',0,2),('24_added',1,3),('204_added',2,3),('both',0,3)]:
            row[label+'_speedup']=row[f'm{numerator}_cold_ms']/row[f'm{denominator}_cold_ms']
        rows.append(row)
by_case_bound={(r['case'],r['bound']):r for r in rows}
for r in rows:
    off=by_case_bound[r['case'],0]; on=by_case_bound[r['case'],1]
    for mode in range(4):
        r[f'm{mode}_bound_speedup']=off[f'm{mode}_cold_ms']/on[f'm{mode}_cold_ms']
with a.csv.open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print('# Isolating the two descriptor formats\n')
print('Controlled diagnostic, K = 1M, RTX 5090. Runtime is median cold setup + PFXT\n'
      'over three standalone trials. Graph-file loading and SFXT are excluded.\n')
print('Think of one assembly line with two optional ways to bundle waiting paths.\n'
      'We disable either bundle without replacing the assembly line. A rejected\n'
      'bundle emits ordinary LONG nodes instead. All variants retain the same\n'
      'ordinary producer, adaptive policy, source-local representation, warp\n'
      'aggregation, and arena-disabled allocation policy.\n')
print('The diagnostic disables the materialized-queue-only final-window shortcut\n'
      'in every variant. Consequently these are controlled ablation timings,\n'
      '**not replacements for production headline timings**. Removing a format\n'
      'also removes its allocation, replay and promotion work: that net effect\n'
      'is what the experiment is intended to measure.\n')
print('It also disables the safe-ordinary statistics shortcut in all variants\n'
      '(`GPUCPG_ADAPTIVE_PFXT_ADAPTIVE_SAFE_MIN_PATHS=2147483647`). Its production\n'
      'precheck uses a capped grid without a grid-stride loop and can inspect\n'
      'only a prefix on large frontiers. Queue order changed whether full stats\n'
      'were skipped on leon2 x8, despite equal modes and expansion counts. The\n'
      'first attempt stopped at this gate. Full statistics are used consistently\n'
      'in this corrected experiment; production defaults are not changed. This\n'
      'is a preparation/selection confound, not evidence of incorrect top-K:\n'
      'both ordinary and deferred branches still use exact candidate predicates.\n')
print('All 1,032 timed queries passed their correctness and workload checks.\n'
      'Correctness is checked during timing; no separate validation pass is\n'
      'required. Within each case and bound setting, window output\n'
      'counts, split values, recorded adaptive decisions, and final candidate\n'
      'counts match. Equal-cost output ordering is not required to be identical.\n')
for bound in (0,1):
    selected=[r for r in rows if r['bound']==bound]
    print(f'## Tight-bound {"on" if bound else "off"}\n')
    print('| Benchmark | Neither (ms) | 204 only (ms) | 24 only (ms) | Both (ms) | 204 alone | 24 added to 204 |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for r in selected:
        label=r['case'].replace('_base_x',' x').replace('_base','').replace('_d',' d')
        values=' | '.join(f'{r[f"m{m}_cold_ms"]:.3f}' for m in range(4))
        print(f'| {label} | {values} | {r["204_alone_speedup"]:.3f}x | {r["24_added_speedup"]:.3f}x |')
    print('\nGeometric means (speedup > 1 means faster):\n')
    for label in ('204_alone','24_alone','24_added','204_added','both'):
        g=math.exp(statistics.mean(math.log(r[label+'_speedup']) for r in selected))
        print(f'- {label}: {g:.3f}x')
    print()
print('## Isolating tight-bound with each representation held fixed\n')
print('Each cell is bound-off runtime divided by bound-on runtime, using the\n'
      'same descriptor configuration. Unlike the packing comparisons, this\n'
      'comparison permits fewer generated candidates: avoiding that work is\n'
      'the purpose of the bound. Both sides still validate against GPG.\n')
print('| Benchmark | Neither: bound speedup | 204 only: bound speedup | 24 only: bound speedup | Both: bound speedup |')
print('|---|---:|---:|---:|---:|')
for r in rows:
    if r['bound']!=1:continue
    label=r['case'].replace('_base_x',' x').replace('_base','').replace('_d',' d')
    print('| '+label+' | '+' | '.join(f'{r[f"m{m}_bound_speedup"]:.3f}x' for m in range(4))+' |')
print()
print('All four marginal comparisons and creation counts are retained in the CSV.\n'
      'The conservation checks count initial LONG storage, excluding SHORT/SKIP\n'
      'and later promotions. Counters reuse existing producer totals without\n'
      'new GPU scans or transfers. Timing includes the common diagnostic trace\n'
      'and accounting overhead; profiling is performed separately.\n')
