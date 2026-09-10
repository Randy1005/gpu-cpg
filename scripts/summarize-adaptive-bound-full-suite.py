#!/usr/bin/env python3
import csv
from pathlib import Path
import re
from statistics import median

root=Path(__file__).resolve().parents[1]
out=root/'experiments/adaptive-bound-full-suite-20260910'
cases=['des_perf_base_x16','leon3mp_base_x16','netcard_base_x16','leon2_d30',
       'netcard_d10','netcard_d50','leon3mp_d50','des_perf_d40','leon2_base',
       'des_perf_base','cage15','M6','nlpkkt120']
rows=[]
for case in cases:
  row={'case':case}
  for variant in ['gpg','fixed','bound']:
    pfxt=[]; query=[]
    for repeat in range(1,4):
      text=(out/'logs'/f'{case}_{variant}_r{repeat}.log').read_text()
      if 'INPROCESS EXACTNESS PASS' not in text:
        raise RuntimeError(f'correctness failed: {case} {variant} r{repeat}')
      match=re.search(r'^exactness_summary (.*)$',text,re.M)
      if not match: raise RuntimeError(f'missing summary: {case} {variant} r{repeat}')
      values=dict(item.split('=') for item in match[1].split())
      if values['pass']!='1' or values['result_count']!='1000000':
        raise RuntimeError(f'invalid result: {case} {variant} r{repeat}')
      pfxt.append(float(values['pfxt_ms']))
      query.append(float(values['query_ms']))
    row[variant+'_pfxt_ms']=median(pfxt)
    row[variant+'_query_ms']=median(query)
  row['fixed_vs_gpg']=row['gpg_pfxt_ms']/row['fixed_pfxt_ms']
  row['bound_vs_fixed']=row['fixed_pfxt_ms']/row['bound_pfxt_ms']
  row['bound_vs_gpg']=row['gpg_pfxt_ms']/row['bound_pfxt_ms']
  rows.append(row)
with (out/'results.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
for row in rows: print(row)
