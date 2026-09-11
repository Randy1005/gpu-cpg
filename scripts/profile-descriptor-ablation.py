"""Profile PFXT kernel-attributed DRAM traffic after standalone validation/timing."""
import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
import time
from ncu_traffic import parse_dram_csv

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--ablation',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--ncu',default='/usr/local/cuda-13.1/bin/ncu')
p.add_argument('--cases',nargs='+',default=['netcard_d10','netcard_d50','leon2_d30','des_perf_base_x16'])
p.add_argument('--bounds',nargs='+',type=int,choices=(0,1),default=[0,1])
p.add_argument('--modes',nargs='+',type=int,choices=range(4),default=list(range(4)))
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
a.out.mkdir(parents=True,exist_ok=False)
env={k:v for k,v in os.environ.items() if not k.startswith('GPUCPG_')}
env.update(GPUCPG_STRIP24='1',GPUCPG_DESCRIPTOR_COVERAGE='1')
env['GPUCPG_ADAPTIVE_PFXT_ADAPTIVE_SAFE_MIN_PATHS']='2147483647'
rows=[]
for case in a.cases:
    for bound in a.bounds:
        for mode in a.modes:
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                           '--format=csv,noheader'],text=True).strip():
                print('WAIT GPU_BUSY',flush=True);time.sleep(10)
            name=f'{case}_b{bound}_m{mode}'
            print('START '+name,flush=True)
            metric_file=(a.out/(name+'.metrics.csv')).resolve()
            app_file=a.out/(name+'.log')
            cmd=[a.ncu,'--replay-mode','application','--cache-control','none',
                 '--clock-control','none','--nvtx','--nvtx-include','descriptor_ablation_pfxt/',
                 '--metrics','dram__bytes_op_read.sum,dram__bytes_op_write.sum',
                 '--csv','--page','raw','--log-file',str(metric_file),
                 str(root/'build-strip/examples/tc-pfxt-inprocess-exactness'),
                 '--benchmark',str(a.data/f'{case}.csrbin'),'--ks','1000000',
                 '--baseline-file',str(a.reference/'goldens'/f'{case}_k1000000.gpg.costs'),
                 '--mode','adaptive']
            with app_file.open('x') as f:
                subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,
                               env=env | {'GPUCPG_DESCRIPTOR_ABLATION':str(mode+1),
                                          'GPUCPG_ADAPTIVE_PFXT_BOUND':str(bound)})
            app=app_file.read_text()
            assert 'INPROCESS EXACTNESS PASS' in app,name
            reference_file=a.ablation/f'{name}_validation.log'
            if not reference_file.exists():reference_file=a.ablation/f'{name}_r1.log'
            reference=reference_file.read_text()
            def windows(t):return re.findall(r'^descriptor_ablation_window .*$',t,re.M)
            assert windows(app)==windows(reference), 'profile workload mismatch '+name
            assert re.search(r'generated_paths=(\d+)',app)[1] == re.search(r'generated_paths=(\d+)',reference)[1]
            raw=metric_file.read_text()
            assert '==ERROR==' not in raw,raw[-2000:]
            row=dict(case=case,bound=bound,mode=mode,**parse_dram_csv(raw))
            rows.append(row)
            with (a.out/'traffic.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n')
                w.writeheader();w.writerows(rows)
            print('PASS '+name,flush=True)
print('TRAFFIC PROFILE COMPLETE',flush=True)
