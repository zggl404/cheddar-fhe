#!/usr/bin/env python3
"""Measure checked CNN forwards using the existing synchronized online boundary."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
    p.add_argument('--cifar',type=Path,required=True);p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--families',nargs='+',default=['vgg11','vgg16','mobilenet','squeezenet','alexnet'])
    p.add_argument('--count',type=int,default=6);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--channel-tile',type=int,default=512);p.add_argument('--diagonal-chunk',type=int,default=64)
    p.add_argument('--baby-step',type=int,default=128)
    p.add_argument('--encode-mode',choices=('auto','pre','stream'),default='auto')
    a=p.parse_args()
    boots={'vgg11':16,'vgg16':26,'mobilenet':54,'squeezenet':36,'alexnet':14}
    if any(f not in boots for f in a.families) or not 0<=a.warmup<a.count<=10000:p.error('invalid family/sample range')
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'report.json').exists():p.error('output already contains a report; use a new directory')
    binary=a.binary.resolve();library=binary.parents[1]/'libcheddar.so'
    env=dict(os.environ);env['LD_LIBRARY_PATH']=str(library.parent)+':'+env.get('LD_LIBRARY_PATH','')
    root=Path(__file__).resolve().parents[1]
    report=dict(status='RUNNING',label=a.label,cells=[],binary_sha256=sha(binary),library_sha256=sha(library),
        cifar_sha256=sha(a.cifar),count=a.count,warmup=a.warmup,
        timing_boundary='unchanged sum of synchronized Online calls; preparation/encryption/checks excluded',
        source_sha256={str(f.relative_to(root)):sha(f) for folder in ('include','src','example','scripts','tests')
                       for f in sorted((root/folder).rglob('*')) if f.is_file() and f.suffix in ('.h','.cuh','.cpp','.cu','.py','.txt')})
    report['source_sha256']['CMakeLists.txt']=sha(root/'CMakeLists.txt')
    report['dependencies']=subprocess.check_output(['ldd',str(binary)],env=env,text=True)
    def gpu():
        return subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,clocks.current.sm,power.draw',
                                        '--format=csv,noheader'],text=True).strip()
    def save():
        temp=a.output/'report.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(a.output/'report.json')
    try:
        for family in a.families:
            model=a.models/f'{family}.bin';golden=json.loads((a.reference_dir/f'float-{family}.json').read_text())
            manifest=json.loads(model.with_suffix('.json').read_text())
            if golden['source_sha256']!=manifest['source_sha256'] or golden['cifar_sha256']!=report['cifar_sha256']:
                raise RuntimeError('reference source/data mismatch')
            if len(golden['images'])<a.count:raise RuntimeError('insufficient independent reference images')
            command=[sys.executable,str(root/'scripts/run_inference.py'),'--binary',str(binary),'--log-format','json','--verbose','1','--encode-mode',a.encode_mode,'--weights',str(model),'--cifar',str(a.cifar),'--count',str(a.count),
                '--channel-tile',str(a.channel_tile),'--diagonal-chunk',str(a.diagonal_chunk),'--baby-step',str(a.baby_step)]
            cell=dict(family=family,command=command,model_sha256=sha(model),images=[],gpu_before=gpu())
            report['cells'].append(cell);save();nodes=[];parameters=None;gpu_only=False;begin=time.monotonic()
            with (a.output/f'{family}.log').open('w',buffering=1) as log:
                process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                try:
                    for line in process.stdout:
                        log.write(line)
                        if line.startswith('PARAM '):parameters=json.loads(line[6:])
                        elif line.startswith('NODE_TIMING '):nodes.append(json.loads(line[12:]))
                        elif line.startswith('GPU_ONLY_PASS '):gpu_only='cpu_attempts=0' in line
                        elif line.startswith('RESULT '):
                            row=json.loads(line[7:]);row['nodes']=nodes;nodes=[];cell['images'].append(row)
                            print(f"IMAGE label={a.label} family={family} image={row['image']} online_ms={row['online_ms']:.3f} error={row['max_abs_error']:.3g}",flush=True)
                    cell['exit_code']=process.wait()
                finally:
                    if process.poll() is None:process.terminate();process.wait()
            (a.output/f'{family}.exit').write_text(str(cell['exit_code'])+'\n')
            cell['wall_seconds']=time.monotonic()-begin;cell['gpu_after']=gpu();save()
            if cell['exit_code'] or len(cell['images'])!=a.count or not gpu_only or not parameters:
                raise RuntimeError(f'{family}: executable/encoding audit failed')
            digest=hashlib.sha256(json.dumps(parameters,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if 'parameter_sha256' in report and report['parameter_sha256']!=digest:raise RuntimeError('parameters changed')
            report['parameters']=parameters;report['parameter_sha256']=digest
            for row in cell['images']:
                ref=golden['images'][row['image']]
                if row['max_abs_error']>1e-2 or row['online_encode_calls'] or row['boot_count']!=boots[family]:
                    raise RuntimeError(f'{family}: numerical/encoding/schedule gate failed')
                if row['prediction']!=row['polynomial_top1'] or row['prediction']!=row['relu_top1'] or row['prediction']!=ref['prediction']:
                    raise RuntimeError(f'{family}: top1 mismatch')
                row['cpp_vs_torch_error']=max(abs(x-y) for x,y in zip(row['relu_logits'],ref['logits']))
                if row['cpp_vs_torch_error']>1e-7:raise RuntimeError('independent float reference mismatch')
                if len(row['nodes'])!=len(manifest['nodes']):raise RuntimeError('missing node timing')
                row['node_time_sum_error_ms']=sum(n['online_ms'] for n in row['nodes'])-row['online_ms']
                if abs(row['node_time_sum_error_ms'])>1e-6:raise RuntimeError('node timing does not sum to online_ms')
            measured=cell['images'][a.warmup:];samples=[x['online_ms'] for x in measured]
            by_operation=defaultdict(list)
            for row in measured:
                sums=defaultdict(float)
                for node in row['nodes']:sums[str(node['operation'])]+=node['online_ms']
                for operation,value in sums.items():by_operation[operation].append(value)
            cell['online_ms']=dict(samples=samples,mean=statistics.mean(samples),median=statistics.median(samples),min=min(samples),max=max(samples))
            cell['operation_ms_mean']={k:statistics.mean(v) for k,v in by_operation.items()}
            cell['status']='PASS';save();print(f'CELL_PASS {a.label} {family} median_ms={cell["online_ms"]["median"]:.3f}',flush=True)
        report['status']='PASS';save()
    except Exception as error:
        report['status']='FAILED';report['error']=str(error);save();print(f'BENCHMARK_FAILED {error}',flush=True);return 1
    print('BENCHMARK_PASS',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
