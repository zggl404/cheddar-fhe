#!/usr/bin/env python3
"""Run staged ciphertext CNN checks and retain source/data/binary evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from inference_timing import validate_image, validate_run


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--cifar',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--families',nargs='+',default=['vgg11','vgg16','mobilenet','squeezenet','alexnet'],choices=['vgg11','vgg16','mobilenet','squeezenet','alexnet'])
    parser.add_argument('--count',type=int,default=3);parser.add_argument('--relu-range',type=float,default=10)
    parser.add_argument('--encode-mode',choices=('auto','pre','stream'),default='auto')
    args=parser.parse_args()
    if not 1<=args.count<=10000:parser.error('count must be in [1,10000]')
    args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    report=dict(status='RUNNING',cells=[],count=args.count,timing_schema_version=2,
                binary_sha256=sha(args.binary),library_sha256=sha(args.binary.resolve().parents[1]/'libcheddar.so'),cifar_sha256=sha(args.cifar),
                source_sha256={str(p.relative_to(root)):sha(p) for folder in ('include','src','example','scripts','tests')
                               for p in sorted((root/folder).rglob('*')) if p.is_file() and p.suffix in ('.h','.cuh','.cpp','.cu','.py','.txt')})
    report['source_sha256']['CMakeLists.txt']=sha(root/'CMakeLists.txt')
    report['hardware']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    def save():
        temporary=args.output/'report.json.tmp';temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(args.output/'report.json')
    def run(family,stop,count):
        name=f'{family}-prefix{stop}' if stop else family
        model=args.models/f'{family}.bin';manifest=json.loads(model.with_suffix('.json').read_text())
        golden=json.loads((args.reference_dir/f'float-{family}.json').read_text())
        if golden['source_sha256']!=manifest['source_sha256'] or golden['cifar_sha256']!=report['cifar_sha256']:
            raise RuntimeError('independent reference source/data hash mismatch')
        command=[sys.executable,str(root/'scripts/run_inference.py'),'--binary',str(args.binary),'--log-format','json','--verbose','1','--encode-mode',args.encode_mode,'--weights',str(model),'--cifar',str(args.cifar),'--stop-after',str(stop),
                 '--count',str(count),'--relu-range',str(args.relu_range)]
        cell=dict(name=name,command=command,model_sha256=sha(model),log=f'{name}.log',images=[],image_timings=[])
        report['cells'].append(cell);save();parameters=None;gpu_only=False
        with (args.output/cell['log']).open('w',buffering=1) as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                for line in process.stdout:
                    log.write(line)
                    if line.startswith('PARAM '):parameters=json.loads(line[6:])
                    if line.startswith('GPU_ONLY_PASS '):gpu_only=True
                    if line.startswith('IMAGE_TIMING '):
                        timing=json.loads(line.removeprefix('IMAGE_TIMING '));validate_image(timing);cell['image_timings'].append(timing)
                    if line.startswith('TIMING '):
                        cell['timing']=json.loads(line.removeprefix('TIMING '))
                    if line.startswith('RESULT '):
                        row=json.loads(line[7:]);cell['images'].append(row)
                        print(f"IMAGE cell={name} image={row['image']} error={row['max_abs_error']:.4g}",flush=True)
                cell['exit_code']=process.wait()
            finally:
                if process.poll() is None:process.terminate();process.wait()
        (args.output/f'{name}.exit').write_text(str(cell['exit_code'])+'\n')
        save()
        if 'timing' not in cell or cell['timing'].get('schema_version') != 2 or not cell['image_timings']:
            raise RuntimeError(f'{name}: missing unified timing records')
        validate_run(dict(cell['timing'], images=cell['image_timings']))
        if [t['image'] for t in cell['image_timings'] if t['image'] >= 0] != [r['image'] for r in cell['images']]:
            raise RuntimeError(f'{name}: result/timing image mismatch')
        if cell['exit_code'] or len(cell['images'])!=count or not gpu_only or not parameters:
            raise RuntimeError(f'{name}: executable/encoding audit failed')
        if parameters['ring_dimension']!=131072 or parameters['slots']!=65536 or parameters['boot_output_level']!=9:
            raise RuntimeError(f'{name}: parameter mismatch')
        if 'parameters' in report and report['parameters']!=parameters:raise RuntimeError('resolved parameters changed')
        report['parameters']=parameters
        report['parameter_sha256']=hashlib.sha256(json.dumps(parameters,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        for row in cell['images']:
            if row['status']!='PASS' or row['max_abs_error']>1e-2 or row['fhe_encode_calls']:
                raise RuntimeError(f'{name}: ciphertext numerical/encoding audit failed')
            if not stop:
                reference=golden['images'][row['image']]
                if reference['image']!=row['image']:raise RuntimeError('independent reference image index mismatch')
                row['cpp_vs_torch_error']=max(abs(a-b) for a,b in zip(row['relu_logits'],reference['logits']))
                if row['cpp_vs_torch_error']>1e-7 or row['prediction']!=reference['prediction'] or row['prediction']!=row['polynomial_top1']:
                    raise RuntimeError(f'{name}: independent reference mismatch')
        cell['status']='PASS';save();print(f'CELL_PASS {name}',flush=True)
    try:
        for family in args.families:
            run(family,1,1)
            run(family,{'vgg11':5,'vgg16':5,'mobilenet':6,'squeezenet':8,'alexnet':5}[family],1)
            run(family,0,args.count)
        report['status']='PASS';save()
    except Exception as error:
        report['status']='FAILED';report['error']=str(error);save();print(f'VALIDATION_FAILED {error}',flush=True);return 1
    print('VALIDATION_PASS',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
