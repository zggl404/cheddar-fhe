#!/usr/bin/env python3
"""Benchmark every model with a coarse wall baseline and a separate NVTX collection."""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

from analyze_gpu_activity import analyze
from inference_timing import validate_run
from run_inference import MODELS, ROOT


def number(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'invalid {name}: {value}')
    return value


def time_values(path):
    values = dict(line.split() for line in path.read_text().splitlines() if line.strip())
    return {f'time_{key}_ms': number(float(values[key])*1000, 'time '+key) for key in ('real', 'user', 'sys')}


def rows_with_prefix(text, prefix):
    return [json.loads(line[len(prefix):]) for line in text.splitlines() if line.startswith(prefix)]


def run_phase(args, model, binary, weights, keys, mode, folder, count, encoding, nvtx=False):
    folder.mkdir()
    command = [args.time_binary, '-p', '-o', str(folder/'time.txt'), str(binary),
        '--weights', str(weights), '--cifar', str(args.cifar), '--count', str(count),
        '--relu-range', str({'resnet56': 12, 'resnet110': 24}.get(model, 10)),
        '--encode-mode', encoding, '--key-mode', mode, '--key-dir', str(keys),
        '--split-hybrid', args.split_hybrid, '--log-format', 'json',
        '--time-cpu', 'off', '--time-cuda-event', 'off', '--time-nvtx', 'on' if nvtx else 'off',
        '--timing-file', str(folder/'wall.json')]
    if nvtx:
        command = [args.nsys, 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                   '--export=sqlite', '--output='+str(folder/'nvtx'), *command]
    (folder/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    print(f'  {model}: {folder.name}', flush=True)
    begin = time.perf_counter()
    with (folder/'stdout.log').open('w') as out, (folder/'stderr.log').open('w') as err:
        result = subprocess.run(command, stdout=out, stderr=err, env=dict(os.environ,
            CHEDDAR_GPU_TRACE='1' if nvtx else '0', CHEDDAR_CUDA_EVENT_TRACE='0',
            LC_ALL='C',
            LD_LIBRARY_PATH=str(args.build_dir)+':'+os.environ.get('LD_LIBRARY_PATH', '')))
    wrapper_ms = (time.perf_counter()-begin)*1000
    if result.returncode:
        raise RuntimeError(f'{folder.name} exited {result.returncode}; see {folder}/stderr.log and stdout.log')
    report = json.loads((folder/'wall.json').read_text())
    validate_run(report)
    if report.get('timing_detail') != 'coarse' or report.get('ks_profile', False):
        raise ValueError('worker must provide coarse timing without synchronized operation profiling; rebuild examples')
    if report['encode_mode'] not in ('pre','stream'):
        raise ValueError('worker did not report resolved encoding mode')
    if len(report['images']) != count or [r['image'] for r in report['images']] != list(range(count)):
        raise ValueError('unexpected image census')
    if mode == 'load' and (report['keys_generated'] != 0 or report['keys_loaded'] != report['key_count']):
        raise ValueError('load mode generated or missed evaluation keys')
    for key in ('total_ms', 'key_load_ms', 'weight_preencode_ms', 'inference_ms'):
        number(report[key], key)
    if mode=='load' and report['key_load_ms']<=0:
        raise ValueError('key loading time was not recorded')
    if report['encode_mode']=='pre' and report['weight_preencode_ms']<=0:
        raise ValueError('preencoding time was not recorded')
    other = report['total_ms']-report['key_load_ms']-report['weight_preencode_ms']-report['inference_ms']
    if other < -1e-6:
        raise ValueError('overlapping preparation/inference intervals')
    text = (folder/'stdout.log').read_text()
    images = rows_with_prefix(text, 'RESULT ')
    if len(images) != count or any(r.get('status') != 'PASS' or len(r.get('fhe_logits', [])) != 10 for r in images):
        raise ValueError('missing successful full inference results')
    if [r['image'] for r in images]!=list(range(count)):
        raise ValueError('result image census disagrees with wall report')
    for row in images:
        observed=row.get('family') or ('resnet'+str(row['depth']) if 'depth' in row else row.get('model','').lower())
        if observed!=model:raise ValueError(f'weights contain {observed}, requested {model}')
    return dict(end_to_end_ms=report['total_ms'], **time_values(folder/'time.txt'),
        key_load_ms=report['key_load_ms'], weight_preencode_ms=report['weight_preencode_ms'],
        inference_total_ms=report['inference_ms'], other_prepare_and_teardown_ms=max(other, 0),
        encode_mode=report['encode_mode'], plaintext_reference=report['plaintext_reference'],
        keys_generated=report['keys_generated'], keys_loaded=report['keys_loaded'], key_count=report['key_count'],
        key_identity=report['key_identity'], wrapper_wall_ms=wrapper_ms,
        image_times=report['images'], results=images)


def pair_result(baseline, profiled, analysis, warmup):
    if baseline['key_identity'] != profiled['key_identity'] or baseline['encode_mode'] != profiled['encode_mode']:
        raise ValueError('paired runs used different keys or encoding modes')
    if baseline['plaintext_reference'] != profiled['plaintext_reference']:
        raise ValueError('paired runs used different plaintext reference builds')
    ids = [r['image'] for r in baseline['image_times']][warmup:]
    if analysis['measured_images'] != ids:
        raise ValueError('NVTX and wall timing used different measured images')
    if any(r['kernel_overlap_ns'] for r in analysis['images']):
        raise ValueError('overlapping kernels: duration sums cannot partition inference wall time')
    maximum = 0
    for before, after in zip(baseline['results'], profiled['results']):
        if before['prediction'] != after['prediction']:
            raise ValueError('paired predictions differ')
        for a, b in zip(before['fhe_logits'], after['fhe_logits']):
            maximum = max(maximum, number(abs(a-b), 'logits delta'))
    if maximum > .01:
        raise ValueError(f'paired logits differ by {maximum}')
    for run in (baseline, profiled):
        run['inference_mean_ms'] = sum(r['inference_ms'] for r in run['image_times'][warmup:])/len(ids)
    mean = analysis['mean_ms']
    key = number(mean.get('kernel_keyswitch', 0), 'NVTX Keyswitch')
    residual = profiled['inference_mean_ms']-key
    number(residual, 'Inference minus NVTX Keyswitch')
    return dict(measured_count=len(ids), warmup_count=warmup, max_logits_delta=maximum,
        nvtx_keyswitch_ms=key, nvtx_hybrid_ms=mean.get('kernel_hybrid', 0),
        nvtx_other_kernel_ms=mean.get('kernel_other', 0), nvtx_kernel_total_ms=mean['kernel_total'],
        non_keyswitch_residual_ms=residual,
        residual_definition='profiled inference wall mean minus same-run NVTX Keyswitch kernel mean; includes Hybrid, host work, idle, input, plaintext reference, validation and runtime encoding')


def write_summary(output, summary):
    temporary=output/'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2)+'\n');temporary.replace(output/'summary.json')
    fields = ['model', 'status', 'keys_prepared_by_script', 'temporary_keys_removed', 'encode_mode', 'plaintext_reference',
        'split_hybrid','measured_count','warmup_count',
        *[f'{kind}_{key}' for kind in ('baseline','profiled') for key in
          ('end_to_end_ms','time_real_ms','time_user_ms','time_sys_ms','key_load_ms','weight_preencode_ms','inference_total_ms','inference_mean_ms','other_prepare_and_teardown_ms')],
        'nvtx_keyswitch_ms','nvtx_hybrid_ms','nvtx_other_kernel_ms','non_keyswitch_residual_ms','error']
    with (output/'summary.csv.tmp').open('w', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=fields);writer.writeheader()
        for row in summary['models']:
            flat={key:row.get(key,'') for key in fields}
            if 'baseline' in row:
                for kind in ('baseline','profiled'):
                    flat.update({f'{kind}_{k}':v for k,v in row[kind].items() if f'{kind}_{k}' in fields})
                flat['encode_mode']=row['profiled']['encode_mode'];flat['plaintext_reference']=row['profiled']['plaintext_reference']
            writer.writerow(flat)
    (output/'summary.csv.tmp').replace(output/'summary.csv')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,default=ROOT/'build')
    parser.add_argument('--models-dir',type=Path,default=ROOT/'models')
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--cifar',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new directory for logs, traces and summaries')
    parser.add_argument('--key-root',type=Path,help='reuse KEY_ROOT/MODEL; missing bundles are temporary')
    parser.add_argument('--count',type=int,default=6)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--encode-mode',choices=('auto','pre','stream'),default='auto')
    parser.add_argument('--split-hybrid',choices=('off','on'),default='on',
                        help='split mixed operations (default: on); off uses the original fused path')
    parser.add_argument('--nsys',default='nsys')
    parser.add_argument('--time-binary',default='/usr/bin/time')
    args=parser.parse_args()
    if not 0<=args.warmup<args.count<=10000:parser.error('require 0 <= warmup < count <= 10000')
    if len(set(args.models))!=len(args.models):parser.error('duplicate models')
    for key in ('build_dir','models_dir','cifar','output','key_root'):
        value=getattr(args,key)
        if value is not None:setattr(args,key,value.expanduser().resolve())
    if not args.cifar.is_file():parser.error('missing CIFAR file')
    for key in ('nsys','time_binary'):
        value=shutil.which(getattr(args,key))
        if not value:parser.error(f'missing {key}; install Nsight Systems CLI and GNU time')
        setattr(args,key,value)
    if args.output.exists():parser.error('output directory already exists')
    args.output.mkdir(parents=True)
    key_root=args.key_root or args.output/'.temporary-keys'
    key_root.mkdir(parents=True,exist_ok=True,mode=0o700)
    summary=dict(schema_version=1,status='RUNNING',models=[],count=args.count,warmup=args.warmup,
        split_hybrid=args.split_hybrid,units='ms',
        timing_policy='paired coarse baseline and NVTX collection; no CUDA Event or synchronized CPU operation timing',
        key_cache_note='temporary key generation and baseline may warm the page cache; not a cold-disk benchmark')
    started=time.perf_counter()
    for model in args.models:
        row=dict(model=model,status='FAILED',keys_prepared_by_script=False,temporary_keys_removed=False,split_hybrid=args.split_hybrid)
        scratch=None
        try:
            binary=args.build_dir/'example'/('resnet_cifar' if model.startswith('resnet') else 'cnn_cifar')
            weights=args.models_dir/(model+'.bin')
            if not binary.is_file() or not os.access(binary,os.X_OK):raise ValueError(f'missing executable: {binary}')
            if not weights.is_file():raise ValueError(f'missing weights: {weights}')
            folder=args.output/model;folder.mkdir()
            existing=key_root/model
            if existing.exists() and not (existing/'manifest.txt').is_file():raise ValueError(f'incomplete existing key bundle: {existing}')
            context=nullcontext(None) if existing.exists() else tempfile.TemporaryDirectory(prefix=f'.{model}-',dir=key_root)
            with context as scratch:
                keys=existing if scratch is None else Path(scratch)/'bundle'
                row['keys_prepared_by_script']=scratch is not None
                row['weights_sha256']=hashlib.sha256(weights.read_bytes()).hexdigest()
                if scratch is not None:
                    run_phase(args,model,binary,weights,keys,'save',folder/'key-setup',1,args.encode_mode)
                baseline=run_phase(args,model,binary,weights,keys,'load',folder/'baseline',args.count,args.encode_mode)
                profiled=run_phase(args,model,binary,weights,keys,'load',folder/'profiled',args.count,baseline['encode_mode'],True)
                analysis=analyze(folder/'profiled/nvtx.sqlite',folder/'profiled/analysis',args.warmup,folder/'profiled/stdout.log',args.split_hybrid=='on')
                row.update(pair_result(baseline,profiled,analysis,args.warmup),baseline=baseline,profiled=profiled,status='PASS')
        except (OSError,ValueError,KeyError,TypeError,RuntimeError,sqlite3.Error,MemoryError) as error:
            row['error']=str(error)
        finally:
            row['temporary_keys_removed']=scratch is not None and not Path(scratch).exists()
        summary['models'].append(row);summary['batch_wall_ms']=(time.perf_counter()-started)*1000
        if len(summary['models'])==len(args.models):
            summary['status']='PASS' if all(r['status']=='PASS' for r in summary['models']) else 'FAILED'
        write_summary(args.output,summary)
        print(f'{model}: {row["status"]}'+(': '+row['error'] if 'error' in row else ''),flush=True)
    print('Reports:',args.output/'summary.csv',args.output/'summary.json',flush=True)
    return int(any(row['status']!='PASS' for row in summary['models']))


if __name__=='__main__':raise SystemExit(main())
