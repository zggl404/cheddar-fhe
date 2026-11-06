#!/usr/bin/env python3
"""Validate all models with independent baseline, Event, NVTX and CPU runs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from benchmark_nvtx import run_phase, rows_with_prefix
from compare_ks_timings import compare
from inference_timing import validate_run
from run_inference import MODELS, ROOT


def launch(args, model, weights, keys, folder, encoding, method):
    command = [sys.executable, str(ROOT/'scripts/run_inference.py'), '--model', model,
        '--build-dir', str(args.build_dir), '--weights', str(weights), '--cifar', str(args.cifar),
        '--key-mode', 'load', '--key-dir', str(keys), '--encode-mode', encoding,
        '--count', str(args.count), '--warmup', str(args.warmup), '--split-hybrid', args.split_hybrid,
        '--numerical-gate', 'strict', '--log-format', 'json', '--timing-dir', str(folder),
        '--time-cpu', 'on' if method == 'CPU' else 'off',
        '--time-cuda-event', 'on' if method == 'CUDA_EVENT' else 'off',
        '--time-nvtx', 'on' if method == 'NVTX' else 'off', '--nsys', args.nsys]
    log = folder.with_suffix('.log')
    print(f'  {model}: {method}', flush=True)
    with log.open('w') as handle:
        code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)
    if code:raise RuntimeError(f'{method} exited {code}; see {log}')
    wall = json.loads((folder/'cpu.json').read_text());validate_run(wall)
    report = json.loads((folder/'summary.json').read_text())[method]
    if not report['numerical_passed'] or wall['keys_generated'] or wall['keys_loaded'] != wall['key_count']:
        raise ValueError(f'{method} failed numerical/key-load validation')
    return wall, report, rows_with_prefix((folder/'worker.log').read_text(), 'RESULT ')


def verify_results(baseline, wall, results, count):
    for key in ('key_identity', 'encode_mode', 'plaintext_reference'):
        if baseline[key] != wall[key]:raise ValueError('baseline configuration mismatch: '+key)
    if len(results) != count or [r['image'] for r in results] != list(range(count)):
        raise ValueError('missing/duplicate inference results')
    maximum = 0.
    for a, b in zip(baseline['results'], results):
        if b['status'] != 'PASS' or a['prediction'] != b['prediction'] or len(b['fhe_logits']) != 10:
            raise ValueError('baseline result/prediction mismatch')
        for x, y in zip(a['fhe_logits'], b['fhe_logits']):
            if not math.isfinite(y):raise ValueError('nonfinite logits')
            maximum = max(maximum, abs(x-y))
    if maximum > .01:raise ValueError('baseline paired logits exceed numerical gate')
    return maximum


def write_summary(output, summary):
    path = output/'summary.json.tmp';path.write_text(json.dumps(summary, indent=2)+'\n');path.replace(output/'summary.json')
    fields = ('model', 'status', 'encode_mode', 'measured_count', 'baseline_end_to_end_ms',
        'baseline_time_real_ms', 'baseline_time_user_ms', 'baseline_time_sys_ms',
        'baseline_key_load_ms', 'baseline_weight_preencode_ms', 'baseline_inference_total_ms',
        'baseline_other_prepare_and_teardown_ms', 'baseline_inference_mean_ms',
        'event_inference_mean_ms', 'nvtx_inference_mean_ms', 'cpu_inference_mean_ms',
        'event_keyswitch_ms', 'nvtx_keyswitch_ms', 'event_non_keyswitch_ms', 'nvtx_non_keyswitch_ms',
        'event_hybrid_ms','nvtx_hybrid_ms','cpu_hybrid_ms',
        'cpu_non_keyswitch_ms', 'delta_ms', 'delta_percent',
        'cpu_keyswitch_ms', 'cpu_fhe_total_ms', 'matched_measured_segments', 'max_baseline_logits_delta',
        'temporary_keys_removed', 'error')
    with (output/'summary.csv.tmp').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader()
        writer.writerows({k:row.get(k, '') for k in fields} for row in summary['models'])
    (output/'summary.csv.tmp').replace(output/'summary.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build')
    parser.add_argument('--models-dir', type=Path, default=ROOT/'models')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--cifar', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--encode-mode', choices=('auto', 'pre', 'stream'), default='auto')
    parser.add_argument('--split-hybrid',choices=('on','off'),default='on')
    parser.add_argument('--nsys', default='nsys')
    parser.add_argument('--time-binary', default='/usr/bin/time')
    args = parser.parse_args()
    if not 0 <= args.warmup < args.count <= 10000:parser.error('require 0 <= warmup < count <= 10000')
    if len(set(args.models)) != len(args.models):parser.error('duplicate models')
    for key in ('build_dir', 'models_dir', 'cifar', 'output'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    for key in ('nsys', 'time_binary'):
        found = shutil.which(getattr(args, key))
        if not found:parser.error('missing '+key)
        setattr(args, key, found)
    if not args.cifar.is_file():parser.error('missing CIFAR file')
    if args.output.exists():parser.error('output directory already exists')
    args.output.mkdir(parents=True)
    args.numerical_gate = 'strict'
    summary = dict(status='RUNNING', count=args.count, warmup=args.warmup,split_hybrid=args.split_hybrid,
        policy='Sequential independent baseline/Event/NVTX/CPU; strict gates; same saved keys; temporary keys removed per model', models=[])
    for model in args.models:
        row = dict(model=model, status='FAILED', measured_count=args.count-args.warmup, temporary_keys_removed=False)
        scratch = None
        try:
            folder = args.output/model;folder.mkdir()
            weights = args.models_dir/(model+'.bin')
            binary = args.build_dir/'example'/('resnet_cifar' if model.startswith('resnet') else 'cnn_cifar')
            if not weights.is_file() or not binary.is_file():raise ValueError('missing model weights or executable')
            row['weights_sha256'] = hashlib.sha256(weights.read_bytes()).hexdigest()
            with tempfile.TemporaryDirectory(prefix='.keys-'+model+'-', dir=args.output) as scratch:
                keys = Path(scratch)/'bundle'
                run_phase(args, model, binary, weights, keys, 'save', folder/'key-setup', 1, args.encode_mode)
                base = run_phase(args, model, binary, weights, keys, 'load', folder/'baseline', args.count, args.encode_mode)
                mode = base['encode_mode'];row['encode_mode'] = mode
                row.update(baseline_end_to_end_ms=base['end_to_end_ms'], baseline_key_load_ms=base['key_load_ms'],
                    baseline_weight_preencode_ms=base['weight_preencode_ms'],
                    baseline_inference_total_ms=base['inference_total_ms'],
                    baseline_other_prepare_and_teardown_ms=base['other_prepare_and_teardown_ms'],
                    **{'baseline_'+k:base[k] for k in ('time_real_ms','time_user_ms','time_sys_ms')},
                    baseline_inference_mean_ms=sum(r['inference_ms'] for r in base['image_times'][args.warmup:])/(args.count-args.warmup))
                maximum = 0.
                for name, method in (('event', 'CUDA_EVENT'), ('nvtx', 'NVTX'), ('cpu', 'CPU')):
                    wall, report, results = launch(args, model, weights, keys, folder/name, mode, method)
                    maximum = max(maximum, verify_results(base, wall, results, args.count))
                    row[name+'_inference_mean_ms'] = sum(r['inference_ms'] for r in wall['images'][args.warmup:])/(args.count-args.warmup)
                    row[name+'_non_keyswitch_ms'] = report['NonKeyswitchTime']
                    row[name+'_hybrid_ms'] = report['Hybrid']
                    if name == 'cpu':row.update(cpu_keyswitch_ms=report['KeyswitchTime'], cpu_fhe_total_ms=report['total'])
                paired = compare(folder/'event', folder/'nvtx', folder/'comparison')
                row.update(status='PASS', event_keyswitch_ms=paired['event_keyswitch_mean_ms'], nvtx_keyswitch_ms=paired['nvtx_keyswitch_mean_ms'],
                    delta_ms=paired['delta_ms'], delta_percent=paired['delta_percent'],
                    matched_measured_segments=paired['matched_measured_segments'], max_baseline_logits_delta=maximum)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            row.update(status='FAILED', error=str(error))
        finally:
            row['temporary_keys_removed'] = scratch is not None and not Path(scratch).exists()
        summary['models'].append(row)
        summary['status'] = 'RUNNING' if len(summary['models']) < len(args.models) else ('PASS' if all(r['status']=='PASS' for r in summary['models']) else 'FAILED')
        write_summary(args.output, summary)
        print(f'{model}: {row["status"]}'+(': '+row['error'] if 'error' in row else ''), flush=True)
    return int(summary['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
