#!/usr/bin/env python3
"""Run CIFAR-10 inference with independently switched CPU, CUDA Event and NVTX timing."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from inference_timing import summarize, validate_run, validate_totals

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
          'vgg11', 'vgg16', 'mobilenet', 'squeezenet', 'alexnet')


def process_wall(fallback_start):
    try:
        fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        born = int(fields[19]) / os.sysconf('SC_CLK_TCK')
        return ((time.clock_gettime(time.CLOCK_BOOTTIME) - born) * 1000,
                'launcher process wall including child startup and teardown')
    except (OSError, ValueError, AttributeError, IndexError):
        return ((time.perf_counter() - fallback_start) * 1000,
                'Python main through child teardown; interpreter startup excluded')


def category_report(method, scope, key, non, hybrid, warmup, measured):
    values = (key, non, hybrid)
    if any(not 0 <= value < float('inf') for value in values):
        raise ValueError('invalid timing category')
    return dict(method=method, scope=scope, unit='ms', warmup_count=warmup,
                measured_count=measured, KeyswitchTime=key, NonKeyswitchTime=non,
                Hybrid=hybrid, total=sum(values))


def print_timing(name, row, json_output):
    if json_output:
        print(name + '_TIMING ' + json.dumps(row, separators=(',', ':')))
    else:
        print(f'\n{name} timing (ms, {row["scope"]}; mean of {row["measured_count"]} images)')
        if not row.get('numerical_passed',True):print('  Numerical checks failed; timings are diagnostic only')
        if name == 'CPU':
            for key in ('end_to_end_ms', 'prepare_ms', 'model_prepare_ms', 'boot_prepare_ms', 'inference_ms'):
                print(f'  {key:24}: {row[key]:.3f}')
        for key in ('KeyswitchTime', 'NonKeyswitchTime', 'Hybrid', 'total'):
            print(f'  {key:24}: {row[key]:.3f}')


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument('--model', choices=MODELS)
    entry.add_argument('--binary', type=Path, help='advanced: direct C++ entry, including probes')
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build')
    parser.add_argument('--weights', type=Path, help='default: models/MODEL.bin')
    parser.add_argument('--cifar', type=Path)
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=0, help='included in count and CPU end-to-end time')
    parser.add_argument('--encode-mode', choices=('auto', 'pre', 'stream'), default='auto')
    parser.add_argument('--key-mode', choices=('generate', 'save', 'load'), default='generate')
    parser.add_argument('--key-dir', type=Path)
    parser.add_argument('--split-hybrid', choices=('off', 'on'), default='off')
    parser.add_argument('--numerical-gate', choices=('strict','warn'), default='strict',
                        help='warn continues finite precision failures and marks results unsuccessful')
    parser.add_argument('--time-cpu', '--ks-profile', dest='time_cpu', choices=('off', 'on'), default='off',
                        help='synchronized CPU wall timing; --ks-profile is a compatibility alias')
    parser.add_argument('--time-cuda-event', choices=('off', 'on'), default='off')
    parser.add_argument('--time-nvtx', choices=('off', 'on'), default='off')
    parser.add_argument('--nsys', default='nsys', help='Nsight Systems executable for --time-nvtx on')
    parser.add_argument('--timing-dir', type=Path, help='retain raw timing evidence in a new directory')
    parser.add_argument('--log-format', choices=('human', 'json'), default='human')
    args, options = parser.parse_known_args()
    verbose = any(options[i:i+2] == ['--verbose', '1'] for i in range(len(options)))
    cpu, events, nvtx = args.time_cpu == 'on', args.time_cuda_event == 'on', args.time_nvtx == 'on'
    if args.key_mode != 'generate' and not args.key_dir:
        parser.error('--key-mode save/load requires --key-dir')
    if args.key_dir:
        options += ['--key-dir', str(args.key_dir.expanduser().resolve())]
    options += ['--key-mode', args.key_mode]
    if not 0 <= args.warmup < args.count <= 10000:
        parser.error('require 0 <= warmup < count <= 10000')
    if any(x.split('=', 1)[0] in ('--timing-file', '--event-file') for x in options):
        parser.error('timing-file and event-file are managed by this launcher')
    if args.model and not args.cifar:
        parser.error('--model requires --cifar')
    binary = args.binary or args.build_dir / 'example' / ('resnet_cifar' if args.model.startswith('resnet') else 'cnn_cifar')
    binary = binary.expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f'missing executable: {binary}; build both examples first')
    weights = args.weights or (ROOT / 'models' / f'{args.model}.bin' if args.model else None)
    if weights:
        options += ['--weights', str(weights.expanduser().resolve())]
    if args.cifar:
        options += ['--cifar', str(args.cifar.expanduser().resolve())]
    if args.model and '--relu-range' not in options:
        options += ['--relu-range', {'resnet56': '12', 'resnet110': '24'}.get(args.model, '10')]
    options += ['--count', str(args.count), '--encode-mode', args.encode_mode, '--log-format', args.log_format,
                '--time-cpu', args.time_cpu, '--time-cuda-event', args.time_cuda_event,
                '--time-nvtx', args.time_nvtx, '--split-hybrid', args.split_hybrid,
                '--numerical-gate', args.numerical_gate]
    nsys = shutil.which(args.nsys) if nvtx else None
    if nvtx and not nsys:
        parser.error('--time-nvtx on requires Nsight Systems; supply --nsys /path/to/nsys')
    env = dict(os.environ, CHEDDAR_GPU_TRACE='1' if nvtx else '0',
               CHEDDAR_CUDA_EVENT_TRACE='1' if events else '0')
    env['LD_LIBRARY_PATH'] = str(binary.parents[1]) + ':' + env.get('LD_LIBRARY_PATH', '')
    try:
        # The OFF path launches the worker directly: no timing report or profiler.
        if not (cpu or events or nvtx):
            code = subprocess.call([str(binary), *options], env=env)
            return code if code >= 0 else 128-code
        with tempfile.TemporaryDirectory(prefix='cheddar-timing-') as temporary:
            folder = args.timing_dir.expanduser().resolve() if args.timing_dir else Path(temporary)
            if args.timing_dir:
                folder.mkdir(parents=True, exist_ok=False)
            path, csv, log = folder/'cpu.json', folder/'events.csv', folder/'worker.log'
            command = [str(binary), *options]
            command += ['--timing-file', str(path)]
            if events:
                command += ['--event-file', str(csv)]
            if nvtx:
                command = [nsys, 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none', '--cuda-event-trace=false',
                           '--force-overwrite=true', '--export=sqlite', '--output='+str(folder/'nvtx'), *command]
            (folder/'command.json').write_text(json.dumps(dict(command=command), indent=2)+'\n')
            result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            log.write_text(result.stdout)
            (folder/'profiler-stderr.log').write_text(result.stderr)
            if result.returncode:
                print(result.stdout, end='')
                print(result.stderr, end='', file=sys.stderr)
                return result.returncode if result.returncode > 0 else 128-result.returncode
            warnings=[line for line in result.stderr.splitlines() if line.startswith('NUMERICAL_WARNING ')]
            for warning in warnings:print(warning,file=sys.stderr)
            results=[json.loads(line[len('RESULT '):]) for line in result.stdout.splitlines() if line.startswith('RESULT ')]
            numerical_passed=not warnings and all(r.get('status','PASS')=='PASS' and r.get('numerical_passed',r.get('success',True)) for r in results)
            # Hide internal audits and nsys progress; retain both in the raw log.
            for line in result.stdout.splitlines():
                if line.startswith(('IMAGE_TIMING ', 'CUDA_EVENT_IMAGE ', 'CUDA_EVENT_KS_IMAGE ', 'GPU_TRACE_AUDIT ')) and not verbose:
                    continue
                if nvtx and line.startswith(('Generating ', 'Processing ', 'Generated:', 'Report files:', '[', '    /')):
                    continue
                print(line)
            reports = {}
            if cpu:
                row = json.loads(path.read_text())
                validate_run(row)
                if all(image['image'] >= 0 for image in row['images']) and len(row['images']) != args.count:
                    raise ValueError('unexpected image timing count')
                if args.warmup and any(image['image'] < 0 for image in row['images']):
                    raise ValueError('warmup is only supported for per-image runs')
                stats = summarize(row['images'], args.warmup)
                row['total_ms'], row['boundary'] = process_wall(start)
                row['model_prepare_ms'] = row['total_ms']-row['boot_prepare_ms']-row['inference_ms']
                validate_totals(row)
                mean = stats['mean_ms']
                key, hybrid = mean.get('keyswitch_ms', 0), mean.get('hybrid_ms', 0)
                reports['CPU'] = category_report('synchronized_cpu_wall', 'fhe_evaluation',
                    key, mean['fhe_eval_ms']-key-hybrid,
                    hybrid, args.warmup, stats['measured_count'])
                reports['CPU'].update(end_to_end_ms=row['total_ms'],
                    prepare_ms=row['model_prepare_ms']+row['boot_prepare_ms'],
                    model_prepare_ms=row['model_prepare_ms'], boot_prepare_ms=row['boot_prepare_ms'],
                    inference_ms=row['inference_ms'], inference_mean_ms=mean['inference_ms'], boundary=row['boundary'])
                if verbose:
                    legacy = dict(row, statistics=stats, image_count=len(row['images']), command=command)
                    legacy.pop('images')
                    print('TIMING ' + json.dumps(legacy, separators=(',', ':')))
            if events:
                from analyze_cuda_events import analyze
                analysis = analyze(csv, log, folder/'event-analysis', args.warmup, args.split_hybrid=='on')
                mean = analysis['mean_ms']
                if analysis['method'] == 'cuda_event_keyswitch_segments_v1':
                    wall = json.loads(path.read_text());validate_run(wall)
                    if [r['image'] for r in wall['images']][args.warmup:] != analysis['measured_images']:
                        raise ValueError('KS Event and wall timing used different images')
                    inference = sum(r['inference_ms'] for r in wall['images'][args.warmup:])/len(analysis['measured_images'])
                    key,hybrid=mean['keyswitch'],mean.get('hybrid',0)
                    reports['CUDA_EVENT'] = category_report(analysis['method'], 'same_run_inference_wall_with_ks_event_segments',
                        key,inference-key-hybrid,hybrid,args.warmup,len(analysis['measured_images']))
                    reports['CUDA_EVENT'].update(profiled_inference_mean_ms=inference,
                        fused_baby_inclusive_ms=mean['fused_baby_inclusive'], boundary=analysis['boundary'])
                else:
                    reports['CUDA_EVENT'] = category_report('cuda_event_exclusive_default_stream',
                        'fhe_default_stream_elapsed', mean['keyswitch'], mean['other'], mean['hybrid'],
                        args.warmup, len(analysis['measured_images']))
            if nvtx:
                from analyze_gpu_activity import analyze
                analysis = analyze(folder/'nvtx.sqlite', folder/'nvtx-analysis', args.warmup, log, args.split_hybrid=='on')
                mean = analysis['mean_ms']
                wall = json.loads(path.read_text())
                validate_run(wall)
                if [r['image'] for r in wall['images']][args.warmup:] != analysis['measured_images']:
                    raise ValueError('NVTX and wall timing used different images')
                if any(r['kernel_overlap_ns'] for r in analysis['images']):
                    raise ValueError('overlapping kernels cannot partition inference elapsed time')
                inference = sum(r['inference_ms'] for r in wall['images'][args.warmup:])/len(analysis['measured_images'])
                key, hybrid = mean['keyswitch_span'], mean.get('hybrid_span', 0)
                reports['NVTX'] = category_report(analysis['method'], 'same_run_inference_wall_with_ks_gpu_spans',
                    key, inference-key-hybrid, hybrid, args.warmup, len(analysis['measured_images']))
                reports['NVTX'].update(profiled_inference_mean_ms=inference,
                    fused_baby_inclusive_ms=mean['fused_baby_inclusive'], policy=analysis['policy'],
                    timestamp_files=analysis['ks_spans']['files'], timing_diagnostic_only=False,
                    kernel_keyswitch_diagnostic_ms=mean.get('kernel_keyswitch', 0),
                    kernel_hybrid_diagnostic_ms=mean.get('kernel_hybrid', 0))
            for name, report in reports.items():
                report.setdefault('policy', 'KS timing follows the declared execution stages; Baby preparation stays fused.')
                report.update(numerical_passed=numerical_passed,diagnostic_only=not numerical_passed)
                report['enabled_methods'] = dict(cpu=cpu, cuda_event=events, nvtx=nvtx)
                print_timing(name, report, args.log_format == 'json')
            (folder/'summary.json').write_text(json.dumps(reports, indent=2)+'\n')
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'FAILED timing: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
