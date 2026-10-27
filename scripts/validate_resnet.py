#!/usr/bin/env python3
"""Gated GPU-only validation; stops at the first failed stage or model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--cifar', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--reference-json', type=Path)
    args = parser.parse_args()
    if not 1 <= args.count <= 10000:
        parser.error('count must be in [1,10000]')
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    reference = json.loads(args.reference_json.read_text()) if args.reference_json else None
    report = dict(status='running', count=args.count, cells=[],
                  timing_role='validation with streamed GPU weight preparation and per-layer synchronization',
                  binary_sha256=sha(args.binary), library_sha256=sha(args.binary.resolve().parents[1] / 'libcheddar.so'),
                  cifar_sha256=sha(args.cifar))
    report['source_sha256'] = {str(p.relative_to(root)): sha(p) for folder in ('include', 'src', 'example', 'scripts', 'tests', 'parameters')
                              for p in sorted((root / folder).rglob('*')) if p.is_file() and p.suffix in ('.h', '.cpp', '.cu', '.py', '.json', '.txt')}
    report['source_sha256']['CMakeLists.txt'] = sha(root / 'CMakeLists.txt')
    report['hardware'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
    (args.output / 'dependencies.txt').write_text(subprocess.check_output(['ldd', str(args.binary)], text=True))

    def save():
        temporary = args.output / 'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.output / 'report.json')

    def run(name, options, expected_images=0):
        command = [str(args.binary), *map(str, options)]
        cell = dict(name=name, command=command, log=f'{name}.log', images=[])
        report['cells'].append(cell)
        begin = time.monotonic()
        parameters = None
        gpu_only_pass = False
        with (args.output / cell['log']).open('w', buffering=1) as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    log.write(line)
                    if line.startswith('PARAM '):
                        parameters = json.loads(line.removeprefix('PARAM '))
                    if line.startswith('GPU_ONLY_PASS '):
                        gpu_only_pass = True
                    if line.startswith('RESULT '):
                        image = json.loads(line.removeprefix('RESULT '))
                        cell['images'].append(image)
                        print(f"IMAGE cell={name} index={image['image']} error={image['max_abs_error']:.3g} prediction={image['prediction']}", flush=True)
                cell['exit_code'] = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        cell['wall_seconds'] = time.monotonic() - begin
        (args.output / f'{name}.exit').write_text(str(cell['exit_code']) + '\n')
        if cell['exit_code']:
            save()
            raise RuntimeError(f'{name} exited {cell["exit_code"]}')
        if len(cell['images']) != expected_images:
            raise RuntimeError(f'{name}: unexpected result count')
        if not parameters or not gpu_only_pass:
            raise RuntimeError(f'{name}: missing parameter/encoding audit')
        if (parameters['initialization'] != 'BootConfig' or parameters['normal_scale_bits'] != 40
                or parameters['slots'] != 16384 or parameters['boot_output_level'] != 8
                or parameters['default_encryption_level'] != 12 or parameters['stc_start_level'] != 12):
            raise RuntimeError(f'{name}: unexpected parameter initialization or level budget')
        if 'parameters' in report and report['parameters'] != parameters:
            raise RuntimeError(f'{name}: resolved parameters changed between stages')
        report['parameters'] = parameters
        report['parameter_sha256'] = hashlib.sha256(
            json.dumps(parameters, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        for image in cell['images']:
            if image['max_abs_error'] > 1e-2 or image['cpu_encode_attempts'] or image['online_encode_calls']:
                raise RuntimeError(f'{name}: numerical/encoding audit failed')
            if image['mode'] == 'full':
                if image['prediction'] != image['polynomial_top1'] or image['prediction'] != image['relu_top1']:
                    raise RuntimeError(f'{name}: reference top1 mismatch')
                if reference:
                    golden = reference[str(image['depth'])]['images'][image['image']]
                    image['cpp_vs_torch_relu_error'] = max(abs(a-b) for a, b in zip(image['relu_logits'], golden['logits']))
                    image['fhe_vs_torch_relu_error'] = max(abs(a-b) for a, b in zip(image['fhe_logits'], golden['logits']))
                    if image['cpp_vs_torch_relu_error'] > 1e-7 or image['prediction'] != golden['prediction']:
                        raise RuntimeError(f'{name}: independent float reference mismatch')
        cell['status'] = 'PASS'
        save()
        print(f'CELL_PASS {name} seconds={cell["wall_seconds"]:.2f}', flush=True)

    try:
        run('encode', ['--mode', 'encode'])
        run('boot', ['--mode', 'boot'])
        run('relu', ['--mode', 'relu'])
        base = ['--weights', args.models / 'resnet20.bin', '--cifar', args.cifar]
        for mode in ('conv', 'stem', 'blocks'):
            run(mode, [*base, '--mode', mode], 1)
        for stage in (2, 3):
            run(f'projection{stage}', [*base, '--mode', 'projection', '--probe-stage', stage], 1)
        for depth, activation_range in ((20,10), (32,10), (44,10), (56,12), (110,24)):
            run(f'resnet{depth}', ['--mode','full','--weights',args.models/f'resnet{depth}.bin',
                '--cifar',args.cifar,'--count',args.count,'--relu-range',activation_range], args.count)
        report['status'] = 'PASS'
        save()
    except Exception as error:
        report['status'] = 'FAILED'
        report['error'] = str(error)
        save()
        print(f'VALIDATION_FAILED {error}', file=sys.stderr)
        return 1
    print('VALIDATION_PASS', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
