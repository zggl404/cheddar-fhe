#!/usr/bin/env python3
"""Run one CIFAR-10 FHE model with unified, additive wall-clock timing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from inference_timing import print_human, summarize, validate_run, validate_totals

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
          'vgg11', 'vgg16', 'mobilenet', 'squeezenet', 'alexnet')


def process_wall(fallback_start):
    # Linux process birth includes Python startup; resolution is one clock tick.
    try:
        fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        born = int(fields[19]) / os.sysconf('SC_CLK_TCK')
        return ((time.clock_gettime(time.CLOCK_BOOTTIME) - born) * 1000,
                'launcher process wall including child startup and teardown')
    except (OSError, ValueError, AttributeError, IndexError):
        return ((time.perf_counter() - fallback_start) * 1000,
                'Python main through child teardown; interpreter startup excluded')


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument('--model', choices=MODELS)
    entry.add_argument('--binary', type=Path, help='advanced: direct C++ entry, including probes')
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build')
    parser.add_argument('--weights', type=Path, help='default: models/MODEL.bin')
    parser.add_argument('--cifar', type=Path)
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=0, help='included in count and end-to-end time')
    parser.add_argument('--encode-mode', choices=('auto', 'pre', 'stream'), default='auto')
    parser.add_argument('--log-format', choices=('human', 'json'), default='human')
    args, options = parser.parse_known_args()
    if not 0 <= args.warmup < args.count <= 10000:
        parser.error('require 0 <= warmup < count <= 10000')
    if any(x.split('=', 1)[0] in ('--timing-file',) for x in options):
        parser.error('timing-file is managed by this launcher')
    if args.model and not args.cifar:
        parser.error('--model requires --cifar')
    binary = args.binary or args.build_dir / 'example' / ('resnet_cifar' if args.model.startswith('resnet') else 'cnn_cifar')
    binary = binary.expanduser().resolve()
    weights = args.weights or (ROOT / 'models' / f'{args.model}.bin' if args.model else None)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f'missing executable: {binary}; build both examples first')
    if weights:
        options += ['--weights', str(weights.expanduser().resolve())]
    if args.cifar:
        options += ['--cifar', str(args.cifar.expanduser().resolve())]
    if args.model and '--relu-range' not in options:
        options += ['--relu-range', {'resnet56': '12', 'resnet110': '24'}.get(args.model, '10')]
    options += ['--count', str(args.count), '--encode-mode', args.encode_mode, '--log-format', args.log_format]
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(binary.parents[1]) + ':' + env.get('LD_LIBRARY_PATH', '')
    try:
        with tempfile.TemporaryDirectory(prefix='cheddar-timing-') as temporary:
            path = Path(temporary) / 'timing.json'
            code = subprocess.call([str(binary), *options, '--timing-file', str(path)], env=env)
            if code:
                return code if code > 0 else 128 - code
            row = json.loads(path.read_text())
            validate_run(row)
        if all(image['image'] >= 0 for image in row['images']) and len(row['images']) != args.count:
            raise ValueError('unexpected image timing count')
        if args.warmup and any(image['image'] < 0 for image in row['images']):
            raise ValueError('warmup is only supported for per-image runs')
        row['image_count'] = len(row['images'])
        row['statistics'] = summarize(row['images'], args.warmup)
        row['command'] = [str(binary), *options]
        row.pop('images')  # Release per-image reports before the final timestamp in both formats.
        # Include report parsing, validation and statistics before taking the
        # final timestamp. Only the fixed-size final summary remains outside.
        row['total_ms'], row['boundary'] = process_wall(start)
        row['model_prepare_ms'] = row['total_ms'] - row['boot_prepare_ms'] - row['inference_ms']
        validate_totals(row)
        if args.log_format == 'json':
            print('TIMING ' + json.dumps(row, separators=(',', ':')))
        else:
            print_human(row)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'FAILED timing: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
