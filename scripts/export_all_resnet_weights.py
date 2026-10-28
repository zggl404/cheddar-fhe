#!/usr/bin/env python3
"""Export all five supported CIFAR ResNets from a directory of folded ONNX files."""
import argparse
from pathlib import Path

from export_resnet_weights import export


def export_all(source, output):
    models = [(depth, source / f'resnet{depth}_cifar10_train.onnx')
              for depth in (20, 32, 44, 56, 110)]
    missing = [str(path) for _, path in models if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing ONNX files:\n' + '\n'.join(missing))
    for depth, path in models:
        export(path, output / f'resnet{depth}.bin', depth)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='directory containing resnet{depth}_cifar10_train.onnx')
    parser.add_argument('--output', type=Path, default=Path('models'), help='output directory (default: ./models)')
    args = parser.parse_args()
    try:
        export_all(args.source, args.output)
    except (OSError, ValueError, ImportError, AssertionError) as error:
        parser.exit(1, f'Export failed: {error}\n')


if __name__ == '__main__':
    main()
