#!/usr/bin/env python3
"""Export AE .npy weights or fixed CIFAR ResNet ONNX weights; no FHE compiler."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


def export(source, output, depth):
    import numpy as np
    assert depth in (20, 32, 44, 56, 110)
    blocks = (depth - 2) // 6
    tensors = {}
    mapping = {}
    if source.is_dir():
        if depth != 20:
            raise ValueError('AE weight directory is ResNet20 only')
        mean, stdev = [.4914, .4822, .4465], [.2023, .1994, .2010]
        names = [('stem', 'conv1_reparam'), ('fc', 'linear')]
        for stage in (1, 2, 3):
            for block in range(blocks):
                prefix = f'layer{stage}.{block}'
                names += [(f'{prefix}.conv{i}', f'{prefix}.conv{i}_reparam') for i in (1, 2)]
                if stage > 1 and block == 0:
                    names.append((f'{prefix}.shortcut', f'{prefix}.shortcut_reparam'))
        for target, origin in names:
            for suffix in ('weight', 'bias'):
                key = f'{target}.{suffix}'
                tensors[key] = np.load(source / f'{origin}.{suffix}', allow_pickle=False)
                mapping[key] = f'{origin}.{suffix}'
    else:
        import onnx
        from onnx import numpy_helper
        model = onnx.load(str(source))
        initializers = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
        mean, stdev = [.485, .456, .406], [.229, .224, .225]
        adds = [0, 0, 0]
        for node in model.graph.node:
            if node.op_type == 'BatchNormalization':
                raise ValueError('export requires BN-folded inference weights')
            if node.op_type == 'Add':
                match = re.fullmatch(r'/layer([123])/layer\1\.(\d+)/Add', node.name)
                if match:
                    adds[int(match[1])-1] += 1
            if node.op_type not in ('Conv', 'Gemm'):
                continue
            if node.name == '/conv1/Conv':
                key = 'stem'
            elif node.name == '/fc/Gemm':
                key = 'fc'
                attrs = {x.name: onnx.helper.get_attribute_value(x) for x in node.attribute}
                if attrs.get('transB', 0) != 1 or attrs.get('alpha', 1) != 1 or attrs.get('beta', 1) != 1:
                    raise ValueError('unsupported classifier Gemm attributes')
            else:
                match = re.fullmatch(r'/layer([123])/layer\1\.(\d+)/(conv[12]|downsample/downsample\.0)/Conv', node.name)
                if not match:
                    raise ValueError(f'unsupported Conv name: {node.name}')
                stage, block, operation = match.groups()
                key = f'layer{stage}.{block}.' + ('shortcut' if operation.startswith('downsample') else operation)
            if len(node.input) != 3:
                raise ValueError('expected folded convolution/classifier bias')
            for suffix, name in zip(('weight', 'bias'), node.input[1:]):
                tensors[f'{key}.{suffix}'] = initializers[name]
                mapping[f'{key}.{suffix}'] = name
        if adds != [blocks]*3:
            raise ValueError(f'depth mismatch: block counts {adds}')
    expected = 2*(1 + 6*blocks + 2 + 1)
    if len(tensors) != expected:
        raise ValueError(f'expected {expected} tensors, got {len(tensors)}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        stream.write(b'CHDRRN01')
        stream.write(struct.pack('<III6d', 1, depth, len(tensors), *mean, *stdev))
        for name, array in sorted(tensors.items()):
            array = np.asarray(array, dtype='<f4', order='C')
            if not np.isfinite(array).all():
                raise ValueError(f'nonfinite tensor: {name}')
            encoded = name.encode()
            stream.write(struct.pack('<II', len(encoded), array.ndim))
            stream.write(encoded)
            stream.write(struct.pack(f'<{array.ndim}I', *array.shape))
            stream.write(array.tobytes())
    manifest = dict(depth=depth, source=str(source.resolve()), mean=mean, stdev=stdev,
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    tensors={k: dict(source=mapping[k], shape=list(v.shape)) for k, v in sorted(tensors.items())})
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'exported depth={depth} tensors={len(tensors)} sha256={manifest["sha256"]} path={output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--depth', type=int, default=20)
    args = parser.parse_args()
    export(args.source, args.output, args.depth)


if __name__ == '__main__':
    main()
