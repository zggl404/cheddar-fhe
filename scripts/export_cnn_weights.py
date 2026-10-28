#!/usr/bin/env python3
"""Export a checked, BN-folded CIFAR CNN operator list for direct Cheddar execution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct


def export(source, output, family):
    import numpy as np
    import onnx
    from onnx import numpy_helper
    graph = onnx.shape_inference.infer_shapes(onnx.load(str(source))).graph
    initializers = {x.name: numpy_helper.to_array(x) for x in graph.initializer}
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in [*graph.input, *graph.output, *graph.value_info]}
    if shapes[graph.input[0].name] != [1, 3, 32, 32] or shapes[graph.output[0].name] != [1, 10]:
        raise ValueError('expected CIFAR input [1,3,32,32] and logits [1,10]')
    expected = {
        'vgg11': dict(Conv=8, Relu=8, AveragePool=5, Reshape=1, Gemm=1),
        'vgg16': dict(Conv=13, Relu=13, AveragePool=5, Reshape=1, Gemm=1),
        'mobilenet': dict(Conv=27, Relu=27, AveragePool=1, Reshape=1, Gemm=1),
        'squeezenet': dict(Conv=26, Relu=18, AveragePool=4, GlobalAveragePool=1, Concat=8, Reshape=1, Gemm=1),
        'alexnet': dict(Conv=5, Relu=7, AveragePool=3, Reshape=1, Gemm=3),
    }
    if dict(Counter(n.op_type for n in graph.node)) != expected[family]:
        raise ValueError('operator list does not match the selected BN-folded CIFAR model')
    ids = {graph.input[0].name: 0}
    tensors, nodes = {}, []
    kinds = dict(Conv=1, Relu=2, AveragePool=3, GlobalAveragePool=4, Concat=5, Gemm=6, Reshape=7)
    for node in graph.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        shape = shapes[node.output[0]]
        if len(shape) == 4 and shape[0] == 1 and shape[2] == shape[3]:
            channels, width = shape[1:3]
        elif len(shape) == 2 and shape[0] == 1:
            channels, width = shape[1], 1
        else:
            raise ValueError(f'unsupported shape at {node.name}: {shape}')
        if width > 32 or width <= 0 or width & (width-1):
            raise ValueError(f'unsupported spatial layout at {node.name}')
        kernel, stride, padding, groups = 1, 1, 0, 1
        weight = bias = ''
        if node.op_type in ('Conv', 'Gemm'):
            if len(node.input) != 3:
                raise ValueError(f'folded weight and bias required at {node.name}')
            weight, bias = node.input[1:]
            for name in (weight, bias):
                tensors[name] = np.asarray(initializers[name], dtype='<f4', order='C')
                if not np.isfinite(tensors[name]).all():
                    raise ValueError(f'nonfinite tensor: {name}')
        if node.op_type in ('Conv', 'AveragePool'):
            k = attrs.get('kernel_shape', [1, 1]); s = attrs.get('strides', [1, 1]); p = attrs.get('pads', [0]*4)
            if len(set(k)) != 1 or len(set(s)) != 1 or len(set(p)) != 1 or attrs.get('dilations', [1,1]) != [1,1]:
                raise ValueError(f'unsupported convolution/pool geometry at {node.name}')
            kernel, stride, padding, groups = k[0], s[0], p[0], attrs.get('group', 1)
            if attrs.get('auto_pad', b'NOTSET') not in (b'NOTSET', b''):
                raise ValueError('explicit padding required')
            if node.op_type == 'AveragePool' and (padding or attrs.get('ceil_mode', 0)):
                raise ValueError('pool currently requires zero padding and ceil_mode=0')
        if node.op_type == 'Gemm' and (attrs.get('transB',0) != 1 or attrs.get('transA',0) or
                                       attrs.get('alpha',1) != 1 or attrs.get('beta',1) != 1):
            raise ValueError('unsupported Gemm attributes')
        if node.op_type == 'Concat' and attrs.get('axis') != 1:
            raise ValueError('Concat must use channel axis')
        if node.op_type == 'Reshape':
            before = shapes[node.input[0]]
            if shape != [1, int(np.prod(before[1:]))]:
                raise ValueError('only a CHW flatten view is supported')
        inputs = [ids[name] for name in node.input if name not in initializers]
        nodes.append(dict(kind=kinds[node.op_type], name=node.name, inputs=inputs,
                          shape=[channels,width], geometry=[kernel,stride,padding,groups], weight=weight,bias=bias))
        ids[node.output[0]] = len(nodes)
    output.parent.mkdir(parents=True, exist_ok=True)
    def string(stream, value):
        encoded = value.encode(); stream.write(struct.pack('<I',len(encoded)));stream.write(encoded)
    with output.open('wb') as stream:
        stream.write(b'CHDRCN01');stream.write(struct.pack('<I',1));string(stream,family)
        stream.write(struct.pack('<6d',.485,.456,.406,.229,.224,.225))
        stream.write(struct.pack('<I',len(tensors)))
        for name, array in sorted(tensors.items()):
            string(stream,name);stream.write(struct.pack('<I',array.ndim))
            stream.write(struct.pack(f'<{array.ndim}I',*array.shape));stream.write(array.tobytes())
        stream.write(struct.pack('<I',len(nodes)))
        for n in nodes:
            stream.write(struct.pack('<I',n['kind']));string(stream,n['name'])
            stream.write(struct.pack('<I',len(n['inputs'])));stream.write(struct.pack(f"<{len(n['inputs'])}I",*n['inputs']))
            stream.write(struct.pack('<6I',*n['shape'],*n['geometry']));string(stream,n['weight']);string(stream,n['bias'])
    manifest = dict(family=family,source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),nodes=nodes,
                    tensors={name:list(value.shape) for name,value in tensors.items()})
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f"exported {family} nodes={len(nodes)} sha256={manifest['sha256']}")


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--family',required=True,choices=('vgg11','vgg16','mobilenet','squeezenet','alexnet'))
    args=parser.parse_args();export(args.source,args.output,args.family)
