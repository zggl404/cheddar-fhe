#!/usr/bin/env python3
"""Independent CPU float64 replay of the exported source ONNX graph."""
import argparse
import hashlib
import json
from pathlib import Path


def reference(source, cifar, start, count):
    import numpy as np
    import onnx
    from onnx import numpy_helper
    import torch
    import torch.nn.functional as functional
    torch.set_num_threads(8)
    graph=onnx.load(str(source)).graph
    arrays={v.name:numpy_helper.to_array(v) for v in graph.initializer}
    parameters={name:torch.from_numpy(array.copy()).to(torch.float64) for name,array in arrays.items()}
    records=np.frombuffer(cifar.read_bytes(),dtype=np.uint8).reshape(-1,3073)
    mean=torch.tensor([.485,.456,.406],dtype=torch.float64).reshape(1,3,1,1)
    stdev=torch.tensor([.229,.224,.225],dtype=torch.float64).reshape(1,3,1,1)
    images=[]
    for image in range(start,start+count):
        values=dict(parameters)
        values[graph.input[0].name]=(torch.from_numpy(records[image,1:].copy()).to(torch.float64).reshape(1,3,32,32)/255-mean)/stdev
        bounds={}
        with torch.no_grad():
            for node in graph.node:
                inputs=[values[name] for name in node.input]
                attrs={v.name:onnx.helper.get_attribute_value(v) for v in node.attribute}
                op=node.op_type
                if op=='Conv':
                    pads=attrs.get('pads',[0]*4)
                    if pads[0]!=pads[2] or pads[1]!=pads[3]:raise ValueError('asymmetric padding')
                    out=functional.conv2d(inputs[0],inputs[1],inputs[2],attrs.get('strides',[1,1]),pads[:2],attrs.get('dilations',[1,1]),attrs.get('group',1))
                elif op=='Relu':
                    bounds[node.name]=inputs[0].abs().max().item();out=torch.relu(inputs[0])
                elif op=='AveragePool':
                    out=functional.avg_pool2d(inputs[0],attrs['kernel_shape'],attrs.get('strides',attrs['kernel_shape']),
                                             attrs.get('pads',[0]*4)[:2],bool(attrs.get('ceil_mode',0)),bool(attrs.get('count_include_pad',0)))
                elif op=='GlobalAveragePool':out=inputs[0].mean(dim=(2,3),keepdim=True)
                elif op=='Concat':out=torch.cat(inputs,dim=attrs['axis'])
                elif op=='Reshape':out=inputs[0].reshape(tuple(int(x) for x in arrays[node.input[1]].reshape(-1)))
                elif op=='Gemm':
                    a=inputs[0].T if attrs.get('transA',0) else inputs[0]
                    b=inputs[1].T if attrs.get('transB',0) else inputs[1]
                    out=attrs.get('alpha',1)*(a@b)+attrs.get('beta',1)*inputs[2]
                else:raise ValueError(f'unsupported source operator: {op}')
                values[node.output[0]]=out
        logits=values[graph.output[0].name].reshape(-1).tolist()
        images.append(dict(image=image,label=int(records[image,0]),prediction=max(range(len(logits)),key=logits.__getitem__),
                           logits=logits,activation_bounds=bounds,max_activation=max(bounds.values())))
        print(f"REFERENCE image={image} prediction={images[-1]['prediction']} max_activation={images[-1]['max_activation']:.9g}",flush=True)
    return dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cifar_sha256=hashlib.sha256(cifar.read_bytes()).hexdigest(),
                dtype='torch.float64',device='cpu',images=images)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--cifar',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--count',type=int,default=11)
    args=parser.parse_args()
    if args.start<0 or args.count<=0 or args.start+args.count>10000:parser.error('invalid image range')
    result=reference(args.source,args.cifar,args.start,args.count)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
