"""Aligned event metadata, event census, independent pairing and mismatch rejection."""
import csv
import json
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_cuda_events import analyze
from compare_ks_timings import compare
from inference_timing import LEAF_FIELDS

with tempfile.TemporaryDirectory(prefix='cheddar-ks-alignment-') as temporary:
    root=Path(temporary);event=root/'event';nvtx=root/'nvtx'
    event.mkdir();nvtx.mkdir();(nvtx/'nvtx-analysis').mkdir()
    rows=[];logs=[];nvtx_rows=[]
    for image in range(2):
        specs=[(1,1,'multkey','core',1,0,False,1.,3.),(2,2,'baby','prepare',8,0,True,3.5,6.5),
               (2,3,'baby','finish',8,3,True,7.,8.),(4,4,'shared_finish','finish',0,5,False,9.,9.5)]
        for ordinal,(unit,segment,kind,part,logical,lt,mixed,start,end) in enumerate(specs):
            meta=dict(ks_id=unit,segment_id=segment,kind=kind,segment=part,logical_ks=logical,
                      lt_id=lt,baby_id=2 if kind!='multkey' else 0,includes_permutation=mixed)
            rows.append(dict(image=image,start_event=2*ordinal+1,end_event=2*ordinal+2,
                start_relative_ms=start,end_relative_ms=end,milliseconds=end-start,layer='conv,"quoted"',
                subpath='spatial_baby' if kind=='baby' else '',metadata=json.dumps(meta)))
            nvtx_rows.append(dict(image=image,measured=int(image>0),layer=rows[-1]['layer'],subpath=rows[-1]['subpath'],
                **meta,duration_ms=.9*(end-start)))
        logs.append('CUDA_EVENT_KS_IMAGE '+json.dumps(dict(image=image,events=9,segments=4,sum_ms=6.5,readback_ms=.1)))
        logs.append('RESULT '+json.dumps(dict(image=image,status='PASS',prediction=3,fhe_logits=[.1]*10)))
    def write(values,path=event/'events.csv'):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    write(rows);(event/'worker.log').write_text('\n'.join(logs)+'\n')
    result=analyze(event/'events.csv',event/'worker.log',event/'event-analysis',1,True)
    assert result['method']=='cuda_event_keyswitch_segments_v1' and result['mean_ms']['keyswitch']==6.5
    assert all(r['logical_ks']==9 and r['execution_units']==2 for r in result['images'])
    assert len(list(csv.DictReader((event/'event-analysis/ks_segments.csv').open())))==8
    for name,change in [('duplicate',lambda r:r.update(start_event=1)),
                        ('missing_id',lambda r:r.update(metadata=json.dumps({**json.loads(r['metadata']),'segment_id':7}))),
                        ('hybrid',lambda r:r.update(metadata=json.dumps({**json.loads(r['metadata']),'category':'hybrid'}))),
                        ('time',lambda r:r.update(end_relative_ms=3.))]:
        bad=[dict(r) for r in rows];change(bad[1]);write(bad)
        try:analyze(event/'events.csv',event/'worker.log',root/name,1,True)
        except ValueError:pass
        else:raise AssertionError('accepted '+name)
    write(rows)
    unsplit=analyze(event/'events.csv',event/'worker.log',root/'unsplit',1,False)
    assert not unsplit['split_hybrid'] and unsplit['mean_ms']['hybrid']==0
    write(nvtx_rows,nvtx/'nvtx-analysis/ks_segments.csv')
    (nvtx/'worker.log').write_text('\n'.join(s for s in logs if s.startswith('RESULT '))+'\n')
    images=[dict(schema_version=2,image=i,**{k:0. for k in LEAF_FIELDS},inference_ms=10.,fhe_eval_ms=0.,
        ks_profile=False,split_hybrid=True,plaintext_reference=False) for i in range(2)]
    for r in images:r['dispatch_ms']=10.
    wall=dict(images[0]);wall.pop('image');wall.update(images=images,inference_ms=20.,dispatch_ms=20.,
        model_prepare_ms=20.,boot_prepare_ms=0.,total_ms=40.,key_mode='load',key_identity='same',
        key_count=2,keys_loaded=2,keys_generated=0,encode_mode='pre')
    for path,name,method,total,enabled in ((event,'CUDA_EVENT','cuda_event_keyswitch_segments_v1',6.5,dict(cpu=False,cuda_event=True,nvtx=False)),
                                          (nvtx,'NVTX','nsys_keyswitch_segment_gpu_span_v1',5.85,dict(cpu=False,cuda_event=False,nvtx=True))):
        (path/'cpu.json').write_text(json.dumps(wall))
        (path/'summary.json').write_text(json.dumps({name:dict(method=method,KeyswitchTime=total,
            numerical_passed=True,enabled_methods=enabled,warmup_count=1,measured_count=1)}))
    paired=compare(event,nvtx,root/'paired')
    assert paired['matched_segments']==8 and paired['matched_measured_segments']==4
    assert math.isclose(paired['delta_ms'],.65) and paired['identities_match']
    report_path=event/'summary.json';original=report_path.read_text()
    combined=json.loads(original);combined['CUDA_EVENT']['enabled_methods']['nvtx']=True
    report_path.write_text(json.dumps(combined))
    try:compare(event,nvtx,root/'combined')
    except ValueError as e:assert 'independent' in str(e)
    else:raise AssertionError('accepted combined profiling as independent')
    report_path.write_text(original)
    wall_path=event/'cpu.json';original=wall_path.read_text();wrong=json.loads(original);wrong['key_identity']='different'
    wall_path.write_text(json.dumps(wrong))
    try:compare(event,nvtx,root/'different-keys')
    except ValueError as e:assert 'key_identity' in str(e)
    else:raise AssertionError('accepted different keys')
    wall_path.write_text(original)
    changed=[dict(r) for r in nvtx_rows];changed[0]['subpath']='wrong';write(changed,nvtx/'nvtx-analysis/ks_segments.csv')
    try:compare(event,nvtx,root/'mismatch')
    except ValueError as e:assert 'metadata mismatch' in str(e)
    else:raise AssertionError('accepted mismatched segment metadata')
    mixed_rows=[dict(r) for r in rows]
    mixed_nvtx=[dict(r,category='keyswitch') for r in nvtx_rows]
    for r in mixed_rows:
        meta=json.loads(r['metadata'])
        if meta['segment_id']==3:meta['category']='hybrid'
        r['metadata']=json.dumps(meta)
    for r in mixed_nvtx:
        if r['segment_id']==3:r['category']='hybrid'
    write(mixed_rows);write(mixed_nvtx,nvtx/'nvtx-analysis/ks_segments.csv')
    mixed=analyze(event/'events.csv',event/'worker.log',event/'event-analysis',1,False)
    assert mixed['mean_ms']['keyswitch']==5.5 and mixed['mean_ms']['hybrid']==1
    for path,name,key,hybrid in ((event,'CUDA_EVENT',5.5,1),(nvtx,'NVTX',4.95,.9)):
        value=json.loads((path/'cpu.json').read_text());value['split_hybrid']=False
        for image in value['images']:image['split_hybrid']=False
        (path/'cpu.json').write_text(json.dumps(value))
        value=json.loads((path/'summary.json').read_text());value[name].update(KeyswitchTime=key,Hybrid=hybrid)
        (path/'summary.json').write_text(json.dumps(value))
    mixed_pair=compare(event,nvtx,root/'mixed-paired')
    assert not mixed_pair['split_hybrid'] and mixed_pair['matched_segments']==8
    assert math.isclose(mixed_pair['event_hybrid_mean_ms'],1) and math.isclose(mixed_pair['hybrid_delta_ms'],.1)
    value=json.loads((nvtx/'cpu.json').read_text());value['split_hybrid']=True
    for image in value['images']:image['split_hybrid']=True
    (nvtx/'cpu.json').write_text(json.dumps(value))
    try:compare(event,nvtx,root/'different-modes')
    except ValueError as error:assert 'split_hybrid' in str(error)
    else:raise AssertionError('accepted different execution modes')
print('KS_EVENT_ALIGNMENT_TEST_PASS census, shared batches, pairing and rejection')
