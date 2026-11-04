"""Profiled-only batch export: first image, measured means, warn and partial failure."""
import copy
import csv
from decimal import Decimal
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from inference_timing import IMAGE_FIELDS
from run_inference import MODELS
from summarize_nvtx import summarize


def write_json(path, value):
    path.write_text(json.dumps(value))


def fixture(source, name, failed=False):
    folder = source/name/'profiled'
    analysis = folder/'analysis'
    analysis.mkdir(parents=True)
    image_times, ks_times = [30,10,12,14,16,18], [6,2,3,4,5,6]
    images = []
    for image, duration in enumerate(image_times):
        row = {key: 0. for key in IMAGE_FIELDS}
        row.update(schema_version=2, image=image, inference_ms=duration, dispatch_ms=duration,
                   ks_profile=False, split_hybrid=True, plaintext_reference=False)
        images.append(row)
    wall = dict(images[0]);wall.pop('image')
    wall.update(inference_ms=100., dispatch_ms=100., images=images, timing_detail='coarse',
                total_ms=150., model_prepare_ms=40., boot_prepare_ms=10., key_mode='load',
                key_load_ms=20., weight_preencode_ms=10., encode_mode='pre',
                keys_generated=0, keys_loaded=2, key_count=2, numerical_gate='warn',
                numerical_passed=not failed, numerical_failures=3 if failed else 0)
    write_json(folder/'wall.json', wall)
    spans = {str(i):dict(keyswitch_span_ns=k*1000000) for i,k in enumerate(ks_times)}
    write_json(analysis/'summary.json', dict(method='nsys_keyswitch_segment_gpu_span_v1', split_hybrid=True,
               ks_spans=dict(coverage_verified=True, images=spans), measured_images=[1,2,3,4,5],
               images=[dict(image=i,kernel_overlap_ns=0) for i in range(1,6)],
               mean_ms=dict(keyswitch_span=4)))
    with (analysis/'ks_segments.csv').open('w',newline='') as handle, gzip.open(analysis/'gpu_activities.csv.gz','wt',newline='') as gpu:
        writer=csv.writer(handle);activities=csv.writer(gpu)
        writer.writerow(['image','measured','segment_id','ks_id','gpu_start_ns','gpu_end_ns','duration_ns','layer','kind','segment'])
        activities.writerow(['image','measured','kind','start_ns','end_ns'])
        for i,(duration,k) in enumerate(zip(image_times,ks_times)):
            base=(100+i*100)*1000000;flag=int(i>0)
            writer.writerow([i,flag,1,1,base+2000000,base+(2+k)*1000000,k*1000000,'conv1','multkey','core'])
            for a,b in [(0,1),(2,2+k),(duration-2,duration-1)]:
                activities.writerow([i,flag,'kernel',base+a*1000000,base+b*1000000])
    return wall


with tempfile.TemporaryDirectory(prefix='cheddar-profiled-report-') as temporary:
    root=Path(temporary);source=root/'batch';source.mkdir()
    models=[]
    for name in MODELS:
        failed=name=='vgg11'
        fixture(source,name,failed)
        models.append(dict(model=name,status='NUMERICAL_FAILED' if failed else 'PASS',diagnostic_only=failed,
                           baseline=dict(end_to_end_ms=9999999)))
    batch=dict(status='NUMERICAL_FAILED',count=6,warmup=1,split_hybrid='on',models=models)
    write_json(source/'summary.json',batch)
    original=(source/'summary.json').read_bytes()
    records=summarize(source,root/'report')
    assert len(records)==len(MODELS) and not any(r.get('error') for r in records)
    assert sorted(p.name for p in (root/'report').iterdir())==sorted(['00_summary.csv',*MODELS])
    rows=list(csv.DictReader((root/'report/00_summary.csv').open()))
    assert len(rows)==len(MODELS)
    expected_fields = ['model','end_to_end_ms','key_load_ms','weight_preencode_ms',
                       'init_and_teardown_ms','inference_total_ms','inference_mean_ms',
                       'keyswitch_mean_ms','non_keyswitch_mean_ms',
                       *[f'image_{i}_{kind}_ms' for i in range(6) for kind in ('keyswitch','non_keyswitch')],
                       'status','error']
    assert list(rows[0])==expected_fields
    for row in rows:
        assert float(row['end_to_end_ms'])==150
        assert float(row['key_load_ms'])==20 and float(row['weight_preencode_ms'])==10
        assert Decimal(row['init_and_teardown_ms'])==20
        inference = [Decimal(row[f'image_{i}_keyswitch_ms'])+Decimal(row[f'image_{i}_non_keyswitch_ms']) for i in range(6)]
        assert inference==[30,10,12,14,16,18]
        assert sum(inference)==Decimal(row['inference_total_ms'])
        assert sum(inference[1:])/5==Decimal(row['inference_mean_ms'])
        assert Decimal(row['keyswitch_mean_ms'])==4 and Decimal(row['non_keyswitch_mean_ms'])==10
        assert Decimal(row['inference_mean_ms'])==Decimal(row['keyswitch_mean_ms'])+Decimal(row['non_keyswitch_mean_ms'])
        assert Decimal(row['end_to_end_ms'])==sum(Decimal(row[k]) for k in
               ('key_load_ms','weight_preencode_ms','inference_total_ms','init_and_teardown_ms'))
        assert sorted(p.name for p in (root/'report'/row['model']).iterdir())==['summary.json','timeline.csv','timeline.txt']
    warning=next(r for r in rows if r['model']=='vgg11')
    assert warning['status']=='NUMERICAL_FAILED' and warning['error']==''
    assert (source/'summary.json').read_bytes()==original
    command=[sys.executable,'-B',str(ROOT/'scripts/summarize_nvtx.py'),'--input',str(source)]
    cli=subprocess.run([*command,'--output',str(root/'cli')],text=True,capture_output=True)
    assert cli.returncode==0,(cli.stdout,cli.stderr)  # Finite numerical warnings retain the report.
    duplicate=subprocess.run([*command,'--output',str(root/'cli')],text=True,capture_output=True)
    assert duplicate.returncode==2 and 'already exists' in duplicate.stderr
    nested=subprocess.run([*command,'--output',str(source/'report')],text=True,capture_output=True)
    assert nested.returncode==2 and 'separate' in nested.stderr

    broken=copy.deepcopy(batch)
    broken['models'][0].update(status='FAILED',error='worker failed')
    write_json(source/'summary.json',broken)
    (source/'alexnet/profiled/wall.json').unlink()
    partial=subprocess.run([*command,'--output',str(root/'partial')],text=True,capture_output=True)
    assert partial.returncode==1,(partial.stdout,partial.stderr)
    failures=list(csv.DictReader((root/'partial/00_summary.csv').open()))
    assert failures[0]['status']=='FAILED' and failures[0]['end_to_end_ms']==''
    assert failures[-1]['status']=='FAILED' and failures[-1]['error']
    assert Decimal(failures[1]['end_to_end_ms'])==150
    assert not (root/'partial'/MODELS[0]).exists() and not (root/'partial/alexnet').exists()

    # A different measured-image census must never silently produce a mean.
    write_json(source/'summary.json',batch)
    analysis_path=source/'resnet20/profiled/analysis/summary.json'
    analysis=json.loads(analysis_path.read_text());analysis['measured_images']=[0,1,2,3,4]
    write_json(analysis_path,analysis)
    mismatch=summarize(source,root/'mismatch')
    assert mismatch[0]['status']=='FAILED' and 'different measured images' in mismatch[0]['error']

    precise=root/'precise';precise.mkdir()
    wall=fixture(precise,'resnet20')
    for key in ('inference_ms','dispatch_ms'):
        wall['images'][0][key]=30.000001
        wall[key]=100.000001
    wall['total_ms']=150.000001
    write_json(precise/'resnet20/profiled/wall.json',wall)
    write_json(precise/'summary.json',dict(batch,models=[models[0]],status='PASS'))
    precise_row=summarize(precise,root/'precise-report')[0]
    assert precise_row['image_0_non_keyswitch_ms']==Decimal('24.000001')
    assert precise_row['end_to_end_ms']==sum(precise_row[k] for k in
           ('key_load_ms','weight_preencode_ms','inference_total_ms','init_and_teardown_ms'))
print('SUMMARIZE_NVTX_TEST_PASS compact columns, per-image ns accounting, measured means, warn and failures')
