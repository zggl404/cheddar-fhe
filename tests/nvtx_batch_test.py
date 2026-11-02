"""Exercise all-model pairing, attribution, key ownership and partial failures without a GPU."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_nvtx import pair_result
from run_inference import MODELS

WORKER=r'''
import json,sys
from pathlib import Path
args=dict(zip(sys.argv[1::2],sys.argv[2::2]))
kind=Path(args['--weights']).read_text()
if kind=='FAIL':raise SystemExit(7)
mode=args['--key-mode'];keys=Path(args['--key-dir'])
failed=kind=='NUMERICAL' or (kind=='SETUP_NUMERICAL' and mode=='save')
if failed and args['--numerical-gate']=='strict':raise SystemExit(1)
if mode=='save':
    assert not keys.exists();keys.mkdir();(keys/'manifest.txt').write_text('mock bundle')
else:assert (keys/'manifest.txt').is_file()
count=int(args['--count']);pre=args['--encode-mode']!='stream'
rows=[]
for image in range(count):
    rows.append(dict(image=image,schema_version=2,inference_ms=10.,fhe_eval_ms=0.,input_ms=0.,runtime_weight_prepare_ms=0.,
        linear_ms=0.,bootstrap_ms=0.,activation_ms=0.,graph_ops_ms=0.,reference_ms=0.,validation_output_ms=0.,dispatch_ms=10.,
        ks_profile=False,split_hybrid=args['--split-hybrid']=='on',plaintext_reference=False))
    logits=[1.]*10
    if kind=='NONFINITE':logits[0]=float('nan')
    print('RESULT '+json.dumps(dict(status='NUMERICAL_FAILED' if failed else 'PASS',success=not failed,
        numerical_passed=not failed,numerical_failures=int(failed),
        model=Path(args['--weights']).stem,image=image,prediction=0,fhe_logits=logits)))
    if failed:print('NUMERICAL_WARNING test.relu image='+str(image)+' max_abs_error=0.02 threshold=0.01',file=sys.stderr)
    if args['--time-nvtx']=='on':print('GPU_TRACE_AUDIT '+json.dumps(dict(image=image,ranges=5)))
report=dict(rows[0]);report.pop('image');report.update(inference_ms=10.*count,dispatch_ms=10.*count,
    images=rows,total_ms=40.+10*count,model_prepare_ms=35.,boot_prepare_ms=5.,timing_detail='coarse',
    key_mode=mode,key_load_ms=4. if mode=='load' else 0.,weight_preencode_ms=8. if pre else 0.,
    encode_mode='pre' if pre else 'stream',keys_generated=5 if mode=='save' else 0,
    numerical_gate=args['--numerical-gate'],numerical_passed=not failed,numerical_failures=count*int(failed),
    keys_loaded=5 if mode=='load' else 0,key_count=5,key_identity='same-test-key-identity')
Path(args['--timing-file']).write_text(json.dumps(report))
'''
TIME=r'''
import subprocess,sys
from pathlib import Path
code=subprocess.call(sys.argv[4:])
Path(sys.argv[3]).write_text('real 0.09\nuser 0.02\nsys 0.01\n')
raise SystemExit(code)
'''
NSYS=r'''
import sqlite3,subprocess,sys
from pathlib import Path
start=next(i for i,a in enumerate(sys.argv) if a.startswith('--output='))
path=Path(sys.argv[start].split('=',1)[1]+'.sqlite')
command=sys.argv[start+1:];code=subprocess.call(command)
if code:raise SystemExit(code)
if Path(command[command.index('--weights')+1]).read_text()=='BADSQL':
    path.write_text('invalid sqlite');raise SystemExit(0)
count=int(command[command.index('--count')+1]);split=command[command.index('--split-hybrid')+1]=='on'
db=sqlite3.connect(path)
db.executescript(''' + '"""' + '''
CREATE TABLE StringIds(id INTEGER,value TEXT);
INSERT INTO StringIds VALUES(1,'test_kernel'),(2,'cudaLaunchKernel_v7000');
CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,contextId INTEGER,streamId INTEGER,deviceId INTEGER,demangledName INTEGER);
''' + '"""' + ''')
u=1000000
scopes=[(900*u,(1000+200*count)*u,'cheddar:run')]
for image in range(count):
    base=1000+200*image
    for a,b,label in [(0,100,'image:'+str(image)),(5,90,'phase:linear_ms'),(10,80,'source:linear_transform'),
                       (20,30,'op:key_mult'),(40,50,'op:'+('key_correction' if split else 'moddown_rescale'))]:
        scopes.append(((base+a)*u,(base+b)*u,'cheddar:'+label))
    for a,b,c,k,l in [(21,22,1,92,95),(41,42,2,97,98),(60,61,3,95,97)]:
        corr=image*10+c
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,7,?,2)',((base+a)*u,(base+b)*u,corr))
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,1,1,0,1)',((base+k)*u,(base+l)*u,corr))
db.executemany('INSERT INTO NVTX_EVENTS VALUES(?,?,7,?,NULL)',scopes);db.commit();db.close()
'''

def executable(path,code):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('#!'+sys.executable+'\n'+textwrap.dedent(code));path.chmod(0o755)

with tempfile.TemporaryDirectory(prefix='cheddar-nvtx-batch-test-') as temporary:
    folder=Path(temporary);build=folder/'build'
    for name in ('resnet_cifar','cnn_cifar'):executable(build/'example'/name,WORKER)
    executable(folder/'time',TIME);executable(folder/'nsys',NSYS)
    models=folder/'models';models.mkdir()
    for model in MODELS:(models/(model+'.bin')).write_text('ok')
    cifar=folder/'test_batch.bin';cifar.write_bytes(b'fixture')
    keys=folder/'keys';(keys/'resnet20').mkdir(parents=True)
    (keys/'resnet20/manifest.txt').write_text('existing');(keys/'resnet20/keep').write_text('never remove')
    base=[sys.executable,str(ROOT/'scripts/benchmark_nvtx.py'),'--build-dir',str(build),'--models-dir',str(models),
          '--cifar',str(cifar),'--key-root',str(keys),'--time-binary',str(folder/'time'),'--nsys',str(folder/'nsys'),
          '--count','2','--warmup','1']
    result=subprocess.run([*base,'--output',str(folder/'all')],text=True,capture_output=True)
    assert result.returncode==0,(result.stdout,result.stderr)
    summary=json.loads((folder/'all/summary.json').read_text());rows=summary['models']
    assert len(rows)==len(MODELS) and all(r['status']=='PASS' for r in rows)
    assert summary['split_hybrid']=='on' and all(r['split_hybrid']=='on' for r in rows)
    assert not rows[0]['keys_prepared_by_script'] and not rows[0]['temporary_keys_removed']
    assert all(r['keys_prepared_by_script'] and r['temporary_keys_removed'] for r in rows[1:])
    assert sorted(p.name for p in keys.iterdir())==['resnet20']
    (models/'vgg11.bin').write_text('BADSQL')
    result=subprocess.run([*base,'--output',str(folder/'bad-sql'),'--models','vgg11','resnet20'],text=True,capture_output=True)
    assert result.returncode==1,result.stderr
    failed=json.loads((folder/'bad-sql/summary.json').read_text())['models']
    assert [r['status'] for r in failed]==['FAILED','PASS'] and failed[0]['temporary_keys_removed']
    assert (keys/'resnet20/keep').read_text()=='never remove'
    for row in rows:
        assert row['nvtx_keyswitch_ms']==4 and row['nvtx_hybrid_ms']==0 and row['nvtx_other_kernel_ms']==2
        assert row['non_keyswitch_residual_ms']==6 and row['measured_count']==row['warmup_count']==1
        assert row['baseline']['time_real_ms']==90 and row['baseline']['key_load_ms']==4
    assert len(list(csv.DictReader((folder/'all/summary.csv').open())))==len(MODELS)
    result=subprocess.run([*base,'--output',str(folder/'fused'),'--models','resnet20','--split-hybrid','off'],text=True,capture_output=True)
    assert result.returncode==0,(result.stdout,result.stderr)
    fused=json.loads((folder/'fused/summary.json').read_text())['models'][0]
    assert fused['split_hybrid']=='off' and fused['nvtx_hybrid_ms']==1
    assert fused['nvtx_keyswitch_ms']==3 and fused['non_keyswitch_residual_ms']==7
    (models/'vgg11.bin').write_text('NUMERICAL')
    result=subprocess.run([*base,'--output',str(folder/'numerical-warn'),'--models','vgg11','resnet20','--numerical-gate','warn'],text=True,capture_output=True)
    assert result.returncode==1,(result.stdout,result.stderr)
    warned=json.loads((folder/'numerical-warn/summary.json').read_text())
    assert warned['status']=='NUMERICAL_FAILED'
    first,second=warned['models']
    assert first['status']=='NUMERICAL_FAILED' and second['status']=='PASS'
    assert first['diagnostic_only'] and not first['numerical_passed'] and first['numerical_failures']==5
    assert first['temporary_keys_removed'] and len(first['profiled']['results'])==2
    assert first['profiled']['inference_total_ms']==20 and first['nvtx_keyswitch_ms']==4
    assert 'NUMERICAL_WARNING' in (folder/'numerical-warn/vgg11/profiled/stderr.log').read_text()
    result=subprocess.run([*base,'--output',str(folder/'numerical-strict'),'--models','vgg11','resnet20'],text=True,capture_output=True)
    assert result.returncode==1
    strict=json.loads((folder/'numerical-strict/summary.json').read_text())['models']
    assert [r['status'] for r in strict]==['FAILED','PASS'] and strict[0]['temporary_keys_removed']
    (models/'vgg11.bin').write_text('SETUP_NUMERICAL')
    result=subprocess.run([*base,'--output',str(folder/'setup-warn'),'--models','vgg11','--numerical-gate','warn'],text=True,capture_output=True)
    assert result.returncode==1,result.stderr
    setup=json.loads((folder/'setup-warn/summary.json').read_text())['models'][0]
    assert setup['status']=='NUMERICAL_FAILED' and setup['numerical_failures']==1
    assert setup['baseline']['numerical_passed'] and setup['profiled']['numerical_passed']
    (models/'vgg11.bin').write_text('NONFINITE')
    result=subprocess.run([*base,'--output',str(folder/'nonfinite'),'--models','vgg11','--numerical-gate','warn'],text=True,capture_output=True)
    assert result.returncode==1
    fatal=json.loads((folder/'nonfinite/summary.json').read_text())['models'][0]
    assert fatal['status']=='FAILED' and 'nonfinite' in fatal['error'] and fatal['temporary_keys_removed']
    analysis=json.loads((folder/'all/resnet20/profiled/analysis/summary.json').read_text())
    bad=copy.deepcopy(rows[0]['profiled']);bad['results'][1]['fhe_logits'][0]+=.02;bad['results'][1]['prediction']=1
    try:pair_result(copy.deepcopy(rows[0]['baseline']),copy.deepcopy(bad),analysis,1)
    except ValueError as error:assert 'paired predictions' in str(error)
    else:raise AssertionError('strict mode accepted paired numerical mismatch')
    mismatch=pair_result(copy.deepcopy(rows[0]['baseline']),bad,analysis,1,'warn')
    assert mismatch['diagnostic_only'] and not mismatch['paired_numerical_passed'] and not mismatch['paired_predictions_match']
    (models/'vgg11.bin').write_text('FAIL')
    result=subprocess.run([*base,'--output',str(folder/'failed'),'--models','vgg11','resnet20'],text=True,capture_output=True)
    assert result.returncode==1
    failed=json.loads((folder/'failed/summary.json').read_text())['models']
    assert [r['status'] for r in failed]==['FAILED','PASS'] and failed[0]['temporary_keys_removed']
    assert sorted(p.name for p in keys.iterdir())==['resnet20']
    result=subprocess.run([*base,'--output',str(folder/'all')],text=True,capture_output=True)
    assert result.returncode==2  # Existing reports must not be overwritten.
    analysis=dict(measured_images=[1],images=[dict(kernel_overlap_ns=1)],mean_ms={})
    try:pair_result(copy.deepcopy(rows[0]['baseline']),copy.deepcopy(rows[0]['profiled']),analysis,1)
    except ValueError as error:assert 'overlapping' in str(error)
    else:raise AssertionError('accepted overlapping kernels')
    bad=copy.deepcopy(rows[0]['profiled']);bad['image_times'][1]['inference_ms']=1
    analysis=json.loads((folder/'all/resnet20/profiled/analysis/summary.json').read_text())
    try:pair_result(copy.deepcopy(rows[0]['baseline']),bad,analysis,1)
    except ValueError as error:assert 'Inference minus' in str(error)
    else:raise AssertionError('accepted negative wall remainder')
print('NVTX_BATCH_TEST_PASS all models, pairing, key cleanup/preservation and failure handling')
