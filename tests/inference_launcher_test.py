"""CPU-only schema, command-default, warmup and process-boundary checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'scripts'))
from inference_timing import IMAGE_FIELDS, LEAF_FIELDS, PROFILE_OPS, summarize, validate_run


def image(index):
    value = 2.0 if index == 0 else 1.0
    row = dict(image=index, schema_version=2, **{k: value for k in LEAF_FIELDS})
    row.update(inference_ms=9.0 * value, fhe_eval_ms=4.0 * value)
    return row


images = [image(0), image(1)]
row = dict(schema_version=2, **{k: sum(i[k] for i in images) for k in IMAGE_FIELDS},
           model_prepare_ms=1.0, boot_prepare_ms=1.0, total_ms=29.0, images=images)
validate_run(row)
for key in ('linear_ms', 'reference_ms', 'dispatch_ms'):
    bad = copy.deepcopy(row);bad[key] += 1
    try: validate_run(bad)
    except ValueError: pass
    else: raise AssertionError(f'accepted broken sum: {key}')
for version in (None, 1):
    bad = copy.deepcopy(row);bad['schema_version'] = version
    try: validate_run(bad)
    except ValueError: pass
    else: raise AssertionError('accepted incompatible schema')
summary = summarize(images, 1)
assert summary['measured_count'] == 1 and summary['mean_ms']['inference_ms'] == 9
disabled = copy.deepcopy(row)
disabled['plaintext_reference'] = False
for item in disabled['images']:
    item['plaintext_reference'] = False
    item['inference_ms'] -= item['reference_ms']
    item['reference_ms'] = 0
disabled['inference_ms'] -= disabled['reference_ms']
disabled['reference_ms'] = 0
disabled['total_ms'] -= row['reference_ms']
validate_run(disabled)
for change in ('nonzero_reference', 'mixed_build'):
    bad = copy.deepcopy(disabled)
    if change == 'nonzero_reference':
        bad['images'][0]['reference_ms'] = 1
        bad['images'][0]['inference_ms'] += 1
    else: bad['images'][0]['plaintext_reference'] = True
    try: validate_run(bad)
    except ValueError: pass
    else: raise AssertionError('accepted invalid disabled-reference timing: ' + change)

with tempfile.TemporaryDirectory(prefix='cheddar-launcher-test-') as temporary:
    folder = Path(temporary)
    binary = folder / 'build/example/resnet_cifar';binary.parent.mkdir(parents=True)
    worker = """import json,os,sys,time
if '--fail' in sys.argv:raise SystemExit(7)
assert os.environ['CHEDDAR_GPU_TRACE']=='0'
assert os.environ['CHEDDAR_CUDA_EVENT_TRACE']=='0'
if '--check-defaults' in sys.argv:
    assert sys.argv[sys.argv.index('--relu-range')+1]=='24'
    assert sys.argv[sys.argv.index('--weights')+1].endswith('/models/resnet110.bin')
    assert sys.argv[sys.argv.index('--encode-mode')+1]=='auto'
print('PARAM {}')
print('RESULT {"fhe_logits":[1,2],"success":true}')
time.sleep(.04)
if sys.argv[sys.argv.index('--time-cpu')+1]=='on':
    report=json.loads(REPORT)
    if '--bad-schema' in sys.argv:report['schema_version']=1
    for r in [report,*report['images']]:
        r.update(ks_profile=True,profile_method='synchronized_scopes',
                 inference_ns=int(r['inference_ms']*1e6),keyswitch_ns=0,hybrid_ns=0,
                 non_keyswitch_ns=int(r['inference_ms']*1e6),keyswitch_ms=0,hybrid_ms=0,
                 non_keyswitch_ms=r['inference_ms'],profile_operations={k:dict(ns=0,calls=0) for k in OPS})
    open(sys.argv[sys.argv.index('--timing-file')+1],'w').write(json.dumps(report))
"""
    binary.write_text('#!' + sys.executable + '\n' + worker.replace('REPORT', repr(json.dumps(row))).replace('OPS', repr(PROFILE_OPS)))
    binary.chmod(0o755)
    command = [sys.executable, str(root/'scripts/run_inference.py'), '--binary', str(binary), '--count', '2', '--warmup', '1', '--log-format', 'json']
    # Inherited variables must not silently turn instrumentation on.
    import os
    environment=dict(os.environ,CHEDDAR_GPU_TRACE='1',CHEDDAR_CUDA_EVENT_TRACE='1')
    result=subprocess.run(command,text=True,capture_output=True,env=environment)
    assert result.returncode==0,result.stderr
    assert result.stdout.splitlines()==['PARAM {}','RESULT {"fhe_logits":[1,2],"success":true}']
    for flag in ('--time-cpu','--ks-profile'):
        result=subprocess.run([*command,flag,'on'],text=True,capture_output=True)
        assert result.returncode==0,result.stderr
        timing=next(json.loads(s[len('CPU_TIMING '):]) for s in result.stdout.splitlines() if s.startswith('CPU_TIMING '))
        assert timing['end_to_end_ms']>30 and timing['warmup_count']==1
        assert timing['total']==timing['KeyswitchTime']+timing['NonKeyswitchTime']+timing['Hybrid']
        assert not any(s.startswith(('CUDA_EVENT_TIMING ','NVTX_TIMING ')) for s in result.stdout.splitlines())
    for option,code in (('--fail',7),('--bad-schema',1)):
        result=subprocess.run([*command,'--time-cpu','on',option,'1'],text=True,capture_output=True)
        assert result.returncode==code and 'CPU_TIMING ' not in result.stdout,result
    result=subprocess.run([*command,'--time-nvtx','on','--nsys',str(folder/'missing')],text=True,capture_output=True)
    assert result.returncode==2 and 'requires Nsight Systems' in result.stderr
    cnn=binary.with_name('cnn_cifar');cnn.write_text(binary.read_text());cnn.chmod(0o755)
    for model,extra in (('resnet110',['--check-defaults','1']),('vgg11',[])):
        result=subprocess.run([sys.executable,str(root/'scripts/run_inference.py'),'--model',model,'--build-dir',str(folder/'build'),'--cifar',str(folder/'test_batch.bin'),'--count','2',*extra],text=True,capture_output=True)
        assert result.returncode==0 and 'TIMING' not in result.stdout,result.stderr

if len(sys.argv)>1:
    for cpu in ('off','on'):
        result=subprocess.run([sys.executable,str(root/'scripts/run_inference.py'),'--binary',sys.argv[1],
            '--count','2','--warmup','1','--time-cpu',cpu,'--log-format','json'],text=True,capture_output=True)
        assert result.returncode==0,result.stderr
        timings=[json.loads(s[len('CPU_TIMING '):]) for s in result.stdout.splitlines() if s.startswith('CPU_TIMING ')]
        assert len(timings)==(cpu=='on')
        if timings:
            assert timings[0]['KeyswitchTime']>0 and timings[0]['Hybrid']>0
    result=subprocess.run([sys.executable,str(root/'scripts/run_inference.py'),'--binary',sys.argv[1],
        '--count','2','--time-cpu','on','--split-hybrid','on','--log-format','json'],text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    timing=next(json.loads(s[len('CPU_TIMING '):]) for s in result.stdout.splitlines() if s.startswith('CPU_TIMING '))
    assert timing['Hybrid']==0  # Shared short-base work is now NonKeyswitch.
    with tempfile.TemporaryDirectory(prefix='cheddar-shared-policy-') as temporary:
        path=Path(temporary)/'timing.json'
        result=subprocess.run([sys.argv[1],'--time-cpu','on','--split-hybrid','on','--timing-file',str(path)],text=True,capture_output=True)
        assert result.returncode==0,result.stderr
        raw=json.loads(path.read_text());validate_run(raw)
        assert raw['hybrid_ns']==0 and raw['profile_operations']['shared_lift_common']['ns']>0
        raw['profile_operations']['shared_lift']['calls']=1
        try:validate_run(raw)
        except ValueError as error:assert 'unsplit hybrid operation' in str(error)
        else:raise AssertionError('accepted full SharedLift in split CPU report')
print('INFERENCE_LAUNCHER_TEST_PASS')
