"""CPU-only schema, command-default, warmup and process-boundary checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'scripts'))
from inference_timing import IMAGE_FIELDS, LEAF_FIELDS, summarize, validate_run


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

with tempfile.TemporaryDirectory(prefix='cheddar-launcher-test-') as temporary:
    folder = Path(temporary)
    binary = folder / 'build/example/resnet_cifar';binary.parent.mkdir(parents=True)
    binary.write_text('#!' + sys.executable + '\n' + '''import json,sys,time
if '--fail' in sys.argv:raise SystemExit(7)
path=sys.argv[sys.argv.index('--timing-file')+1]
open(path+'.args','w').write(json.dumps(sys.argv))
report=json.loads(''' + repr(json.dumps(row)) + ''')
if '--bad-schema' in sys.argv:report['schema_version']=1
if '--check-defaults' in sys.argv:
    assert sys.argv[sys.argv.index('--relu-range')+1]=='24'
    assert sys.argv[sys.argv.index('--weights')+1].endswith('/models/resnet110.bin')
    assert sys.argv[sys.argv.index('--encode-mode')+1]=='auto'
time.sleep(.04)
open(path,'w').write(json.dumps(report))
print('worker output')
''')
    binary.chmod(0o755)
    command = [sys.executable, str(root/'scripts/run_inference.py'), '--binary', str(binary), '--count', '2', '--warmup', '1', '--log-format', 'json']
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(s[7:]) for s in result.stdout.splitlines() if s.startswith('TIMING ')]
    assert len(rows) == 1
    actual = rows[0];actual['images'] = images;validate_run(actual)
    assert actual['total_ms'] > 30 and actual['statistics']['warmup_count'] == 1
    assert 'online_ms' not in actual and 'process_overhead_ms' not in actual
    for option, code in (('--fail', 7), ('--bad-schema', 1)):
        result = subprocess.run([*command, option, '1'], text=True, capture_output=True)
        assert result.returncode == code and 'TIMING ' not in result.stdout, result
    # Ordinary model selection needs no binary path or numerical tuning flags.
    cnn = binary.with_name('cnn_cifar');cnn.write_text(binary.read_text());cnn.chmod(0o755)
    for model, extra in (('resnet110', ['--check-defaults', '1']), ('vgg11', [])):
        result = subprocess.run([sys.executable, str(root/'scripts/run_inference.py'), '--model', model, '--build-dir', str(folder/'build'), '--cifar', str(folder/'test_batch.bin'), '--count', '2', *extra], text=True, capture_output=True)
        assert result.returncode == 0 and 'FHE evaluation' in result.stdout, result.stderr

if len(sys.argv) > 1:
    # Optional integration with the real C++ RunReport CPU worker.
    result = subprocess.run([sys.executable, str(root/'scripts/run_inference.py'), '--binary', sys.argv[1], '--count', '2', '--warmup', '1', '--log-format', 'json'], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    records = [json.loads(s[13:]) for s in result.stdout.splitlines() if s.startswith('IMAGE_TIMING ')]
    total = next(json.loads(s[7:]) for s in result.stdout.splitlines() if s.startswith('TIMING '))
    total['images'] = records;validate_run(total)
    assert len(records) == 2 and records[0]['bootstrap_ms'] > 0 and records[0]['activation_ms'] > 0
    result = subprocess.run([sys.executable, str(root/'scripts/run_inference.py'), '--binary', sys.argv[1], '--count', '2', '--warmup', '1', '--ks-profile', 'on', '--log-format', 'json'], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    records = [json.loads(s[13:]) for s in result.stdout.splitlines() if s.startswith('IMAGE_TIMING ')]
    total = next(json.loads(s[7:]) for s in result.stdout.splitlines() if s.startswith('TIMING '))
    total['images'] = records;validate_run(total)
    assert total['keyswitch_ns'] > 0 and total['hybrid_ns'] > 0
    assert total['inference_ns'] == sum(r['inference_ns'] for r in records)
    split_result = subprocess.run([sys.executable, str(root/'scripts/run_inference.py'), '--binary', sys.argv[1], '--count', '2', '--ks-profile', 'on', '--split-hybrid', 'on', '--log-format', 'json'], text=True, capture_output=True)
    assert split_result.returncode == 0, split_result.stderr
    split_total = next(json.loads(s[7:]) for s in split_result.stdout.splitlines() if s.startswith('TIMING '))
    assert split_total['split_hybrid'] and split_total['hybrid_ns'] == 0
    bad = copy.deepcopy(total);bad['keyswitch_ns'] += 1
    try: validate_run(bad)
    except ValueError: pass
    else: raise AssertionError('accepted broken profiling sum')
print('INFERENCE_LAUNCHER_TEST_PASS')
