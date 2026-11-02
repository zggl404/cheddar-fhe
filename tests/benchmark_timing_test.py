"""Check baseline pairing gates and durable all-model timing summaries."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_timing import verify_results, write_summary

results=[dict(image=i,status='PASS',prediction=3,fhe_logits=[.1]*10) for i in range(2)]
baseline=dict(key_identity='same',encode_mode='stream',plaintext_reference=False,results=results)
wall={k:baseline[k] for k in ('key_identity','encode_mode','plaintext_reference')}
paired=copy.deepcopy(results);paired[1]['fhe_logits'][0]+=.001
assert abs(verify_results(baseline,wall,paired,2)-.001)<1e-12
for label in ('keys','prediction','nonfinite','large_delta','missing'):
    bad_wall=copy.deepcopy(wall);bad=copy.deepcopy(results)
    if label=='keys':bad_wall['key_identity']='other'
    elif label=='prediction':bad[0]['prediction']=4
    elif label=='nonfinite':bad[0]['fhe_logits'][0]=float('nan')
    elif label=='large_delta':bad[0]['fhe_logits'][0]+=.02
    else:bad.pop()
    try:verify_results(baseline,bad_wall,bad,2)
    except ValueError:pass
    else:raise AssertionError('accepted '+label)
with tempfile.TemporaryDirectory(prefix='cheddar-all-timing-') as folder:
    out=Path(folder)
    row=dict(model='resnet20',status='PASS',encode_mode='pre',temporary_keys_removed=True,nvtx_keyswitch_ms=2.)
    summary=dict(status='RUNNING',models=[row]);write_summary(out,summary)
    assert json.loads((out/'summary.json').read_text())==summary
    rows=list(csv.DictReader((out/'summary.csv').open()))
    assert rows[0]['temporary_keys_removed']=='True' and rows[0]['nvtx_keyswitch_ms']=='2.0'
    summary['models'].append(dict(model='vgg11',status='FAILED',error='test failure',temporary_keys_removed=True))
    summary['status']='FAILED';write_summary(out,summary)
    assert len(list(csv.DictReader((out/'summary.csv').open())))==2
    assert not list(out.glob('*.tmp'))
print('BENCHMARK_TIMING_TEST_PASS pairing gates and partial summaries')
