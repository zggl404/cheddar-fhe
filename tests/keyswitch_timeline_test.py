"""Timestamp complements: overlapping/interleaved KS, window edges and warmup."""
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from keyswitch_timeline import generate, partition


def write_csv(path, fields, values):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(values)


with tempfile.TemporaryDirectory(prefix='cheddar-ks-timeline-') as temporary:
    root = Path(temporary)
    segments, bounds = root/'ks_segments.csv', root/'bounds.csv'
    fields = ['image', 'measured', 'segment_id', 'ks_id', 'gpu_start_ns', 'gpu_end_ns', 'duration_ns']
    values = [(1,1,3,1,180,200,20), (0,0,1,1,10,20,10), (1,1,2,2,150,160,10),
              (1,1,1,1,100,130,30), (1,1,4,3,120,140,20)]
    write_csv(segments, fields, values)
    write_csv(bounds, ['image','start_ns','end_ns','measured'],
              [(0,0,40,0),(1,90,220,1),(2,300,400,1)])
    summary = generate(segments, root/'full', bounds_path=bounds)
    timeline = list(csv.DictReader((root/'full/timeline.csv').open()))
    actual = [(r['category'],int(r['start_ns']),int(r['end_ns'])) for r in timeline if r['image']=='1']
    assert actual == [('NonKS',90,100),('KS',100,140),('NonKS',140,150),('KS',150,160),
                      ('NonKS',160,180),('KS',180,200),('NonKS',200,220)]
    merged = next(r for r in timeline if r['image']=='1' and r['category']=='KS')
    assert merged['segment_ids']=='1;4' and merged['ks_ids']=='1;3'
    assert float(merged['start_from_t0_us'])==.01
    assert summary['images'][1]['keyswitch_ns']==70
    assert summary['images'][1]['non_keyswitch_ns']==60
    assert summary['images'][1]['merged_overlap_ns']==10
    assert summary['images'][2]['non_keyswitch_ns']==100
    assert summary['measured_images']==[1,2] and summary['warmup_images']==[0]
    assert summary['measured_totals_ns']==dict(window_ns=230,keyswitch_ns=70,non_keyswitch_ns=160)
    assert summary['mean_ms']['keyswitch']==35/1e6
    assert summary['complete_inference_boundaries']
    assert json.loads((root/'full/summary.json').read_text())==summary
    text=(root/'full/timeline.txt').read_text()
    assert 'T0 = 90 ns' in text and 'T1 = 220 ns' in text and '[KS_0]' in text
    for image in range(3):
        rr=[r for r in timeline if int(r['image'])==image]
        assert all(a['end_ns']==b['start_ns'] for a,b in zip(rr,rr[1:]))
        assert all(int(r['end_ns'])>int(r['start_ns']) for r in rr)

    activities = root/'gpu_activities.csv.gz'
    with gzip.open(activities,'wt',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['image','measured','start_ns','end_ns'])
        writer.writerows([(-1,0,1,2),(0,0,5,25),(1,1,80,85),(1,1,215,230)])
    activity_summary=generate(segments,root/'activity',activities_path=activities)
    assert activity_summary['scope']=='gpu_activity_envelope'
    assert not activity_summary['complete_inference_boundaries']
    assert activity_summary['images'][1]['window_ns']==150
    assert activity_summary['images'][1]['non_keyswitch_ns']==80
    partial=generate(segments,root/'partial')
    assert partial['scope']=='ks_envelope' and partial['images'][1]['window_ns']==100
    assert partial['images'][1]['non_keyswitch_ns']==30
    adjacent, _ = partition(1,[(10,20,{'segment_id':'1'}),(20,30,{'segment_id':'2'})],
                             (10,30),1,'inference')
    assert [r['category'] for r in adjacent]==['KS','KS']

    def reject(fragment, function):
        try:function()
        except ValueError as error:assert fragment in str(error),str(error)
        else:raise AssertionError('accepted invalid input: '+fragment)

    bad=root/'bad.csv'
    write_csv(bad,['image','start_ns','end_ns'],[(0,0,40),(1,110,220)])
    reject('outside',lambda:generate(segments,root/'bad-bounds',bounds_path=bad))
    write_csv(bad,fields,values+[values[0]])
    reject('duplicate',lambda:generate(bad,root/'duplicate'))
    write_csv(bad,fields,[(1,1,1,1,100,90,10)])
    reject('invalid interval',lambda:generate(bad,root/'reversed'))
    write_csv(bad,fields,[(1,1,1,1,100,130,20)])
    reject('duration_ns',lambda:generate(bad,root/'duration'))
    write_csv(bad,fields,[(1,1,1,1,100,130,30),(1,0,2,2,150,160,10)])
    reject('measured flag',lambda:generate(bad,root/'flags'))
    write_csv(bad,['image','ks_id','gpu_start_ns','gpu_end_ns'],[(1,1,100,200)])
    reject('required columns',lambda:generate(bad,root/'units'))
    write_csv(bad,['image','start_ns','end_ns'],[(0,0,40)])
    reject('missing boundaries',lambda:generate(segments,root/'missing',bounds_path=bad))
    unsafe=root/'timeline.csv'
    unsafe.write_bytes(segments.read_bytes())
    reject('overwrite',lambda:generate(unsafe,root))
    assert unsafe.read_bytes()==segments.read_bytes()
    command=[sys.executable,'-B',str(Path(__file__).resolve().parents[1]/'scripts/keyswitch_timeline.py'),
             str(segments),'--activities',str(activities),'--output',str(root/'cli')]
    completed=subprocess.run(command,check=True,capture_output=True,text=True)
    assert json.loads(completed.stdout)['scope']=='gpu_activity_envelope'
print('KEYSWITCH_TIMELINE_TEST_PASS complement, overlaps, interleaved IDs, boundaries, warmup and CLI')
