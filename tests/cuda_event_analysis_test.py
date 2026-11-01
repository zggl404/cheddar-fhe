"""Check exclusive event attribution and reject repeated/overlapping intervals."""
import csv
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_cuda_events import analyze

with tempfile.TemporaryDirectory(prefix='cheddar-event-analysis-') as folder:
    root=Path(folder)
    rows=[(0,0,0,1,'linear_ms','linear_transform','shared_modup','modup',2),
          (0,1,1,2,'linear_ms','linear_transform','giant','moddown_rescale',3),
          (0,2,3,4,'graph_ops_ms','none','none','other',1)]
    def write(values):
        with (root/'events.csv').open('w',newline='') as handle:
            w=csv.writer(handle);w.writerow(['image','segment','start_event','end_event','phase','source','part','operation','milliseconds']);w.writerows(values)
    write(rows)
    (root/'worker.log').write_text('CUDA_EVENT_IMAGE '+json.dumps(dict(image=0,events=5,segments=3,sum_ms=6,readback_ms=.1))+'\n')
    result=analyze(root/'events.csv',root/'worker.log',root/'valid',0)
    assert result['mean_ms']==dict(keyswitch=2,hybrid=3,other=1,total=6)
    assert result['sources']['linear_transform']['keyswitch']==2
    try:analyze(root/'events.csv',root/'worker.log',root/'unsplit',0,True)
    except ValueError:pass
    else:raise AssertionError('accepted ModDownRescale in split event report')
    semantic=[(*rows[0][:7],'fused_baby',2),
              (*rows[1][:5],'sparse_to_dense','none','shared_lift_common',3),rows[2]]
    write(semantic)
    result=analyze(root/'events.csv',root/'worker.log',root/'semantic',0,True)
    assert result['mean_ms']==dict(keyswitch=2,hybrid=0,other=4,total=6)
    assert result['sources']['sparse_to_dense']['other']==3
    for bad in [rows+[rows[-1]], [rows[0],(0,1,0,1,*rows[1][4:]),rows[2]]]:
        write(bad)
        try:analyze(root/'events.csv',root/'worker.log',root/'invalid',0)
        except ValueError:pass
        else:raise AssertionError('accepted duplicate/overlapping event intervals')
print('CUDA_EVENT_ANALYSIS_TEST_PASS')
