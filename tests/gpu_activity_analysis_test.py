"""Small launch-correlation test: nested owners, delayed GPU work, exclusions and rejection."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_gpu_activity import analyze, classify, union_ns

assert union_ns([(1, 5), (3, 7), (10, 12)]) == 8
assert classify({'op':'fused_baby','source':'linear_transform'}) == 'hybrid'
assert classify({'op':'key_correction','source':'rotation'}) == 'keyswitch'
for labels in ({'op': 'unknown'}, {'op': 'modup'}):
    try:
        classify(labels)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted an unknown operation or missing source')

with tempfile.TemporaryDirectory(prefix='cheddar-activity-test-') as temporary:
    folder = Path(temporary)
    database = folder / 'trace.sqlite'
    db = sqlite3.connect(database)
    db.executescript('''
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        INSERT INTO StringIds VALUES(1,'test_kernel');
        INSERT INTO StringIds VALUES(2,'cudaLaunchKernel_v7000');
        INSERT INTO StringIds VALUES(3,'cudaMemcpyAsync_v3020');
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER DEFAULT 2);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,contextId INTEGER,streamId INTEGER,deviceId INTEGER,demangledName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,correlationId INTEGER,contextId INTEGER,streamId INTEGER,deviceId INTEGER);
    ''')
    scopes = [(-1,2100,'run'), (0,2000,'image:0'), (10,1000,'phase:linear_ms'), (20,900,'source:linear_transform'),
              (30,200,'op:key_mult'), (40,100,'op:modup'), (50,90,'source:rotation'),
              (110,140,'op:key_mult'), (290,310,'op:moddown_rescale'),
              (1050,1900,'phase:validation_output_ms')]
    db.executemany('INSERT INTO NVTX_EVENTS VALUES(?,?,7,?,NULL)',
                   [(s,e,'cheddar:'+name) for s,e,name in scopes])
    calls = [(60,61,1), (120,121,2), (250,251,3), (300,301,4), (1100,1101,5), (70,71,6)]
    db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(start,end,globalTid,correlationId) VALUES(?,?,7,?)', calls)
    db.execute('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET nameId=3 WHERE correlationId=6')
    # GPU work runs after its CPU NVTX owners have returned.
    kernels = [(1000,1010,1), (1010,1030,2), (1030,1035,3), (1035,1045,4), (1102,1110,5)]
    db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,1,1,0,1)', kernels)
    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(1045,1048,6,1,1,0)')
    db.commit()
    log = folder/'worker.log'
    log.write_text('GPU_TRACE_AUDIT '+json.dumps({'image': 0, 'ranges': len(scopes)-1})+'\n')
    result = analyze(database, folder/'valid', 0, log)
    row = result['images'][0]
    assert row['kernel_total_ns'] == 45 and row['kernel_keyswitch_ns'] == 30
    assert row['kernel_hybrid_ns'] == 10 and row['kernel_other_ns'] == 5
    assert row['memcpy_keyswitch_ns'] == 3 and result['accounting_error_ns'] == 0
    assert result['sources']['linear_transform']['keyswitch'] == 30/1e6
    assert result['records']['non_fhe_records'] == 1
    assert result['nvtx_range_audit']['verified'] and result['measured_kernel_launches'] == 4
    assert json.loads((folder/'valid/summary.json').read_text())['images'] == result['images']
    db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=1')
    db.commit()
    try:
        analyze(database, folder/'invalid', 0)
    except ValueError as error:
        assert 'uncorrelated' in str(error)
    else:
        raise AssertionError('accepted uncorrelated GPU work')
    db.close()
print('GPU_ACTIVITY_ANALYSIS_TEST_PASS')
