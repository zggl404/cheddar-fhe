"""GPU timestamps, launch gaps, shared tails, coverage and unsafe attribution."""
import csv
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_gpu_activity import analyze

with tempfile.TemporaryDirectory(prefix='cheddar-ks-segments-') as temporary:
    root = Path(temporary)
    path = root/'trace.sqlite'
    db = sqlite3.connect(path)
    db.executescript('''
CREATE TABLE StringIds(id INTEGER,value TEXT);
INSERT INTO StringIds VALUES(1,'kernel'),(2,'cudaLaunchKernel_v7000');
CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,contextId INTEGER,streamId INTEGER,deviceId INTEGER,demangledName INTEGER);
''')
    def scope(a, b, label):
        db.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,7,?,NULL)', (a,b,'cheddar:'+label))
    scope(1,4000,'run')
    for image in range(2):
        base, gpu = image*2000, 10000+image*2000
        for a,b,label in [(100,1000,'image:'+str(image)),(120,950,'phase:linear_ms'),
                          (130,940,'layer:conv1'),(140,930,'source:linear_transform'),
                          (185,225,'op:key_mult'),(240,280,'op:moddown'),(310,330,'op:automorphism'),
                          (355,380,'op:modup'),(400,440,'op:fused_baby'),(555,595,'op:moddown'),
                          (623,640,'op:moddown'),(650,675,'op:key_mult'),(690,710,'op:automorphism'),
                          (725,755,'op:moddown')]:
            scope(base+a,base+b,label)
        specs = [(180,300,1,1,'multkey','core',1,0,False),
                 (350,450,2,2,'baby','prepare',8,0,True),
                 (550,600,2,3,'baby','finish',8,77,True),
                 (620,680,3,4,'giant','core',1,78,False),
                 (720,760,4,5,'shared_finish','finish',0,78,False)]
        for a,b,unit,segment,kind,name,logical,lt,mixed in specs:
            meta=dict(ks_id=unit,segment_id=segment,kind=kind,segment=name,logical_ks=logical,
                      lt_id=lt,baby_id=2 if kind!='multkey' else 0,includes_permutation=mixed)
            scope(base+a,base+b,'ks:'+json.dumps(meta))
        launches = [(190,0,10),(210,20,30),(250,40,50),(320,60,65),
                    (360,70,80),(410,100,120),(510,130,140),(560,150,155),
                    (580,160,165),(630,170,180),(660,190,200),(700,210,215),
                    (730,220,225),(750,230,235)]
        for index,(cpu,start,end) in enumerate(launches):
            corr=image*100+index
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,7,?,2)',(base+cpu,base+cpu+1,corr))
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,1,1,0,1)',(gpu+start,gpu+end,corr))
    db.commit()
    result=analyze(path,root/'valid',1,split_hybrid=True)
    assert result['method']=='nsys_keyswitch_segment_gpu_span_v1'
    assert result['measured_images']==[1]
    assert result['mean_ms']['keyswitch_span']==160/1e6
    assert result['mean_ms']['kernel_keyswitch']==100/1e6
    assert result['mean_ms']['kernel_other']==20/1e6
    assert result['images'][0]['logical_ks']==10 and result['images'][0]['execution_units']==3
    rows=list(csv.DictReader((root/'valid/ks_segments.csv').open()))
    assert len(rows)==10 and rows[0]['gpu_start_ns']=='10000' and rows[0]['gpu_end_ns']=='10050'
    assert float(rows[0]['start_relative_us'])==0 and float(rows[0]['end_relative_us'])==.05
    units=list(csv.DictReader((root/'valid/ks_units.csv').open()))
    baby=[r for r in units if r['image']=='1' and r['kind']=='baby'][0]
    assert baby['logical_ks']=='8' and baby['execution_units']=='1' and baby['segment_count']=='2'
    assert float(baby['duration_ms'])==65/1e6
    assert int(baby['gpu_end_ns'])-int(baby['gpu_start_ns'])==95
    layers=list(csv.DictReader((root/'valid/ks_layers.csv').open()))
    assert len(layers)==2 and math.isclose(float(layers[1]['keyswitch_ms']),160/1e6)

    def reject(name, sql, error):
        target=root/(name+'.sqlite')
        bad=sqlite3.connect(target);db.backup(bad);bad.executescript(sql);bad.close()
        try:analyze(target,root/name,1,split_hybrid=True)
        except ValueError as exc:assert error in str(exc),(name,str(exc))
        else:raise AssertionError('accepted '+name)
    reject('missing',"DELETE FROM NVTX_EVENTS WHERE text LIKE 'cheddar:ks:%'",'missing KS segment')
    reject('unowned',"DELETE FROM NVTX_EVENTS WHERE text LIKE '%multkey%'",'unowned KS/Hybrid kernel')
    reject('duplicate',"INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS WHERE text LIKE '%multkey%'",'nested KS segments')
    reject('concurrent',"UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET streamId=2 WHERE correlationId=100",'one GPU kernel stream')
    reject('foreign',"UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=12015,end=12018 WHERE correlationId=106",'unrelated GPU work')
    reject('wrong_category',"UPDATE NVTX_EVENTS SET text='cheddar:op:automorphism' WHERE text='cheddar:op:key_mult'",'category disagrees')
    # Fused final b/a reductions share one continuous Hybrid stage, including its internal gap.
    fused_path=root/'fused.sqlite'
    fused=sqlite3.connect(fused_path);db.backup(fused)
    fused.execute("UPDATE NVTX_EVENTS SET text='cheddar:op:moddown_rescale' WHERE text='cheddar:op:moddown' AND start IN (555,2555)")
    fused.execute("UPDATE NVTX_EVENTS SET text='cheddar:op:automorphism_accum' WHERE text='cheddar:op:automorphism' AND start IN (310,2310)")
    for image in range(2):
        base=image*2000
        for unit,segment,kind,a,b in [(5,6,'automorphism_accum',305,340),(6,7,'shared_lift',500,530)]:
            meta=dict(ks_id=unit,segment_id=segment,kind=kind,segment='core',logical_ks=0,
                      lt_id=0,baby_id=0,includes_permutation=False,category='hybrid')
            fused.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,7,?,NULL)',(base+a,base+b,'cheddar:ks:'+json.dumps(meta)))
        fused.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,7,?,NULL)',(base+502,base+528,'cheddar:op:shared_lift'))
    for rowid,text in fused.execute("SELECT rowid,text FROM NVTX_EVENTS WHERE text LIKE 'cheddar:ks:%'").fetchall():
        meta=json.loads(text.split(':',2)[2])
        if meta['segment_id']==3:
            meta['category']='hybrid'
            fused.execute('UPDATE NVTX_EVENTS SET text=? WHERE rowid=?',('cheddar:ks:'+json.dumps(meta),rowid))
    fused.commit()
    mixed=analyze(fused_path,root/'fused',1,split_hybrid=False)
    assert mixed['method']==result['method'] and not mixed['timing_diagnostic_only']
    assert mixed['mean_ms']['keyswitch_span']==145/1e6 and mixed['mean_ms']['hybrid_span']==30/1e6
    assert mixed['mean_ms']['kernel_hybrid']==25/1e6  # The additional 5 ns is inside the finish stage.
    assert mixed['images'][0]['logical_ks']==10 and mixed['images'][0]['execution_units']==3
    assert mixed['images'][0]['hybrid_segments']==3
    fused_units=list(csv.DictReader((root/'fused/ks_units.csv').open()))
    baby=next(r for r in fused_units if r['image']=='1' and r['kind']=='baby')
    assert float(baby['keyswitch_ms'])==50/1e6 and float(baby['hybrid_ms'])==15/1e6
    assert baby['logical_ks']=='8' and baby['segment_count']=='2'
    fused.execute("DELETE FROM NVTX_EVENTS WHERE text LIKE '%\"category\": \"hybrid\"%'");fused.commit()
    try:analyze(fused_path,root/'fused-missing',1,split_hybrid=False)
    except ValueError as error:assert 'unowned KS/Hybrid kernel' in str(error)
    else:raise AssertionError('accepted unmarked fused Hybrid work')
    fused.close()
    db.close()
print('KEYSWITCH_SEGMENTS_TEST_PASS timestamps, gaps, shared tails, warmup and coverage rejection')
