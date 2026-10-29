"""Exercise the external timing boundary and failure propagation without CUDA."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='cheddar-launcher-test-') as temporary:
    binary=Path(temporary)/'build/example/worker'
    binary.parent.mkdir(parents=True)
    binary.write_text('#!'+sys.executable+'\n'+'''import json,sys,time
if '--fail' in sys.argv:raise SystemExit(7)
time.sleep(.01)
path=sys.argv[sys.argv.index('--timing-file')+1]
open(path,'w').write(json.dumps(dict(model_prepare_ms=1,boot_prepare_ms=1,inference_ms=2,total_ms=4)))
print('worker output')
''')
    binary.chmod(0o755)
    command=[sys.executable,str(root/'scripts/run_inference.py'),'--binary',str(binary),'--log-format','json']
    process=subprocess.run(command,text=True,capture_output=True)
    assert process.returncode==0,process.stderr
    timings=[json.loads(line[7:]) for line in process.stdout.splitlines() if line.startswith('TIMING ')]
    assert len(timings)==1
    row=timings[0]
    assert abs(row['model_prepare_ms']+row['boot_prepare_ms']+row['inference_ms']-row['total_ms'])<1e-6
    assert row['total_ms']>10 and row['process_overhead_ms']>0
    failed=subprocess.run([*command,'--fail'],text=True,capture_output=True)
    assert failed.returncode==7 and 'TIMING ' not in failed.stdout
print('INFERENCE_LAUNCHER_TEST_PASS')
