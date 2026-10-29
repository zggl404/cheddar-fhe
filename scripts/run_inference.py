#!/usr/bin/env python3
"""Run a CNN binary with three additive wall-clock categories, including process overhead."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def process_age_ms(fallback_start):
    # Linux exposes process birth before Python imports/argument parsing. Its
    # clock-tick resolution limits agreement with external time to ~one tick.
    try:
        fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
        born=int(fields[19])/os.sysconf('SC_CLK_TCK')
        return (time.clock_gettime(time.CLOCK_BOOTTIME)-born)*1000
    except (OSError,ValueError,AttributeError,IndexError):
        return (time.perf_counter()-fallback_start)*1000


def main():
    start=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    args,options=parser.parse_known_args()
    if any(x in ('--timing-file','--timing-summary') for x in options):
        parser.error('timing-file and timing-summary are managed by this launcher')
    binary=args.binary.resolve()
    env=dict(os.environ);env['LD_LIBRARY_PATH']=str(binary.parents[1])+':'+env.get('LD_LIBRARY_PATH','')
    with tempfile.TemporaryDirectory(prefix='cheddar-timing-') as temporary:
        report=Path(temporary)/'timing.json'
        code=subprocess.call([str(binary),*options,'--timing-file',str(report),'--timing-summary','0'],env=env)
        if code:return code if code>0 else 128-code
        try:row=json.loads(report.read_text())
        except (OSError,ValueError) as error:
            print(f'FAILED timing report: {error}',file=sys.stderr);return 1
    total=process_age_ms(start)
    row['process_overhead_ms']=total-row['total_ms']
    row['model_prepare_ms']=total-row['boot_prepare_ms']-row['inference_ms']
    row['total_ms']=total
    row['boundary']='launcher process wall including child startup and teardown'
    if row['model_prepare_ms']<0:
        print('FAILED overlapping timing categories',file=sys.stderr);return 1
    if '--log-format' in options and options[options.index('--log-format')+1]=='json':
        print('TIMING '+json.dumps(row,separators=(',',':')))
    else:
        print('\nTiming (wall clock, ms)')
        print(f"  Model resources / process overhead : {row['model_prepare_ms']:.3f}")
        print(f"  Boot resources                     : {row['boot_prepare_ms']:.3f}")
        print(f"  Inference stage, all images        : {row['inference_ms']:.3f}")
        print(f"  Sum / end-to-end wall              : {total:.3f}")
        print('  Inference includes input, FHE, validation/output and streamed encoding.')
    return 0


if __name__=='__main__':raise SystemExit(main())
