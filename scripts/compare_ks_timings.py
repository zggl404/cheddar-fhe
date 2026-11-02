#!/usr/bin/env python3
"""Compare independent CUDA Event/NVTX runs by identical KS segment identities."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from inference_timing import validate_run
from keyswitch_segments import write_csv

IDENTITY = ('layer', 'subpath', 'ks_id', 'segment_id', 'kind', 'segment',
            'logical_ks', 'lt_id', 'baby_id', 'includes_permutation')


def compare(event_dir, nvtx_dir, output):
    event_dir, nvtx_dir, output = map(Path, (event_dir, nvtx_dir, output))
    output.mkdir(parents=True, exist_ok=False)
    method_reports = []
    walls = []
    logs = []
    for path, name, enabled in ((event_dir, 'CUDA_EVENT', dict(cpu=False, cuda_event=True, nvtx=False)),
                                (nvtx_dir, 'NVTX', dict(cpu=False, cuda_event=False, nvtx=True))):
        report = json.loads((path/'summary.json').read_text())[name]
        if report['enabled_methods'] != enabled or not report['numerical_passed']:
            raise ValueError('comparison requires independent, numerically passing runs')
        method_reports.append(report)
        wall = json.loads((path/'cpu.json').read_text());validate_run(wall)
        if wall['ks_profile'] or not wall['split_hybrid'] or wall['key_mode'] != 'load':
            raise ValueError('comparison requires split load runs without CPU operation profiling')
        if wall['keys_generated'] != 0 or wall['keys_loaded'] != wall['key_count']:
            raise ValueError('keys must be loaded without generation')
        walls.append(wall)
        logs.append((path/'worker.log').read_text())
    if method_reports[0]['method'] != 'cuda_event_keyswitch_segments_v1' or method_reports[1]['method'] != 'nsys_keyswitch_segment_gpu_span_v1':
        raise ValueError('recollect both methods with aligned KS segment support')
    for key in ('key_identity', 'key_count', 'encode_mode', 'plaintext_reference'):
        if walls[0][key] != walls[1][key]:raise ValueError('paired run mismatch: '+key)
    if 'GPU_TRACE_AUDIT ' in logs[0] or 'CUDA_EVENT_KS_IMAGE ' in logs[1] or 'CUDA_EVENT_IMAGE ' in logs[1]:
        raise ValueError('unexpected cross-method instrumentation')
    for key in ('warmup_count', 'measured_count'):
        if method_reports[0][key] != method_reports[1][key]:raise ValueError('different image selection')
    results = [{r['image']: r for line in text.splitlines() if line.startswith('RESULT ')
                for r in [json.loads(line.split(' ', 1)[1])]} for text in logs]
    if not results[0] or set(results[0]) != set(results[1]):raise ValueError('different/missing inference results')
    logits_delta = 0.
    for image, before in results[0].items():
        after = results[1][image]
        if before['status'] != 'PASS' or after['status'] != 'PASS' or before['prediction'] != after['prediction']:
            raise ValueError('paired numerical gate/prediction mismatch')
        if len(before['fhe_logits']) != 10 or len(after['fhe_logits']) != 10:raise ValueError('missing logits')
        for a, b in zip(before['fhe_logits'], after['fhe_logits']):
            if not math.isfinite(a) or not math.isfinite(b):raise ValueError('nonfinite logits')
            logits_delta = max(logits_delta, abs(a-b))
    if logits_delta > .01:raise ValueError('paired logits exceed numerical gate')
    tables = []
    for path in (event_dir/'event-analysis/ks_segments.csv', nvtx_dir/'nvtx-analysis/ks_segments.csv'):
        with path.open(newline='') as handle:rows = list(csv.DictReader(handle))
        table = {(int(r['image']), int(r['segment_id'])): r for r in rows}
        if len(table) != len(rows):raise ValueError('duplicate KS segment ID')
        tables.append(table)
    if not tables[0] or set(tables[0]) != set(tables[1]):raise ValueError('missing/extra KS segments across runs')
    paired, per_image = [], defaultdict(lambda: [0., 0., 0])
    for key in sorted(tables[0]):
        event, nvtx = tables[0][key], tables[1][key]
        if any(event[field] != nvtx[field] for field in (*IDENTITY, 'measured')):
            raise ValueError(f'KS segment metadata mismatch: {key}')
        e, n = float(event['duration_ms']), float(nvtx['duration_ms'])
        if not all(math.isfinite(v) and v >= 0 for v in (e, n)) or n == 0:
            raise ValueError('invalid comparison duration')
        image, _ = key
        per_image[image][0] += e;per_image[image][1] += n;per_image[image][2] += 1
        paired.append(dict(image=image, measured=int(event['measured']),
            **{field: event[field] for field in IDENTITY}, event_ms=e, nvtx_ms=n,
            delta_ms=e-n, delta_percent=(e/n-1)*100))
    measured = sorted({r['image'] for r in paired if r['measured']})
    if len(measured) != method_reports[0]['measured_count'] or set(per_image) != set(results[0]):
        raise ValueError('different measured images or missing result images')
    event_mean = statistics.fmean(per_image[i][0] for i in measured)
    nvtx_mean = statistics.fmean(per_image[i][1] for i in measured)
    if not math.isclose(event_mean, method_reports[0]['KeyswitchTime'], rel_tol=1e-10) or not math.isclose(nvtx_mean, method_reports[1]['KeyswitchTime'], rel_tol=1e-10):
        raise ValueError('segment totals disagree with method summaries')
    write_csv(output/'segments.csv', paired)
    for field, filename in (('kind', 'kinds.csv'), ('layer', 'layers.csv')):
        grouped = defaultdict(lambda: [0., 0., 0])
        for row in paired:
            if row['measured']:
                group = grouped[row[field]];group[0] += row['event_ms'];group[1] += row['nvtx_ms'];group[2] += 1
        write_csv(output/filename, [dict(**{field: name}, segments=v[2],
            event_mean_ms=v[0]/len(measured), nvtx_mean_ms=v[1]/len(measured),
            delta_mean_ms=(v[0]-v[1])/len(measured), delta_percent=(v[0]/v[1]-1)*100)
            for name, v in grouped.items()])
    deltas = sorted(abs(r['delta_ms']) for r in paired if r['measured'])
    result = dict(status='PASS', identities_match=True, independent_runs=True,
        measured_images=measured, matched_segments=len(paired), matched_measured_segments=sum(r['measured'] for r in paired),
        event_keyswitch_mean_ms=event_mean, nvtx_keyswitch_mean_ms=nvtx_mean,
        delta_ms=event_mean-nvtx_mean, delta_percent=(event_mean/nvtx_mean-1)*100,
        median_segment_absolute_delta_ms=statistics.median(deltas), p95_segment_absolute_delta_ms=deltas[int(.95*(len(deltas)-1))],
        max_paired_logits_delta=logits_delta,
        images=[dict(image=i, event_ms=v[0], nvtx_ms=v[1], segments=v[2]) for i, v in sorted(per_image.items())],
        boundary='Same application KsTraceSegment boundaries and IDs. Event includes edge gaps; NVTX uses first/last GPU kernel. Independent runs also differ in profiler/marker overhead and scheduling.',
        interpretation='PASS verifies identity coverage, numerical gates and accounting; it does not assert a latency tolerance or pure KS instruction time.')
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event-dir', type=Path, required=True)
    parser.add_argument('--nvtx-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.event_dir, args.nvtx_dir, args.output)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'FAILED KS comparison: {error}\n')
