#!/usr/bin/env python3
"""Aggregate disjoint default-stream CUDA event intervals using the NVTX policy."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from analyze_gpu_activity import FHE_PHASES, SOURCES, classify
from keyswitch_segments import metadata, write_csv, POLICY


def analyze_ks(csv_path, log_path, output, warmup, split_hybrid):
    audits = [json.loads(line.split(' ', 1)[1]) for line in Path(log_path).read_text().splitlines()
              if line.startswith('CUDA_EVENT_KS_IMAGE ')]
    expected = {r['image']: r for r in audits}
    images = sorted(expected)
    if not expected or len(expected) != len(audits) or not 0 <= warmup < len(images):
        raise ValueError('missing/duplicate KS event audits or invalid warmup')
    measured = set(images[warmup:])
    rows, counts, totals, units, last = [], Counter(), Counter(), {}, {}
    categories = defaultdict(Counter)
    with Path(csv_path).open(newline='') as handle:
        for raw in csv.DictReader(handle):
            image = int(raw['image'])
            meta = metadata(raw['metadata'])
            if split_hybrid and meta['category'] != 'keyswitch':
                raise ValueError('Hybrid stage remains in split Event execution')
            begin, end = int(raw['start_event']), int(raw['end_event'])
            start_ms, end_ms, ms = (float(raw[k]) for k in ('start_relative_ms', 'end_relative_ms', 'milliseconds'))
            if image not in expected or meta['segment_id'] != counts[image]+1 or begin != 2*counts[image]+1 or end != begin+1:
                raise ValueError('missing/duplicate/overlapping KS event boundary or segment ID')
            if not all(math.isfinite(v) and v >= 0 for v in (start_ms, end_ms, ms)) or start_ms > end_ms:
                raise ValueError('invalid KS event timestamp')
            # CUDA elapsed timestamps are float32 offsets; allow their rounding error.
            tolerance = max(1e-5, (start_ms+end_ms)*2e-7)
            if start_ms < last.get(image, 0)-tolerance or abs(end_ms-start_ms-ms) > tolerance:
                raise ValueError('inconsistent/overlapping KS event timestamps')
            if raw['layer'] == 'none' or not raw['layer']:
                raise ValueError('missing KS event layer')
            key = image, meta['ks_id']
            identity = meta['kind'], meta['logical_ks'], raw['layer'], meta['baby_id']
            if key in units and units[key] != identity:
                raise ValueError('inconsistent KS event unit metadata')
            units[key] = identity
            counts[image] += 1;totals[image] += ms;last[image] = end_ms
            categories[image][meta['category']] += ms
            rows.append(dict(image=image, measured=int(image in measured), layer=raw['layer'], subpath=raw['subpath'],
                **meta, start_event=begin, end_event=end, start_relative_us=start_ms*1000,
                end_relative_us=end_ms*1000, duration_ms=ms))
    for image, audit in expected.items():
        if counts[image] != audit['segments'] or audit['events'] != 2*counts[image]+1:
            raise ValueError('KS event CSV disagrees with program event census')
        if not math.isclose(totals[image], audit['sum_ms'], rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError('KS event CSV disagrees with program duration sum')
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'ks_segments.csv', rows)
    image_rows = [dict(image=i, keyswitch_ms=categories[i]['keyswitch'],hybrid_ms=categories[i]['hybrid'],
        segments=counts[i], events=expected[i]['events'],
        logical_ks=sum(v[1] for (image, _), v in units.items() if image == i),
        execution_units=sum(v[1] > 0 for (image, _), v in units.items() if image == i),
        readback_ms=expected[i]['readback_ms']) for i in images]
    result = dict(method='cuda_event_keyswitch_segments_v1', split_hybrid=split_hybrid,
        measured_images=sorted(measured), warmup_images=images[:warmup],
        mean_ms=dict(keyswitch=sum(categories[i]['keyswitch'] for i in measured)/len(measured),
            hybrid=sum(categories[i]['hybrid'] for i in measured)/len(measured),
            fused_baby_inclusive=sum(r['duration_ms'] for r in rows if r['image'] in measured and r['includes_permutation'] and r['category']=='keyswitch')/len(measured)),
        mean_readback_ms=sum(expected[i]['readback_ms'] for i in measured)/len(measured),
        images=image_rows, image_audits=audits, census_verified=True,
        timestamp_origin='CUDA event recorded at BeginImage; offsets are not absolute GPU timestamps',
        boundary='same KsTraceSegment entry/exit and IDs as NVTX; includes event-to-first/last-kernel edge gaps',
        policy=POLICY)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def analyze(csv_path, log_path, output, warmup=1, split_hybrid=False):
    with Path(csv_path).open(newline='') as handle:
        aligned = 'metadata' in (csv.DictReader(handle).fieldnames or [])
    if aligned:
        return analyze_ks(csv_path, log_path, output, warmup,split_hybrid)
    audits = [json.loads(line[len('CUDA_EVENT_IMAGE '):]) for line in Path(log_path).read_text().splitlines()
              if line.startswith('CUDA_EVENT_IMAGE ')]
    expected = {r['image']: r for r in audits}
    if not expected or len(expected) != len(audits):
        raise ValueError('missing or duplicate CUDA event image audits')
    images = sorted(expected)
    if not 0 <= warmup < len(images):
        raise ValueError('warmup leaves no CUDA event measurements')
    measured = set(images[warmup:])
    counts, total, last, by_image = Counter(), Counter(), {}, defaultdict(Counter)
    breakdown = defaultdict(lambda: [0, 0.0])
    with Path(csv_path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            image, ordinal = int(row['image']), int(row['segment'])
            begin, end, ms = int(row['start_event']), int(row['end_event']), float(row['milliseconds'])
            if image not in expected or ordinal != counts[image] or end != begin+1 or not 0 <= ms < float('inf'):
                raise ValueError('invalid CUDA event segment')
            if begin != 0 and (image not in last or begin not in (last[image], last[image]+1)):
                raise ValueError('overlapping or missing CUDA event boundaries')
            if ordinal and begin == 0:
                raise ValueError('CUDA event sequence restarted')
            last[image] = end
            if row['phase'] not in FHE_PHASES:
                raise ValueError('non-FHE CUDA event segment')
            category = classify({'op': row['operation'], 'source': row['source']}, split_hybrid)
            counts[image] += 1
            total[image] += ms
            by_image[image][category] += ms
            if image in measured:
                item = breakdown[row['source'], row['part'], row['operation'], category]
                item[0] += 1
                item[1] += ms
    for image, audit in expected.items():
        if counts[image] != audit['segments'] or last.get(image, -1)+1 != audit['events']:
            raise ValueError('CUDA event CSV disagrees with program event census')
        if abs(total[image]-audit['sum_ms']) > 1e-8:
            raise ValueError('CUDA event CSV disagrees with program duration sum')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'breakdown.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['source', 'part', 'operation', 'category', 'segments', 'total_ms', 'mean_ms'])
        for key, (count, ms) in sorted(breakdown.items()):
            writer.writerow([*key, count, ms, ms/len(measured)])
    mean = {c: sum(by_image[i][c] for i in measured)/len(measured) for c in ('keyswitch', 'hybrid', 'other')}
    result = dict(method='cuda_event_exclusive_default_stream', split_hybrid=split_hybrid, measured_images=sorted(measured),
        warmup_images=images[:warmup], mean_ms={**mean, 'total':sum(mean.values())},
        mean_readback_ms=sum(expected[i]['readback_ms'] for i in measured)/len(measured),
        sources={s: {c:sum(v[1] for k,v in breakdown.items() if k[0]==s and k[3]==c)/len(measured)
                        for c in ('keyswitch', 'hybrid', 'other')} for s in SOURCES},
        images=[dict(image=i, **by_image[i], total_ms=total[i], **{k:expected[i][k] for k in ('events','segments','readback_ms')})
                for i in sorted(measured)],
        image_audits=audits, census_verified=True,
        policy='same source/operation policy as NVTX; elapsed intervals include memory operations, event markers and device idle gaps')
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--split-hybrid', choices=('off','on'), default='off')
    args = parser.parse_args()
    result = analyze(args.csv, args.log, args.output, args.warmup, args.split_hybrid=='on')
    print(json.dumps({k:result[k] for k in ('method','mean_ms','mean_readback_ms')}, indent=2))
