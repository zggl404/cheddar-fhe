#!/usr/bin/env python3
"""Aggregate disjoint default-stream CUDA event intervals using the NVTX policy."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from analyze_gpu_activity import FHE_PHASES, SOURCES, classify


def analyze(csv_path, log_path, output, warmup=1, split_hybrid=False):
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
