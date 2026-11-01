#!/usr/bin/env python3
"""Attribute Nsight Systems SQLite GPU records to exclusive Cheddar NVTX labels."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import sqlite3
from inference_timing import KEY_OPS, HYBRID_OPS, NON_KEY_OPS, SPLIT_FORBIDDEN_OPS

FHE_PHASES = {'linear_ms', 'bootstrap_ms', 'activation_ms', 'graph_ops_ms'}
SOURCES = ('rotation', 'linear_transform', 'conjugation', 'relinearization',
           'dense_to_sparse', 'sparse_to_dense')


def classify(labels, split_hybrid=False):
    op = labels.get('op', 'other')
    if split_hybrid and op in SPLIT_FORBIDDEN_OPS:
        raise ValueError(f'unsplit hybrid operation remains in split execution: {op}')
    if op in KEY_OPS:
        category = 'keyswitch'
    elif op in HYBRID_OPS:
        category = 'hybrid'
    elif op == 'other' or op in NON_KEY_OPS:
        category = 'other'
    else:
        raise ValueError(f'unknown operation: {op}')
    if category != 'other' and labels.get('source') not in SOURCES:
        raise ValueError(f'missing source for {op}: {labels}')
    return category


def label_launches(ranges, apis):
    """Use the CPU launch enclosure, never GPU-time overlap, for ownership."""
    by_thread = defaultdict(list)
    for row in ranges:
        by_thread[row['globalTid']].append(row)
    launches = defaultdict(list)
    for row in apis:
        launches[row['globalTid']].append(row)
    result = {}
    for thread, calls in launches.items():
        nested = sorted(by_thread[thread], key=lambda r: (r['start'], -r['end']))
        stack, index = [], 0
        for api in sorted(calls, key=lambda r: r['start']):
            while index < len(nested) and nested[index]['start'] <= api['start']:
                scope = nested[index]
                stack = [r for r in stack if r['end'] >= scope['start']]
                if stack and scope['end'] > stack[-1]['end']:
                    raise ValueError('crossing Cheddar NVTX scopes')
                stack.append(scope)
                index += 1
            stack = [r for r in stack if r['end'] >= api['start']]
            labels = {}
            sources = []
            for scope in stack:
                if scope['end'] < api['end']:
                    raise ValueError('CUDA API crosses its NVTX scope')
                fields = scope['text'].split(':', 2)
                if len(fields) == 3:
                    labels[fields[1]] = fields[2]
                    if fields[1] == 'source':
                        sources.append(fields[2])
            if 'linear_transform' in sources:
                labels['source'] = 'linear_transform'  # MinKS can nest HRot.
            key = api['correlationId']
            if key in result:
                raise ValueError(f'duplicate runtime correlation ID: {key}')
            result[key] = (labels, api)
    return result


def union_ns(intervals):
    total, end = 0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = stop if end is None else max(end, stop)
    return total


def analyze(database, output, warmup=1, log=None, split_hybrid=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings = dict(connection.execute('SELECT id,value FROM StringIds'))
    ranges = []
    for raw in connection.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        row = dict(raw)
        row['text'] = row.get('text') or strings.get(row.get('textId'), '')
        if row['text'].startswith('cheddar:'):
            if row['end'] < row['start']:
                raise ValueError('invalid NVTX timestamp')
            ranges.append(row)
    images = sorted({int(r['text'].split(':', 2)[2]) for r in ranges if r['text'].startswith('cheddar:image:')})
    if not 0 <= warmup < len(images):
        raise ValueError(f'warmup leaves no measured images: {images}')
    measured = set(images[warmup:])
    range_audit = {'verified': False}
    if log is not None:
        expected = {row['image']: row['ranges'] for line in Path(log).read_text().splitlines()
                    if line.startswith('GPU_TRACE_AUDIT ')
                    for row in [json.loads(line[len('GPU_TRACE_AUDIT '):])]}
        if set(expected) != set(images):
            raise ValueError('program-side image census does not match NVTX images')
        observed = {}
        for image in images:
            roots = [r for r in ranges if r['text'] == f'cheddar:image:{image}']
            if len(roots) != 1:
                raise ValueError('duplicate NVTX image root')
            root = roots[0]
            observed[image] = sum(r['globalTid'] == root['globalTid'] and r['start'] >= root['start']
                                  and r['end'] <= root['end'] for r in ranges)
        if observed != expected:
            raise ValueError(f'NVTX ranges lost or duplicated: {observed} != {expected}')
        expected_total = sum(expected.values()) + 1
        if sum(r['text'] == 'cheddar:run' for r in ranges) != 1 or len(ranges) != expected_total:
            raise ValueError('whole-run NVTX range census mismatch')
        range_audit = dict(verified=True, expected=expected, observed=observed,
                           expected_total=expected_total, observed_total=len(ranges))
    apis = [dict(r) for r in connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
    owners = label_launches(ranges, apis)
    counters, breakdown, breakdown_counts, per_image = Counter(), Counter(), Counter(), defaultdict(Counter)
    intervals = defaultdict(list)
    seen = set()
    kernel_correlations = Counter()
    fields = ('kind', 'image', 'measured', 'phase', 'layer', 'source', 'part', 'op',
              'category', 'start_ns', 'end_ns', 'duration_ns', 'correlation_id',
              'device', 'context', 'stream', 'name', 'api_start_ns', 'api_end_ns')
    with gzip.open(output / 'gpu_activities.csv.gz', 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for kind, table in [('kernel', 'CUPTI_ACTIVITY_KIND_KERNEL'),
                            ('memcpy', 'CUPTI_ACTIVITY_KIND_MEMCPY'),
                            ('memset', 'CUPTI_ACTIVITY_KIND_MEMSET')]:
            if table not in tables:
                if kind == 'kernel':
                    raise ValueError('missing kernel records')
                continue
            for raw in connection.execute(f'SELECT * FROM {table}'):
                row = dict(raw)
                counters[f'{kind}_records'] += 1
                start, end, correlation = row['start'], row['end'], row['correlationId']
                counters[f'{kind}_total_ns'] += end-start
                if start <= 0 or end <= start:
                    raise ValueError(f'invalid {kind} timestamp: {row}')
                identity = (kind, start, end, row.get('contextId'), row.get('streamId'), correlation, row.get('gridId'))
                if identity in seen:
                    raise ValueError('duplicate GPU activity')
                seen.add(identity)
                if correlation not in owners:
                    raise ValueError(f'uncorrelated GPU activity: {kind} {correlation}')
                labels, api = owners[correlation]
                image = int(labels.get('image', -1))
                phase = labels.get('phase', 'dispatch')
                if image < 0 and any(key in labels for key in ('phase', 'source', 'op')):
                    raise ValueError('Cheddar operation has no image owner')
                if image < 0:
                    counters['outside_image_records'] += 1
                category = classify(labels, split_hybrid) if image >= 0 else 'outside'
                fhe = phase in FHE_PHASES
                if not fhe:
                    counters['non_fhe_records'] += 1
                selected = image in measured and fhe
                op, source = labels.get('op', 'other'), labels.get('source', 'none')
                name = strings.get(row.get('demangledName'), strings.get(row.get('shortName'), kind))
                writer.writerow(dict(kind=kind, image=image, measured=int(image in measured), phase=phase,
                    layer=labels.get('layer', 'none'), source=source, part=labels.get('part', 'none'), op=op,
                    category=category, start_ns=start, end_ns=end, duration_ns=end-start,
                    correlation_id=correlation, device=row.get('deviceId'), context=row.get('contextId'),
                    stream=row.get('streamId'), name=name, api_start_ns=api['start'], api_end_ns=api['end']))
                if selected:
                    breakdown[kind, source, labels.get('part', 'none'), op, category] += end-start
                    breakdown_counts[kind, source, labels.get('part', 'none'), op, category] += 1
                    per_image[image][f'{kind}_{category}_ns'] += end-start
                    per_image[image][f'{kind}_{category}_count'] += 1
                    per_image[image][f'{kind}_count'] += 1
                    if kind == 'kernel':
                        kernel_correlations[correlation] += 1
                        intervals[image].append((start, end))
                        intervals[image, category].append((start, end))
    if not per_image or any(per_image[i]['kernel_count'] == 0 for i in measured):
        raise ValueError('missing measured image kernels')
    launch_count = 0
    for correlation, (labels, api) in owners.items():
        if int(labels.get('image', -1)) not in measured or labels.get('phase') not in FHE_PHASES:
            continue
        name = strings.get(api.get('nameId'), '')
        if name.startswith('cudaGraphLaunch'):
            raise ValueError('CUDA Graph launch needs node-level correlation; unsupported by this analyzer')
        if name.startswith(('cudaLaunchKernel', 'cudaLaunchCooperativeKernel')):
            launch_count += 1
            if kernel_correlations[correlation] != 1:
                raise ValueError(f'kernel launch has missing/duplicate GPU records: {correlation}')
    if launch_count != sum(per_image[i]['kernel_count'] for i in measured):
        raise ValueError('runtime launch census does not match measured GPU kernels')
    image_rows = []
    for image in sorted(measured):
        row = dict(image=image, **per_image[image])
        row['kernel_total_ns'] = sum(row.get(f'kernel_{c}_ns', 0) for c in ('keyswitch', 'hybrid', 'other'))
        row['kernel_union_ns'] = union_ns(intervals[image])
        row['kernel_overlap_ns'] = row['kernel_total_ns'] - row['kernel_union_ns']
        for category in ('keyswitch', 'hybrid', 'other'):
            row[f'kernel_{category}_union_ns'] = union_ns(intervals[image, category])
        image_rows.append(row)
    with (output / 'breakdown.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['kind', 'source', 'part', 'operation', 'category', 'activity_count', 'total_ns', 'mean_ms'])
        for key, ns in sorted(breakdown.items()):
            writer.writerow([*key, breakdown_counts[key], ns, ns / len(measured) / 1e6])
    kernel_total = sum(r['kernel_total_ns'] for r in image_rows)
    classified_total = sum(ns for key, ns in breakdown.items() if key[0] == 'kernel')
    if kernel_total != classified_total:
        raise ValueError('kernel accounting mismatch')
    for kind, table in [('kernel', 'CUPTI_ACTIVITY_KIND_KERNEL'), ('memcpy', 'CUPTI_ACTIVITY_KIND_MEMCPY'),
                        ('memset', 'CUPTI_ACTIVITY_KIND_MEMSET')]:
        if table in tables:
            count, ns = connection.execute(f'SELECT COUNT(*),COALESCE(SUM(end-start),0) FROM {table}').fetchone()
            if count != counters[f'{kind}_records'] or ns != counters[f'{kind}_total_ns']:
                raise ValueError('raw SQLite GPU totals disagree with exported records')
    diagnostics = [dict(r) for r in connection.execute('SELECT * FROM DIAGNOSTIC_EVENT')] if 'DIAGNOSTIC_EVENT' in tables else []
    summary = dict(schema_version=1, method='nsys_gpu_activity_launch_correlation', split_hybrid=split_hybrid,
        warmup_images=images[:warmup], measured_images=sorted(measured), records=dict(counters),
        uncorrelated_records=0, duplicate_records=0, accounting_error_ns=kernel_total-classified_total,
        nvtx_range_audit=range_audit, measured_kernel_launches=launch_count,
        mean_ms={key.removesuffix('_ns'): sum(r.get(key, 0) for r in image_rows)/len(image_rows)/1e6
                 for key in sorted({key for r in image_rows for key in r if key.endswith('_ns')})},
        sources={source: {category: sum(ns for key, ns in breakdown.items()
                  if key[0]=='kernel' and key[1]==source and key[4]==category)/len(measured)/1e6
                  for category in ('keyswitch', 'hybrid', 'other')} for source in SOURCES},
        source_kernel_counts={source: {category: sum(n for key, n in breakdown_counts.items()
                  if key[0]=='kernel' and key[1]==source and key[4]==category)
                  for category in ('keyswitch', 'hybrid', 'other')} for source in SOURCES},
        diagnostics=diagnostics,
        images=image_rows,
        policy='fused_baby and ciphertext correction are keyswitch-related; shared_lift_common is non-keyswitch; shared_lift, moddown_rescale and automorphism_accum are hybrid; sum of durations is not wall time')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    connection.close()
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sqlite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--split-hybrid', choices=('off','on'), default='off')
    parser.add_argument('--log', type=Path, help='Worker log containing program-side NVTX range counts')
    args = parser.parse_args()
    result = analyze(args.sqlite, args.output, args.warmup, args.log, args.split_hybrid=='on')
    print(json.dumps({key: result[key] for key in ('measured_images', 'mean_ms', 'sources', 'accounting_error_ns')}, indent=2))
