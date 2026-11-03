#!/usr/bin/env python3
"""Partition a trace window into KS spans and their elapsed-time complement."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path


PROVENANCE = ('segment_id', 'ks_id', 'layer', 'subpath', 'kind', 'segment')
FIELDS = ('image', 'measured', 'scope', 'interval_id', 'category', 'start_ns', 'end_ns',
          'start_from_t0_us', 'end_from_t0_us', 'duration_ns', 'duration_ms',
          'segment_ids', 'ks_ids', 'layers', 'subpaths', 'kinds', 'segments')


def rows(path, required):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as handle:
        reader = csv.DictReader(handle)
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f'{path}: required columns: {", ".join(required)}')
        yield from reader


def integer(row, key):
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f'invalid integer {key}: {row.get(key)!r}') from None
    if value < 0:
        raise ValueError(f'{key} must be nonnegative')
    return value


def span(row, start_key, end_key):
    start, end = integer(row, start_key), integer(row, end_key)
    if end <= start:
        raise ValueError(f'invalid interval [{start}, {end})')
    return start, end


def measured_flag(row):
    value = integer(row, 'measured') if 'measured' in row else 1
    if value not in (0, 1):
        raise ValueError('measured must be 0 or 1')
    return value


def load_segments(path):
    grouped, flags, seen = defaultdict(list), {}, set()
    for row in rows(path, ('image', 'segment_id', 'gpu_start_ns', 'gpu_end_ns')):
        image, segment_id = integer(row, 'image'), integer(row, 'segment_id')
        start, end = span(row, 'gpu_start_ns', 'gpu_end_ns')
        if 'duration_ns' in row and integer(row, 'duration_ns') != end-start:
            raise ValueError('KS duration_ns disagrees with GPU endpoints; use ks_segments.csv')
        flag = measured_flag(row)
        if image in flags and flags[image] != flag:
            raise ValueError(f'inconsistent measured flag for image {image}')
        if (image, segment_id) in seen:
            raise ValueError(f'duplicate segment_id for image {image}: {segment_id}')
        flags[image] = flag
        seen.add((image, segment_id))
        grouped[image].append((start, end, row))
    if not grouped:
        raise ValueError('no KS segments')
    return grouped, flags


def load_bounds(path, grouped, flags, activities=False):
    bounds = {}
    for row in rows(path, ('image', 'start_ns', 'end_ns')):
        if activities and row['image'] == '-1':
            continue
        image = integer(row, 'image')
        if activities and image not in grouped:
            continue
        start, end = span(row, 'start_ns', 'end_ns')
        if 'measured' in row:
            flag = measured_flag(row)
            if image in flags and flag != flags[image]:
                raise ValueError(f'boundary measured flag disagrees for image {image}')
            flags[image] = flag
        flags.setdefault(image, 1)
        if activities:
            previous = bounds.get(image, (start, end))
            bounds[image] = min(start, previous[0]), max(end, previous[1])
        else:
            if image in bounds:
                raise ValueError(f'duplicate inference boundary for image {image}')
            bounds[image] = start, end
    missing = set(grouped)-set(bounds)
    if missing:
        raise ValueError(f'missing boundaries for images {sorted(missing)}')
    return bounds


def partition(image, segments, bounds, measured, scope):
    start, end = bounds
    merged = []
    for a, b, row in sorted(segments, key=lambda r: (r[0], r[1])):
        if a < start or b > end:
            raise ValueError(f'image {image}: KS outside supplied boundaries')
        # Keep adjacent stages separate so their original identities remain visible.
        if merged and a < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
            merged[-1][2].append(row)
        else:
            merged.append([a, b, [row]])
    result, counts = [], {'KS': 0, 'NonKS': 0}

    def append(category, a, b, sources=()):
        if a == b:
            return
        entry = dict(image=image, measured=measured, scope=scope,
                     interval_id=f'{category}_{counts[category]}', category=category,
                     start_ns=a, end_ns=b, start_from_t0_us=(a-start)/1000,
                     end_from_t0_us=(b-start)/1000, duration_ns=b-a, duration_ms=(b-a)/1e6)
        for key, target in zip(PROVENANCE, FIELDS[-6:]):
            entry[target] = ';'.join(dict.fromkeys(r.get(key, '') for r in sources if r.get(key, '')))
        result.append(entry)
        counts[category] += 1

    cursor = start
    for a, b, sources in merged:
        append('NonKS', cursor, a)
        append('KS', a, b, sources)
        cursor = b
    append('NonKS', cursor, end)
    ks_ns = sum(r['duration_ns'] for r in result if r['category'] == 'KS')
    nonks_ns = sum(r['duration_ns'] for r in result if r['category'] == 'NonKS')
    error = end-start-ks_ns-nonks_ns
    if error or result[0]['start_ns'] != start or result[-1]['end_ns'] != end:
        raise ValueError(f'image {image}: partition does not cover the window')
    summary = dict(image=image, measured=measured, t0_ns=start, t1_ns=end,
                   window_ns=end-start, keyswitch_ns=ks_ns, non_keyswitch_ns=nonks_ns,
                   window_ms=(end-start)/1e6, keyswitch_ms=ks_ns/1e6, non_keyswitch_ms=nonks_ns/1e6,
                   input_ks_segments=len(segments), ks_intervals=counts['KS'],
                   nonks_intervals=counts['NonKS'],
                   merged_overlap_ns=sum(b-a for a,b,_ in segments)-ks_ns, accounting_error_ns=error)
    return result, summary


def generate(segments_path, output, bounds_path=None, activities_path=None):
    segments_path, output = Path(segments_path), Path(output)
    if bounds_path and activities_path:
        raise ValueError('choose inference bounds or GPU activity bounds, not both')
    grouped, flags = load_segments(segments_path)
    source = bounds_path or activities_path
    if source:
        source = Path(source)
        bounds = load_bounds(source, grouped, flags, activities=bool(activities_path))
        scope = 'gpu_activity_envelope' if activities_path else 'inference'
    else:
        bounds = {image: (min(r[0] for r in ss), max(r[1] for r in ss))
                  for image, ss in grouped.items()}
        scope = 'ks_envelope'
    partitions, images = [], []
    for image, window in sorted(bounds.items()):
        intervals, summary = partition(image, grouped.get(image, []), window, flags[image], scope)
        partitions.append(intervals)
        images.append(summary)
    measured = [r for r in images if r['measured']]
    totals = {key: sum(r[key] for r in measured)
              for key in ('window_ns', 'keyswitch_ns', 'non_keyswitch_ns')}
    summary = dict(schema_version=1, method='ks_interval_complement_v1', scope=scope,
                   complete_inference_boundaries=scope == 'inference',
                   timestamp_unit='ns', interval_convention='[start_ns, end_ns)',
                   relative_time_origin='per-image T0; not necessarily the first GPU kernel',
                   segments_file=str(segments_path.resolve()),
                   bounds_file=str(source.resolve()) if source else None,
                   interpretation='NonKS is elapsed time outside the union of KS segments, not pure kernel time.',
                   boundary_note={'inference': 'Caller-supplied Inference endpoints on the same trace clock.',
                                  'gpu_activity_envelope': 'First to last recorded GPU activity; excludes Inference head/tail outside this envelope.',
                                  'ks_envelope': 'First KS start to last KS end; excludes all work outside the KS envelope.'}[scope],
                   measured_images=[r['image'] for r in measured],
                   warmup_images=[r['image'] for r in images if not r['measured']],
                   measured_totals_ns=totals,
                   mean_ms={key.removesuffix('_ns'): value/len(measured)/1e6 if measured else None
                            for key, value in totals.items()}, images=images)
    targets = [output/name for name in ('timeline.csv', 'timeline.txt', 'summary.json')]
    inputs = {segments_path.resolve()} | ({source.resolve()} if source else set())
    if any(path.resolve() in inputs for path in targets):
        raise ValueError('output would overwrite an input file')
    output.mkdir(parents=True, exist_ok=True)
    with targets[0].open('w', newline='') as handle, targets[1].open('w') as text:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        text.write(f'Scope: {scope}\n{summary["boundary_note"]}\n'
                   'All absolute endpoints are trace ns; relative times are us from T0.\n'
                   'NonKS includes computation, memory activity and gaps outside KS spans.\n'
                   'Layout: T0 -> NonKS_0 -> [KS_0] -> NonKS_1 -> [KS_1] -> ... -> T1\n')
        for image, intervals in zip(images, partitions):
            text.write(f'\nimage={image["image"]} measured={image["measured"]}\nT0 = {image["t0_ns"]} ns\n')
            for row in intervals:
                writer.writerow(row)
                label = f'[{row["interval_id"]}]' if row['category'] == 'KS' else row['interval_id']
                owner = f' layers={row["layers"]} kinds={row["kinds"]} segments={row["segment_ids"]}' if row['category'] == 'KS' else ''
                text.write(f'  -> {label} [{row["start_ns"]}, {row["end_ns"]}) ns'
                           f' relative=[{row["start_from_t0_us"]:.3f}, {row["end_from_t0_us"]:.3f}) us'
                           f' duration={row["duration_ms"]:.6f} ms{owner}\n')
            text.write(f'T1 = {image["t1_ns"]} ns\n'
                       f'Window={image["window_ms"]:.6f} ms; KS={image["keyswitch_ms"]:.6f} ms; '
                       f'NonKS={image["non_keyswitch_ms"]:.6f} ms; accounting_error_ns=0\n')
        text.write(f'\nMeasured images: {summary["measured_images"]}\nMean ms: {json.dumps(summary["mean_ms"])}\n')
    targets[2].write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('segments', type=Path, help='NVTX ks_segments.csv (not ks_units.csv)')
    parser.add_argument('--output', type=Path, required=True, help='directory for timeline.csv/txt and summary.json')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--bounds', type=Path, help='Inference CSV: image,start_ns,end_ns[,measured], on the same trace clock')
    group.add_argument('--activities', type=Path, help='gpu_activities.csv[.gz] from the same run; use its per-image envelope')
    args = parser.parse_args()
    try:
        summary = generate(args.segments, args.output, args.bounds, args.activities)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    print(json.dumps(dict(output=str(args.output), scope=summary['scope'],
                          boundary_note=summary['boundary_note'], measured_images=summary['measured_images'],
                          mean_ms=summary['mean_ms']), indent=2))


if __name__ == '__main__':
    main()
