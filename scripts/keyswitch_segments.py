"""Reconstruct non-overlapping KS GPU spans from NVTX launch ownership."""
from bisect import bisect_left
from collections import defaultdict
import csv
import json

KINDS = ('multkey', 'relinearize_rescale', 'baby', 'giant', 'sparse_to_dense',
         'shared_prepare', 'shared_finish')
POLICY = ('KS spans follow declared execution units; Baby preparation stays fused. '
          'Shared finishes are counted once. '
          'Duration is the sum of disjoint first-kernel-to-last-kernel spans, '
          'including internal gaps; it is not pure kernel busy time.')


def metadata(text):
    row = json.loads(text)
    for key in ('ks_id', 'segment_id', 'logical_ks', 'lt_id', 'baby_id'):
        if type(row.get(key)) is not int or row[key] < 0:
            raise ValueError(f'invalid KS metadata: {key}')
    if not row['ks_id'] or not row['segment_id'] or row.get('kind') not in KINDS:
        raise ValueError('invalid KS identity/kind')
    if not isinstance(row.get('segment'), str) or not row['segment']:
        raise ValueError('missing KS segment name')
    if type(row.get('includes_permutation')) is not bool:
        raise ValueError('missing fused permutation policy')
    if bool(row['logical_ks']) == row['kind'].startswith('shared_'):
        raise ValueError('shared stages must have zero logical KS; execution units must be nonzero')
    return row


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'no rows for {path.name}')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct(ranges, activities, output, images, measured):
    roots = [r for r in ranges if r['text'].startswith('cheddar:image:')]
    declared = {}
    for scope in ranges:
        if not scope['text'].startswith('cheddar:ks:'):
            continue
        owners = [r for r in roots if r['globalTid'] == scope['globalTid'] and
                  r['start'] <= scope['start'] and r['end'] >= scope['end']]
        if len(owners) != 1:
            raise ValueError('KS segment must have one image owner')
        image = int(owners[0]['text'].split(':', 2)[2])
        meta = metadata(scope['text'].split(':', 2)[2])
        key = image, meta['segment_id']
        if key in declared:
            raise ValueError('duplicate KS segment identity')
        declared[key] = meta
    if not declared:
        raise ValueError('missing KS segment metadata; rebuild the inference binary')
    kernels = [r for r in activities if r['kind'] == 'kernel']
    streams = {(r['device'], r['context'], r['stream']) for r in kernels if r['fhe']}
    if len(streams) != 1:
        raise ValueError('KS span attribution requires one GPU kernel stream; concurrent streams need a separate policy')
    grouped, origins = defaultdict(list), {}
    for row in kernels:
        image = row['image']
        origins[image] = min(origins.get(image, row['start_ns']), row['start_ns'])
        meta = row['ks']
        if row['fhe'] and row['category'] == 'keyswitch' and meta is None:
            raise ValueError(f'unowned KS kernel: image={image} op={row["op"]} layer={row["layer"]}')
        if meta is None:
            continue
        if not row['fhe'] or row['layer'] == 'none':
            raise ValueError('KS kernel has no FHE/layer owner')
        if row['category'] != 'keyswitch' and not (
                meta['includes_permutation'] and row['op'] == 'automorphism'):
            raise ValueError(f'NonKS kernel inside KS span: {row["op"]}')
        key = image, meta['segment_id']
        if declared.get(key) != meta:
            raise ValueError('KS launch metadata disagrees with its declared segment')
        grouped[key].append(row)
    if set(grouped) != set(declared):
        raise ValueError('KS segment has no correlated GPU kernel')
    ordered = sorted(kernels, key=lambda r: r['start_ns'])
    starts = [r['start_ns'] for r in ordered]
    segments = []
    for (image, _), rows in grouped.items():
        meta = rows[0]['ks']
        start, end = min(r['start_ns'] for r in rows), max(r['end_ns'] for r in rows)
        if len({r['layer'] for r in rows}) != 1:
            raise ValueError('KS segment crosses network layers')
        # Reject unrelated GPU work inside a span, including work on another stream.
        lo, hi = max(0, bisect_left(starts, start)-1), bisect_left(starts, end)
        for other in ordered[lo:hi]:
            if other['end_ns'] > start and (other['image'] != image or other['ks'] != meta):
                raise ValueError('KS span overlaps unrelated GPU work')
        segments.append(dict(image=image, measured=int(image in measured), layer=rows[0]['layer'],
            subpath=rows[0]['subpath'], **meta, gpu_start_ns=start, gpu_end_ns=end,
            start_relative_us=(start-origins[image])/1e3, end_relative_us=(end-origins[image])/1e3,
            duration_ns=end-start, duration_ms=(end-start)/1e6, kernel_count=len(rows),
            kernel_sum_ms=sum(r['end_ns']-r['start_ns'] for r in rows)/1e6))
    segments.sort(key=lambda r: (r['image'], r['gpu_start_ns']))
    previous = {}
    for row in segments:
        if row['gpu_start_ns'] < previous.get(row['image'], 0):
            raise ValueError('overlapping KS spans would double count time')
        previous[row['image']] = row['gpu_end_ns']
    units_by_id = defaultdict(list)
    for row in segments:
        units_by_id[row['image'], row['ks_id']].append(row)
    units = []
    for rows in units_by_id.values():
        head = rows[0]
        if len({(r['kind'], r['logical_ks'], r['layer'], r['baby_id']) for r in rows}) != 1:
            raise ValueError('inconsistent KS unit metadata')
        units.append(dict(image=head['image'], measured=head['measured'], layer=head['layer'],
            subpath=head['subpath'], ks_id=head['ks_id'], kind=head['kind'], logical_ks=head['logical_ks'],
            execution_units=int(head['logical_ks'] > 0), baby_id=head['baby_id'],
            lt_ids=';'.join(str(i) for i in sorted({r['lt_id'] for r in rows})),
            includes_permutation=head['includes_permutation'], segment_count=len(rows),
            gpu_start_ns=min(r['gpu_start_ns'] for r in rows), gpu_end_ns=max(r['gpu_end_ns'] for r in rows),
            duration_ms=sum(r['duration_ns'] for r in rows)/1e6))
    layers = defaultdict(lambda: dict(logical_ks=0, execution_units=0, **{k+'_ms': 0. for k in KINDS}))
    for row in units:
        layer = layers[row['image'], row['layer']]
        layer['logical_ks'] += row['logical_ks']
        layer['execution_units'] += row['execution_units']
        layer[row['kind']+'_ms'] += row['duration_ms']
    layer_rows = [dict(image=image, measured=int(image in measured), layer=name, **row,
                       keyswitch_ms=sum(row[k+'_ms'] for k in KINDS))
                  for (image, name), row in layers.items()]
    per_image = {}
    for image in images:
        selected = [r for r in segments if r['image'] == image]
        if not selected:
            raise ValueError(f'image {image} has no KS spans')
        per_image[image] = dict(keyswitch_span_ns=sum(r['duration_ns'] for r in selected),
            fused_baby_inclusive_ns=sum(r['duration_ns'] for r in selected if r['includes_permutation']),
            logical_ks=sum(r['logical_ks'] for r in units if r['image'] == image),
            execution_units=sum(r['execution_units'] for r in units if r['image'] == image),
            keyswitch_segments=len(selected))
    write_csv(output/'ks_segments.csv', segments)
    write_csv(output/'ks_units.csv', units)
    write_csv(output/'ks_layers.csv', layer_rows)
    return dict(method='nsys_keyswitch_segment_gpu_span_v1', policy=POLICY,
                coverage_verified=True, images=per_image,
                files=['ks_segments.csv', 'ks_units.csv', 'ks_layers.csv'])
