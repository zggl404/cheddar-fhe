#!/usr/bin/env python3
"""Export one profiled-only timing table and per-model timelines from benchmark_nvtx.py."""
import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
import shutil

from benchmark_nvtx import number
from inference_timing import close, validate_run
from keyswitch_timeline import generate


FIELDS = (
    'model', 'end_to_end_ms', 'key_load_ms', 'weight_preencode_ms',
    'init_and_teardown_ms', 'inference_total_ms', 'inference_mean_ms',
    'keyswitch_mean_ms', 'hybrid_mean_ms', 'non_keyswitch_mean_ms',
)


def read_json(path):
    return json.loads(path.read_text())


def milliseconds(ns):
    return Decimal(ns)/1000000


def summarize_model(source, target, model, count, warmup, split_hybrid=True):
    folder = source/model['model']/'profiled'
    wall = read_json(folder/'wall.json')
    validate_run(wall)
    if wall.get('timing_detail') != 'coarse' or wall.get('ks_profile'):
        raise ValueError('require coarse profiled timing')
    if wall.get('split_hybrid') is not split_hybrid:
        raise ValueError('wall split-hybrid mode disagrees with benchmark summary')
    if wall.get('key_mode') != 'load' or wall.get('keys_generated') != 0 or wall.get('keys_loaded') != wall.get('key_count'):
        raise ValueError('profiled run must load all keys without generation')
    images = wall['images']
    if len(images) != count:
        raise ValueError('wall image count disagrees with benchmark count')
    measured = images[warmup:]
    ids = [r['image'] for r in images]
    if ids != list(range(count)):
        raise ValueError('image IDs must match the benchmark sequence 0..count-1')
    measured_ids = [r['image'] for r in measured]
    analysis = read_json(folder/'analysis/summary.json')
    if analysis.get('split_hybrid') is not split_hybrid:
        raise ValueError('NVTX split-hybrid mode disagrees with wall report')
    if analysis['measured_images'] != measured_ids:
        raise ValueError('NVTX and wall timing use different measured images')
    if any(r['kernel_overlap_ns'] for r in analysis['images']):
        raise ValueError('overlapping kernels cannot partition inference elapsed time')
    spans = analysis.get('ks_spans')
    if (analysis.get('method') != 'nsys_keyswitch_segment_gpu_span_v1' or
            not spans or not spans.get('coverage_verified')):
        raise ValueError('missing verified KS/Hybrid stage spans; rebuild and recollect stage labels')
    ks_ns = {int(image): row['keyswitch_span_ns'] for image, row in spans['images'].items()}
    hybrid_ns = {int(image): row.get('hybrid_span_ns', 0) for image, row in spans['images'].items()}
    if split_hybrid and any(hybrid_ns.values()):
        raise ValueError('Hybrid stage remains in split execution')
    if any(type(value) is not int or value < 0 for value in (*ks_ns.values(), *hybrid_ns.values())):
        raise ValueError('KS/Hybrid intervals must be nonnegative integer nanoseconds')
    if set(ks_ns) != set(ids):
        raise ValueError('KS and wall image census disagree')
    # Recover the worker's integer ns, avoiding floating-point subtraction in the CSV.
    timing_ns = {key: round(number(wall[key], key)*1e6) for key in
                 ('total_ms', 'key_load_ms', 'weight_preencode_ms', 'inference_ms')}
    image_ns = {r['image']: round(r['inference_ms']*1e6) for r in images}
    if sum(image_ns.values()) != timing_ns['inference_ms']:
        raise ValueError('image Inference nanoseconds do not sum to the total')
    residual_ns = {i: image_ns[i]-ks_ns[i]-hybrid_ns[i] for i in ids}
    if any(value < 0 for value in residual_ns.values()):
        raise ValueError('negative Inference minus KS and Hybrid')
    ks_mean = milliseconds(sum(ks_ns[i] for i in measured_ids))/len(measured)
    hybrid_mean = milliseconds(sum(hybrid_ns[i] for i in measured_ids))/len(measured)
    close(ks_mean, analysis['mean_ms']['keyswitch_span'], 'KS mean')
    close(hybrid_mean, analysis['mean_ms'].get('hybrid_span', 0), 'Hybrid mean')
    other_ns = timing_ns['total_ms']-timing_ns['key_load_ms']-timing_ns['weight_preencode_ms']-timing_ns['inference_ms']
    if other_ns < 0:
        raise ValueError('negative other preparation and teardown')
    status = model['status']
    if status == 'PASS' and (model.get('diagnostic_only') or not wall['numerical_passed']):
        status = 'NUMERICAL_FAILED'
    row = dict(model=model['model'], status=status, split_hybrid='on' if split_hybrid else 'off',
               timing_method='nvtx_ks_segment_spans', end_to_end_ms=milliseconds(timing_ns['total_ms']),
               key_load_ms=milliseconds(timing_ns['key_load_ms']),
               weight_preencode_ms=milliseconds(timing_ns['weight_preencode_ms']),
               init_and_teardown_ms=milliseconds(other_ns),
               inference_total_ms=milliseconds(timing_ns['inference_ms']),
               inference_mean_ms=milliseconds(sum(image_ns[i] for i in measured_ids))/len(measured),
               keyswitch_mean_ms=ks_mean,
               hybrid_mean_ms=hybrid_mean,
               non_keyswitch_mean_ms=milliseconds(sum(residual_ns[i] for i in measured_ids))/len(measured))
    for i in ids:
        row[f'image_{i}_keyswitch_ms'] = milliseconds(ks_ns[i])
        row[f'image_{i}_hybrid_ms'] = milliseconds(hybrid_ns[i])
        row[f'image_{i}_non_keyswitch_ms'] = milliseconds(residual_ns[i])
    # Reuse the existing interval exporter, including its warmup and coverage checks.
    timeline = generate(folder/'analysis/ks_segments.csv', target,
                        activities_path=folder/'analysis/gpu_activities.csv.gz')
    if timeline['measured_images'] != measured_ids or [r['image'] for r in timeline['images']] != sorted(ids):
        raise ValueError('timeline and wall image census disagree')
    for image in timeline['images']:
        if image['merged_overlap_ns']:
            raise ValueError('timeline has overlapping KS spans')
        if image['keyswitch_ns'] != ks_ns[image['image']]:
            raise ValueError('timeline KS nanoseconds disagree')
        if image['hybrid_ns'] != hybrid_ns[image['image']]:
            raise ValueError('timeline Hybrid nanoseconds disagree')
        inference = next(r['inference_ms'] for r in images if r['image'] == image['image'])
        number(inference-image['window_ms'], 'Inference minus GPU activity envelope')
    return row


def summarize(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    batch = read_json(source/'summary.json')
    if batch.get('status') not in ('PASS', 'NUMERICAL_FAILED', 'FAILED'):
        raise ValueError('benchmark must be finished before exporting')
    count, warmup = batch['count'], batch['warmup']
    if type(count) is not int or type(warmup) is not int or not 0 <= warmup < count:
        raise ValueError('require 0 <= warmup < count in benchmark summary')
    if batch.get('split_hybrid') not in ('on', 'off'):
        raise ValueError('require benchmark_nvtx.py output with split_hybrid=on/off')
    models = batch['models']
    names = [r['model'] for r in models]
    if not names or len(set(names)) != len(names) or any(
            not isinstance(name, str) or Path(name).name != name or name in ('.', '..') for name in names):
        raise ValueError('invalid or duplicate model names')
    if output == source or source in output.parents:
        raise ValueError('output must be separate from the benchmark input directory')
    if output.exists():
        raise ValueError('output directory already exists; choose a new directory')
    output.mkdir(parents=True)
    records = []
    with (output/'00_summary.csv').open('w', newline='') as handle:
        fields = (*FIELDS, *[f'image_{i}_{kind}_ms' for i in range(count)
                             for kind in ('keyswitch', 'hybrid', 'non_keyswitch')],
                  'split_hybrid', 'timing_method', 'status', 'error')
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in models:
            target = output/model['model']
            try:
                if model['status'] == 'FAILED':
                    raise ValueError(model.get('error', 'benchmark model failed'))
                row = summarize_model(source, target, model, count, warmup, batch['split_hybrid']=='on')
            except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
                if target.exists():
                    shutil.rmtree(target)
                row = dict(model=model['model'], status='FAILED', error=str(error))
            records.append(row)
            writer.writerow(row)
            handle.flush()
            print(f'{row["model"]}: {row["status"]}'+(': '+row['error'] if row.get('error') else ''), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='completed benchmark_nvtx.py output directory')
    parser.add_argument('--output', type=Path, required=True, help='new directory for 00_summary.csv and model timelines')
    args = parser.parse_args()
    try:
        records = summarize(args.input, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print('Report:', args.output/'00_summary.csv')
    return int(any(r.get('error') for r in records))


if __name__ == '__main__':
    raise SystemExit(main())
