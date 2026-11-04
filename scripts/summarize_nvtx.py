#!/usr/bin/env python3
"""Export one profiled-only timing table and per-model timelines from benchmark_nvtx.py."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from benchmark_nvtx import number, time_values
from inference_timing import close, validate_run
from keyswitch_timeline import generate


FIELDS = (
    'model', 'status', 'diagnostic_only', 'numerical_gate', 'numerical_passed', 'numerical_failures',
    'encode_mode', 'plaintext_reference', 'image_count', 'warmup_count', 'measured_count',
    'time_real_ms', 'time_user_ms', 'time_sys_ms', 'end_to_end_ms',
    'model_prepare_ms', 'boot_prepare_ms', 'prepare_total_ms', 'key_load_ms',
    'weight_preencode_ms', 'other_prepare_and_teardown_ms', 'inference_total_ms',
    'first_image', 'first_image_inference_ms', 'warmup_inference_total_ms',
    'measured_inference_total_ms', 'inference_mean_ms',
    'keyswitch_total_ms', 'non_keyswitch_total_ms',
    'first_image_keyswitch_ms', 'first_image_non_keyswitch_ms',
    'keyswitch_mean_ms', 'non_keyswitch_mean_ms',
    'timeline_scope', 'timeline_window_mean_ms', 'inference_outside_timeline_mean_ms',
    'timeline_directory', 'error',
)


def read_json(path):
    return json.loads(path.read_text())


def summarize_model(source, target, model, count, warmup):
    folder = source/model['model']/'profiled'
    wall = read_json(folder/'wall.json')
    validate_run(wall)
    if wall.get('timing_detail') != 'coarse' or wall.get('ks_profile') or not wall.get('split_hybrid'):
        raise ValueError('require coarse profiled timing with split-hybrid on')
    if wall.get('key_mode') != 'load' or wall.get('keys_generated') != 0 or wall.get('keys_loaded') != wall.get('key_count'):
        raise ValueError('profiled run must load all keys without generation')
    images = wall['images']
    if len(images) != count:
        raise ValueError('wall image count disagrees with benchmark count')
    measured = images[warmup:]
    ids = [r['image'] for r in images]
    measured_ids = [r['image'] for r in measured]
    analysis = read_json(folder/'analysis/summary.json')
    spans = analysis.get('ks_spans')
    if (analysis.get('method') != 'nsys_keyswitch_segment_gpu_span_v1' or
            not analysis.get('split_hybrid') or not spans or not spans.get('coverage_verified')):
        raise ValueError('missing verified KS stage spans; split-hybrid on collection is required')
    if analysis['measured_images'] != measured_ids:
        raise ValueError('NVTX and wall timing use different measured images')
    if any(r['kernel_overlap_ns'] for r in analysis['images']):
        raise ValueError('overlapping kernels cannot partition inference elapsed time')
    ks = {int(image): number(row['keyswitch_span_ns'], 'KS span')/1e6
          for image, row in spans['images'].items()}
    if set(ks) != set(ids):
        raise ValueError('KS and wall image census disagree')
    residual = {r['image']: number(r['inference_ms']-ks[r['image']], 'Inference minus KS') for r in images}
    inference_mean = sum(r['inference_ms'] for r in measured)/len(measured)
    ks_mean = sum(ks[i] for i in measured_ids)/len(measured)
    close(ks_mean, analysis['mean_ms']['keyswitch_span'], 'KS mean')
    other = number(wall['total_ms']-wall['key_load_ms']-wall['weight_preencode_ms']-wall['inference_ms'],
                   'other preparation and teardown')
    row = dict(model=model['model'], status=model['status'],
               diagnostic_only=bool(model.get('diagnostic_only') or not wall['numerical_passed'] or model['status'] != 'PASS'),
               numerical_gate=wall['numerical_gate'], numerical_passed=wall['numerical_passed'],
               numerical_failures=wall['numerical_failures'], encode_mode=wall['encode_mode'],
               plaintext_reference=wall['plaintext_reference'], image_count=count,
               warmup_count=warmup, measured_count=len(measured), **time_values(folder/'time.txt'),
               end_to_end_ms=wall['total_ms'], model_prepare_ms=wall['model_prepare_ms'],
               boot_prepare_ms=wall['boot_prepare_ms'],
               prepare_total_ms=wall['model_prepare_ms']+wall['boot_prepare_ms'],
               key_load_ms=number(wall['key_load_ms'], 'key load'),
               weight_preencode_ms=number(wall['weight_preencode_ms'], 'weight preencode'),
               other_prepare_and_teardown_ms=other, inference_total_ms=wall['inference_ms'],
               first_image=ids[0], first_image_inference_ms=images[0]['inference_ms'],
               warmup_inference_total_ms=sum(r['inference_ms'] for r in images[:warmup]),
               measured_inference_total_ms=sum(r['inference_ms'] for r in measured),
               inference_mean_ms=inference_mean, keyswitch_total_ms=sum(ks.values()),
               non_keyswitch_total_ms=sum(residual.values()), first_image_keyswitch_ms=ks[ids[0]],
               first_image_non_keyswitch_ms=residual[ids[0]], keyswitch_mean_ms=ks_mean,
               non_keyswitch_mean_ms=inference_mean-ks_mean, timeline_directory=model['model'])
    # Reuse the existing interval exporter, including its warmup and coverage checks.
    timeline = generate(folder/'analysis/ks_segments.csv', target,
                        activities_path=folder/'analysis/gpu_activities.csv.gz')
    if timeline['measured_images'] != measured_ids or [r['image'] for r in timeline['images']] != sorted(ids):
        raise ValueError('timeline and wall image census disagree')
    for image in timeline['images']:
        if image['merged_overlap_ns']:
            raise ValueError('timeline has overlapping KS spans')
        close(image['keyswitch_ms'], ks[image['image']], 'timeline KS')
        inference = next(r['inference_ms'] for r in images if r['image'] == image['image'])
        number(inference-image['window_ms'], 'Inference minus GPU activity envelope')
    row.update(timeline_scope=timeline['scope'], timeline_window_mean_ms=timeline['mean_ms']['window'],
               inference_outside_timeline_mean_ms=inference_mean-timeline['mean_ms']['window'])
    return row


def summarize(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    batch = read_json(source/'summary.json')
    if batch.get('status') not in ('PASS', 'NUMERICAL_FAILED', 'FAILED'):
        raise ValueError('benchmark must be finished before exporting')
    count, warmup = batch['count'], batch['warmup']
    if type(count) is not int or type(warmup) is not int or not 0 <= warmup < count:
        raise ValueError('require 0 <= warmup < count in benchmark summary')
    if batch.get('split_hybrid') != 'on':
        raise ValueError('require benchmark_nvtx.py --split-hybrid on output')
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
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for model in models:
            target = output/model['model']
            try:
                if model['status'] == 'FAILED':
                    raise ValueError(model.get('error', 'benchmark model failed'))
                row = summarize_model(source, target, model, count, warmup)
            except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
                if target.exists():
                    shutil.rmtree(target)
                row = dict(model=model['model'], status='FAILED', diagnostic_only=True, error=str(error))
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
