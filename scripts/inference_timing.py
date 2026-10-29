"""The single timing schema used by launchers and validation reports (milliseconds)."""
import math
import statistics

FHE_FIELDS = ('linear_ms', 'bootstrap_ms', 'activation_ms', 'graph_ops_ms')
LEAF_FIELDS = ('input_ms', 'runtime_weight_prepare_ms', *FHE_FIELDS,
               'reference_ms', 'validation_output_ms', 'dispatch_ms')
IMAGE_FIELDS = ('inference_ms', 'fhe_eval_ms', *LEAF_FIELDS)


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-6):
        raise ValueError(f'{label}: timing components do not sum ({actual} != {expected})')


def validate_image(row):
    if row.get('schema_version') != 2:
        raise ValueError('timing schema_version must be 2; rebuild both inference binaries')
    for key in IMAGE_FIELDS:
        value = row[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid timing value: {key}')
    close(row['fhe_eval_ms'], sum(row[k] for k in FHE_FIELDS), 'FHE evaluation')
    close(row['inference_ms'], sum(row[k] for k in LEAF_FIELDS), 'inference')


def validate_totals(row):
    validate_image(row)
    for key in ('model_prepare_ms', 'boot_prepare_ms', 'total_ms'):
        if not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or row[key] < 0:
            raise ValueError(f'invalid timing value: {key}')
    close(row['total_ms'], row['model_prepare_ms'] + row['boot_prepare_ms'] + row['inference_ms'], 'run')


def validate_run(row):
    validate_totals(row)
    images = row['images']
    if not images:
        raise ValueError('missing image timings')
    for image in images:
        validate_image(image)
    if any(type(image.get('image')) is not int for image in images):
        raise ValueError('invalid image index in timing report')
    if len({image['image'] for image in images}) != len(images):
        raise ValueError('duplicate image timing')
    for key in IMAGE_FIELDS:
        close(row[key], sum(image[key] for image in images), f'image sum: {key}')


def summarize(images, warmup):
    if not 0 <= warmup < len(images):
        raise ValueError('warmup must leave at least one measured image')
    measured = images[warmup:]
    means = {k: statistics.fmean(row[k] for row in measured) for k in IMAGE_FIELDS}
    validate_image(dict(schema_version=2, **means))
    return dict(warmup_count=warmup, measured_count=len(measured), mean_ms=means,
                median_ms={k: statistics.median(row[k] for row in measured)
                           for k in ('inference_ms', 'fhe_eval_ms')})


def print_human(row):
    summary = row['statistics']
    print('\nTiming (wall clock, ms; all images including warmup)')
    for label, key in (('Model resources / process overhead', 'model_prepare_ms'),
                       ('Boot resource preparation', 'boot_prepare_ms'),
                       ('Inference stage', 'inference_ms')):
        print(f'  {label:36}: {row[key]:.3f}')
    for label, key in (('Input preparation', 'input_ms'),
                       ('Runtime weight preparation', 'runtime_weight_prepare_ms'),
                       ('FHE evaluation', 'fhe_eval_ms'),
                       ('  Linear layers', 'linear_ms'),
                       ('  Bootstrap execution', 'bootstrap_ms'),
                       ('  Activation excluding Boot', 'activation_ms'),
                       ('  Other ciphertext operations', 'graph_ops_ms'),
                       ('Plaintext reference', 'reference_ms'),
                       ('Validation and output', 'validation_output_ms'),
                       ('Dispatch and other overhead', 'dispatch_ms')):
        print(f'    {label:34}: {row[key]:.3f}')
    print(f"  Sum / end-to-end wall                : {row['total_ms']:.3f}")
    if row['image_count'] > 1 or summary['warmup_count']:
        mean = summary['mean_ms']
        print(f"\nMean per measured image (warmup={summary['warmup_count']}, measured={summary['measured_count']})")
        for key in ('inference_ms', 'runtime_weight_prepare_ms', 'fhe_eval_ms'):
            print(f'  {key:36}: {mean[key]:.3f}')
