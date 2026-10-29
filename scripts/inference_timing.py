"""The single timing schema used by launchers and validation reports (milliseconds)."""
import math
import statistics

FHE_FIELDS = ('linear_ms', 'bootstrap_ms', 'activation_ms', 'graph_ops_ms')
LEAF_FIELDS = ('input_ms', 'runtime_weight_prepare_ms', *FHE_FIELDS,
               'reference_ms', 'validation_output_ms', 'dispatch_ms')
PROFILE_FIELDS = ('keyswitch_ms', 'non_keyswitch_ms', 'hybrid_ms')
KEY_OPS = ('modup', 'pseudo_modup', 'key_mult', 'moddown', 'automorphism', 'sparse_to_dense')
HYBRID_OPS = ('fused_baby', 'moddown_rescale', 'automorphism_accum', 'shared_lift')
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

    if row.get('ks_profile', False):
        if row.get('profile_method') != 'synchronized_scopes':
            raise ValueError('unknown operation profiling method')
        for key in ('inference_ns', 'keyswitch_ns', 'non_keyswitch_ns', 'hybrid_ns'):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f'invalid profiling nanoseconds: {key}')
            close(row[key.replace('_ns', '_ms')], row[key] / 1e6, key)
        if row['inference_ns'] != row['keyswitch_ns'] + row['non_keyswitch_ns'] + row['hybrid_ns']:
            raise ValueError('operation wall-clock nanoseconds do not sum')
        ops = row['profile_operations']
        if row.get('split_hybrid', False) and any(ops[k]['ns'] or ops[k]['calls'] for k in HYBRID_OPS):
            raise ValueError('hybrid operation remains in split execution')
        for name in (*KEY_OPS, *HYBRID_OPS):
            for key in ('ns', 'calls'):
                if type(ops[name][key]) is not int or ops[name][key] < 0:
                    raise ValueError(f'invalid operation counter: {name}.{key}')
        if sum(ops[k]['ns'] for k in KEY_OPS) != row['keyswitch_ns']:
            raise ValueError('keyswitch operation scopes do not sum')
        if sum(ops[k]['ns'] for k in HYBRID_OPS) != row['hybrid_ns']:
            raise ValueError('hybrid operation scopes do not sum')


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
    if any(image.get('split_hybrid', False) != row.get('split_hybrid', False) for image in images):
        raise ValueError('mixed execution modes')
    if any(image.get('ks_profile', False) != row.get('ks_profile', False) for image in images):
        raise ValueError('mixed operation profiling modes')
    fields = (*IMAGE_FIELDS, *PROFILE_FIELDS) if row.get('ks_profile', False) else IMAGE_FIELDS
    for key in fields:
        close(row[key], sum(image[key] for image in images), f'image sum: {key}')
    if row.get('ks_profile', False):
        for key in ('inference_ns', 'keyswitch_ns', 'non_keyswitch_ns', 'hybrid_ns'):
            if row[key] != sum(image[key] for image in images):
                raise ValueError(f'integer image sum mismatch: {key}')
        for name in (*KEY_OPS, *HYBRID_OPS):
            for key in ('ns', 'calls'):
                if row['profile_operations'][name][key] != sum(image['profile_operations'][name][key] for image in images):
                    raise ValueError(f'operation image sum mismatch: {name}.{key}')


def summarize(images, warmup):
    if not 0 <= warmup < len(images):
        raise ValueError('warmup must leave at least one measured image')
    measured = images[warmup:]
    fields = (*IMAGE_FIELDS, *PROFILE_FIELDS) if measured[0].get('ks_profile', False) else IMAGE_FIELDS
    means = {k: statistics.fmean(row[k] for row in measured) for k in fields}
    validate_image(dict(schema_version=2, **means))
    if measured[0].get('ks_profile', False):
        close(means['inference_ms'], sum(means[k] for k in PROFILE_FIELDS), 'mean operation profile')
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
    if row.get('ks_profile', False):
        print('\n  Inference by operation (same stage; synchronized profiling)')
        for key in PROFILE_FIELDS[:2] if row.get('split_hybrid', False) else PROFILE_FIELDS:
            print(f'    {key:34}: {row[key]:.3f}')
    print(f"  Sum / end-to-end wall                : {row['total_ms']:.3f}")
    if row['image_count'] > 1 or summary['warmup_count']:
        mean = summary['mean_ms']
        print(f"\nMean per measured image (warmup={summary['warmup_count']}, measured={summary['measured_count']})")
        for key in ('inference_ms', 'runtime_weight_prepare_ms', 'fhe_eval_ms'):
            print(f'  {key:36}: {mean[key]:.3f}')
