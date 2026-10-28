"""Run with: python3 tests/test_export_all_resnet_weights.py"""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_all_resnet_weights import export_all


with tempfile.TemporaryDirectory() as directory:
    source = Path(directory) / 'onnx with spaces'
    source.mkdir()
    output = Path(directory) / 'output with spaces'
    with patch('export_all_resnet_weights.export') as export:
        try:
            export_all(source, output)
        except FileNotFoundError as error:
            assert 'resnet110_cifar10_train.onnx' in str(error)
        else:
            raise AssertionError('Missing models must fail before exporting')
        export.assert_not_called()
        for depth in (20, 32, 44, 56, 110):
            (source / f'resnet{depth}_cifar10_train.onnx').touch()
        export_all(source, output)
        assert [call.args for call in export.call_args_list] == [
            (source / f'resnet{depth}_cifar10_train.onnx', output / f'resnet{depth}.bin', depth)
            for depth in (20, 32, 44, 56, 110)]
print('PASS: missing-file preflight and all five export paths/depths')
