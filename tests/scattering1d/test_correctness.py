import pytest
import numpy as np
import warnings
from kymatio.scattering1d.backend.agnostic_backend import pad

run_without_pytest = 0


def _test_padding(backend_name):
    def _arange(N):
        if backend_name == 'tensorflow':
            return backend.range(N)
        return backend.arange(N)

    if backend_name == 'numpy':
        backend = np
    elif backend_name == 'torch':
        import torch
        backend = torch
    elif backend_name == 'tensorflow':
        import tensorflow as tf
        backend = tf

    for N in (128, 129):  # even, odd
        x = backend.reshape(_arange(6 * N), (2, 3, N))
        for pad_factor in (1, 2, 3, 4):
            pad_left = (N // 2) * pad_factor
            pad_right = int(np.ceil(N / 4) * pad_factor)

            for pad_mode in ('zero', 'reflect'):
                out0 = pad(x, pad_left, pad_right, pad_mode=pad_mode,
                           backend_name=backend_name)
                out1 = np.pad(x,
                              [[0, 0]] * (x.ndim - 1) + [[pad_left, pad_right]],
                              mode=pad_mode if pad_mode != 'zero' else 'constant')

                out0 = out0.numpy() if hasattr(out0, 'numpy') else out0
                assert np.allclose(out0, out1), (
                    "{} | (N, pad_mode, pad_left, pad_right) = ({}, {}, {}, {})"
                    ).format(backend_name, N, pad_mode, pad_left, pad_right)


def test_pad_numpy():
    _test_padding('numpy')


def test_pad_torch():
    try:
        import torch
    except ImportError:
        warnings.warn("Failed to import torch")
        return
    _test_padding('torch')


def test_pad_tensorflow():
    try:
        import tensorflow
    except ImportError:
        warnings.warn("Failed to import tensorflow")
        return
    _test_padding('tensorflow')


if __name__ == '__main__':
    if run_without_pytest:
        test_pad_numpy()
        test_pad_torch()
        test_pad_tensorflow()
    else:
        pytest.main([__file__, "-s"])
