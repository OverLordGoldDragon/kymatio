import tensorflow as tf

from ...backend.tensorflow_backend import TensorFlowBackend
from .agnostic_backend import pad as agnostic_pad


class TensorFlowBackend1D(TensorFlowBackend):
    @classmethod
    def subsample_fourier(cls, x, k):
        """Subsampling in the Fourier domain
        Subsampling in the temporal domain amounts to periodization in the Fourier
        domain, so the input is periodized according to the subsampling factor.
        Parameters
        ----------
        x : tensor
            Input tensor with at least 3 dimensions, where the next to last
            corresponds to the frequency index in the standard PyTorch FFT
            ordering. The length of this dimension should be a power of 2 to
            avoid errors. The last dimension should represent the real and
            imaginary parts of the Fourier transform.
        k : int
            The subsampling factor.
        Returns
        -------
        res : tensor
            The input tensor periodized along the next to last axis to yield a
            tensor of size x.shape[-2] // k along that dimension.
        """
        cls.complex_check(x)

        y = tf.reshape(x, (*x.shape[:-1], k, x.shape[-1] // k))

        return tf.reduce_mean(y, axis=-2)

    @staticmethod
    def pad(x, pad_left, pad_right, pad_mode='reflect'):
        """Pad real 1D tensors
        1D implementation of the padding function for real PyTorch tensors.
        Parameters
        ----------
        x : tensor
            Three-dimensional input tensor with the third axis being the one to
            be padded.
        pad_left : int
            Amount to add on the left of the tensor (at the beginning of the
            temporal axis).
        pad_right : int
            amount to add on the right of the tensor (at the end of the temporal
            axis).
        pad_mode : str
            name of padding to use.
        Returns
        -------
        res : tensor
            The tensor passed along the third dimension.
        """
        return agnostic_pad(x, pad_left, pad_right, pad_mode, 'tensorflow')

    @staticmethod
    def unpad(x, i0, i1):
        """Unpad real 1D tensor
        Slices the input tensor at indices between i0 and i1 along the last axis.
        Parameters
        ----------
        x : tensor
            Input tensor with least one axis.
        i0 : int
            Start of original signal before padding.
        i1 : int
            End of original signal before padding.
        Returns
        -------
        x_unpadded : tensor
            The tensor x[..., i0:i1].
        """
        return x[..., i0:i1]

    @classmethod
    def zeros_like(cls, ref, shape=None):
        shape = shape if shape is not None else ref.shape
        return tf.zeros(shape, dtype=ref.dtype)

    @classmethod
    def fft(cls, x, axis=-1):  # TODO transpose?
        return tf.signal.fft(x, name='fft1d')

    @classmethod
    def rfft(cls, x):
        cls.real_check(x)

        return tf.signal.fft(tf.cast(x, tf.complex64), name='rfft1d')

    @classmethod
    def irfft(cls, x):
        cls.complex_check(x)

        return tf.math.real(tf.signal.ifft(x, name='irfft1d'))

    @classmethod
    def ifft(cls, x):
        cls.complex_check(x)

        return tf.signal.ifft(x, name='ifft1d')

    @classmethod
    def transpose(cls, x):
        """Permute time and frequency dimension for time-frequency scattering"""
        D = x.ndim
        return tf.transpose(x, (*list(range(D - 2)), D - 1, D - 2))

    @classmethod
    def mean(cls, x, axis=-1):
        """Take mean along specified axis, without collapsing the axis."""
        return tf.reduce_mean(x, axis=axis, keepdims=True)

backend = TensorFlowBackend1D
