import tensorflow as tf
import warnings

from ...frontend.tensorflow_frontend import ScatteringTensorFlow
from ..core.scattering1d import scattering1d
from ..core.timefrequency_scattering import timefrequency_scattering
from ..utils import precompute_size_scattering
from .base_frontend import (ScatteringBase1D, TimeFrequencyScatteringBase1D,
                            _check_runtime_args_jtfs)


class ScatteringTensorFlow1D(ScatteringTensorFlow, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, T=None, max_order=2, average=True,
            oversampling=0, out_type='array', pad_mode='reflect',
            max_pad_factor=2, backend='tensorflow', name='Scattering1D'):
        ScatteringTensorFlow.__init__(self, name=name)
        ScatteringBase1D.__init__(self, J, shape, Q, T, max_order, average,
                oversampling, out_type, pad_mode, max_pad_factor, backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)

    def scattering(self, x):
        # basic checking, should be improved
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        if not self.out_type in ('array', 'list'):
            raise RuntimeError("The out_type must be one of 'array' or 'list'.")

        if self.out_type == 'array' and not self.average:
            raise ValueError("out_type=='array' and average==False are mutually "
                             "incompatible. Please set out_type='list'.")

        batch_shape = tf.shape(x)[:-1]
        signal_shape = tf.shape(x)[-1:]

        x = tf.reshape(x, tf.concat(((-1, 1), signal_shape), 0))

        # get the arguments before calling the scattering
        # treat the arguments
        if self.average:
            size_scattering = precompute_size_scattering(
                self.J, self.Q, max_order=self.max_order, detail=True)
        else:
            size_scattering = 0

        S = scattering1d(x, self.backend.pad, self.backend.unpad, self.backend, self.J, self.psi1_f, self.psi2_f,
                         self.phi_f, max_order=self.max_order, average=self.average, pad_left=self.pad_left,
                         pad_right=self.pad_right, ind_start=self.ind_start, ind_end=self.ind_end,
                         oversampling=self.oversampling,
                         size_scattering=size_scattering,
                         out_type=self.out_type)

        if self.out_type == 'array':
            scattering_shape = tf.shape(S)[-2:]
            new_shape = tf.concat((batch_shape, scattering_shape), 0)

            S = tf.reshape(S, new_shape)
        else:
            for x in S:
                scattering_shape = tf.shape(x['coef'])[-1:]
                new_shape = tf.concat((batch_shape, scattering_shape), 0)

                x['coef'] = tf.reshape(x['coef'], new_shape)

        return S


ScatteringTensorFlow1D._document()


class TimeFrequencyScatteringTensorFlow1D(TimeFrequencyScatteringBase1D,
                                          ScatteringTensorFlow1D):
    def __init__(self, J, shape, Q, J_fr=None, Q_fr=2, T=None, F=None,
                 average=True, average_fr=False, oversampling=0,
                 oversampling_fr=None, aligned=True, resample_filters_fr=True,
                 out_type="array", out_3D=False, pad_mode='zero', max_pad_factor=2,
                 max_pad_factor_fr=None, backend='tensorflow',
                 name='TimeFrequencyScattering1D'):
        if oversampling_fr is None:
            oversampling_fr = oversampling
        # Second-order scattering object for the time variable
        max_order_tm = 2
        scattering_out_type = out_type.lstrip('dict:')
        ScatteringTensorFlow1D.__init__(
            self, J, shape, Q, T, max_order_tm, average, oversampling,
            scattering_out_type, pad_mode, max_pad_factor, backend)

        TimeFrequencyScatteringBase1D.__init__(
            self, J_fr, Q_fr, F, average_fr, oversampling_fr, aligned,
            resample_filters_fr, max_pad_factor_fr, out_3D, out_type)
        TimeFrequencyScatteringBase1D.build(self)

    def scattering(self, x):
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        _check_runtime_args_jtfs(self.average, self.average_fr, self.out_type,
                                 self.out_3D)

        signal_shape = tf.shape(x)[-1:]
        x = tf.reshape(x, tf.concat(((-1, 1), signal_shape), 0))

        S = timefrequency_scattering(
            x,
            self.backend.pad, self.backend.unpad,
            self.backend,
            self.J,
            self.log2_T,
            self.psi1_f, self.psi2_f, self.phi_f,
            self.sc_freq,
            average=self.average,
            average_global=self.average_global,
            pad_left=self.pad_left, pad_right=self.pad_right,
            ind_start=self.ind_start, ind_end=self.ind_end,
            oversampling=self.oversampling,
            oversampling_fr=self.oversampling_fr,
            aligned=self.aligned,
            out_type=self.out_type,
            out_3D=self.out_3D,
            pad_mode=self.pad_mode)
        return S

    def sc_freq_compute_padding_fr(self):
        raise NotImplementedError("Here for docs; implemented in "
                                  "`_FrequencyScatteringBase`.")

    def sc_freq_compute_J_pad(self):
        raise NotImplementedError("Here for docs; implemented in "
                                  "`_FrequencyScatteringBase`.")

TimeFrequencyScatteringTensorFlow1D._document()


__all__ = ['ScatteringTensorFlow1D', 'TimeFrequencyScatteringTensorFlow1D']
