import numpy as np
import math
import warnings
from scipy.fft import ifft

def adaptive_choice_P(sigma, eps=1e-7):
    """
    Adaptive choice of the value of the number of periods in the frequency
    domain used to compute the Fourier transform of a Morlet wavelet.

    This function considers a Morlet wavelet defined as the sum
    of
    * a Gabor term hat psi(omega) = hat g_{sigma}(omega - xi)
    where 0 < xi < 1 is some frequency and g_{sigma} is
    the Gaussian window defined in Fourier by
    hat g_{sigma}(omega) = e^{-omega^2/(2 sigma^2)}
    * a low pass term \\hat \\phi which is proportional to \\hat g_{\\sigma}.

    If \\sigma is too large, then these formula will lead to discontinuities
    in the frequency interval [0, 1] (which is the interval used by numpy.fft).
    We therefore choose a larger integer P >= 1 such that at the boundaries
    of the Fourier transform of both filters on the interval [1-P, P], the
    magnitude of the entries is below the required machine precision.
    Mathematically, this means we would need P to satisfy the relations:

    |\\hat \\psi(P)| <= eps and |\\hat \\phi(1-P)| <= eps

    Since 0 <= xi <= 1, the latter implies the former. Hence the formula which
    is easily derived using the explicit formula for g_{\\sigma} in Fourier.

    Parameters
    ----------
    sigma: float
        Positive number controlling the bandwidth of the filters
    eps : float, optional
        Positive number containing required precision. Defaults to 1e-7

    Returns
    -------
    P : int
        integer controlling the number of periods used to ensure the
        periodicity of the final Morlet filter in the frequency interval
        [0, 1[. The value of P will lead to the use of the frequency
        interval [1-P, P[, so that there are 2*P - 1 periods.
    """
    val = math.sqrt(-2 * (sigma**2) * math.log(eps))
    P = int(math.ceil(val + 1))
    return P


def periodize_filter_fourier(h_f, nperiods=1, aggregation='sum'):
    """
    Computes a periodization of a filter provided in the Fourier domain.
    Parameters
    ----------
    h_f : array_like
        complex numpy array of shape (N*n_periods,)
    n_periods: int, optional
        Number of periods which should be used to periodize
    aggregation: str['sum', 'mean'], optional
        'sum' will multiply subsampled time-domain signal by subsampling
        factor to conserve energy during scattering (rather not double-account
        for it since we already subsample after convolving).
        'mean' will only subsample the input.

    Returns
    -------
    v_f : array_like
        complex numpy array of size (N,), which is a periodization of
        h_f as described in the formula:
        v_f[k] = sum_{i=0}^{n_periods - 1} h_f[i * N + k]
    """
    N = h_f.shape[0] // nperiods
    h_f_re = h_f.reshape(nperiods, N)
    v_f = (h_f_re.sum(axis=0) if aggregation == 'sum' else
           h_f_re.mean(axis=0))
    v_f = v_f if h_f.ndim == 1 else v_f[:, None]  # preserve dim
    return v_f


def morlet_1d(N, xi, sigma, normalize='l1', P_max=5, eps=1e-7):
    """
    Computes the Fourier transform of a Morlet filter.

    A Morlet filter is the sum of a Gabor filter and a low-pass filter
    to ensure that the sum has exactly zero mean in the temporal domain.
    It is defined by the following formula in time:
    psi(t) = g_{sigma}(t) (e^{i xi t} - beta)
    where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
    the cancelling parameter.

    Parameters
    ----------
    N : int
        size of the temporal support
    xi : float
        central frequency (in [0, 1])
    sigma : float
        bandwidth parameter
    normalize : string, optional
        normalization types for the filters. Defaults to 'l1'.
        Supported normalizations are 'l1' and 'l2' (understood in time domain).
    P_max: int, optional
        integer controlling the maximal number of periods to use to ensure
        the periodicity of the Fourier transform. (At most 2*P_max - 1 periods
        are used, to ensure an equal distribution around 0.5). Defaults to 5
        Should be >= 1
    eps : float
        required machine precision (to choose the adequate P)

    Returns
    -------
    morlet_f : array_like
        numpy array of size (N,) containing the Fourier transform of the Morlet
        filter at the frequencies given by np.fft.fftfreq(N).
    """
    if type(P_max) != int:
        raise ValueError('P_max should be an int, got {}'.format(type(P_max)))
    if P_max < 1:
        raise ValueError('P_max should be non-negative, got {}'.format(P_max))
    # Find the adequate value of P
    P = min(adaptive_choice_P(sigma, eps=eps), P_max)
    assert P >= 1
    # Define the frequencies over [1-P, P[
    freqs = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    if P == 1:
        # in this case, make sure that there is continuity around 0
        # by using the interval [-0.5, 0.5]
        freqs_low = np.fft.fftfreq(N)
    elif P > 1:
        freqs_low = freqs
    # define the gabor at freq xi and the low-pass, both of width sigma
    gabor_f = np.exp(-(freqs - xi)**2 / (2 * sigma**2))
    low_pass_f = np.exp(-(freqs_low**2) / (2 * sigma**2))
    # discretize in signal <=> periodize in Fourier
    gabor_f = periodize_filter_fourier(gabor_f, nperiods=2 * P - 1)
    low_pass_f = periodize_filter_fourier(low_pass_f, nperiods=2 * P - 1)
    # find the summation factor to ensure that morlet_f[0] = 0.
    kappa = gabor_f[0] / low_pass_f[0]
    morlet_f = gabor_f - kappa * low_pass_f
    # normalize the Morlet if necessary
    morlet_f *= get_normalizing_factor(morlet_f, normalize=normalize)
    return morlet_f


def get_normalizing_factor(h_f, normalize='l1'):
    """
    Computes the desired normalization factor for a filter defined in Fourier.

    Parameters
    ----------
    h_f : array_like
        numpy vector containing the Fourier transform of a filter
    normalized : string, optional
        desired normalization type, either 'l1' or 'l2'. Defaults to 'l1'.

    Returns
    -------
    norm_factor : float
        such that h_f * norm_factor is the adequately normalized vector.
    """
    h_real = ifft(h_f)
    if np.abs(h_real).sum() < 1e-7:
        raise ValueError('Zero division error is very likely to occur, ' +
                         'aborting computations now.')
    if normalize == 'l1':
        norm_factor = 1. / (np.abs(h_real).sum())
    elif normalize == 'l2':
        norm_factor = 1. / np.sqrt((np.abs(h_real)**2).sum())
    else:
        raise ValueError("Supported normalizations only include 'l1' and 'l2'")
    return norm_factor


def gauss_1d(N, sigma, normalize='l1', P_max=5, eps=1e-7):
    """
    Computes the Fourier transform of a low pass gaussian window.

    \\hat g_{\\sigma}(\\omega) = e^{-\\omega^2 / 2 \\sigma^2}

    Parameters
    ----------
    N : int
        size of the temporal support
    sigma : float
        bandwidth parameter
    normalize : string, optional
        normalization types for the filters. Defaults to 'l1'
        Supported normalizations are 'l1' and 'l2' (understood in time domain).
    P_max : int, optional
        integer controlling the maximal number of periods to use to ensure
        the periodicity of the Fourier transform. (At most 2*P_max - 1 periods
        are used, to ensure an equal distribution around 0.5). Defaults to 5
        Should be >= 1
    eps : float, optional
        required machine precision (to choose the adequate P)

    Returns
    -------
    g_f : array_like
        numpy array of size (N,) containing the Fourier transform of the
        filter (with the frequencies in the np.fft.fftfreq convention).
    """
    # Find the adequate value of P
    if type(P_max) != int:
        raise ValueError('P_max should be an int, got {}'.format(type(P_max)))
    if P_max < 1:
        raise ValueError('P_max should be non-negative, got {}'.format(P_max))
    P = min(adaptive_choice_P(sigma, eps=eps), P_max)
    assert P >= 1
    # switch cases
    if P == 1:
        freqs_low = np.fft.fftfreq(N)
    elif P > 1:
        freqs_low = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    # define the low pass
    g_f = np.exp(-freqs_low**2 / (2 * sigma**2))
    # periodize it
    g_f = periodize_filter_fourier(g_f, nperiods=2 * P - 1)
    # normalize the signal
    g_f *= get_normalizing_factor(g_f, normalize=normalize)
    # return the Fourier transform
    return g_f


def compute_sigma_psi(xi, Q, r=math.sqrt(0.5)):
    """
    Computes the frequential width sigma for a Morlet filter of frequency xi
    belonging to a family with Q wavelets.

    The frequential width is adapted so that the intersection of the
    frequency responses of the next filter occurs at a r-bandwidth specified
    by r, to ensure a correct coverage of the whole frequency axis.

    Parameters
    ----------
    xi : float
        frequency of the filter in [0, 1]
    Q : int
        number of filters per octave, Q is an integer >= 1
    r : float, optional
        Positive parameter defining the bandwidth to use.
        Should be < 1. We recommend keeping the default value.
        The larger r, the larger the filters in frequency domain.

    Returns
    -------
    sigma : float
        frequential width of the Morlet wavelet.

    Refs
    ----
    Convolutional operators in the time-frequency domain, V. Lostanlen,
    PhD Thesis, 2017
    https://tel.archives-ouvertes.fr/tel-01559667
    """
    factor = 1. / math.pow(2, 1. / Q)
    term1 = (1 - factor) / (1 + factor)
    term2 = 1. / math.sqrt(2 * math.log(1. / r))
    return xi * term1 * term2


def compute_temporal_support(h_f, criterion_amplitude=1e-3, warn=False):
    """
    Computes the (half) temporal support of a family of centered,
    symmetric filters h provided in the Fourier domain

    This function computes the support N which is the smallest integer
    such that for all signals x and all filters h,

    \\| x \\conv h - x \\conv h_{[-N, N]} \\|_{\\infty} \\leq \\epsilon
        \\| x \\|_{\\infty}  (1)

    where 0<\\epsilon<1 is an acceptable error, and h_{[-N, N]} denotes the
    filter h whose support is restricted in the interval [-N, N]

    The resulting value N used to pad the signals to avoid boundary effects
    and numerical errors.

    If the support is too small, no such N might exist.
    In this case, N is defined as the half of the support of h, and a
    UserWarning is raised.

    Parameters
    ----------
    h_f : array_like
        a numpy array of size batch x time, where each row contains the
        Fourier transform of a filter which is centered and whose absolute
        value is symmetric
    criterion_amplitude : float, optional
        value \\epsilon controlling the numerical
        error. The larger criterion_amplitude, the smaller the temporal
        support and the larger the numerical error. Defaults to 1e-3
    warn: bool (default False)
        Whether to raise a warning upon `h_f` leading to boundary effects.

    Returns
    -------
    t_max : int
        temporal support which ensures (1) for all rows of h_f

    """
    h = ifft(h_f, axis=1)
    half_support = h.shape[1] // 2
    # check if any value in half of worst case of abs(h) is below criterion
    hhalf = np.max(np.abs(h[:, :half_support]), axis=0)
    max_amplitude = hhalf.max()
    meets_criterion_idxs = np.where(hhalf <= criterion_amplitude * max_amplitude
                                    )[0]
    if len(meets_criterion_idxs) != 0:
        # if it is possible
        N = meets_criterion_idxs.min() + 1
        # in this case pretend it's 1 less so external computations don't
        # have to double support since this is close enough
        if N == half_support:
            N -= 1
    else:
        # if there are none
        N = half_support
        if warn:
            # Raise a warning to say that there will be border effects
            warnings.warn('Signal support is too small to avoid border effects')
    return N


def compute_minimum_required_length(fn, N_init, max_N=None,
                                    criterion_amplitude=1e-3):
    """Computes minimum required number of samples for `fn(N)` to have temporal
    support less than `N`, as determined by `compute_temporal_support`.

    Parameters
    ----------
    fn: FunctionType
        Function / lambda taking `N` as input and returning a filter in
        frequency domain.
    N_init: int
        Initial input to `fn`, will keep doubling until `N == max_N` or
        temporal support of `fn` is `< N`.
    max_N: int / None
        See `N_init`; if None, will raise `N` indefinitely.
    criterion_amplitude : float, optional
        value \\epsilon controlling the numerical
        error. The larger criterion_amplitude, the smaller the temporal
        support and the larger the numerical error. Defaults to 1e-3

    Returns
    -------
    N: int
        Minimum required number of samples for `fn(N)` to have temporal
        support less than `N`.
    """
    N = 2**math.ceil(math.log2(N_init))  # ensure pow 2
    while True:
        try:
            p_fr = fn(N)
        except ValueError:  # get_normalizing_factor()
            N *= 2
            continue

        p_halfwidth = compute_temporal_support(
            p_fr.reshape(1, -1), criterion_amplitude=criterion_amplitude)

        if N > 1e9:  # avoid crash
            raise Exception("couldn't satisfy stop criterion before `N > 1e9`; "
                            "check `fn`")
        if 2 * p_halfwidth < N or (max_N is not None and N > max_N):
            break
        N *= 2
    return N


def get_max_dyadic_subsampling(xi, sigma, alpha=4.):
    """
    Computes the maximal dyadic subsampling which is possible for a Gabor
    filter of frequency xi and width sigma

    Finds the maximal integer j such that:
    omega_0 < 2^{-(j + 1)}
    where omega_0 is the boundary of the filter, defined as
    omega_0 = xi + alpha * sigma

    This ensures that the filter can be subsampled by a factor 2^j without
    aliasing.

    We use the same formula for Gabor and Morlet filters.

    Parameters
    ----------
    xi : float
        frequency of the filter in [0, 1]
    sigma : float
        frequential width of the filter
    alpha : float, optional
        parameter controlling the error done in the aliasing.
        The larger alpha, the smaller the error. Defaults to 4.

    Returns
    -------
    j : int
        integer such that 2^j is the maximal subsampling accepted by the
        Gabor filter without aliasing.
    """
    upper_bound = min(xi + alpha * sigma, 0.5)
    j = math.floor(-math.log2(upper_bound)) - 1
    j = int(j)
    return j


def move_one_dyadic_step(cv, Q, alpha=4.):
    """
    Computes the parameters of the next wavelet on the low frequency side,
    based on the parameters of the current wavelet.

    This function is used in the loop defining all the filters, starting
    at the wavelet frequency and then going to the low frequencies by
    dyadic steps. This makes the loop in compute_params_filterbank much
    simpler to read.

    The steps are defined as:
    xi_{n+1} = 2^{-1/Q} xi_n
    sigma_{n+1} = 2^{-1/Q} sigma_n

    Parameters
    ----------
    cv : dictionary
        stands for current_value. Is a dictionary with keys:
        *'key': a tuple (j, n) where n is a counter and j is the maximal
            dyadic subsampling accepted by this wavelet.
        *'xi': central frequency of the wavelet
        *'sigma': width of the wavelet
    Q : int
        number of wavelets per octave. Controls the relationship between
        the frequency and width of the current wavelet and the next wavelet.
    alpha : float, optional
        tolerance parameter for the aliasing. The larger alpha,
        the more conservative the algorithm is. Defaults to 4.

    Returns
    -------
    new_cv : dictionary
        a dictionary with the same keys as the ones listed for cv,
        whose values are updated
    """
    factor = 1. / math.pow(2., 1. / Q)
    n = cv['key']
    new_cv = {'xi': cv['xi'] * factor, 'sigma': cv['sigma'] * factor}
    # compute the new j
    new_cv['j'] = get_max_dyadic_subsampling(new_cv['xi'], new_cv['sigma'], alpha=alpha)
    new_cv['key'] = n + 1
    return new_cv


def compute_xi_max(Q):
    """
    Computes the maximal xi to use for the Morlet family, depending on Q.

    Parameters
    ----------
    Q : int
        number of wavelets per octave (integer >= 1)

    Returns
    -------
    xi_max : float
        largest frequency of the wavelet frame.
    """
    xi_max = max(1. / (1. + math.pow(2., 3. / Q)), 0.35)
    return xi_max


def compute_params_filterbank(sigma_min, Q, r_psi=math.sqrt(0.5), alpha=4.,
                              xi_min=None):
    """
    Computes the parameters of a Morlet wavelet filterbank.

    This family is defined by constant ratios between the frequencies and
    width of adjacent filters, up to a minimum frequency where the frequencies
    are translated. sigma_min specifies the smallest frequential width
    among all filters, while preserving the coverage of the whole frequency
    axis.

    The keys of the dictionaries are tuples of integers (j, n) where n is a
    counter (starting at 0 for the highest frequency filter) and j is the
    maximal dyadic subsampling accepted by this filter.

    Parameters
    ----------
    sigma_min : float
        This acts as a lower-bound on the frequential widths of the band-pass
        filters. The low-pass filter may be wider (if T < 2**J_scattering), making
        invariants over shorter time scales than longest band-pass filter.
    Q : int
        number of wavelets per octave.
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets).
        Defaults to sqrt(0.5).
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 4.
    xi_min : float, optional
        Lower bound on `xi` to ensure every bandpass is a valid wavelet
        (doesn't peak at FFT bin 1) within `2*len(x)` padding.

    Returns
    -------
    xi : dictionary
        dictionary containing the central frequencies of the wavelets.
    sigma : dictionary
        dictionary containing the frequential widths of the wavelets.
    # TODO j

    Refs
    ----
    Convolutional operators in the time-frequency domain, 2.1.3, V. Lostanlen,
    PhD Thesis, 2017
    https://tel.archives-ouvertes.fr/tel-01559667
    """
    xi_min = xi_min if xi_min is not None else -1
    xi_max = compute_xi_max(Q)
    sigma_max = compute_sigma_psi(xi_max, Q, r=r_psi)

    xi = []
    sigma = []
    j = []

    if sigma_max <= sigma_min or xi_max <= xi_min:
        # in this exceptional case, we will not go through the loop, so
        # we directly assign
        last_xi = sigma_max
    else:
        # fill all the dyadic wavelets as long as possible
        current = {'key': 0, 'j': 0, 'xi': xi_max, 'sigma': sigma_max}
        # while we can attribute something
        while current['sigma'] > sigma_min and current['xi'] > xi_min:
            xi.append(current['xi'])
            sigma.append(current['sigma'])
            j.append(current['j'])
            current = move_one_dyadic_step(current, Q, alpha=alpha)
        # get the last key
        last_xi = xi[-1]
    # fill num_interm wavelets between last_xi and 0, both excluded
    num_intermediate = Q - 1
    for q in range(1, num_intermediate + 1):
        factor = (num_intermediate + 1. - q) / (num_intermediate + 1.)
        new_xi = factor * last_xi
        new_sigma = sigma_min
        if new_xi < xi_min:
            break
        xi.append(new_xi)
        sigma.append(new_sigma)
        j.append(get_max_dyadic_subsampling(new_xi, new_sigma, alpha=alpha))
    # return results
    return xi, sigma, j


def calibrate_scattering_filters(J, Q, T, r_psi=math.sqrt(0.5), sigma0=0.1,
                                 alpha=4., xi_min=None):
    """
    Calibrates the parameters of the filters used at the 1st and 2nd orders
    of the scattering transform.

    These filterbanks share the same low-pass filterbank, but use a
    different Q: Q_1 = Q and Q_2 = 1.

    The dictionaries for the band-pass filters have keys which are 2-tuples
    of the type (j, n), where n is an integer >=0 counting the filters (for
    identification purposes) and j is an integer >= 0 denoting the maximal
    subsampling 2**j which can be performed on a signal convolved with this
    filter without aliasing.

    Parameters
    ----------
    J : int
        maximal scale of the scattering (controls the number of wavelets)
    Q : int / tuple[int]
        The number of first-order wavelets per octave. Defaults to `1`.
        If tuple, sets `Q = (Q1, Q2)`, where `Q2` is the number of
        second-order wavelets per octave (which defaults to `1`).
            - Q1: For audio signals, a value of `>= 12` is recommended in
              order to separate partials.
            - Q2: Recommended `2` or `1` for most applications.
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and maximum subsampling
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets).
        Defaults to sqrt(0.5)
    sigma0 : float, optional
        frequential width of the low-pass filter at scale J=0
        (the subsequent widths are defined by sigma_J = sigma0 / 2^J).
        Defaults to 1e-1
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 4.
    xi_min : float, optional
        Lower bound on `xi` to ensure every bandpass is a valid wavelet
        (doesn't peak at FFT bin 1) within `2*len(x)` padding.

    Returns
    -------
    sigma_low : float
        frequential width of the low-pass filter
    xi1 : dictionary
        dictionary containing the center frequencies of the first order
        filters. See above for a decsription of the keys.
    sigma1 : dictionary
        dictionary containing the frequential width of the first order
        filters. See above for a description of the keys.
    xi2 : dictionary
        dictionary containing the center frequencies of the second order
        filters. See above for a decsription of the keys.
    sigma2 : dictionary
        dictionary containing the frequential width of the second order
        filters. See above for a description of the keys.
    # TODO j1 & j2
    """
    Q1, Q2 = Q if isinstance(Q, tuple) else (Q, 1)
    if Q1 < 1 or Q2 < 1:
        raise ValueError('Q should always be >= 1, got {}'.format(Q))

    # lower bound of band-pass filter frequential widths:
    # for default T = 2**(J), this coincides with sigma_low
    sigma_min = sigma0 / math.pow(2, J)

    xi1, sigma1, j1 = compute_params_filterbank(sigma_min, Q1, r_psi=r_psi,
                                                alpha=alpha, xi_min=xi_min)
    xi2, sigma2, j2 = compute_params_filterbank(sigma_min, Q2, r_psi=r_psi,
                                                alpha=alpha, xi_min=xi_min)

    # width of the low-pass filter
    sigma_low = sigma0 / T
    return sigma_low, xi1, sigma1, j1, xi2, sigma2, j2


def scattering_filter_factory(J_support, J_scattering, Q, T,
                              r_psi=math.sqrt(0.5),
                              criterion_amplitude=1e-3, normalize='l1',
                              max_subsampling=None, sigma0=0.1, alpha=4.,
                              P_max=5, eps=1e-7, **kwargs):
    """
    Builds in Fourier the Morlet filters used for the scattering transform.

    Each single filter is provided as a dictionary with the following keys:
    * 'xi': central frequency, defaults to 0 for low-pass filters.
    * 'sigma': frequential width
    * k where k is an integer bounded below by 0. The maximal value for k
        depends on the type of filter, it is dynamically chosen depending
        on max_subsampling and the characteristics of the filters.
        Each value for k is an array (or tensor) of size 2**(J_support - k)
        containing the Fourier transform of the filter after subsampling by
        2**k

    Parameters
    ----------
    J_support : int
        2**J_support is the desired support size of the filters
    J_scattering : int
        parameter for the scattering transform (2**J_scattering
        corresponds to maximal temporal support of any filter)
    Q : int >= 1 / tuple[int]
        The number of first-order wavelets per octave. Defaults to `1`.
        If tuple, sets `Q = (Q1, Q2)`, where `Q2` is the number of
        second-order wavelets per octave (which defaults to `1`).
            - Q1: For audio signals, a value of `>= 12` is recommended in
              order to separate partials.
            - Q2: Recommended `1` for most (`Scattering1D)` applications.
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and maximum subsampling
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets).
        Defaults to sqrt(0.5).
    criterion_amplitude : float, optional
        Represents the numerical error which is allowed to be lost after
        convolution and padding. Defaults to 1e-3.
    normalize : string, optional
        Normalization convention for the filters (in the
        temporal domain). Supported values include 'l1' and 'l2'; a ValueError
        is raised otherwise. Defaults to 'l1'.
    max_subsampling: int or None, optional
        maximal dyadic subsampling to compute, in order
        to save computation time if it is not required. Defaults to None, in
        which case this value is dynamically adjusted depending on the filters.
    sigma0 : float, optional
        parameter controlling the frequential width of the
        low-pass filter at J_scattering=0; at a an absolute J_scattering, it
        is equal to sigma0 / 2**J_scattering. Defaults to 1e-1
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 4.
    P_max : int, optional
        maximal number of periods to use to make sure that the Fourier
        transform of the filters is periodic. P_max = 5 is more than enough for
        double precision. Defaults to 5. Should be >= 1
    eps : float, optional
        required machine precision for the periodization (single
        floating point is enough for deep learning applications).
        Defaults to 1e-7

    Returns
    -------
    phi_f : dictionary
        a dictionary containing the low-pass filter at all possible
        subsamplings. See above for a description of the dictionary structure.
        The possible subsamplings are controlled by the inputs they can
        receive, which correspond to the subsamplings performed on top of the
        1st and 2nd order transforms.
    psi1_f : dictionary
        a dictionary containing the band-pass filters of the 1st order,
        only for the base resolution as no subsampling is used in the
        scattering tree.
        Each value corresponds to a dictionary for a single filter, see above
        for an exact description.
        The keys of this dictionary are of the type (j, n) where n is an
        integer counting the filters and j the maximal dyadic subsampling
        which can be performed on top of the filter without aliasing.
    psi2_f : dictionary
        a dictionary containing the band-pass filters of the 2nd order
        at all possible subsamplings. The subsamplings are determined by the
        input they can receive, which depends on the scattering tree.
        Each value corresponds to a dictionary for a single filter, see above
        for an exact description.
        The keys of this dictionary are of th etype (j, n) where n is an
        integer counting the filters and j is the maximal dyadic subsampling
        which can be performed on top of this filter without aliasing.
    t_max_phi : int
        temporal size to use to pad the signal on the right and on the
        left by making at most criterion_amplitude error. Assumes that the
        temporal support of the low-pass filter is larger than all filters.

    Refs
    ----
    Convolutional operators in the time-frequency domain, V. Lostanlen,
    PhD Thesis, 2017
    https://tel.archives-ouvertes.fr/tel-01559667
    """
    N = 2**J_support
    xi_min = 2 / N  # minimal peak at bin 2
    # compute the spectral parameters of the filters
    (sigma_low, xi1, sigma1, j1s, xi2, sigma2, j2s
     ) = calibrate_scattering_filters(J_scattering, Q, T, r_psi=r_psi,
                                      sigma0=sigma0, alpha=alpha, xi_min=xi_min)

    # instantiate the dictionaries which will contain the filters
    phi_f = {}
    psi1_f = []
    psi2_f = []

    # compute the band-pass filters of the second order,
    # which can take as input a subsampled
    for (n2, j2) in enumerate(j2s):
        # compute the current value for the max_subsampling,
        # which depends on the input it can accept.
        if max_subsampling is None:
            possible_subsamplings_after_order1 = [
                j1 for j1 in j1s if j2 > j1]
            if len(possible_subsamplings_after_order1) > 0:
                max_sub_psi2 = max(possible_subsamplings_after_order1)
            else:
                max_sub_psi2 = 0
        else:
            max_sub_psi2 = max_subsampling

        # We first compute the filter without subsampling
        psi_f = {}
        psi_f[0] = morlet_1d(
            N, xi2[n2], sigma2[n2], normalize=normalize, P_max=P_max, eps=eps)
        # compute the filter after subsampling at all other subsamplings
        # which might be received by the network, based on this first filter
        for subsampling in range(1, max_sub_psi2 + 1):
            factor_subsampling = 2**subsampling
            psi_f[subsampling] = periodize_filter_fourier(
                psi_f[0], nperiods=factor_subsampling)
        psi2_f.append(psi_f)

    # for the 1st order filters, the input is not subsampled so we
    # can only compute them with N=2**J_support
    for (n1, j1) in enumerate(j1s):
        psi1_f.append({0: morlet_1d(
            N, xi1[n1], sigma1[n1], normalize=normalize, P_max=P_max, eps=eps)})

    # compute the low-pass filters phi
    # Determine the maximal subsampling for phi, which depends on the
    # input it can accept (both 1st and 2nd order)
    if max_subsampling is None:
        max_subsampling_after_psi1 = max(j1s)
        max_subsampling_after_psi2 = max(j2s)
        log2_T = math.floor(math.log2(T))
        max_sub_phi = min(max(max_subsampling_after_psi1,
                              max_subsampling_after_psi2), log2_T)
    else:
        max_sub_phi = max_subsampling

    # compute the filters at all possible subsamplings
    phi_f[0] = gauss_1d(N, sigma_low, P_max=P_max, eps=eps)
    for subsampling in range(1, max_sub_phi + 1):
        factor_subsampling = 2**subsampling
        # compute the low_pass filter
        phi_f[subsampling] = periodize_filter_fourier(
            phi_f[0], nperiods=factor_subsampling)

    # Embed the meta information within the filters
    for (n1, j1) in enumerate(j1s):
        psi1_f[n1]['xi'] = xi1[n1]
        psi1_f[n1]['sigma'] = sigma1[n1]
        psi1_f[n1]['j'] = j1
    for (n2, j2) in enumerate(j2s):
        psi2_f[n2]['xi'] = xi2[n2]
        psi2_f[n2]['sigma'] = sigma2[n2]
        psi2_f[n2]['j'] = j2
    phi_f['xi'] = 0.
    phi_f['sigma'] = sigma_low
    phi_f['j'] = log2_T

    # compute the support size allowing to pad without boundary errors
    # at the finest resolution
    t_max_phi = compute_temporal_support(
        phi_f[0].reshape(1, -1), criterion_amplitude=criterion_amplitude)

    # return results
    return phi_f, psi1_f, psi2_f, t_max_phi


def psi_fr_factory(J_pad_fr_max_init, J_fr, Q_fr, shape_fr,
                   subsample_equiv_relative_to_max_pad_init,
                   sampling_psi_fr='resample', sigma_max_to_min_max_ratio=1.2,
                   r_psi=math.sqrt(0.5), normalize='l1', criterion_amplitude=1e-3,
                   sigma0=0.1, alpha=4., P_max=5, eps=1e-7):
    """
    Builds in Fourier the Morlet filters used for the scattering transform.

    Each single filter is provided as a dictionary with the following keys:
    * 'xi': central frequency, defaults to 0 for low-pass filters.
    * 'sigma': frequential width
    * k where k is an integer bounded below by 0. The maximal value for k
        depends on the type of filter, it is dynamically chosen depending
        on max_subsampling and the characteristics of the filters.
        Each value for k is an array (or tensor) of size 2**(J_support - k)
        containing the Fourier transform of the filter after subsampling by 2**k

    Parameters
    ----------
    J_pad_fr_max : int  # TODO
        `2**J_pad_fr_max` is the desired support size (length) of the filters.

    J_fr : int
        The maximum log-scale of frequential scattering in joint scattering
        transform, and number of octaves of frequential filters. That is,
        the maximum (bandpass) scale is given by :math:`2^J_fr`.

    Q_fr : int
        Number of wavelets per octave for frequential scattering.

    subsample_equiv_relative_to_max_pad_init : int
        Amount of *equivalent subsampling* of frequential padding relative to
        `J_pad_fr_max`, indexed by `n2`. See `help(sc_freq.compute_padding_fr())`.

    sampling_psi_fr : str['resample', 'recalibrate', 'exclude']
        See `help(TimeFrequencyScattering1D)`.
        In terms of effect on maximum `j` per `n1_fr`:

            - 'resample': no variation (by design, all temporal properties are
              preserved, including subsampling factor).
            - 'recalibrate': `j1_fr_max` is (likely) lesser with greater
              `subsample_equiv_due_to_pad` (by design, temporal width is halved
              for shorter `shape_fr`). The limit, however, is set by
              `sigma_max_to_min_max_ratio` (see its docs).
            - 'exclude': approximately same as 'recalibrate'. By design, excludes
              temporal widths above `min_width * 2**downsampling_factor`, which
              is likely to reduce `j1_fr_max` with greater
              `subsample_equiv_due_to_pad`.
                - It's "approximately" same because center frequencies and
                  widths are different; depending how loose our alias tolerance
                  (`alpha`), they're exactly the same.

    sigma_max_to_min_max_ratio : float
        Largest permitted `max(sigma) / min(sigma)`.
        See `help(TimeFrequencyScattering1D)`.

    r_psi, normalize, criterion_amplitude, sigma0, alpha, P_max, eps:
        See `help(kymatio.scattering1d.filter_bank.scattering_filter_factory)`.

    Returns
    -------
    psi1_f_fr_up : list[dict]
        List of dicts containing the band-pass filters of frequential scattering
        with "up" spin at all possible downsamplings. Each element corresponds to
        a dictionary for a single filter, see above for an exact description.
        Downsampling factors are indexed by integers, where `2**index` is the
        amount of downsampling, and the `'j'` key holds the value of maximum
        subsampling which can be performed on each filter without aliasing.

        The kind of downsampling done is controlled by `sampling_psi_fr`
        (but it is *never* subsampling).

        Example (`J_fr = 2`, `n1_fr = 8`, lists hold subsampling factors):
            - 'resample':
                0: [2, 1, 0]
                1: [2, 1, 0]
                2: [2, 1, 0]
            - 'recalibrate':
                0: [2, 1, 0]
                1: [1, 1, 0]
                2: [0, 0, 0]
            - 'exclude':
                0: [2, 1, 0]
                1: [1, 0]
                2: [0]

    psi1_f_fr_down : list[dict]
        Same as `psi1_f_fr_up` but with "down" spin (analytic, whereas "up"
        is anti-analytic wavelet).

    j0_max : int / None
        Sets `max_subsample_equiv_before_psi_fr`, see its docs in
        `help(TimeFrequencyScattering1D)`.
    """
    # compute the spectral parameters of the filters
    J_support = J_pad_fr_max_init  # begin with longest
    N = 2**J_support
    xi_min = 2 / N  # minimal peak at bin 2
    T = 1  # for computing `sigma_low`, unused
    _, xi1, sigma1, j1s, *_ = calibrate_scattering_filters(
        J_fr, Q_fr, T=T, r_psi=r_psi, sigma0=sigma0, alpha=alpha, xi_min=xi_min)

    shape_fr_scale_max = int(np.ceil(np.log2(max(shape_fr))))
    shape_fr_scale_min = int(np.ceil(np.log2(min(
        N_fr for N_fr in shape_fr if N_fr > 0))))

    # instantiate the dictionaries which will contain the filters
    psi1_f_fr_up = []
    psi1_f_fr_down = []

    j0_max, scale_diff_max = None, None
    kw = dict(criterion_amplitude=criterion_amplitude)
    if sampling_psi_fr == 'recalibrate':
        # recalibrate filterbank to each j0
        xi1_new, sigma1_new, j1s_new, scale_diff_max = _recalibrate_psi_fr_v2(
            xi1, sigma1, j1s, N, alpha, shape_fr_scale_max, shape_fr_scale_min,
            sigma_max_to_min_max_ratio)
    elif sampling_psi_fr == 'resample':
        # in this case filter temporal behavior is preserved across all lengths
        # so we must restrict lowest length such that widest filter still decays
        j0 = 0
        while True:
            psi_widest = morlet_1d(N // 2**j0, xi1[-1], sigma1[-1], P_max=P_max,
                                   normalize=normalize, eps=eps)[:, None]
            psi_widest_halfwidth = compute_temporal_support(psi_widest.T, **kw)
            if psi_widest_halfwidth == len(psi_widest) // 2:
                j0_max = max(j0 - 1, 0)
                if j0_max < 0:  # TODO
                    raise Exception("got `j0_max = %s < 0`, meaning " % j0_max
                                    + "`J_pad_fr_max_init` computed incorrectly.")
                break
            elif len(psi_widest) == shape_fr_scale_min:
                # smaller pad length is impossible
                break
            j0 += 1
    elif sampling_psi_fr == 'exclude':
        # this is built precisely to enable `j0_max=None` while preserving
        # temporal behavior
        pass

    def get_params(n1_fr, scale_diff):
        if sampling_psi_fr in ('resample', 'exclude'):
            return xi1[n1_fr], sigma1[n1_fr], j1s[n1_fr]
        elif sampling_psi_fr == 'recalibrate':
            return (xi1_new[scale_diff][n1_fr], sigma1_new[scale_diff][n1_fr],
                    j1s_new[scale_diff][n1_fr])

    # keep a mapping from `j0` to `scale_diff`
    j0_to_scale_diff = {}
    # sample spin down and up wavelets
    for n1_fr in range(len(j1s)):
        psi_down = {}
        # expand dim to multiply along freq like (2, 32, 4) * (32, 1)
        psi_down[0] = morlet_1d(N, xi1[n1_fr], sigma1[n1_fr], normalize=normalize,
                                P_max=P_max, eps=eps)[:, None]
        psi_down['width'] = {0: compute_temporal_support(psi_down[0].T, **kw)}

        # j0 is ordered greater to lower, so reverse
        j0_prev = -1
        for j0, N_fr in zip(subsample_equiv_relative_to_max_pad_init[::-1],
                            shape_fr[::-1]):
            # ensure we compute at valid `j0` and don't recompute
            if j0 <= 0 or j0 == j0_prev or (j0_max is not None and j0 > j0_max):
                continue
            # compute scale params
            j0_prev = j0
            factor = 2**j0
            shape_fr_scale = math.ceil(math.log2(N_fr))
            scale_diff = shape_fr_scale_max - shape_fr_scale
            if scale_diff_max is not None and scale_diff > scale_diff_max:
                # subsequent `scale_diff` are only greater
                break

            # ensure every `j0` maps to one `scale_diff`
            if j0 in j0_to_scale_diff and j0_to_scale_diff[j0] != scale_diff:
                raise Exception("same `J_pad_fr` mapped to different "
                                "`scale_diff`.")
            elif list(j0_to_scale_diff.values()).count(scale_diff) > 1:
                raise Exception("same `scale_diff` yielded multiple `J_pad_fr`")
            elif j0 not in j0_to_scale_diff:
                j0_to_scale_diff[j0] = scale_diff

            # fetch wavelet params, sample wavelet, compute its spatial width
            xi, sigma, j = get_params(n1_fr, scale_diff)
            psi = morlet_1d(N // factor, xi, sigma, normalize=normalize,
                            P_max=P_max, eps=eps)[:, None]
            psi_width = compute_temporal_support(psi.T, **kw)
            if sampling_psi_fr == 'exclude':
                # if wavelet exceeds max possible width at this scale, exclude it
                shape_fr_scale = math.ceil(math.log2(N_fr))
                if psi_width > 2**shape_fr_scale:
                    # subsequent `shape_fr_scale` are only lesser, and `psi_width`
                    # doesn't change (approx w/ discretization error)
                    break

            psi_down[j0] = psi
            psi_down['width'][j0] = psi_width

        psi1_f_fr_down.append(psi_down)
        # compute spin up
        psi_up = {}
        j0s = [j0 for j0 in psi_down if isinstance(j0, int)]
        for j0 in psi_down:
            if isinstance(j0, int):
                # compute spin up by conjugating spin down in frequency domain
                psi_up[j0] = conj_fr(psi_down[j0])
        psi_up['width'] = psi_down['width'].copy()
        psi1_f_fr_up.append(psi_up)

    # Embed meta information within the filters
    for (n1_fr, j1_fr) in enumerate(j1s):
        for psi_f in (psi1_f_fr_down, psi1_f_fr_up):
            # create initial meta
            meta = {'xi': xi1[n1_fr], 'sigma': sigma1[n1_fr], 'j': j1_fr}
            for field, value in meta.items():
                psi_f[n1_fr][field] = {0: value}
            # fill for j0s
            j0s = [k for k in psi_f[n1_fr] if (isinstance(k, int) and k != 0)]
            for j0 in j0s:
                xi, sigma, j = get_params(n1_fr, j0_to_scale_diff[j0])
                psi_f[n1_fr]['xi'][j0] = xi
                psi_f[n1_fr]['sigma'][j0] = sigma
                psi_f[n1_fr]['j'][j0] = j

    # return results
    return psi1_f_fr_up, psi1_f_fr_down, j0_max


def phi_fr_factory(J_pad_fr_max, F, log2_F, sampling_phi_fr='resample',
                   criterion_amplitude=1e-3, sigma0=0.1, P_max=5, eps=1e-7):
    """
    Builds in Fourier the lowpass Gabor filters used for JTFS.

    Each single filter is provided as a dictionary with the following keys:
    * 'xi': central frequency, defaults to 0 for low-pass filters.
    * 'sigma': frequential width
    * 'j': subsampling factor from 0 to `log2_F` (or potentially less if
      `sampling_phi_fr=False`).

    Parameters
    ----------
    J_pad_fr_max : int
        `2**J_pad_fr_max` is the desired support size (length) of the filters.
    F : int
        temporal support of frequential low-pass filter, controlling amount of
        imposed frequency transposition invariance and maximum frequential
        subsampling.
    log2_F : int
        Equal to `log2(prevpow2(F))`, sets maximum subsampling factor.
        If `sampling_phi_fr=True`, this factor may not be reached *by the filter*,
        as temporal width is preserved upon resampling rather than halved as
        with subsampling. Subsampling by `log2_F` *after* convolving with
        `phi_f_fr` is fine, thus the restriction is to not subsample by more than
        the most subsampled `phi_f_fr` *before* convolving with it - set by
        `max_subsample_before_phi_fr`.
    sampling_phi_fr : str['resample', 'recalibrate']
        See `help(TimeFrequencyScattering1D)`.
    criterion_amplitude, sigma, P_max, eps:
        See `help(kymatio.scattering1d.filter_bank.scattering_filter_factory)`.

    Returns
    -------
    phi_f_fr : dict[list]
        A dictionary containing the low-pass filter at all possible lengths.
        A distinction is made between input length difference due to trimming
        (or padding less) and subsampling (in frequential scattering with `psi`):
            `phi = phi_f_fr[subsample_equiv_due_to_pad][n1_fr_subsample]`
        so lists hold subsamplings of each trimming.

        Example (`log2_F = 2`, lists hold subsampling factors):
            - 'resample':
                0: [2, 1, 0]
                1: [2, 1, 0]
                2: [2, 1, 0]
                3: [2, 1, 0]
            - 'recalibrate':
                0: [2, 1, 0]
                1: [1, 1, 0]
                2: [0, 0, 0]
                3: [0, 0, 0]
    """
    # compute the spectral parameters of the filters
    sigma_low = sigma0 / F
    J_support = J_pad_fr_max
    N = 2**J_support

    # initial lowpass
    phi_f_fr = {}
    # expand dim to multiply along freq like (2, 32, 4) * (32, 1)
    phi_f_fr[0] = [gauss_1d(N, sigma_low, P_max=P_max, eps=eps)[:, None]]

    def compute_all_subsamplings(phi_f_fr, j_fr):
        for j_fr_sub in range(1, 1 + log2_F):
            phi_f_fr[j_fr].append(periodize_filter_fourier(
                phi_f_fr[j_fr][0], nperiods=2**j_fr_sub))

    compute_all_subsamplings(phi_f_fr, j_fr=0)

    # lowpass filters at all possible input lengths
    for j_fr in range(1, 1 + log2_F):
        factor = 2**j_fr
        if sampling_phi_fr == 'resample':
            prev_phi = phi_f_fr[j_fr - 1][0].reshape(1, -1)
            prev_phi_halfwidth = compute_temporal_support(
                prev_phi, criterion_amplitude=criterion_amplitude)

            if prev_phi_halfwidth == prev_phi.size // 2:
                # This means width is already too great for own length,
                # so lesser length will distort lowpass.
                # Frontend will adjust "all possible input lengths" accordingly
                break
            phi_f_fr[j_fr] = [gauss_1d(N // factor, sigma_low, P_max=P_max,
                                       eps=eps)[:, None]]
            # dedicate separate filters for *subsampled* as opposed to *trimmed*
            # inputs (i.e. `n1_fr_subsample` vs `J_pad_fr_max_init - J_pad_fr`)
            # note this increases maximum subsampling of phi_fr relative to
            # J_pad_fr_max_init
            compute_all_subsamplings(phi_f_fr, j_fr=j_fr)
        else:
            # These won't differ from plain subsampling but we still index
            # via `subsample_equiv_relative_to_max_pad_init` and
            # `n1_fr_subsample` so just copy pointers.
            # `phi[::factor] == gauss_1d(N // factor, sigma_low * factor)`
            # when not aliased
            phi_f_fr[j_fr] = [phi_f_fr[0][j_fr_sub]
                              for j_fr_sub in range(j_fr, 1 + log2_F)]

    # embed meta info in filters
    phi_f_fr.update({field: {} for field in ('xi', 'sigma', 'j')})
    j_frs = [j for j in phi_f_fr if isinstance(j, int)]
    for j_fr in j_frs:
        xi_fr_0 = 0.
        sigma_fr_0 = (sigma_low if sampling_phi_fr == 'resample' else
                      sigma_low * 2**j_fr)
        j_fr_0 = (log2_F if sampling_phi_fr == 'resample' else
                  log2_F - j_fr)
        for field in ('xi', 'sigma', 'j'):
            phi_f_fr[field][j_fr] = []
        for j_fr_sub in range(len(phi_f_fr[j_fr])):
            phi_f_fr['xi'][j_fr].append(xi_fr_0)
            phi_f_fr['sigma'][j_fr].append(sigma_fr_0 * 2**j_fr_sub)
            phi_f_fr['j'][j_fr].append(j_fr_0 - j_fr_sub)

    for j_fr in j_frs:
        for j_fr_sub in range(len(phi_f_fr[j_fr])):
            # no negative subsampling
            assert phi_f_fr['j'][j_fr][j_fr_sub] >= 0
            # no sigma exceeding `F==1` case
            assert phi_f_fr['sigma'][j_fr][j_fr_sub] <= sigma0

    # return results
    return phi_f_fr


def _recalibrate_psi_fr_v2(xi1, sigma1, j1s, N, alpha,
                           shape_fr_scale_max, shape_fr_scale_min,
                           sigma_max_to_min_max_ratio):
    # recalibrate filterbank to each j0
    # j0=0 is the original length, no change needed
    xi1_new, sigma1_new, j1s_new = {0: xi1}, {0: sigma1}, {0: j1s}
    scale_diff_max = None

    for shape_fr_scale in range(shape_fr_scale_max - 1, shape_fr_scale_min - 1,
                                -1):
        scale_diff = shape_fr_scale_max - shape_fr_scale
        xi1_new[scale_diff], sigma1_new[scale_diff], j1s_new[scale_diff] = (
            [], [], [])
        factor = 2**scale_diff

        # contract largest temporal width of any wavelet by 2**j0,
        # but not above sigma_max/sigma_max_to_min_max_ratio
        sigma_min_max = max(sigma1) / sigma_max_to_min_max_ratio
        new_sigma_min = min(sigma1) * factor
        if new_sigma_min > sigma_min_max:
            scale_diff_max = scale_diff
            break

        # halve distance from existing xi_max to .5 (max possible)
        new_xi_max = .5 - (.5 - max(xi1)) / factor
        new_xi_min = 2 / (N // factor)
        # logarithmically distribute
        new_xi = np.logspace(np.log10(new_xi_min), np.log10(new_xi_max),
                             len(xi1), endpoint=True)[::-1]
        xi1_new[scale_diff].extend(new_xi)
        new_sigma = np.logspace(np.log10(new_sigma_min), np.log10(max(sigma1)),
                                len(xi1), endpoint=True)[::-1]
        sigma1_new[scale_diff].extend(new_sigma)
        for xi, sigma in zip(new_xi, new_sigma):
            j1s_new[scale_diff].append(get_max_dyadic_subsampling(
                xi, sigma, alpha=alpha))

    return xi1_new, sigma1_new, j1s_new, scale_diff_max


def _recalibrate_psi_fr(xi1, sigma1, j1s, N, alpha,
                        subsample_equiv_relative_to_max_pad_init,
                        sigma_max_to_min_max_ratio):
    # recalibrate filterbank to each j0
    # j0=0 is the original length, no change needed
    xi1_new, sigma1_new, j1s_new = {0: xi1}, {0: sigma1}, {0: j1s}
    j0_max = None
    j0_prev = -1
    for j0 in subsample_equiv_relative_to_max_pad_init[::-1]:
        if j0 <= 0 or j0 == j0_prev:
            continue
        xi1_new[j0], sigma1_new[j0], j1s_new[j0] = [], [], []
        factor = 2**j0
        j0_prev = j0

        # contract largest temporal width of any wavelet by 2**j0,
        # but not above sigma_max/sigma_max_to_min_max_ratio
        sigma_min_max = max(sigma1) / sigma_max_to_min_max_ratio
        new_sigma_min = min(sigma1) * factor
        if new_sigma_min >= sigma_min_max:
            j0_max = j0
            new_sigma_min = sigma_min_max

        # halve distance from existing xi_max to .5 (theoretical max)
        new_xi_max = .5 - (.5 - max(xi1)) / factor
        new_xi_min = 2 / (N // factor)
        # logarithmically distribute
        new_xi = np.logspace(np.log10(new_xi_min), np.log10(new_xi_max),
                             len(xi1), endpoint=True)[::-1]
        xi1_new[j0].extend(new_xi)
        new_sigma = np.logspace(np.log10(new_sigma_min), np.log10(max(sigma1)),
                                len(xi1), endpoint=True)[::-1]
        sigma1_new[j0].extend(new_sigma)
        for xi, sigma in zip(new_xi, new_sigma):
            j1s_new[j0].append(get_max_dyadic_subsampling(xi, sigma, alpha=alpha))

        if j0_max is not None:
            break
    return xi1_new, sigma1_new, j1s_new, j0_max


def conj_fr(x):
    """Conjugate in frequency domain by swapping all bins (except dc);
    assumes frequency along first axis.
    """
    out = np.zeros_like(x)
    out[0] = x[0]
    out[1:] = x[:0:-1]
    return out


def assert_not_pure_sine(psi_f, ratio_threshold=100):  # TODO apply this?
    psi_fr_sorted = np.sort(psi_f)[::-1]
    assert psi_fr_sorted[0] / psi_fr_sorted[1] < ratio_threshold
