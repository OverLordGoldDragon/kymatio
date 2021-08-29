def timefrequency_scattering(
        x, pad, unpad, backend, J, psi1, psi2, phi, sc_freq,
        pad_left=0, pad_right=0, ind_start=None, ind_end=None, oversampling=0,
        max_order=2, average=True, size_scattering=(0, 0, 0), out_type='array'):
    """
    Main function implementing the joint time-frequency scattering transform.
    """
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    rfft = backend.rfft
    ifft = backend.ifft
    irfft = backend.irfft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate

    batch_size = x.shape[0]
    kJ = max(J - oversampling, 0)
    temporal_size = ind_end[kJ] - ind_start[kJ]
    out_S_0, out_S_1, out_S_2 = [], [], []

    # pad to a dyadic size and make it complex
    U_0 = pad(x, pad_left=pad_left, pad_right=pad_right)

    # compute the Fourier transform
    U_0_hat = rfft(U_0)

    # First order:
    U_1_hat_list = []
    S_1_list = []
    for n1 in range(len(psi1)):
        # Convolution + downsampling
        j1 = psi1[n1]['j']
        k1 = max(j1 - oversampling, 0)
        U_1_c = cdgmm(U_0_hat, psi1[n1][0])
        U_1_hat = subsample_fourier(U_1_c, 2**k1)
        U_1_c = ifft(U_1_hat)

        # Modulus
        U_1_m = modulus(U_1_c)

        # Map to Fourier domain
        U_1_hat = rfft(U_1_m)
        U_1_hat_list.append(U_1_hat)

        # Apply low-pass filtering phi (optional) and unpad
        if average:
            # Convolve with phi_J
            k1_J = max(J - k1 - oversampling, 0)
            S_1_c = cdgmm(U_1_hat, phi[k1])
            S_1_hat = subsample_fourier(S_1_c, 2**k1_J)
            S_1_r = irfft(S_1_hat)
            S_1 = unpad(S_1_r, ind_start[k1_J + k1], ind_end[k1_J + k1])
        else:
            S_1 = unpad(U_1_m, ind_start[k1], ind_end[k1])

        out_S_1.append({'coef': S_1,
                        'j': (j1,),
                        'n': (n1,)})

    # Second order:
    S_2_list = []
    for n2 in range(len(psi2)):
        j2 = psi2[n2]['j']

        if j2 == 0:
            continue

        # Wavelet transform over time
        Y_2_list = []
        for n1 in range(len(psi1)):
            # Retrieve first-order coefficient in the list
            j1 = psi1[n1]['j']
            if j1 >= j2:
                continue
            U_1_hat = U_1_hat_list[n1]

            # Convolution and downsampling
            k1 = max(j1 - oversampling, 0)
            k2 = max(j2 - j1 - oversampling, 0)
            Y_2_c = cdgmm(U_1_hat, psi2[n2][k1])
            Y_2_hat = subsample_fourier(Y_2_c, 2**k2)
            Y_2_list.append(ifft(Y_2_hat))

        # Concatenate along the frequency axis
        Y_2 = concatenate(Y_2_list)
        print(Y_2.shape)


        # Complex FFT is not implemented in the backend, only RFFT and IFFT
        # so we use IFFT which is equivalent up to conjugation.
        Y_2_hat = ifft(Y_2)

        for n_fr in range(len(sc_freq.psi1_f)):
            # Wavelet transform over frequency
            j_fr = sc_freq.psi1_f[n_fr]['j']
            k_fr = max(j_fr - oversampling, 0)
            Y_fr_c = cdgmm(Y_2_hat, sc_freq.psi1_f[n_fr][0])
            Y_fr_hat = subsample_fourier(Y_fr_c, 2**k_fr)

            # Modulus
            U_2_m = modulus(Y_fr_hat)

            # Low-pass filtering over frequency
            k_J_fr = max(sc_freq.J - k_fr - oversampling, 0)
            U_2_hat = rfft(U_2_m)
            S_2_fr_c = cdgmm(U_2_hat, sc_freq.phi_f[k_fr])
            S_2_fr_hat = subsample_fourier(S_2_fr_c, 2**k_J_fr)
            S_2_fr = irfft(S_2_fr_hat)

            # TODO unpad frequency domain iff out_type == "list"

            # Swap time and frequency subscripts again
            S_2_fr = backend.transpose(S_2_fr)

            # Low-pass filtering over time
            k2_J = max(J - j2 - oversampling, 0)
            U_2_hat = rfft(S_2_fr)
            S_2_c = cdgmm(U_2_hat, phi[j2])
            S_2_hat = subsample_fourier(S_2_c, 2**k2_J)
            S_2_r = irfft(S_2_hat)

            # TODO unpad time domain

            S_2_list.append(S_2_r)

        # Complex FFT is not implemented in the backend, only RFFT and IFFT
        # so we use IFFT which is equivalent up to conjugation.
        Y_2_hat = ifft(Y_2)

        for n_fr in range(len(sc_freq.psi1_f)):
            # Wavelet transform over frequency
            j_fr = sc_freq.psi1_f[n_fr]['j']
            k_fr = max(j_fr - oversampling, 0)
            Y_fr_c = cdgmm(Y_2_hat, sc_freq.psi1_f[n_fr][0])
            Y_fr_hat = subsample_fourier(Y_fr_c, 2**k_fr)

            # Modulus
            U_2_m = modulus(Y_fr_hat)

    out_S = []
    out_S.extend(out_S_1)
    #out_S.extend(out_S_2)

    if out_type == 'array':
        out_S = concatenate([x['coef'] for x in out_S])
    elif out_type == 'list':
        for x in out_S:
            x.pop('n')

    return out_S

__all__ = ['timefrequency_scattering']
