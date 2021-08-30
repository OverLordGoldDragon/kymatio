"""
Compute the Joint Time-Frequency Scattering of a synthetic signal
=================================================================

We compute and visualize JTFS of an exponential chirp and illustrate
FDTS-(frequency-dependent time shifts) discriminability.
"""


###############################################################################
# Import the necessary packages
# -----------------------------
from kymatio.numpy import TimeFrequencyScattering1D
from kymatio import visuals
from kymatio.visuals import plot, imshow
from kymatio.toolkit import echirp
import numpy as np

#%%# Generate echirp and create scattering object #############################
N = 4096
# span low to Nyquist; assume duration of 1 second
x = echirp(N, fmin=64, fmax=N/2)

# 9 temporal octaves
# largest scale is 2**9 [samples] / 4096 [samples / sec] == 125 ms
J = 9
# 8 bandpass wavelets per octave
# J*Q ~= 72 total temporal coefficients in first-order scattering
Q = 8
# scale of temporal invariance, 31.25 ms
T = 2**7
# 4 frequential octaves
J_fr = 4
# 2 bandpass wavelets per octave
Q_fr = 2
# scale of frequential invariance, F/Q == 1 cycle per octave
F = 8
# do frequential averaging to enable 4D concatenation
average_fr = True
# return packed as dict keyed by pair names for easy inspection
out_type = 'dict:array'

jtfs = TimeFrequencyScattering1D(shape=N, J=J, Q=Q, T=T, J_fr=J_fr, Q_fr=Q_fr,
                                 F=F, average_fr=average_fr, out_type=out_type)

#%%# Take JTFS, print pair names and shapes ###################################
Scx = jtfs(x)
print("JTFS pairs:")
for pair in Scx:
    print("{:<12} -- {}".format(str(Scx[pair].shape), pair))

#%%# Show `x` and its (time-averaged) scalogram ###############################
plot(x, show=1,
     xlabel="time [samples]",
     title="Exponential chirp | fmin=64, fmax=2048, 4096 samples")
imshow(Scx['S1'].squeeze(), abs=1,
       xlabel="time [samples] (subsampled)",
       ylabel="frequency [Hz]",
       title="Scalogram, time-averaged (first-order scattering)")
