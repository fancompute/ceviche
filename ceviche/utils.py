import numpy as np
import scipy.sparse as sp
import copy
import matplotlib.pylab as plt

""" Just some utilities for easier testing and debugging"""

def make_sparse(N, random=True, density=1):
    """ Makes a sparse NxN matrix. """
    if not random:
        np.random.seed(0)
    D = sp.random(N, N, density=density) + 1j * sp.random(N, N, density=density)
    return D


def grad_num(fn, arg, step_size=1e-7):
    """ numerically differentiate `fn` w.r.t. its argument `arg` """
    N = arg.size
    gradient = np.zeros((N,))
    f_old = fn(arg)
    for i in range(N):
        arg_new = copy.copy(arg)
        arg_new[i] += step_size
        f_new_i = fn(arg_new)
        gradient[i] = (f_new_i - f_old) / step_size
    return gradient


""" Plotting and measurement utilities for FDTD, may be moved later"""


def aniplot(F, source, steps, component='Ez', num_panels=10):
    """ Animate an FDTD (F) with `source` for `steps` time steps.
    display the `component` field components at `num_panels` equally spaced.
    """
    F.initialize_fields()

    # initialize the plot
    f, ax_list = plt.subplots(1, num_panels, figsize=(20*num_panels,20))
    Nx, Ny, _ = F.eps_r.shape
    ax_index = 0

    # fdtd time loop
    for t_index in range(steps):
        fields = F.forward(Jz=source(t_index))
    
        # if it's one of the num_panels panels
        if t_index % (steps // num_panels) == 0:
            print('working on axis {}/{} for time step {}'.format(ax_index, num_panels, t_index))

            # grab the axis
            ax = ax_list[ax_index]

            # plot the fields
            im_t = ax.pcolormesh(np.zeros((Nx, Ny)), cmap='RdBu')
            max_E = np.abs(fields[component]).max()
            im_t.set_array(fields[component][:, :, 0].ravel().T)
            im_t.set_clim([-max_E / 2.0, max_E / 2.0])
            ax.set_title('time = {} seconds'.format(F.dt*t_index))

            # update the axis
            ax_index += 1


def measure_fields(F, source, steps, probes, component='Ez'):
    """ Returns a time series of the measured `component` fields from FDFD `F`
        driven by `source and measured at `probe`.
    """
    F.initialize_fields()
    if not isinstance(probes, list):
        probes = [probes]
    N_probes = len(probes)
    measured = np.zeros((steps, N_probes))
    for t_index in range(steps):
        if t_index % (steps//20) == 0:
            print('{:.2f} % done'.format(float(t_index)/steps*100.0))
        fields = F.forward(Jz=source(t_index))
        for probe_index, probe in enumerate(probes):
            field_probe = np.sum(fields[component] * probe)
            measured[t_index, probe_index] = field_probe
    return measured

""" FFT Utilities """

from numpy.fft import fft, fftfreq
from librosa.core import stft
from librosa.display import specshow

def get_spectrum(series, dt):
    """ Get FFT of series """

    steps = len(series)
    times = np.arange(steps) * dt

    # multiply with hamming window to get rid of numerical errors
    hamming_window = np.hamming(steps).reshape((steps, 1))
    signal_f = np.fft.fft(hamming_window * series)
    freqs = np.fft.fftfreq(steps, d=dt)
    return freqs, signal_f

def get_max_power_freq(series, dt):

    freqs, signal_f = get_spectrum(series, dt)
    return freqs[np.argmax(signal_f)]

def get_spectral_power(series, dt):

    freqs, signal_f = get_spectrum(series, dt)
    return freqs, np.square(np.abs(signal_f))

def plot_spectral_power(series, dt, f_top=2e14):
    steps = len(series)
    freqs, signal_f_power = get_spectral_power(series, dt)

    # only plot half (other is redundant)
    plt.plot(freqs[:steps//2], signal_f_power[:steps//2])
    plt.xlim([0, f_top])
    plt.xlabel('frequency (Hz)')
    plt.ylabel('power (|signal|^2)')
    plt.show()
