import numpy as np
import scipy.sparse as sp
import copy

""" ==================== FDTD AND FDFD UTILITIES ==================== """

import autograd.numpy as npa


def grid_center_to_xyz(Q_mid, averaging=True):
    """ Computes the interpolated value of the quantity Q_mid felt at the Ex, Ey, Ez positions of the Yee latice
        Returns these three components
    """

    # initialize
    Q_xx = copy.copy(Q_mid)
    Q_yy = copy.copy(Q_mid)
    Q_zz = copy.copy(Q_mid)

    # if averaging, set the respective xx, yy, zz components to the midpoint in the Yee lattice.
    if averaging:

        # get the value from the middle of the next cell over
        Q_x_r = npa.roll(Q_mid, shift=1, axis=0)
        Q_y_r = npa.roll(Q_mid, shift=1, axis=1)
        Q_z_r = npa.roll(Q_mid, shift=1, axis=2)

        # average with the two middle values
        Q_xx = (Q_mid + Q_x_r)/2
        Q_yy = (Q_mid + Q_y_r)/2
        Q_zz = (Q_mid + Q_z_r)/2

    return Q_xx, Q_yy, Q_zz


def grid_xyz_to_center(Q_xx, Q_yy, Q_zz):
    """ Computes the interpolated value of the quantitys Q_xx, Q_yy, Q_zz at the center of Yee latice
        Returns these three components
    """

    # compute the averages
    Q_xx_avg = (Q_xx.astype('float') + npa.roll(Q_xx, shift=1, axis=0))/2
    Q_yy_avg = (Q_yy.astype('float') + npa.roll(Q_yy, shift=1, axis=1))/2
    Q_zz_avg = (Q_zz.astype('float') + npa.roll(Q_zz, shift=1, axis=2))/2

    return Q_xx_avg, Q_yy_avg, Q_zz_avg


def vec_zz_to_xy(info_dict, vec_zz, grid_averaging=True):
    """ does grid averaging on z vector vec_zz """
    arr_zz = vec_zz.reshape(info_dict['shape'])[:,:,None]
    arr_xx, arr_yy, _ = grid_center_to_xyz(arr_zz, averaging=grid_averaging)
    vec_xx, vec_yy = arr_xx.flatten(), arr_yy.flatten()
    return vec_xx, vec_yy

""" ===================== TESTING AND DEBUGGING ===================== """


def make_sparse(N, random=True, density=1):
    """ Makes a sparse NxN matrix. """
    if not random:
        np.random.seed(0)
    D = sp.random(N, N, density=density) + 1j * sp.random(N, N, density=density)
    return D


def float_2_array(x):
    if not isinstance(x, np.ndarray):
        return np.array([x])
    else:
        return x


def grad_num(fn, arg, step_size=1e-7):
    """ DEPRICATED: use 'numerical' in jacobians.py instead
    numerically differentiate `fn` w.r.t. its argument `arg` 
    `arg` can be a numpy array of arbitrary shape
    `step_size` can be a number or an array of the same shape as `arg` """

    in_array = float_2_array(arg).flatten()
    out_array = float_2_array(fn(arg)).flatten()

    m = in_array.size
    n = out_array.size
    shape = (m, n)
    jacobian = np.zeros(shape)

    for i in range(m):
        input_i = in_array.copy()
        input_i[i] += step_size
        arg_i = input_i.reshape(in_array.shape)
        output_i = fn(arg_i).flatten()
        grad_i = (output_i - out_array) / step_size
        jacobian[i, :] = get_value(grad_i)

    return jacobian


def reshape_to_ND(arr, N):
    """ Adds dimensions to arr until it is dimension N
    """

    ND = len(arr.shape)
    if ND > N:
        raise ValueError("array is larger than {} dimensional, given shape {}".format(N, arr.shape))
    extra_dims = (N - ND) * (1,)
    return arr.reshape(arr.shape + extra_dims)


""" =========================== AUTOGRAD =========================== """


import autograd
from autograd.extend import primitive, vspace, defvjp, defjvp


def get_value(x):
    if type(x) == autograd.numpy.numpy_boxes.ArrayBox:
        return x._value
    else:
        return x

get_value_arr = np.vectorize(get_value)


def get_shape(x):
    """ Gets the shape of x, even if it is not an array """
    if isinstance(x, float) or isinstance(x, int):
        return (1,)
    elif isinstance(x, tuple) or isinstance(x, list):
        return (len(x),)
    else:
        return vspace(x).shape


def vjp_maker_num(fn, arg_inds, steps):
    """ Makes a vjp_maker for the numerical derivative of a function `fn`
    w.r.t. argument at position `arg_ind` using step sizes `steps` """

    def vjp_single_arg(ia):
        arg_ind = arg_inds[ia]
        step = steps[ia]

        def vjp_maker(fn_out, *args):
            shape = args[arg_ind].shape
            num_p = args[arg_ind].size
            step = steps[ia]

            def vjp(v):

                vjp_num = np.zeros(num_p)
                for ip in range(num_p):
                    args_new = list(args)
                    args_rav = args[arg_ind].flatten()
                    args_rav[ip] += step
                    args_new[arg_ind] = args_rav.reshape(shape)
                    dfn_darg = (fn(*args_new) - fn_out)/step
                    vjp_num[ip] = np.sum(v * dfn_darg)

                return vjp_num

            return vjp

        return vjp_maker

    vjp_makers = []
    for ia in range(len(arg_inds)):
        vjp_makers.append(vjp_single_arg(ia=ia))

    return tuple(vjp_makers)


@primitive
def spdot(A, x):
    """ Dot product of sparse matrix A and dense matrix x (Ax = b) """
    return A.dot(x)

def vjp_maker_spdot(b, A, x):
    """ Gives vjp for b = spdot(A, x) w.r.t. x"""
    def vjp(v):
        return spdot(A.T, v)
    return vjp

def jvp_spdot(g, b, A, x):
    """ Gives jvp for b = spdot(A, x) w.r.t. x"""
    return spdot(A, g)

defvjp(spdot, None, vjp_maker_spdot)
defjvp(spdot, None, jvp_spdot)


""" =================== PLOTTING AND MEASUREMENT =================== """


import matplotlib.pylab as plt


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

            if ax_index < num_panels:   # extra safety..sometimes tries to access num_panels-th elemet of ax_list, leading to error

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
    plt.show()


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


def imarr(arr):
    """ puts array 'arr' into form ready to plot """
    arr_value = get_value(arr)
    arr_plot = arr_value.copy()
    if len(arr.shape) == 3:
        arr_plot = arr_plot[:,:,0]
    return np.flipud(arr_plot.T)


""" ====================== FOURIER TRANSFORMS  ======================"""

from autograd.extend import primitive, defjvp
from numpy.fft import fft, fftfreq


@primitive
def my_fft(x):    
    """ 
    Wrapper for numpy's FFT, so I can add a primitive to it
        FFT(x) is like a DFT matrix (D) dot with x
    """
    return np.fft.fft(x)


def fft_grad(g, ans, x):
    """ 
    Define the jacobian-vector product of my_fft(x)
        The gradient of FFT times g is the vjp
        ans = fft(x) = D @ x
        jvp(fft(x))(g) = d{fft}/d{x} @ g
                       = D @ g
        Therefore, it looks like the FFT of g
    """
    return np.fft.fft(g)

defjvp(my_fft, fft_grad)


def get_spectrum(series, dt):
    """ Get FFT of series """

    steps = len(series)
    times = np.arange(steps) * dt

    # reshape to be able to multiply by hamming window
    series = series.reshape((steps, -1))

    # multiply with hamming window to get rid of numerical errors
    hamming_window = np.hamming(steps).reshape((steps, 1))
    signal_f = my_fft(hamming_window * series)

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

""" ========================= LINEAR ALGEBRA ========================= """


def block_4(A, B, C, D):
    """ Constructs a big matrix out of four sparse blocks
        returns [A B]
                [C D]
    """
    left = sp.vstack([A, C])
    right = sp.vstack([B, D])
    return sp.hstack([left, right])    

