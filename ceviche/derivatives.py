import numpy as np
import autograd.numpy as npa
import scipy.sparse as sp

from .constants import *

"""
This file contains functions related to performing derivative operations used in the simulation tools.
-  The FDTD method requires autograd-compatible curl operations, which are performed using numpy.roll
-  The FDFD method requires sparse derivative matrices, with PML added, which are constructed here.
"""


"""================================== CURLS FOR FDTD ======================================"""

def curl_E(axis, Ex, Ey, Ez, dL):
    if axis == 0:
        return (npa.roll(Ez, shift=-1, axis=1) - Ez) / dL - (npa.roll(Ey, shift=-1, axis=2) - Ey) / dL
    elif axis == 1:
        return (npa.roll(Ex, shift=-1, axis=2) - Ex) / dL - (npa.roll(Ez, shift=-1, axis=0) - Ez) / dL
    elif axis == 2:
        return (npa.roll(Ey, shift=-1, axis=0) - Ey) / dL - (npa.roll(Ex, shift=-1, axis=1) - Ex) / dL

def curl_H(axis, Hx, Hy, Hz, dL):
    if axis == 0:
        return (Hz - npa.roll(Hz, shift=1, axis=1)) / dL - (Hy - npa.roll(Hy, shift=1, axis=2)) / dL
    elif axis == 1:
        return (Hx - npa.roll(Hx, shift=1, axis=2)) / dL - (Hz - npa.roll(Hz, shift=1, axis=0)) / dL
    elif axis == 2:
        return (Hy - npa.roll(Hy, shift=1, axis=0)) / dL - (Hx - npa.roll(Hx, shift=1, axis=1)) / dL

"""======================= STUFF THAT CONSTRUCTS THE DERIVATIVE MATRIX ==========================="""

def compute_derivative_matrices(omega, shape, npml, dL, bloch_x=0.0, bloch_y=0.0):
    """ Returns sparse derivative matrices.  Currently works for 2D and 1D
            omega: angular frequency (rad/sec)
            shape: shape of the FDFD grid
            npml: list of number of PML cells in x and y.
            dL: spatial grid size (m)
            block_x: bloch phase (phase across periodic boundary) in x
            block_y: bloch phase (phase across periodic boundary) in y
    """

    # Construct derivate matrices without PML
    Dxf = createDws('x', 'f', shape, dL, bloch_x=bloch_x, bloch_y=bloch_y)
    Dxb = createDws('x', 'b', shape, dL, bloch_x=bloch_x, bloch_y=bloch_y)
    Dyf = createDws('y', 'f', shape, dL, bloch_x=bloch_x, bloch_y=bloch_y)
    Dyb = createDws('y', 'b', shape, dL, bloch_x=bloch_x, bloch_y=bloch_y)

    # make the S-matrices for PML
    (Sxf, Sxb, Syf, Syb) = create_S_matrices(omega, shape, npml, dL)

    # apply PML to derivative matrices
    Dxf = Sxf.dot(Dxf)
    Dxb = Sxb.dot(Dxb)
    Dyf = Syf.dot(Dyf)
    Dyb = Syb.dot(Dyb)

    return Dxf, Dxb, Dyf, Dyb

""" Derivative Matrices (no PML) """

def createDws(component, dir, shape, dL, bloch_x=0.0, bloch_y=0.0):
    """ creates the derivative matrices
            component: one of 'x' or 'y' for derivative in x or y direction
            dir: one of 'f' or 'b', whether to take forward or backward finite difference
            shape: shape of the FDFD grid
            dL: spatial grid size (m)
            block_x: bloch phase (phase across periodic boundary) in x
            block_y: bloch phase (phase across periodic boundary) in y
    """

    Nx, Ny = shape

    # special case, a 1D problem
    if component == 'x' and Nx == 1:
        return sp.eye(Ny)
    if component is 'y' and Ny == 1:
        return sp.eye(Nx)

    # select a `make_D` function based on the component and direction
    component_dir = component + dir
    if component_dir == 'xf':
        return make_Dxf(dL, shape, bloch_x=bloch_x)
    elif component_dir == 'xb':
        return make_Dxb(dL, shape, bloch_x=bloch_x)
    elif component_dir == 'yf':
        return make_Dyf(dL, shape, bloch_y=bloch_y)
    elif component_dir == 'yb':
        return make_Dyb(dL, shape, bloch_y=bloch_y)
    else:
        raise ValueError("component and direction {} and {} not recognized".format(component, dir))

def make_Dxf(dL, shape, bloch_x=0.0):
    """ Forward derivative in x """
    Nx, Ny = shape
    phasor_x = np.exp(1j * bloch_x)
    Dxf = sp.diags([-1, 1, phasor_x], [0, 1, -Nx+1], shape=(Nx, Nx), dtype=np.complex128)
    Dxf = 1 / dL * sp.kron(Dxf, sp.eye(Ny))
    return Dxf

def make_Dxb(dL, shape, bloch_x=0.0):
    """ Backward derivative in x """
    Nx, Ny = shape
    phasor_x = np.exp(1j * bloch_x)
    Dxb = sp.diags([1, -1, -np.conj(phasor_x)], [0, -1, Nx-1], shape=(Nx, Nx), dtype=np.complex128)
    Dxb = 1 / dL * sp.kron(Dxb, sp.eye(Ny))
    return Dxb

def make_Dyf(dL, shape, bloch_y=0.0):
    """ Forward derivative in y """
    Nx, Ny = shape
    phasor_y = np.exp(1j * bloch_y)
    Dyf = sp.diags([-1, 1, phasor_y], [0, 1, -Ny+1], shape=(Ny, Ny))
    Dyf = 1 / dL * sp.kron(sp.eye(Nx), Dyf)
    return Dyf

def make_Dyb(dL, shape, bloch_y=0.0):
    """ Backward derivative in y """
    Nx, Ny = shape
    phasor_y = np.exp(1j * bloch_y)
    Dyb = sp.diags([1, -1, -np.conj(phasor_y)], [0, -1, Ny-1], shape=(Ny, Ny))
    Dyb = 1 / dL * sp.kron(sp.eye(Nx), Dyb)
    return Dyb


""" PML Functions """

def create_S_matrices(omega, shape, npml, dL):
    """ Makes the 'S-matrices'.  When dotted with derivative matrices, they add PML """

    # strip out some information needed
    Nx, Ny = shape
    N = Nx * Ny
    x_range = [0, float(dL * Nx)]
    y_range = [0, float(dL * Ny)]
    Nx_pml, Ny_pml = npml

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor('f', omega, dL, Nx, Nx_pml)
    s_vector_x_b = create_sfactor('b', omega, dL, Nx, Nx_pml)
    s_vector_y_f = create_sfactor('f', omega, dL, Ny, Ny_pml)
    s_vector_y_b = create_sfactor('b', omega, dL, Ny, Ny_pml)

    # Fill the 2D space with layers of appropriate s-factors
    Sx_f_2D = np.zeros(shape, dtype=np.complex128)
    Sx_b_2D = np.zeros(shape, dtype=np.complex128)
    Sy_f_2D = np.zeros(shape, dtype=np.complex128)
    Sy_b_2D = np.zeros(shape, dtype=np.complex128)

    # insert the cross sections into the S-grids (could be done more elegantly)
    for i in range(0, Ny):
        Sx_f_2D[:, i] = 1 / s_vector_x_f
        Sx_b_2D[:, i] = 1 / s_vector_x_b
    for i in range(0, Nx):
        Sy_f_2D[i, :] = 1 / s_vector_y_f
        Sy_b_2D[i, :] = 1 / s_vector_y_b

    # Reshape the 2D s-factors into a 1D s-vecay
    Sx_f_vec = Sx_f_2D.flatten()
    Sx_b_vec = Sx_b_2D.flatten()
    Sy_f_vec = Sy_f_2D.flatten()
    Sy_b_vec = Sy_b_2D.flatten()

    # Construct the 1D total s-vecay into a diagonal matrix
    Sx_f = sp.spdiags(Sx_f_vec, 0, N, N)
    Sx_b = sp.spdiags(Sx_b_vec, 0, N, N)
    Sy_f = sp.spdiags(Sy_f_vec, 0, N, N)
    Sy_b = sp.spdiags(Sy_b_vec, 0, N, N)

    return Sx_f, Sx_b, Sy_f, Sy_b

def create_sfactor(dir, omega, dL, N, N_pml):
    """ creates the S-factor cross section needed in the S-matrices """

    #  for no PNL, this should just be zero
    if N_pml == 0:
        return np.ones(N, dtype=np.complex128)

    # otherwise, get different profiles for forward and reverse derivative matrices
    dw = N_pml * dL
    if dir == 'f':
        return create_sfactor_f(omega, dL, N, N_pml, dw)
    elif dir == 'b':
        return create_sfactor_b(omega, dL, N, N_pml, dw)
    else:
        raise ValueError("Dir value {} not recognized".format(dir))

def create_sfactor_f(omega, dL, N, N_pml, dw):
    """ S-factor profile for forward derivative matrix """
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i <= N_pml:
            sfactor_array[i] = s_value(dL * (N_pml - i + 0.5), dw, omega)
        elif i > N - N_pml:
            sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 0.5), dw, omega)
    return sfactor_array

def create_sfactor_b(omega, dL, N, N_pml, dw):
    """ S-factor profile for backward derivative matrix """
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i <= N_pml:
            sfactor_array[i] = s_value(dL * (N_pml - i + 1), dw, omega)
        elif i > N - N_pml:
            sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 1), dw, omega)
    return sfactor_array

def sig_w(l, dw, m=3, lnR=-30):
    """ Fictional conductivity, note that these values might need tuning """
    sig_max = -(m + 1) * lnR / (2 * ETA_0 * dw)
    return sig_max * (l / dw)**m

def s_value(l, dw, omega):
    """ S-value to use in the S-matrices """
    return 1 - 1j * sig_w(l, dw) / (omega * EPSILON_0)

""" Index expansion """

import numpy as np
import scipy.sparse as sp

def check_args(*dims, shifts=None):
    """ if shifts is NOne, makes all shifts = 0, checks that shifts is same length as dims """
    ndim = len(dims)
    if shifts is None:
        shifts = ndim * (0,)
    assert len(shifts) == ndim
    return shifts

def shift_arange(N, shift=0):
    """ make an arange 0 to N-1, shifted by shift """
    a = np.arange(N)
    return np.roll(a, shift=shift)

def expand_dims(*dims, shifts=None):
    """ expands dims = Nx, Ny, ... into several aranges, each with shift[dim] """
    shifts = check_args(*dims, shifts=shifts)

    dim_ranges = (shift_arange(N, s) for (N, s) in zip(list(dims), shifts))

    return dim_ranges

def get_subs(*dims, shifts=None):
    """ Return tuple of subscipts into each dimension """
    dim_ranges = expand_dims(*dims, shifts=shifts)
    subs_exp = np.meshgrid(*dim_ranges, indexing='ij')
    return (s.flatten() for s in subs_exp)

def get_inds(*dims, shifts=None):
    """ Return i and j indices into big array """
    subs_exp = get_subs(*dims, shifts=shifts)
    return np.ravel_multi_index(list(subs_exp), dims)

def shift_inds(*dims, shifts=None):
    i = get_inds(*dims, shifts=None)
    j = get_inds(*dims, shifts=shifts)
    return i, j

def shift_mat(*dims, shifts=None):
    N = np.prod(dims)
    i, j = shift_inds(*dims, shifts=shifts)
    entries = np.ones_like(i)
    indices = np.vstack((i, j))
    return sp.coo_matrix((entries, indices), shape=(N, N))

def roll_mat(*dims, shift=0, axis=0):
    shifts = len(dims) * [0]
    shifts[axis] = shift
    return shift_mat(*dims, shifts=shifts)

def der_mat(*dims, axis=0, fb='f'):
    I = roll_mat(*dims)
    if fb == 'f':
        return roll_mat(*dims, shift=1, axis=axis) - I
    elif fb == 'b':
        return I - roll_mat(*dims, shift=-1, axis=axis)
    else:
        raise ValueError(f'fb must be f or b, given {fb}')

def Dxf(*dims):
    return Der(*dims, axis=0, fb='f')

def Dxb(*dims):
    return Der(*dims, axis=0, fb='b')

def Dyf(*dims):
    return Der(*dims, axis=1, fb='f')

def Dyb(*dims):
    return Der(*dims, axis=1, fb='b')

def Dzf(*dims):
    return Der(*dims, axis=2, fb='f')

def Dzb(*dims):
    return Der(*dims, axis=2, fb='b')
