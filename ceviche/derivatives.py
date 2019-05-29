import numpy as np
import scipy.sparse as sp

from ceviche.constants import *

def compute_derivatives(omega, shape, npml, x_range, y_range, N):

    # make the S-matrices for PML
    (Sxf, Sxb, Syf, Syb) = S_create(omega, shape, npml, x_range, y_range)

    # Construct derivate matrices without PML
    Dxf_0 = createDws('x', 'f', dL(N, x_range, y_range), shape)
    Dxb_0 = createDws('x', 'b', dL(N, x_range, y_range), shape)
    Dyf_0 = createDws('y', 'f', dL(N, x_range, y_range), shape)
    Dyb_0 = createDws('y', 'b', dL(N, x_range, y_range), shape)

    # apply PML to derivative matrices
    Dxf = Sxf.dot(Dxf_0)
    Dxb = Sxb.dot(Dxb_0)
    Dyf = Syf.dot(Dyf_0)
    Dyb = Syb.dot(Dyb_0)

    return Dxf, Dxb, Dyf, Dyb

def createDws(w, s, dL, N):
    """ creates the derivative matrices
            NOTE: python uses C ordering rather than Fortran ordering. Therefore the
            derivative operators are constructed slightly differently than in MATLAB
    """

    Nx = N[0]
    dx = dL[0]
    if len(N) is not 1:
        Ny = N[1]
        dy = dL[1]
    else:
        Ny = 1
        dy = 1
    if w is 'x':
        if Nx > 1:
            if s is 'f':
                dxf = sp.diags([-1, 1, 1], [0, 1, -Nx+1], shape=(Nx, Nx))
                Dws = 1/dx*sp.kron(dxf, sp.eye(Ny))
            else:
                dxb = sp.diags([1, -1, -1], [0, -1, Nx-1], shape=(Nx, Nx))
                Dws = 1/dx*sp.kron(dxb, sp.eye(Ny))
        else:
            Dws = sp.eye(Ny)            
    if w is 'y':
        if Ny > 1:
            if s is 'f':
                dyf = sp.diags([-1, 1, 1], [0, 1, -Ny+1], shape=(Ny, Ny))
                Dws = 1/dy*sp.kron(sp.eye(Nx), dyf)
            else:
                dyb = sp.diags([1, -1, -1], [0, -1, Ny-1], shape=(Ny, Ny))
                Dws = 1/dy*sp.kron(sp.eye(Nx), dyb)
        else:
            Dws = sp.eye(Nx)
    return Dws

def dL(N, x_range, y_range=None):
    # solves for the grid spacing

    if y_range is None:
        L = np.array([np.diff(x_range)[0]])  # Simulation domain lengths
    else:
        L = np.array([np.diff(x_range)[0],
                      np.diff(y_range)[0]])  # Simulation domain lengths
    return L/N

# def sig_w(l, dw, m=4, lnR=-12):
def sig_w(l, dw, m=3, lnR=-12e4):
    # helper for S()

    sig_max = -(m+1)*lnR/(2*ETA_0*dw)
    return sig_max*(l/dw)**m


def S(l, dw, omega):
    # helper for create_sfactor()

    return 1 - 1j*sig_w(l, dw)/(omega*EPSILON_0)


def create_sfactor(wrange, s, omega, Nw, Nw_pml):
    # used to help construct the S matrices for the PML creation

    sfactor_array = np.ones(Nw, dtype=np.complex128)
    if Nw_pml < 1:
        return sfactor_array
    hw = np.diff(wrange)[0]/Nw
    dw = Nw_pml*hw
    for i in range(0, Nw):
        if s is 'f':
            if i <= Nw_pml:
                sfactor_array[i] = S(hw * (Nw_pml - i + 0.5), dw, omega)
            elif i > Nw - Nw_pml:
                sfactor_array[i] = S(hw * (i - (Nw - Nw_pml) - 0.5), dw, omega)
        if s is 'b':
            if i <= Nw_pml:
                sfactor_array[i] = S(hw * (Nw_pml - i + 1), dw, omega)
            elif i > Nw - Nw_pml:
                sfactor_array[i] = S(hw * (i - (Nw - Nw_pml) - 1), dw, omega)
    return sfactor_array


def S_create(omega, N, Npml, x_range, y_range=None):
    # creates S matrices for the PML creation

    M = np.prod(N)
    if np.isscalar(Npml):
        Npml = np.array([Npml])
    if len(N) < 2:
        N = np.append(N, 1)
        Npml = np.append(Npml, 0)
    Nx = N[0]
    Nx_pml = Npml[0]
    Ny = N[1]
    Ny_pml = Npml[1]

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor(x_range, 'f', omega, Nx, Nx_pml)
    s_vector_x_b = create_sfactor(x_range, 'b', omega, Nx, Nx_pml)
    s_vector_y_f = create_sfactor(y_range, 'f', omega, Ny, Ny_pml)
    s_vector_y_b = create_sfactor(y_range, 'b', omega, Ny, Ny_pml)

    # Fill the 2D space with layers of appropriate s-factors
    Sx_f_2D = np.zeros(N, dtype=np.complex128)
    Sx_b_2D = np.zeros(N, dtype=np.complex128)
    Sy_f_2D = np.zeros(N, dtype=np.complex128)
    Sy_b_2D = np.zeros(N, dtype=np.complex128)

    for i in range(0, Ny):
        Sx_f_2D[:, i] = 1/s_vector_x_f
        Sx_b_2D[:, i] = 1/s_vector_x_b

    for i in range(0, Nx):
        Sy_f_2D[i, :] = 1/s_vector_y_f
        Sy_b_2D[i, :] = 1/s_vector_y_b

    # Reshape the 2D s-factors into a 1D s-array
    Sx_f_vec = Sx_f_2D.reshape((-1,))
    Sx_b_vec = Sx_b_2D.reshape((-1,))
    Sy_f_vec = Sy_f_2D.reshape((-1,))
    Sy_b_vec = Sy_b_2D.reshape((-1,))

    # Construct the 1D total s-array into a diagonal matrix
    Sx_f = sp.spdiags(Sx_f_vec, 0, M, M)
    Sx_b = sp.spdiags(Sx_b_vec, 0, M, M)
    Sy_f = sp.spdiags(Sy_f_vec, 0, M, M)
    Sy_b = sp.spdiags(Sy_b_vec, 0, M, M)

    return (Sx_f, Sx_b, Sy_f, Sy_b)
