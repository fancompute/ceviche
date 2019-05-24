import autograd.numpy as np
import scipy.sparse as sp

from ceviche.primitives import *
from ceviche.utils import make_sparse

import copy
from time import time

class fdfd():
    """ Base class for FDFD simulation """

    def __init__(self, omega, dL, eps_r, source, npml):
        """ initialize with a given structure and source """

        self.omega = omega
        self.dL = dL
        self.npml = npml

        self.eps_r = eps_r
        self.source = source

        self.matrices = {'omega': self.omega}
        self.setup_derivatives()

    def setup_derivatives(self):

        # Creates all of the operators needed for later
        self.Dxf, self.Dxb, self.Dyf, self.Dyb = compute_derivatives(self.omega, 1, self.shape, self.npml, self.x_range, self.y_range, self.N)

        # save to a dictionary for convenience passing to primitives
        self.matrices['Dxf'] = self.Dxf
        self.matrices['Dxb'] = self.Dxb
        self.matrices['Dyf'] = self.Dyf
        self.matrices['Dyb'] = self.Dyb

    @property
    def eps_r(self):
        """ Returns the relative permittivity grid """
        return self.__eps_r

    @eps_r.setter
    def eps_r(self, new_eps):
        """ Defines some attributes when eps_r is set. """
        self.__eps_r = new_eps
        self.eps_arr = self.__eps_r.flatten()
        self.N = self.eps_arr.size
        self.shape = self.__eps_r.shape
        self.Nx = self.shape[0]
        self.Ny = self.shape[1]
        self.x_range = [0.0, self.Nx * self.dL]
        self.y_range = [0.0, self.Ny * self.dL]

    @property
    def source(self):
        """ Returns the source grid """
        return self.__source

    @source.setter
    def source(self, new_source):
        """ Defines some attributes when source is set. """
        self.__source = new_source
        self.source_arr = self.__source.flatten()

    @staticmethod
    def make_A(matrices, eps_r):
        # eventually make these functions in the fdfd class
        raise NotImplementedError("need to make a solve() method")

    def solve(self):
        raise NotImplementedError("need to make a solve() method")

    def _arr_to_grid(self, arr):
        return np.reshape(arr, self.shape)

""" These are the fdfd classes that you'll actually want to use """

class fdfd_hz(fdfd):
    """ FDFD class for Hz polarization """

    def __init__(self, omega, L0, eps_r, source, npml):
        super().__init__(omega, L0, eps_r, source, npml)
        self.A = make_A_Hz(self.matrices, self.eps_arr)
        print('norm A: ', spl.norm(self.A))

    def solve(self):
        """ Solves the electromagnetic fields of the system """

        Hz_arr = solve_Hz(self.matrices, self.eps_arr, self.source_arr)
        Ex_arr, Ey_arr = H_to_E(Hz_arr, self.matrices, self.eps_arr)
        Ex = self._arr_to_grid(Ex_arr)
        Ey = self._arr_to_grid(Ey_arr)
        Hz = self._arr_to_grid(Hz_arr)
        return Ex, Ey, Hz


class fdfd_ez(fdfd):
    """ FDFD class for Ez polarization """

    def __init__(self, omega, L0, eps_r, source, npml):
        super().__init__(omega, L0, eps_r, source, npml)
        self.A = make_A_Ez(self.matrices, self.eps_arr)
        print('norm A: ', spl.norm(self.A))

    def solve(self):
        """ Solves the electromagnetic fields of the system """

        Ez_arr = solve_Ez(self.matrices, self.eps_arr, self.source_arr)
        Hx_arr, Hy_arr = E_to_H(Ez_arr, self.matrices)
        Hx = self._arr_to_grid(Hx_arr)
        Hy = self._arr_to_grid(Hy_arr)
        Ez = self._arr_to_grid(Ez_arr)
        return Hx, Hy, Ez


""" These are helper functions for the fdfd classes above """

EPSILON_0 = 1
ETA_0 = 1


def compute_derivatives(omega, L0, shape, npml, x_range, y_range, N):

    # t = time()
    (Sxf, Sxb, Syf, Syb) = S_create(omega, L0, shape, npml, x_range, y_range)

    # Construct derivate matrices
    Dxf = Sxf.dot(createDws('x', 'f', dL(N, x_range, y_range), shape))
    Dxb = Sxb.dot(createDws('x', 'b', dL(N, x_range, y_range), shape))
    Dyf = Syf.dot(createDws('y', 'f', dL(N, x_range, y_range), shape))
    Dyb = Syb.dot(createDws('y', 'b', dL(N, x_range, y_range), shape))
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

def sig_w(l, dw, m=4, lnR=-12):
    # helper for S()

    sig_max = -(m+1)*lnR/(2*ETA_0*dw)
    return sig_max*(l/dw)**m


def S(l, dw, omega, L0):
    # helper for create_sfactor()

    return 1 - 1j*sig_w(l, dw)/(omega*EPSILON_0*L0)


def create_sfactor(wrange, L0, s, omega, Nw, Nw_pml):
    # used to help construct the S matrices for the PML creation

    sfactor_array = np.ones(Nw, dtype=np.complex128)
    if Nw_pml < 1:
        return sfactor_array
    hw = np.diff(wrange)[0]/Nw
    dw = Nw_pml*hw
    for i in range(0, Nw):
        if s is 'f':
            if i <= Nw_pml:
                sfactor_array[i] = S(hw * (Nw_pml - i + 0.5), dw, omega, L0)
            elif i > Nw - Nw_pml:
                sfactor_array[i] = S(hw * (i - (Nw - Nw_pml) - 0.5), dw, omega, L0)
        if s is 'b':
            if i <= Nw_pml:
                sfactor_array[i] = S(hw * (Nw_pml - i + 1), dw, omega, L0)
            elif i > Nw - Nw_pml:
                sfactor_array[i] = S(hw * (i - (Nw - Nw_pml) - 1), dw, omega, L0)
    return sfactor_array


def S_create(omega, L0, N, Npml, x_range, y_range=None):
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
    s_vector_x_f = create_sfactor(x_range, L0, 'f', omega, Nx, Nx_pml)
    s_vector_x_b = create_sfactor(x_range, L0, 'b', omega, Nx, Nx_pml)
    s_vector_y_f = create_sfactor(y_range, L0, 'f', omega, Ny, Ny_pml)
    s_vector_y_b = create_sfactor(y_range, L0, 'b', omega, Ny, Ny_pml)

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
