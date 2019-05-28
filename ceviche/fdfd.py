import autograd.numpy as np
import scipy.sparse as sp

from ceviche.primitives import *
from ceviche.derivatives import compute_derivatives
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

        L0 = 1  # can everything be in SI units?  when I make L0 = 1 things don't work

        # Creates all of the operators needed for later
        self.Dxf, self.Dxb, self.Dyf, self.Dyb = compute_derivatives(self.omega, L0, self.shape, self.npml, self.x_range, self.y_range, self.N)

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
        self.x_range = [0.0, float(self.Nx * self.dL)]
        self.y_range = [0.0, float(self.Ny * self.dL)]

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
        # self.A = make_A_Hz(self.matrices, self.eps_arr)

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
        # self.A = make_A_Ez(self.matrices, self.eps_arr)

    def solve(self, source):
        """ Solves the electromagnetic fields of the system """

        Ez_arr = solve_Ez(self.matrices, self.eps_arr, self.source_arr)
        Hx_arr, Hy_arr = E_to_H(Ez_arr, self.matrices)
        Hx = self._arr_to_grid(Hx_arr)
        Hy = self._arr_to_grid(Hy_arr)
        Ez = self._arr_to_grid(Ez_arr)
        return Hx, Hy, Ez
