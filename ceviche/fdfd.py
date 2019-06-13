import autograd.numpy as np
import scipy.sparse as sp

from ceviche.primitives import *
from ceviche.derivatives import compute_derivative_matrices
from ceviche.utils import make_sparse

import copy
from time import time

class fdfd():
    """ Base class for FDFD simulation """

    def __init__(self, omega, dL, eps_r, npml):
        """ initialize with a given structure and source """

        self.omega = omega
        self.dL = dL
        self.npml = npml

        self.eps_r = eps_r
        # self.source = source

        self.info_dict = {'omega': self.omega}
        self.setup_derivatives()

    def setup_derivatives(self):

        # Creates all of the operators needed for later
        info_dict = compute_derivative_matrices(self.omega, self.shape, self.npml, self.dL)
        self.Dxf, self.Dxb, self.Dyf, self.Dyb = info_dict

        # save to a dictionary for convenience passing to primitives
        self.info_dict['Dxf'] = self.Dxf
        self.info_dict['Dxb'] = self.Dxb
        self.info_dict['Dyf'] = self.Dyf
        self.info_dict['Dyb'] = self.Dyb

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
        self.shape = self.Nx, self.Ny = self.__eps_r.shape

    # @property
    # def source(self):
    #     """ Returns the source grid """
    #     return self.__source

    # @source.setter
    # def source(self, new_source):
    #     """ Defines some attributes when source is set. """
    #     self.__source = new_source
    #     self.source_arr = self.__source.flatten()

    @staticmethod
    def make_A(info_dict, eps_r):
        # eventually make these functions in the fdfd class
        raise NotImplementedError("need to make a solve() method")

    def solve(self):
        raise NotImplementedError("need to make a solve() method")

    def _arr_to_grid(self, arr):
        return np.reshape(arr, self.shape)

""" These are the fdfd classes that you'll actually want to use """

class fdfd_hz(fdfd):
    """ FDFD class for Hz polarization """

    def __init__(self, omega, L0, eps_r, npml):
        super().__init__(omega, L0, eps_r, npml)
        self.A = make_A_Hz(self.info_dict, self.eps_arr)

    def solve(self, source):
        """ Solves the electromagnetic fields of the system """

        source_arr = source.flatten()

        Hz_arr = solve_Hz(self.info_dict, self.eps_arr, source_arr)
        Ex_arr, Ey_arr = H_to_E(Hz_arr, self.info_dict, self.eps_arr)
        Ex = self._arr_to_grid(Ex_arr)
        Ey = self._arr_to_grid(Ey_arr)
        Hz = self._arr_to_grid(Hz_arr)
        return Ex, Ey, Hz


class fdfd_ez(fdfd):
    """ FDFD class for Ez polarization """

    def __init__(self, omega, L0, eps_r, npml):
        super().__init__(omega, L0, eps_r, npml)
        self.A = make_A_Ez(self.info_dict, self.eps_arr)

    def solve(self, source):
        """ Solves the electromagnetic fields of the system """

        source_arr = source.flatten()

        Ez_arr = solve_Ez(self.info_dict, self.eps_arr, source_arr)
        Hx_arr, Hy_arr = E_to_H(Ez_arr, self.info_dict)
        Hx = self._arr_to_grid(Hx_arr)
        Hy = self._arr_to_grid(Hy_arr)
        Ez = self._arr_to_grid(Ez_arr)
        return Hx, Hy, Ez
