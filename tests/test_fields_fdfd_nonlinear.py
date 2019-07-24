import unittest

import numpy as np
import autograd.numpy as npa
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import copy
import matplotlib.pylab as plt

from numpy.linalg import norm

from autograd.extend import primitive, defvjp
from autograd import grad

import sys
sys.path.append('../ceviche')

from ceviche.utils import grad_num, get_value, imarr
from ceviche.fdfd import fdfd_hz, fdfd_ez, fdfd_ez_nl
from ceviche.jacobians import jacobian
from ceviche.constants import *
"""
This file tests the autograd gradients of an FDFD and makes sure that they
equal the numerical derivatives
"""

# test parameters
ALLOWED_RATIO = 1e-4    # maximum allowed ratio of || grad_num - grad_auto || vs. || grad_num ||
VERBOSE = False         # print out full gradients?
DEPS = 1e-6             # numerical gradient step size

class TestFDFD(unittest.TestCase):

    """ Tests the flexible objective function specifier """

    def setUp(self):

        # basic simulation parameters
        self.Nx = 100
        self.Ny = 100
        self.omega = 2*np.pi*200e12
        self.dL = 5e-8
        self.pml = [10, 10]

        # sources (chosen to have objectives around 1)
        self.source_amp_ez = 1e3
        self.source_amp_hz = 1e3

        self.source_ez = np.zeros((self.Nx, self.Ny))
        self.source_ez[self.Nx//2, self.Ny//2] = self.source_amp_ez

        # starting relative permittivity (random for debugging)
        self.eps_lin = np.ones((self.Nx, self.Ny))
        self.chi3 = 2000
        self.eps_nl = lambda Ez: self.eps_lin + 3 * self.chi3 * np.square(np.abs(Ez))

        f = fdfd_ez(self.omega, self.dL, self.eps_lin, self.pml)
        Hx, Hy, Ez = f.solve(self.source_ez)
        self.Ez = Ez

        eps_nl = lambda Ez: self.eps_lin + 3 * self.eps_lin * self.chi3 * npa.square(npa.abs(Ez))
        f_nl = fdfd_ez_nl(self.omega, self.dL, self.eps_nl, self.pml)
        Hx_nl, Hy_nl, Ez_nl = f_nl.solve(self.source_ez)
        self.Ez_nl = Ez_nl

    def test_Ez_lin(self):

        print('\nplotting linear fields')

        E_max = np.max(np.abs(self.Ez))
        # plt.imshow(self.eps_lin)
        plt.imshow(np.real(self.Ez), cmap='RdBu', vmin=-E_max/2, vmax=E_max/2)
        plt.title('linear')
        plt.colorbar()
        plt.show()

    def test_Ez_nl(self):

        print('\nplotting nonlinear fields')

        mod_strength = np.max(np.abs(self.eps_nl(self.Ez) - self.eps_lin))
        print('max nonlinear change in epsilon = {}'.format(mod_strength))

        E_max = np.max(np.abs(self.Ez_nl))
        # plt.imshow(self.eps_nl(self.Ez))
        plt.imshow(np.real(self.Ez_nl), cmap='RdBu', vmin=-E_max/2, vmax=E_max/2)
        plt.title('nonlinear')
        plt.colorbar()
        plt.show()

    def test_z_field_diff(self):

        print('\nplotting the difference between linear and nonlinear')

        rel_diff = np.abs(self.Ez - self.Ez_nl) / np.abs(self.Ez)

        E_max = np.max(np.abs(rel_diff))
        plt.imshow(np.abs(rel_diff), cmap='magma', vmin=-E_max/2, vmax=E_max/2)
        plt.title('relative difference |Ez - Ez_nl| / |Ez|')
        plt.colorbar()
        plt.show()




if __name__ == '__main__':
    unittest.main()
