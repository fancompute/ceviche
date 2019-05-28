import unittest

import numpy as np
import autograd.numpy as npa
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import copy

from autograd.extend import primitive, defvjp
from autograd import grad

from ceviche.utils import make_sparse, grad_num
from ceviche.primitives import *
from ceviche.fdfd import fdfd_hz, fdfd_ez

"""
This file tests the autograd gradients of an FDFD and makes sure that they
equal the numerical derivatives
"""

# test parameters
ALLOWED_RATIO = 1e-4    # maximum allowed ratio of || grad_num - grad_auto || vs. || grad_num ||
VERBOSE = False         # print out full gradients?
DEPS = 1e-6             # numerical gradient step size

class TestGrads(unittest.TestCase):

    """ Tests the flexible objective function specifier """

    def setUp(self):

        # basic simulation parameters
        self.Nx = 30
        self.Ny = 30
        self.omega = 2*np.pi*200e12
        self.dL = 1e-6
        self.pml = [0, 0]

        self.source_mask = np.ones((self.Nx, self.Ny))
        self.source_mask[5, 5] = 0   # comment this out and look at the result
        self.source_mask = np.random.random((self.Nx, self.Ny))

        # print(self.source_mask)

        # sources (chosen to be around 1)
        self.source_amp_ez = 1e-8
        self.source_amp_hz = 1e-8

        self.source_ez = np.zeros((self.Nx, self.Ny))
        self.source_ez[self.Nx//2, self.Ny//2] = self.source_amp_ez

        self.source_hz = np.zeros((self.Nx, self.Ny))
        self.source_hz[self.Nx//2, self.Ny//2] = self.source_amp_hz

        # starting relative permittivity (random for debugging)
        self.eps_r   = np.random.random((self.Nx, self.Ny)) + 1
        self.eps_arr = self.eps_r.flatten()

    def check_gradient_error(self, grad_num, grad_auto):
        """ Checks the test case:
            compares the norm of the gradient to the norm of the difference
            Throws error if this is greater than ALLOWED RATIO
        """
        norm_grad = np.linalg.norm(grad_num)
        print('\t\tnorm of gradient:   ', norm_grad)
        norm_diff = np.linalg.norm(grad_num - grad_auto)
        print('\t\tnorm of difference: ', norm_diff)
        norm_ratio = norm_diff / norm_grad        
        print('\t\tratio of norms:     ', norm_ratio)
        self.assertLessEqual(norm_ratio, ALLOWED_RATIO)
        print('')

    def t1est_Hz(self):
        print('\ttesting Hz in FDFD')

        f = fdfd_hz(self.omega, self.dL, self.eps_r, self.source_hz, self.pml)

        def J_fdfd(eps_arr):

            eps_r = eps_arr.reshape((self.Nx, self.Ny))

            f.eps_r = eps_r
            f.source = self.source_hz * eps_r

            Ex, Ey, Hz = f.solve()

            return npa.sum(npa.square(npa.abs(Hz))) \
                 + npa.sum(npa.square(npa.abs(Ex))) \
                 + npa.sum(npa.square(npa.abs(Ey)))

        grad_autograd = grad(J_fdfd)(self.eps_arr)
        grad_numerical = grad_num(J_fdfd, self.eps_arr, step_size=DEPS)

        if VERBOSE:
            print('\tobjective function value: ', J_fdfd(self.eps_arr))
            print('\tgrad (auto):  \n\t\t', grad_autograd)
            print('\tgrad (num):   \n\t\t\n', grad_numerical)

        self.check_gradient_error(grad_autograd, grad_numerical)

    def test_Ez(self):

        print('\ttesting Ez in FDFD')

        f = fdfd_ez(self.omega, self.dL, self.eps_r, self.source_hz, self.pml)

        def J_fdfd(eps_arr):

            eps_r = eps_arr.reshape((self.Nx, self.Ny))

            f.eps_r = eps_r
            f.source = self.source_amp_ez * self.source_mask * eps_r

            Hx, Hy, Ez = f.solve(eps_arr)

            return npa.sum(npa.square(npa.abs(Ez))) \
                 + npa.sum(npa.square(npa.abs(Hx))) \
                 + npa.sum(npa.square(npa.abs(Hy)))

        grad_autograd = grad(J_fdfd)(self.eps_arr)
        grad_numerical = grad_num(J_fdfd, self.eps_arr, step_size=DEPS)

        if VERBOSE:
            print('\tobjective function value: ', J_fdfd(self.eps_arr))
            print('\tgrad (auto):  \n\t\t', grad_autograd)
            print('\tgrad (num):   \n\t\t', grad_numerical)

        self.check_gradient_error(grad_autograd, grad_numerical)


if __name__ == '__main__':
    unittest.main()
