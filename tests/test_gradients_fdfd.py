import unittest

import numpy as np
import autograd.numpy as npa
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import copy

from autograd.extend import primitive, defvjp
from autograd import grad

import sys
sys.path.append('../ceviche')

from ceviche.utils import grad_num
from ceviche import jacobian, fdfd_hz, fdfd_ez

"""
This file tests the autograd gradients of an FDFD and makes sure that they
equal the numerical derivatives
"""

# test parameters
ALLOWED_RATIO = 1e-4    # maximum allowed ratio of || grad_num - grad_auto || vs. || grad_num ||
VERBOSE = False         # print out full gradients?
DEPS = 1e-6             # numerical gradient step size

print("Testing the Linear FDFD gradients")

class TestFDFD(unittest.TestCase):

    """ Tests the flexible objective function specifier """

    def setUp(self):

        # basic simulation parameters
        self.Nx = 30
        self.Ny = 30
        self.omega = 2*np.pi*200e12
        self.dL = 1e-6
        self.pml = [10, 10]

        self.source_mask = np.ones((self.Nx, self.Ny))
        self.source_mask[15, 15] = 1

        # sources (chosen to have objectives around 1)
        self.source_amp_ez = 1
        self.source_amp_hz = 1

        self.source_ez = np.zeros((self.Nx, self.Ny))
        self.source_ez[self.Nx//2, self.Ny//2] = self.source_amp_ez

        self.source_hz = np.zeros((self.Nx, self.Ny))
        self.source_hz[self.Nx//2, self.Ny//2] = self.source_amp_hz

        # starting relative permittivity (random for debugging)
        self.eps_r   = np.random.random((self.Nx, self.Ny)) + 1
        # self.eps_r   = np.ones((self.Nx, self.Ny)) + 1
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

    def test_Hz_reverse(self):

        print('\ttesting reverse-mode Hz in FDFD')

        f = fdfd_hz(self.omega, self.dL, self.eps_r, self.pml)

        def J_fdfd(eps_arr):

            eps_r = eps_arr.reshape((self.Nx, self.Ny))

            # set the permittivity
            f.eps_r = eps_r

            # set the source amplitude to the permittivity at that point
            Ex, Ey, Hz = f.solve(eps_r * self.source_hz)

            return npa.sum(npa.square(npa.abs(Hz))) \
                 + npa.sum(npa.square(npa.abs(Ex))) \
                 + npa.sum(npa.square(npa.abs(Ey)))


        grad_autograd_rev = jacobian(J_fdfd, mode='reverse')(self.eps_arr)
        grad_numerical = jacobian(J_fdfd, mode='numerical')(self.eps_arr)

        if VERBOSE:
            print('\tobjective function value: ', J_fdfd(self.eps_arr))
            print('\tgrad (auto):  \n\t\t', grad_autograd_rev)
            print('\tgrad (num):   \n\t\t\n', grad_numerical)

        self.check_gradient_error(grad_numerical, grad_autograd_rev)

    def test_Hz_forward(self):

        print('\ttesting forward-mode Hz in FDFD')

        f = fdfd_hz(self.omega, self.dL, self.eps_r, self.pml)

        def J_fdfd(c):

            # set the permittivity
            f.eps_r = c * self.eps_r

            # set the source amplitude to the permittivity at that point
            Ex, Ey, Hz = f.solve(c * self.eps_r * self.source_hz)

            return npa.square(npa.abs(Hz)) \
                 + npa.square(npa.abs(Ex)) \
                 + npa.square(npa.abs(Ey))

        grad_autograd_for = jacobian(J_fdfd, mode='forward')(1.0)
        grad_numerical = jacobian(J_fdfd, mode='numerical')(1.0)

        if VERBOSE:
            print('\tobjective function value: ', J_fdfd(1.0))
            print('\tgrad (auto):  \n\t\t', grad_autograd_for)
            print('\tgrad (num):   \n\t\t', grad_numerical)

        self.check_gradient_error(grad_numerical, grad_autograd_for)

    def test_Ez_reverse(self):

        print('\ttesting reverse-mode Ez in FDFD')

        f = fdfd_ez(self.omega, self.dL, self.eps_r, self.pml)

        def J_fdfd(eps_arr):

            eps_r = eps_arr.reshape((self.Nx, self.Ny))

            # set the permittivity
            f.eps_r = eps_r

            # set the source amplitude to the permittivity at that point
            Hx, Hy, Ez = f.solve(eps_r * self.source_ez)

            return npa.sum(npa.square(npa.abs(Ez))) \
                 + npa.sum(npa.square(npa.abs(Hx))) \
                 + npa.sum(npa.square(npa.abs(Hy)))

        grad_autograd_rev = jacobian(J_fdfd, mode='reverse')(self.eps_arr)
        grad_numerical = jacobian(J_fdfd, mode='numerical')(self.eps_arr)

        if VERBOSE:
            print('\tobjective function value: ', J_fdfd(self.eps_arr))
            print('\tgrad (auto):  \n\t\t', grad_autograd_rev)
            print('\tgrad (num):   \n\t\t', grad_numerical)

        self.check_gradient_error(grad_numerical, grad_autograd_rev)

    def test_Ez_forward(self):

        print('\ttesting forward-mode Ez in FDFD')

        f = fdfd_ez(self.omega, self.dL, self.eps_r, self.pml)

        def J_fdfd(c):

            # set the permittivity
            f.eps_r = c * self.eps_r

            # set the source amplitude to the permittivity at that point
            Hx, Hy, Ez = f.solve(c * self.eps_r * self.source_ez)

            return npa.square(npa.abs(Ez)) \
                 + npa.square(npa.abs(Hx)) \
                 + npa.square(npa.abs(Hy))

        grad_autograd_for = jacobian(J_fdfd, mode='forward')(1.0)
        grad_numerical = jacobian(J_fdfd, mode='numerical')(1.0)

        if VERBOSE:
            print('\tobjective function value: ', J_fdfd(1.0))
            print('\tgrad (auto):  \n\t\t', grad_autograd_for)
            print('\tgrad (num):   \n\t\t', grad_numerical)

        self.check_gradient_error(grad_numerical, grad_autograd_for)


if __name__ == '__main__':
    unittest.main()
