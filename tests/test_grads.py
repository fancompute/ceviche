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

RELATIVE_TOLERANCE = 1e-3
VERBOSE = True
DEPS = 1e-6

class TestGrads(unittest.TestCase):

    """ Tests the flexible objective function specifier """

    def setUp(self):

        print('setting up...')
        self.Nx = 10
        self.Ny = 10
        N = self.Nx * self.Ny

        self.omega = 2*np.pi*200e12
        self.dL = 1e-7

        # make the FDFD matrices (random for now)
        Dxf = make_sparse(N, 0)
        Dxb = make_sparse(N, 0)
        Dyf = make_sparse(N, 0)
        Dyb = make_sparse(N, 0)
        self.matrices = {'Dxf':Dxf,
                         'Dxb':Dxb,
                         'Dyf':Dyf,
                         'Dyb':Dyb,
                         'omega': self.omega}

        # source
        self.source_amp = 1
        self.b = self.source_amp * np.ones((N,))

        # starting relative permittivity
        self.eps_r   = np.random.random((self.Nx, self.Ny)) + 1
        self.eps_arr = self.eps_r.flatten()

    def test_Hz_direct(self):
        print('\ttesting Hz direct')

        # a function using special solve directly
        def J_direct(eps_r):

            # get the fields
            Hz = solve_Ez(self.matrices, eps_r, self.b)
            Ex, Ey = H_to_E(Hz, self.matrices, eps_r)

            # some objective function of Hz, Ex, Ey.
            return npa.sum(npa.square(npa.abs(Hz))) \
                 + npa.sum(npa.square(npa.abs(Ex))) \
                 + npa.sum(npa.square(npa.abs(Ey)))

        grad_autograd = grad(J_direct)(self.eps_arr)
        grad_numerical = grad_num(J_direct, self.eps_arr, step_size=DEPS)

        if VERBOSE:
            print('\tobjective function value: ', J_direct(self.eps_arr))
            print('\tgrad (auto):  \n\t\t', grad_autograd)
            print('\tgrad (num):   \n\t\t', grad_numerical)

        np.testing.assert_allclose(grad_autograd, grad_numerical, rtol=RELATIVE_TOLERANCE)
        
    def test_Hz_fdfd(self):
        print('\ttesting Hz in FDFD')

        # a function using the fdfd object
        f = fdfd_hz(self.omega, self.dL, self.eps_r, self.b, [0,0])

        def J_fdfd(eps_arr):

            f.source = 1j * eps_arr
            f.eps_r = eps_arr.reshape((self.Nx, self.Ny))
            Ex, Ey, Hz = f.solve()
            return npa.sum(npa.square(npa.abs(Hz))) \
                 + npa.sum(npa.square(npa.abs(Ex))) \
                 + npa.sum(npa.square(npa.abs(Ey)))

        grad_autograd = grad(J_fdfd)(self.eps_arr)
        grad_numerical = grad_num(J_fdfd, self.eps_arr, step_size=DEPS)

        if VERBOSE:
            print('\tobjective function value: ', J_fdfd(self.eps_arr))
            print('\tgrad (auto):  \n\t\t', grad_autograd)
            print('\tgrad (num):   \n\t\t', grad_numerical)

        np.testing.assert_allclose(grad_autograd, grad_numerical, rtol=RELATIVE_TOLERANCE)

    def test_Ez_direct(self):
        print('\ttesting Ez direct')

        # a function using special solve directly
        def J_direct(eps_r):

            # get the fields
            Ez = solve_Ez(self.matrices, eps_r, self.b)
            Hx, Hy = E_to_H(Ez, self.matrices)

            # some objective function of Hz, Ex, Ey.
            return npa.sum(npa.square(npa.abs(Ez))) \
                 + npa.sum(npa.square(npa.abs(Hx))) \
                 + npa.sum(npa.square(npa.abs(Hy)))

        grad_autograd = grad(J_direct)(self.eps_arr)
        grad_numerical = grad_num(J_direct, self.eps_arr, step_size=DEPS)

        if VERBOSE:
            print('\tobjective function value: ', J_direct(self.eps_arr))
            print('\tgrad (auto):  \n\t\t', grad_autograd)
            print('\tgrad (num):   \n\t\t', grad_numerical)

        np.testing.assert_allclose(grad_autograd, grad_numerical, rtol=RELATIVE_TOLERANCE)
        
    def test_Ez_fdfd(self):

        print('\ttesting Ez in FDFD')

        # a function using the fdfd object
        f = fdfd_ez(self.omega, self.dL, self.eps_r, self.b, [3, 4])

        def J_fdfd(eps_arr):
            f.source = 1j * self.source_amp * eps_arr
            f.eps_r = eps_arr.reshape((self.Nx, self.Ny))
            Hx, Hy, Ez = f.solve()
            Hx += 1e9
            Hy += 1e9
            Ez += 1e9
            return npa.sum(npa.square(npa.abs(Ez))) \
                 + npa.sum(npa.square(npa.abs(Hx))) \
                 + npa.sum(npa.square(npa.abs(Hy)))

        grad_autograd = grad(J_fdfd)(self.eps_arr)
        grad_numerical = grad_num(J_fdfd, self.eps_arr, step_size=DEPS)

        if VERBOSE:
            print('\tobjective function value: ', J_fdfd(self.eps_arr))
            print('\tgrad (auto):  \n\t\t', grad_autograd)
            print('\tgrad (num):   \n\t\t', grad_numerical)

        np.testing.assert_allclose(grad_autograd, grad_numerical, rtol=RELATIVE_TOLERANCE)


if __name__ == '__main__':
    unittest.main()



