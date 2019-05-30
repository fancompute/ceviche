import unittest
import sys
import autograd.numpy as npa
import numpy as np
from autograd import grad
from autograd import checkpoint

from copy import deepcopy, copy
from time import time

from ceviche.fdtd import fdtd
from ceviche.sources import Gaussian
from ceviche.constants import *
from ceviche.utils import grad_num

# gradient error tolerance
ALLOWED_RATIO = 1e-4    # maximum allowed ratio of || grad_num - grad_auto || vs. || grad_num ||
DEPS = 1e-6             # numerical gradient step size
VERBOSE = False

class TestFDTD(unittest.TestCase):
    '''Tests for Gradient Correctness'''

    def setUp(self):

        # basic simulation parameters
        self.Nx = 8
        self.Ny = 8
        self.Nz = 1

        self.omega = 2*np.pi*200e12
        self.dL = 5e-8
        self.pml = [2, 2, 0]

        # source parameters
        self.steps = 1500
        self.t0 = 300
        self.sigma = 20        
        self.source_amp = 1
        self.source = np.zeros((self.Nx, self.Ny, self.Nz))
        self.source[self.Nx//2, self.Ny//2, self.Nz//2] = self.source_amp

        # starting relative permittivity (random for debugging)
        self.eps_r   = np.random.random((self.Nx, self.Ny, self.Nz)) + 1
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

    def test_grad_E(self):

        print('\ttesting E fields in FDTD')

        F = fdtd(self.eps_r, dL=self.dL, npml=self.pml)

        G1 = Gaussian(mask=self.source, component='Jz', amp=self.source_amp, sigma=self.sigma, t0=self.t0)
        G2 = Gaussian(mask=self.source, component='Jx', amp=self.source_amp, sigma=self.sigma, t0=self.t0)
        G3 = Gaussian(mask=self.source, component='Jy', amp=self.source_amp, sigma=self.sigma, t0=self.t0)

        F.add_src(G1)
        F.add_src(G2)
        F.add_src(G3)
        F.add_src(G1)

        def objective(eps_arr):
            F.eps_r = eps_arr.reshape((self.Nx, self.Ny, self.Nz))
            S = 0.0
            for t_index, fields in enumerate(F.run(self.steps)):
                S += npa.sum(fields['Ex'] + fields['Ey'] + fields['Ez'])
            return S

        grad_autograd = grad(objective, 0)(self.eps_arr)
        grad_numerical = grad_num(objective, self.eps_arr, step_size=DEPS)

        if VERBOSE:
            print('\tobjective function value: ', objective(self.eps_arr))
            print('\tgrad (auto):  \n\t\t', grad_autograd)
            print('\tgrad (num):   \n\t\t', grad_numerical)

        self.check_gradient_error(grad_numerical, grad_autograd)

    def test_grad_H(self):

        print('\ttesting H fields in FDTD')

        F = fdtd(self.eps_r, dL=self.dL, npml=self.pml)

        G1 = Gaussian(mask=self.source, component='Jz', amp=self.source_amp, sigma=self.sigma, t0=self.t0)
        G2 = Gaussian(mask=self.source, component='Jx', amp=self.source_amp, sigma=self.sigma, t0=self.t0)
        G3 = Gaussian(mask=self.source, component='Jy', amp=self.source_amp, sigma=self.sigma, t0=self.t0)

        F.add_src(G1)
        F.add_src(G2)
        F.add_src(G3)

        def objective(eps_arr):
            F.eps_r = eps_arr.reshape((self.Nx, self.Ny, self.Nz))
            S = 0.0
            for t_index, fields in enumerate(F.run(self.steps)):
                S += npa.sum(fields['Hx'] + fields['Hy'] + fields['Hz'])
            return S

        grad_autograd = grad(objective, 0)(self.eps_arr)
        grad_numerical = grad_num(objective, self.eps_arr, step_size=DEPS)

        if VERBOSE:
            print('\tobjective function value: ', objective(self.eps_arr))
            print('\tgrad (auto):  \n\t\t', grad_autograd)
            print('\tgrad (num):   \n\t\t', grad_numerical)

        self.check_gradient_error(grad_numerical, grad_autograd)


if __name__ == "__main__":
    unittest.main()
