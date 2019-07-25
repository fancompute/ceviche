import unittest
import sys
import autograd.numpy as npa
import numpy as np
from autograd import grad
from autograd import checkpoint

from copy import deepcopy, copy
from time import time

import sys
sys.path.append('../ceviche')

from ceviche import fdtd
from ceviche.utils import grad_num
from ceviche.jacobians import jacobian

# gradient error tolerance
ALLOWED_RATIO = 1e-4    # maximum allowed ratio of || grad_num - grad_auto || vs. || grad_num ||
DEPS = 1e-6             # numerical gradient step size
VERBOSE = False

print("Testing the FDTD gradients")

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
        self.steps = 500
        self.t0 = 300
        self.sigma = 20        
        self.source_amp = 1
        self.source_pos = np.zeros((self.Nx, self.Ny, self.Nz))
        self.source_pos[self.Nx//2, self.Ny//2, self.Nz//2] = self.source_amp
        self.gaussian = lambda t: self.source_pos * self.source_amp * np.exp(-(t - self.t0)**2 / 2 / self.sigma**2)

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

    def test_grad_rev_E(self):

        print('\ttesting E fields in FDTD (reverse mode)')

        F = fdtd(self.eps_r, dL=self.dL, npml=self.pml)

        def objective(eps_arr):
            F.eps_r = eps_arr.reshape((self.Nx, self.Ny, self.Nz))
            S = 0.0
            for t_index in range(self.steps):
                fields = F.forward(Jz=self.gaussian(t_index))
                S += npa.sum(fields['Ex'] + fields['Ey'] + fields['Ez'])
            return S

        jac_autograd_rev = jacobian(objective, mode='reverse')(self.eps_arr)
        jac_numerical = jacobian(objective, mode='numerical', step_size=DEPS)(self.eps_arr)

        if VERBOSE:
            print('\tobjective function value: ', objective(self.eps_arr))
            print('\tjacobian (auto):  \n\t\t', jac_autograd_rev)
            print('\tjacobian (num):   \n\t\t', jac_numerical)

        self.check_gradient_error(jac_numerical, jac_autograd_rev)


    def test_grad_for_E(self):

        print('\ttesting E fields in FDTD (forward mode)')

        F = fdtd(self.eps_r, dL=self.dL, npml=self.pml)

        def objective(c):

            F = fdtd(c * self.eps_r, dL=self.dL, npml=self.pml)

            S = 0.0
            for t_index in range(self.steps):
                fields = F.forward(Jx=self.gaussian(t_index))
                S += fields['Ex'] + fields['Ey'] + fields['Ez']
            return S

        c0 = 2.0
        jac_autograd_for = jacobian(objective, mode='forward')(c0)
        jac_numerical = jacobian(objective, mode='numerical', step_size=DEPS)(c0)

        if VERBOSE:
            print('\tobjective function value: ', objective(self.eps_arr))
            print('\tjacobian (auto):  \n\t\t', jac_autograd_for)
            print('\tjacobian (num):   \n\t\t', jac_numerical)

        self.check_gradient_error(jac_numerical, jac_autograd_for)

    def test_grad_rev_H(self):

        print('\ttesting H fields in FDTD (reverse mode)')

        F = fdtd(self.eps_r, dL=self.dL, npml=self.pml)

        def objective(eps_arr):
            F.eps_r = eps_arr.reshape((self.Nx, self.Ny, self.Nz))
            S = 0.0
            for t_index in range(self.steps):
                fields = F.forward(Jx=self.gaussian(t_index))
                S += npa.sum(fields['Hx'] + fields['Hy'] + fields['Hz'])
            return S

        jac_autograd_rev = jacobian(objective)(self.eps_arr)
        jac_numerical = jacobian(objective, mode='numerical', step_size=DEPS)(self.eps_arr)

        if VERBOSE:
            print('\tobjective function value: ', objective(self.eps_arr))
            print('\tjacobian (auto):  \n\t\t', jac_autograd_rev)
            print('\tjacobian (num):   \n\t\t', jac_numerical)

        self.check_gradient_error(jac_numerical, jac_autograd_rev)

    def test_grad_for_H(self):

        print('\ttesting H fields in FDTD (forward mode)')    

        def objective(c):

            F = fdtd(c * self.eps_r, dL=self.dL, npml=self.pml)

            S = 0.0
            for t_index in range(self.steps):
                fields = F.forward(Jx=self.gaussian(t_index))
                S += fields['Hx'] + fields['Hy'] + fields['Hz']
            return S

        c0 = 2.0
        jac_autograd_for = jacobian(objective, mode='forward')(c0)
        jac_numerical = jacobian(objective, mode='numerical', step_size=DEPS)(c0)

        if VERBOSE:
            print('\tobjective function value: ', objective(c0))
            print('\tjacobian (auto):  \n\t\t', jac_autograd_for)
            print('\tjacobian (num):   \n\t\t', jac_numerical)

        self.check_gradient_error(jac_numerical, jac_autograd_for)


if __name__ == "__main__":
    unittest.main()
