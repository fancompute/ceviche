import unittest
import sys
import autograd.numpy as npa
import numpy as np
from autograd import grad
from autograd import checkpoint

from copy import deepcopy, copy
from time import time

from ceviche.fdtd import FDTD
from ceviche.sources import Gaussian
from ceviche.constants import *

# gradient error tolerance
EPSILON = 1e-4

class TestAG(unittest.TestCase):
    '''Tests for Gradient Correctness'''

    def setUp(self):

        # basic simulation parameters
        self.Nx = 100
        self.Ny = 100
        self.Nz = 1

        self.omega = 2*np.pi*200e12
        self.dL = 1e-6
        self.pml = [10, 10, 0]

        self.source_mask = np.ones((self.Nx, self.Ny))
        self.source_mask[5, 5] = 1

        # sources (chosen to have objectives around 1)
        self.source_amp = 1e-8

        self.source = np.zeros((self.Nx, self.Ny, self.Nz, 1))
        self.source[self.Nx//2, self.Ny//2] = self.source_amp

        # starting relative permittivity (random for debugging)
        self.eps_r   = np.random.random((self.Nx, self.Ny, self.Nz, 1)) + 1

    def test_grad(self):

        F = FDTD(self.eps_r, dL=self.dL, NPML=self.pml)

        source_loc1 = np.zeros(F.grid_shape)
        source_loc1[self.Nx//2, self.Ny//2, 0, 0] = 1
        G1 = Gaussian(mask=self.source, component='Jz', amp=self.source_amp, sigma=20, t0=300)

        F.add_src(G1)

        steps = 500

        measuring_point = np.zeros(F.grid_shape)
        measuring_point[self.Nx//2, self.Ny//2, 0, 0] = 1

        def objective(eps_r):
            F.eps_r = eps_r
            S = 0.0
            for t_index, fields in enumerate(F.run(steps)):
                S += npa.sum(fields['Ex'] + fields['Ey'] + fields['Ez'], axis=(0,1,2,3))
            return S

        grad_ag = grad(objective, 0)(copy(F.eps_r))


if __name__ == "__main__":
    unittest.main()
