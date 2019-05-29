import unittest
import sys
import autograd.numpy as npa
import numpy as np
import autograd as ag
from autograd import checkpoint

from copy import deepcopy, copy
from time import time

sys.path.append('..')
from ceviche.fdtd import FDTD
from ceviche.sources import Gaussian
from ceviche.gradients import grad_forward, grad_adjoint, grad_numerical, grad_autograd, sum_of_squares
from ceviche.constants import *

# gradient error tolerance
EPSILON = 1e-4

class TestAG(unittest.TestCase):
    '''Tests for Gradient Correctness'''

    # self.setUp() sets up the FDTD and runs a numerical gradient.
    # gradients and such are stored as Class variables to get around this
    # This flag checks whether setUp() has been run, if True, dont need to run it again.
    SETUP_DONE = False

    def setUp(self):

        if not TestAG.SETUP_DONE:

            print('setting up...')

            Nx = 40
            Ny = 40
            Nz = 1

            TestAG.eps_r = np.ones((Nx, Ny, Nz, 1))
            TestAG.eps_r[Nx//2 - Nx//4 : Nx//2 + Nx//4, Ny//2 - Ny//4 : Ny//2 + Ny//4, 0, 0] = 4
            TestAG.F = FDTD(TestAG.eps_r, dL=1e-8, NPML=[0, 0, 0])

            source_loc1 = np.zeros(TestAG.F.grid_shape)
            source_loc1[Nx//2-4, Ny//2+4, 0, 0] = 1
            G1 = Gaussian(mask=source_loc1, component='Jx', amp=1, sigma=20, t0=300)

            source_loc2 = np.zeros(TestAG.F.grid_shape)
            source_loc2[Nx//2+4, Ny//2-4, 0, 0] = 1
            G2 = Gaussian(mask=source_loc2, component='Jz', amp=-1, sigma=20, t0=300)

            TestAG.F.add_src(G1)
            TestAG.F.add_src(G2)

            TestAG.steps = 500

            measuring_point = np.zeros(TestAG.F.grid_shape)
            measuring_point[Nx//2, Ny//2, 0, 0] = 1

            def objective(Ex, Ey, Ez, t):
                # return np.sum(Ez * measuring_point)
                # I = sum_of_squares(Ex, Ey, Ez)
                return npa.sum(npa.abs(Ez) * measuring_point, axis=(0,1,2))

            TestAG.objfn = objective

            TestAG.design_region = np.zeros(TestAG.F.grid_shape)
            TestAG.design_region[Nx//2-9:Nx//2+6, Ny//2-3:Ny//2+6, 0, 0] = 1

            print('computing adjoint gradient...')
            t = time()
            TestAG.grad_adj = grad_adjoint(F=TestAG.F, objective=TestAG.objfn, design_region=TestAG.design_region, steps=TestAG.steps)
            print('    done, took {} seconds\n'.format(time() - t))

            TestAG.SETUP_DONE = True

    def test_autograd(self):

        print('computing gradient using autograd...')

        F = TestAG.F

        # @checkpoint
        def full_objective(eps_r):
            F_ag = FDTD(eps_r, copy(F.dL), copy(F.NPML))
            F_ag.sources = deepcopy(F.sources)
            S = 0.0
            for t_index, fields in enumerate(F_ag.run(TestAG.steps)):
                S += TestAG.objfn(fields['Ex'], fields['Ey'], fields['Ez'], t_index)
            return S
        t = time()        
        grad_ag = ag.grad(full_objective, 0)(copy(F.eps_r))
        design_pts = np.where(TestAG.design_region > 0)
        grad_ag = grad_ag[design_pts]
        print('    done, took {} seconds'.format(time() - t))
        error = np.linalg.norm(TestAG.grad_adj - grad_ag) / np.linalg.norm(TestAG.grad_adj)
        print('error adjoint vs. autograd = {:.4E}\n'.format(error))
        self.assertLess(error, EPSILON)

    def test_autograd_fn(self):

        print('computing gradient using the grad_autograd() function...')

        t = time()        
        grad_ag = grad_autograd(TestAG.F, TestAG.objfn, TestAG.steps, design_region=TestAG.design_region)
        print('    done, took {} seconds'.format(time() - t))
        error = np.linalg.norm(TestAG.grad_adj - grad_ag) / np.linalg.norm(TestAG.grad_adj)
        print('error adjoint vs. autograd = {:.4E}\n'.format(error))
        self.assertLess(error, EPSILON)


if __name__ == "__main__":
    unittest.main()
