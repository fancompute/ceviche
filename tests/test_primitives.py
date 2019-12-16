import unittest
import numpy as np
import matplotlib.pylab as plt

from ceviche.primitives import *
from ceviche.constants import *

import ceviche    # use the ceviche wrapper for autograd derivatives
DECIMAL = 3       # number of decimals to check to

## Setup
np.random.seed(2)  # note sometimes the random matrices are singular and therefore the solver doesn't work (gives really large answers)

class TestPlaneWave(unittest.TestCase):

    """ Tests whether a plane wave has the right wavelength """

    def setUp(self):

        self.N = 8                  # size of matrix dimensions.  matrix shape = (N, N)
        self.M = self.N**2 - 20     # number of non-zeros (make it dense for numerical stability)
        self.M2 = self.M - 2        # number of non-zeros in the second sparse matrix (for spsp_mult tests)

        # these are the default values used within the test functions
        self.indices_const = make_rand_indeces(self.N, self.M)
        self.entries_const = make_rand_complex(self.M)
        self.indices_const2 = make_rand_indeces(self.N, self.M2)
        self.entries_const2 = make_rand_complex(self.M2)
        self.x_const = make_rand_complex(self.N)
        self.b_const = make_rand_complex(self.N)

    def out_fn(self, output_vector):
        # this function takes the output of each primitive and returns a real scalar (sort of like the objective function)
        return npa.abs(npa.sum(output_vector))

    def err_msg(self, fn_name, mode):
        return '{}-mode gradients failed for fn: {}'.format(mode, fn_name)

    def test_mult_entries(self):

        def fn_mult_entries(entries):
            # sparse matrix multiplication (Ax = b) as a function of matrix entries 'A(entries)'
            b = sp_mult(entries, self.indices_const, self.x_const)
            return self.out_fn(b)

        entries = make_rand_complex(self.M)

        grad_rev = ceviche.jacobian(fn_mult_entries, mode='reverse')(entries)[0]
        grad_for = ceviche.jacobian(fn_mult_entries, mode='forward')(entries)[0]
        grad_true = grad_num(fn_mult_entries, entries)

        np.testing.assert_almost_equal(grad_rev, grad_true, decimal=DECIMAL, err_msg=self.err_msg('fn_mult_entries', 'reverse'))
        np.testing.assert_almost_equal(grad_for, grad_true, decimal=DECIMAL, err_msg=self.err_msg('fn_mult_entries', 'forward'))

    def test_mult_x(self):

        def fn_mult_x(x):
            # sparse matrix multiplication (Ax = b) as a function of dense vector 'x'
            b = sp_mult(self.entries_const, self.indices_const, x)
            return self.out_fn(b)

        x = make_rand_complex(self.N)

        grad_rev = ceviche.jacobian(fn_mult_x, mode='reverse')(x)[0]
        grad_for = ceviche.jacobian(fn_mult_x, mode='forward')(x)[0]
        grad_true = grad_num(fn_mult_x, x)

        np.testing.assert_almost_equal(grad_rev, grad_true, decimal=DECIMAL, err_msg=self.err_msg('fn_mult_x', 'reverse'))
        np.testing.assert_almost_equal(grad_for, grad_true, decimal=DECIMAL, err_msg=self.err_msg('fn_mult_x', 'forward'))

    def test_solve_entries(self):

        def fn_solve_entries(entries):
            # sparse matrix solve (x = A^{-1}b) as a function of matrix entries 'A(entries)'
            x = sp_solve(entries, self.indices_const, self.b_const)
            return self.out_fn(x)

        entries = make_rand_complex(self.M)

        grad_rev = ceviche.jacobian(fn_solve_entries, mode='reverse')(entries)[0]
        grad_for = ceviche.jacobian(fn_solve_entries, mode='forward')(entries)[0]
        grad_true = grad_num(fn_solve_entries, entries)

        np.testing.assert_almost_equal(grad_rev, grad_true, decimal=DECIMAL, err_msg=self.err_msg('fn_solve_entries', 'reverse'))
        np.testing.assert_almost_equal(grad_for, grad_true, decimal=DECIMAL, err_msg=self.err_msg('fn_solve_entries', 'forward'))

    def test_solve_b(self):

        def fn_solve_b(b):
            # sparse matrix solve (x = A^{-1}b) as a function of source 'b'
            x = sp_solve(self.entries_const, self.indices_const, b)
            return self.out_fn(x)

        b = make_rand_complex(self.N)

        grad_rev = ceviche.jacobian(fn_solve_b, mode='reverse')(b)[0]
        grad_for = ceviche.jacobian(fn_solve_b, mode='forward')(b)[0]
        grad_true = grad_num(fn_solve_b, b)

        np.testing.assert_almost_equal(grad_rev, grad_true, decimal=DECIMAL, err_msg=self.err_msg('fn_solve_b', 'reverse'))
        np.testing.assert_almost_equal(grad_for, grad_true, decimal=DECIMAL, err_msg=self.err_msg('fn_solve_b', 'forward'))

    def test_spmut_entries(self):

        def fn_spsp_entries_a(entries):
            # sparse matrix - sparse matrix dot procut as function of entries into first matrix (A)
            entries_c, indices_c = spsp_mult(entries, self.indices_const, self.entries_const2, self.indices_const2, N=self.N)
            # and then as a function of entries in the second matrix (X)
            entries_d, indices_d = spsp_mult(self.entries_const, self.indices_const, entries_c, indices_c, N=self.N)
            # do a sparse linear solve using the resulting matrix and return out_fn of result
            x = sp_solve(entries_d, indices_d, self.b_const)
            return self.out_fn(x)

        entries = make_rand_complex(self.M)

        grad_rev = ceviche.jacobian(fn_spsp_entries_a, mode='reverse')(entries)[0]
        grad_for = ceviche.jacobian(fn_spsp_entries_a, mode='forward')(entries)[0]
        grad_true = grad_num(fn_spsp_entries_a, entries)

        np.testing.assert_almost_equal(grad_rev, grad_true, decimal=DECIMAL, err_msg=self.err_msg('fn_solve_entries', 'reverse'))
        np.testing.assert_almost_equal(grad_for, grad_true, decimal=DECIMAL, err_msg=self.err_msg('fn_solve_entries', 'forward'))

if __name__ == '__main__':
    unittest.main()
