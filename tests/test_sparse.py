import unittest
import autograd.numpy as np
import matplotlib.pylab as plt
import autograd as ag
import scipy.sparse as sp
import scipy.signal as sps

from numpy.testing import assert_allclose

import sys
sys.path.append('../ceviche')

from ceviche.sparse import Sparse, Diagonal, from_csr_matrix, diags, convmat_1d, convmat_2d
from ceviche import jacobian

class TestSparse(unittest.TestCase):

    """ Tests the field patterns by inspection """

    def setUp(self):
        N = 20

        shape = (N, N)
        self.A_csr_matrix = sp.random(*shape, format='csr')
        self.B_csr_matrix = sp.random(*shape, format='csr')

        self.A_Sparse = from_csr_matrix(self.A_csr_matrix)
        self.B_Sparse = from_csr_matrix(self.B_csr_matrix)

        self.A_ndarray = self.A_Sparse.A
        self.B_ndarray = self.B_Sparse.A

        self.diag_vec = np.random.random(N)
        self.D = Diagonal(self.diag_vec)
        self.D_csr_matrix = self.D.csr_matrix

    """ transpose """
    def test_transpose(self):
        C = self.A_Sparse.T
        true_C = self.A_ndarray.T
        assert_allclose(C.A, true_C)
        assert isinstance(C, Sparse)

    """ addition """

    def test_add_Sparse(self):
        C = self.A_Sparse + self.B_Sparse
        true_C = self.A_ndarray + self.B_ndarray
        assert_allclose(C.A, true_C)
        assert isinstance(C, Sparse)

    def test_add_csr_matrix(self):
        C = self.A_Sparse + self.B_csr_matrix
        true_C = self.A_ndarray + self.B_ndarray
        assert_allclose(C.A, true_C)
        assert isinstance(C, Sparse)

    def test_add_ndarray(self):
        C = self.A_Sparse + self.B_ndarray
        true_C = self.A_ndarray + self.B_ndarray
        assert_allclose(C, true_C)
        assert isinstance(C, np.ndarray)

    """ subtraction """

    def test_sub_Sparse(self):
        C = self.A_Sparse - self.B_Sparse
        true_C = self.A_ndarray - self.B_ndarray
        assert_allclose(C.A, true_C)
        assert isinstance(C, Sparse)

    def test_sub_csr_matrix(self):
        C = self.A_Sparse - self.B_csr_matrix
        true_C = self.A_ndarray - self.B_ndarray
        assert_allclose(C.A, true_C)
        assert isinstance(C, Sparse)

    def test_sub_ndarray(self):
        C = self.A_Sparse - self.B_ndarray
        true_C = self.A_ndarray - self.B_ndarray
        assert_allclose(C, true_C)
        assert isinstance(C, np.ndarray)

    """ negative """

    def test_neg_Sparse(self):
        C = -self.A_Sparse
        true_C = -self.A_ndarray
        assert_allclose(C.A, true_C)
        assert isinstance(C, Sparse)

    """ matrix multiplication """

    def test_matmul_Sparse(self):
        C = self.A_Sparse @ self.B_Sparse
        true_C = self.A_ndarray @ self.B_ndarray
        assert_allclose(C.A, true_C)
        assert isinstance(C, Sparse)

    def test_matmul_csr_matrix(self):
        C = self.A_Sparse @ self.B_csr_matrix
        true_C = self.A_ndarray @ self.B_ndarray
        assert_allclose(C.A, true_C)
        assert isinstance(C, Sparse)

    def test_matmul_ndarray(self):
        C = self.A_Sparse @ self.B_ndarray
        true_C = self.A_ndarray @ self.B_ndarray
        assert_allclose(C, true_C)
        assert isinstance(C, np.ndarray)

    """ diagonal matrices """

    def test_diag(self):
        D_ndarray = self.D.A
        assert_allclose(D_ndarray.diagonal(), self.diag_vec)

    """ diags constructor """

    def test_diags_toeplitz(self):
        values = np.random.rand(3,)
        offsets = [-1, 0, 2]
        shape = (5, 4)
        C = diags(values, offsets, shape).A
        true_C = sp.diags(values, offsets, shape).A
        assert_allclose(C, true_C)

    def test_diags_sequence(self):
        diagonals = [np.random.rand(4,) for i in range(3)]
        offsets = [-1, 0, 2]
        shape = (5, 4)
        C = diags(diagonals, offsets, shape).A
        true_C = sp.diags(diagonals, offsets, shape).A
        assert_allclose(C, true_C)

    """ convolution matrices """

    def test_conv1d(self):
        in_shape = 6
        inp = np.random.rand(in_shape)

        # Odd-sized kernel 
        kernel = np.random.rand(3)
        c = convmat_1d(kernel, in_shape) @ inp
        true_c = sps.correlate(inp, kernel, mode='same')
        assert_allclose(c, true_c)

        # Even-sized kernel 
        kernel = np.random.rand(4)
        c = convmat_1d(kernel, in_shape) @ inp
        true_c = sps.correlate(inp, kernel, mode='same')
        assert_allclose(c, true_c)

    def test_conv2d(self):
        in_shape = (5, 6)
        inp = np.random.rand(in_shape[0], in_shape[1])
        kernel = np.random.rand(4, 3)
        c = convmat_2d(kernel, in_shape) @ inp.ravel()
        true_c = sps.correlate(inp, kernel, mode='same')
        assert_allclose(c, true_c.ravel())

    """ autograd stuff """

    """ ag multiplication """

    def test_ag_matmul_Sparse(self):

        def f(v):
            D1 = Diagonal(v)
            D2 = Diagonal(v)
            D3 = D1 @ D2
            return np.abs(np.sum(D3.entries))

        grad = jacobian(f, mode='reverse')(self.diag_vec)
        assert_allclose(grad[0,:], 2 * self.diag_vec)

        grad = jacobian(f, mode='forward')(self.diag_vec)
        assert_allclose(grad[0,:], 2 * self.diag_vec)

    def test_ag_matmul_csr_matrix(self):

        def f(v):
            D1 = Diagonal(v)
            # Gradient only supported w.r.t. first argument of @
            D2 = self.D_csr_matrix
            D3 = D1 @ D2
            return np.abs(np.sum(D3.entries))

        grad = jacobian(f, mode='reverse')(self.diag_vec)
        assert_allclose(grad[0,:], self.diag_vec)

        grad = jacobian(f, mode='forward')(self.diag_vec)
        assert_allclose(grad[0,:], self.diag_vec)

    def test_ag_matmul_ndarray(self):

        def f(v):
            D1 = Diagonal(v)
            v2 = D1 @ v
            return np.abs(np.sum(v2))

        grad = jacobian(f, mode='reverse')(self.diag_vec)
        assert_allclose(grad[0,:], 2 * self.diag_vec)

        grad = jacobian(f, mode='forward')(self.diag_vec)
        assert_allclose(grad[0,:], 2 * self.diag_vec)

    """ ag addition """

    def test_ag_add_Sparse(self):

        def f(v):
            D1 = Diagonal(v)
            D2 = Diagonal(v)
            D3 = D1 + D2
            v2 = D3 @ v
            return np.abs(np.sum(v2))

        grad = jacobian(f, mode='reverse')(self.diag_vec)
        assert_allclose(grad[0,:], 4 * self.diag_vec)

        grad = jacobian(f, mode='forward')(self.diag_vec)
        assert_allclose(grad[0,:], 4 * self.diag_vec)

    def test_ag_add_csr_matrix(self):

        def f(v):
            D1 = Diagonal(v)
            # Gradient only supported w.r.t. first argument of +
            D2 = self.D_csr_matrix
            D3 = D1 + D2
            v2 = D3 @ v
            return np.abs(np.sum(v2))

        grad = jacobian(f, mode='reverse')(self.diag_vec)
        assert_allclose(grad[0,:], 3 * self.diag_vec)

        grad = jacobian(f, mode='forward')(self.diag_vec)
        assert_allclose(grad[0,:], 3 * self.diag_vec)

    def test_ag_add_ndarray(self):
        """ This is not ag compatible and also shouldn't be used in general,
        as it returns a dense ndarray"""
        pass

    """ ag diags constructor """

    def test_ag_diag_toeplitz(self):

        def f(vals):
            D1 = diags(vals, [-1, 0, 2], shape=(5, 4))
            return np.abs(np.sum(D1.entries))

        vals = np.random.rand(3)
        grad_r = jacobian(f, mode='reverse')(vals)
        grad_f = jacobian(f, mode='forward')(vals)
        grad_n = jacobian(f, mode='numerical')(vals)
        assert_allclose(grad_r, grad_n)
        assert_allclose(grad_f, grad_n)

    def test_ag_diag_sequence(self):

        def f(vals):
            diagonals = [vals[:3], vals[3:]]
            D1 = diags(diagonals, [-1, 2], shape=(5, 4))
            return np.abs(np.sum(D1.entries))

        vals = np.random.rand(6)
        grad_r = jacobian(f, mode='reverse')(vals)
        grad_f = jacobian(f, mode='forward')(vals)
        grad_n = jacobian(f, mode='numerical')(vals)
        assert_allclose(grad_r, grad_n)
        assert_allclose(grad_f, grad_n)

if __name__ == '__main__':
    unittest.main()
