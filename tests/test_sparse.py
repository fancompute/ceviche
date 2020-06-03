import unittest
import autograd.numpy as np
from numpy.testing import assert_allclose
import matplotlib.pylab as plt
import autograd as ag

import sys
sys.path.append('../ceviche')

from ceviche.sparse import Sparse, Diagonal, from_csr_matrix
import scipy.sparse as sp

class TestSparse(unittest.TestCase):

    """ Tests the field patterns by inspection """

    def setUp(self):
        N = 10

        shape = (N, N)
        self.A_csr_matrix = sp.random(*shape, format='csr')
        self.B_csr_matrix = sp.random(*shape, format='csr')

        self.A_Sparse = from_csr_matrix(self.A_csr_matrix)
        self.B_Sparse = from_csr_matrix(self.B_csr_matrix)

        # self.A_ndarray = self.A_csr_matrix.A
        # self.B_ndarray = self.B_csr_matrix.A

        self.diag_vec = np.random.random(N)
        self.D = Diagonal(self.diag_vec)

    """ addition """

    def _test_add_Sparse(self):
        C = self.A_Sparse + self.B_Sparse
        true_C = self.A_ndarray + self.B_ndarray
        assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def _test_add_csr_matrix(self):
        C = self.A_Sparse + self.B_csr_matrix
        true_C = self.A_ndarray + self.B_ndarray
        assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def _test_add_ndarray(self):
        C = self.A_Sparse + self.B_ndarray
        true_C = self.A_ndarray + self.B_ndarray
        assert_allclose(C, true_C)
        assert isinstance(C, np.ndarray)

    """ subtraction """

    def _test_sub_Sparse(self):
        C = self.A_Sparse - self.B_Sparse
        true_C = self.A_ndarray - self.B_ndarray
        assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def _test_sub_csr_matrix(self):
        C = self.A_Sparse - self.B_csr_matrix
        true_C = self.A_ndarray - self.B_ndarray
        assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def _test_sub_ndarray(self):
        C = self.A_Sparse - self.B_ndarray
        true_C = self.A_ndarray - self.B_ndarray
        assert_allclose(C, true_C)
        assert isinstance(C, np.ndarray)


    """ negative """

    def _test_neg_Sparse(self):
        C = -self.A_Sparse
        true_C = -self.A_ndarray
        assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    """ matrix multiplication """

    def _test_matmul_Sparse(self):
        C = self.A_Sparse @ self.B_Sparse
        true_C = self.A_ndarray @ self.B_ndarray
        assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def _test_matmul_csr_matrix(self):
        C = self.A_Sparse @ self.B_csr_matrix
        true_C = self.A_ndarray @ self.B_ndarray
        assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def _test_matmul_ndarray(self):
        C = self.A_Sparse @ self.B_ndarray
        true_C = self.A_ndarray @ self.B_ndarray
        assert_allclose(C, true_C)
        assert isinstance(C, np.ndarray)

    """ diagonal matrices """

    def _test_diag(self):
        D_ndarray = self.D.csr_matrix.A
        assert_allclose(D_ndarray.diagonal(), self.diag_vec)

    """ autograd shit """

    def test_ag_matmul_Sparse(self):

        def f(v):
            D1 = Diagonal(v)
            D2 = Diagonal(v)
            D3 = D1 @ D2
            return np.abs(np.sum(D3.entries))

        val = f(self.diag_vec)

        grad = ag.grad(f)(self.diag_vec)
        # analytical val  = np.abs(np.sum(v ** 2))
        # analytical grad = 2 * np.abs(v)
        assert_allclose(grad, 2 * np.abs(self.diag_vec))

    def test_ag_matmul_ndarray(self):

        def f(v):
            D1 = Diagonal(v)
            v2 = D1 @ v
            return np.abs(np.sum(v2))

        grad = ag.grad(f)(self.diag_vec)
        assert_allclose(grad, 2 * np.abs(self.diag_vec))


if __name__ == '__main__':
    unittest.main()
