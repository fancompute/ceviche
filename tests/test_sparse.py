import unittest
import autograd.numpy as np
from numpy.testing import assert_allclose
import matplotlib.pylab as plt
import sys
sys.path.append('../ceviche')

from ceviche.sparse import Sparse, Diagonal, from_csr_matrix
import scipy.sparse as sp

class TestSparse(unittest.TestCase):

    """ Tests the field patterns by inspection """

    def setUp(self):
        shape = (100, 100)

        self.A_csr_matrix = sp.random(*shape, format='csr')
        self.B_csr_matrix = sp.random(*shape, format='csr')

        self.A_Sparse = from_csr_matrix(self.A_csr_matrix)
        self.B_Sparse = from_csr_matrix(self.B_csr_matrix)

        self.A_ndarray = self.A_csr_matrix.A
        self.B_ndarray = self.B_csr_matrix.A

    """ addition """

    def test_add_Sparse(self):
        C = self.A_Sparse + self.B_Sparse
        true_C = self.A_ndarray + self.B_ndarray
        assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def test_add_csr_matrix(self):
        C = self.A_Sparse + self.B_csr_matrix
        true_C = self.A_ndarray + self.B_ndarray
        assert_allclose(C.csr_matrix.A, true_C)
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
        assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def test_sub_csr_matrix(self):
        C = self.A_Sparse - self.B_csr_matrix
        true_C = self.A_ndarray - self.B_ndarray
        assert_allclose(C.csr_matrix.A, true_C)
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
        assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    """ matrix multiplication """

    def test_matmul_Sparse(self):
        C = self.A_Sparse @ self.B_Sparse
        true_C = self.A_ndarray @ self.B_ndarray
        assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def test_matmul_csr_matrix(self):
        C = self.A_Sparse @ self.B_csr_matrix
        true_C = self.A_ndarray @ self.B_ndarray
        assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def test_matmul_ndarray(self):
        C = self.A_Sparse @ self.B_ndarray
        true_C = self.A_ndarray @ self.B_ndarray
        assert_allclose(C, true_C)
        assert isinstance(C, np.ndarray)

    """ diagonal matrices """

    def test_diag(self):
        N = 100
        diag_vec = np.random.random(N)
        D = Diagonal(diag_vec)
        D_ndarray = D.csr_matrix.A

        assert_allclose(D_ndarray[np.arange(N), np.arange(N)], diag_vec)


if __name__ == '__main__':
    unittest.main()
