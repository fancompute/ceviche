import unittest
import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append('../ceviche')

from ceviche.sparse import Sparse, from_csr_matrix
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
        np.testing.assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def test_add_csr_matrix(self):
        C = self.A_Sparse + self.B_csr_matrix
        true_C = self.A_ndarray + self.B_ndarray
        np.testing.assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def test_add_ndarray(self):
        C = self.A_Sparse + self.B_ndarray
        true_C = self.A_ndarray + self.B_ndarray
        np.testing.assert_allclose(C, true_C)
        assert isinstance(C, np.ndarray)

    """ subtraction """

    def test_sub_Sparse(self):
        C = self.A_Sparse - self.B_Sparse
        true_C = self.A_ndarray - self.B_ndarray
        np.testing.assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def test_sub_csr_matrix(self):
        C = self.A_Sparse - self.B_csr_matrix
        true_C = self.A_ndarray - self.B_ndarray
        np.testing.assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def test_sub_ndarray(self):
        C = self.A_Sparse - self.B_ndarray
        true_C = self.A_ndarray - self.B_ndarray
        np.testing.assert_allclose(C, true_C)
        assert isinstance(C, np.ndarray)


    """ negative """

    def test_neg_Sparse(self):
        C = -self.A_Sparse
        true_C = -self.A_ndarray
        np.testing.assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    """ matrix multiplication """

    def test_matmul_Sparse(self):
        C = self.A_Sparse @ self.B_Sparse
        true_C = self.A_ndarray @ self.B_ndarray
        np.testing.assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def test_matmul_csr_matrix(self):
        C = self.A_Sparse @ self.B_csr_matrix
        true_C = self.A_ndarray @ self.B_ndarray
        np.testing.assert_allclose(C.csr_matrix.A, true_C)
        assert isinstance(C, Sparse)

    def test_matmul_ndarray(self):
        C = self.A_Sparse @ self.B_ndarray
        true_C = self.A_ndarray @ self.B_ndarray
        np.testing.assert_allclose(C, true_C)
        assert isinstance(C, np.ndarray)


if __name__ == '__main__':
    unittest.main()
