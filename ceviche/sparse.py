import numpy as np
import autograd.numpy as npa
import scipy.sparse as sp

from .utils import make_sparse, get_entries_indices
from .primitives import sp_mult, spsp_mult, sp_solve

class Sparse:
    """ A sparse matrix with arbitrary entries, indices, and shape """

    def __init__(self, entries, indices, shape):
        self.entries = entries
        self.indices = indices
        self.shape = shape

    @property
    def csr_matrix(self):
        return make_sparse(self.entries, self.indices, self.shape)

    @property
    def T(self):
        return Sparse(self.entries, npa.roll(self.indices, 1, axis=0), self.shape)

    def __neg__(self):
        return Sparse(-self.entries, self.indices, self.shape)

    def __add__(self, other):
        if isinstance(other, Sparse):
            res_csr = self.csr_matrix + other.csr_matrix
            return from_csr_matrix(res_csr)
        elif isinstance(other, sp.csr_matrix):
            res_csr = self.csr_matrix + other
            return from_csr_matrix(res_csr)
        elif isinstance(other, np.ndarray):
            res_ndarray = self.csr_matrix.A + other
            return res_ndarray

    # def __sub__(self, other):
    #     if isinstance(other, Sparse):
    #         res_csr = self.csr_matrix - other.csr_matrix
    #         return from_csr_matrix(res_csr)
    #     elif isinstance(other, sp.csr_matrix):
    #         res_csr = self.csr_matrix - other
    #         return from_csr_matrix(res_csr)
    #     elif isinstance(other, np.ndarray):
    #         res_ndarray = self.csr_matrix.A - other
    #         return res_ndarray

    def __sub__(self, other):
        return self + (-other)

    def __matmul__(self, other):
        if isinstance(other, Sparse):
            res_entries, res_indices = spsp_mult(self.entries, self.indices, other.entries, other.indices, self.shape[0])
            return Sparse(res_entries, res_indices, self.shape)
        elif isinstance(other, sp.csr_matrix):
            other_entries, other_indices = get_entries_indices(other)
            res_entries, res_indices = spsp_mult(self.entries, self.indices, other_entries, other_indices, self.shape[0])
            return Sparse(res_entries, res_indices, self.shape)
        elif isinstance(other, np.ndarray) or isinstance(other, npa.numpy_boxes.ArrayBox):
            res = sp_mult(self.entries, self.indices, other)
            return res

class Diagonal(Sparse):
    """ A sparse matrix with `diag_vector` along the diagonal and zeros otherwise """

    def __init__(self, diag_vector):
        N = diag_vector.size
        shape = (N, N)
        indices = np.vstack((np.arange(N), np.arange(N)))
        super().__init__(diag_vector, indices, shape)

def from_csr_matrix(csr_matrix):
    """ Creates `sparse` object from explicit scipy.sparse.csr_matrix """

    entries, indices = get_entries_indices(csr_matrix)
    shape = csr_matrix.shape
    return Sparse(entries, indices, shape)

