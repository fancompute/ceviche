import numpy as np
import autograd.numpy as npa
import scipy.sparse as sp

from .utils import make_sparse, get_entries_indices, is_array
from .primitives import spsp_add, sp_mult, spsp_mult, sp_solve
from .derivatives import der_mat, shift_mat

class Sparse:
    """ A sparse matrix with arbitrary entries, indices, and shape """

    def __init__(self, entries, indices, shape):
        self.entries = entries
        self.indices = indices
        self.shape = shape

    @property
    def A(self):
        """ numpy.ndarray explicit dense matrix representation version of self """
        array = np.zeros(self.shape, dtype=complex)
        i, j = self.indices
        array[i, j] = self.entries
        return array

    @property
    def csr_matrix(self):
        """ scipy.sparse.csr_matrix explicit sparse matrix representation of self """
        return make_sparse(self.entries, self.indices, self.shape)

    @property
    def T(self):
        """ transpose of self """
        return Sparse(self.entries, npa.roll(self.indices, shift=1, axis=0), self.shape)

    def solve(self, other):
        """ linear solve """
        if is_array(other):
            return sp_solve(self.entries, self.indices, other)
        else:
            raise ValueError("can't solve with anything but array-like")

    def __neg__(self):
        return Sparse(-self.entries, self.indices, self.shape)

    def __add__(self, other):
        if isinstance(other, Sparse):
            entries, indices = spsp_add(self.entries, self.indices, other.entries, other.indices, self.shape)
            return Sparse(entries, indices, self.shape)
        elif isinstance(other, sp.csr_matrix):
            other_entries, other_indices = get_entries_indices(other)
            entries, indices = spsp_add(self.entries, self.indices, other_entries, other_indices, self.shape)
            return Sparse(entries, indices, self.shape)
        elif is_array(other):
            # This should generally *not* be needed! Also, it's not autograd compatible.
            res_ndarray = self.A + other
            return res_ndarray

    def __sub__(self, other):
        return self + (-other)

    def __matmul__(self, other):
        if isinstance(other, Sparse):
            entries, indices = spsp_mult(self.entries, self.indices, other.entries, other.indices, self.shape[0])
            return Sparse(entries, indices, self.shape)
        elif isinstance(other, sp.csr_matrix):
            other_entries, other_indices = get_entries_indices(other)
            entries, indices = spsp_mult(self.entries, self.indices, other_entries, other_indices, self.shape[0])
            return Sparse(entries, indices, self.shape)
        elif is_array(other):
            res = sp_mult(self.entries, self.indices, other)
            return res

class Diagonal(Sparse):
    """ A sparse matrix with `diag_vector` along the diagonal and zeros otherwise """

    def __init__(self, diag_vector):
        N = diag_vector.size
        shape = (N, N)
        indices = np.vstack((np.arange(N), np.arange(N)))
        super().__init__(diag_vector, indices, shape)

class Derivative(Sparse):

    def __init__(self, shape, axis, fb):
        der_csr = der_mat(*shape, axis=axis, fb=fb)
        entries, indices = get_entries_indices(der_csr)
        N = np.prod(shape)
        super().__init__(entries, indices, shape=(N, N))

class Convolution(Sparse):

    def __init__(self, shape, kernel):

        res_csr = Sparse(entries=np.zeros(0), indices=np.zeros((2, 0)), shape=shape)

        # loop through indices of flattened kernel
        for k, i in enumerate(kernel.flatten()):

            # get the i,j,k... indices as a tuple
            k_subs = np.unravel_index(i, kernel.shape)

            # this is the permutation matrix that shifts a vector by k_subs
            perm_mat = shift_mat(*shape, shift=k_subs)

            # add together with the kernel value as constant in front
            res_csr = res_csr + k * perm_mat

        # get the entries and indices  of the final array
        entries, indices = get_entries_indices(res_csr)

        # save as Sparse()
        super().__init__(entries, indices, shape)


def from_csr_matrix(csr_matrix):
    """ Creates `sparse` object from explicit scipy.sparse.csr_matrix """

    entries, indices = get_entries_indices(csr_matrix)
    shape = csr_matrix.shape
    return Sparse(entries, indices, shape)

def diags(diagonals, offsets=0, shape=None):
    """
    Similar to scipy.sparse.diags, returns a Sparse object.
    `shape` works slightly differently (more general, see below).
    Works with autograd when w.r.t. `diagonals` and the returned
    `entries` of the Sparse object.

    Parameters
    ----------
    diagonals: np.ndarray or sequence of np.ndarray
        If `diagonals` is a single array, `offsets` must either be a single int
        defining the diagonal, or a sequence of the same legnth, in which case a
        Toeplitz matrix is created.
        If `diagonals` is a sequence of arrays, its length must be the same as
        the length of the `offsets` sequence.
    offsets: sequence of int or an int, optional
        Diagonals to set, > 0 means above main diagonal.
    shape: tuple of int, optional
        Shape of the result. If omitted, a square matrix large enough to contain
        the diagonals is returned. Some diagonals may be padded with zeros if
        needed (the zeros are not stored in the sparse representation).
    """

    if shape is not None:
        Ndmax = np.amax([shape[0], shape[1]])
        sp_shape = shape

    if is_array(diagonals):
        if not isinstance(offsets, int):
            # Will use broadcasting to make a Toeplitz matrix
            if diagonals.size == len(offsets):
                if shape==None:
                    Ndmax = np.amax(np.abs(offsets))
                    sp_shape = (Ndmax, Ndmax)

                diag_seq = []
                for (dind, diag) in enumerate(diagonals):
                    rep = Ndmax - offsets[dind]
                    diag_seq.append(diag*np.ones((rep,)))
            else:
                raise ValueError("If `diagonals` is a single array, `offsets` "
                    "should either be a single integer, or a sqeuence with the "
                    "same length as `diagonals`.")
        else:
            # Single diagonal and single offset
            diag_seq = [diagonals]
            offsets = [offsets]
            if shape is None:
                Ndmax = diagonals.size
                sp_shape = (Ndmax, Ndmax)

    else:
        # diagonals is a sequence
        diag_seq = diagonals
        if shape is None:
            Ndmax = np.amax([d.size for d in diagonals])
            sp_shape = (Ndmax, Ndmax)

    entries = []
    row_inds, col_inds = [], []

    # Construct entries and indices one diagonal at a time
    for (dind, diag) in enumerate(diag_seq):
        Nd = diag.size
        entries.append(diag.ravel())
        if offsets[dind] >= 0:
            row_inds.append(np.arange(Nd))
            col_inds.append(np.arange(Nd) + offsets[dind])
        else:
            row_inds.append(np.arange(Nd) - offsets[dind])
            col_inds.append(np.arange(Nd))

    # Stack arrays and trim things that lie outside the matrix and entries equal to 0
    entries = npa.hstack(entries)
    indices = np.vstack((np.hstack(row_inds), np.hstack(col_inds)))
    inds_keep = np.nonzero((indices[0, :] < sp_shape[0]) &
                            (indices[1, :] < sp_shape[1]) &
                            (entries!=0))[0]

    entries = entries[inds_keep]
    indices = indices[:, inds_keep]

    return Sparse(entries, indices, sp_shape)


def convmat_1d(kernel, in_shape):
    """
    Note
    ----
    Strictly speaking, "convolution" in CNNs is a misnomer, it's technically correlation.
    The difference being whether we do `kernel(i+j)*input(i)` or `kernel(i-j)*input(i)`,
    or in other words whether to flip the kernel in the construction below.
    Anyway, here we implement a matrix that does it as is usually defined in CNNs, i.e.
    `convmat_1d(kernel, in_shape) @ input` yields a vector c defined as
        `c_j = sum_i(kernel(i - j)*input(i))`

    Parameters
    ----------
    kernel : np.ndarray
        1D array defining the kernel
    in_shape : int
        Size of the 1D input

    Returns
    -------
    conv1d : Sparse
        A `Sparse` matrix of shape (in_shape, in_shape) such that `conv1d @ input`
        is the convolution of an `input` vector of shape `in_shape` with the `kernel`.
    """

    Nk = kernel.size
    Nkodd = np.mod(Nk, 2)
    offsets = list(range(-Nk//2+Nkodd, Nk//2+Nkodd))

    # Sparse matrix shape
    shape = (in_shape, in_shape)

    return diags(kernel, offsets, shape)

def convmat_2d(kernel, in_shape):
    """
    Note
    ----
    In terms of signal processing, this is technically self-correlation. See note
    on convmat_1d.

    Parameters
    ----------
    kernel : np.ndarray
        2D array defining the kernel
    in_shape : tuple of int
        tuple of length 2 defining the shape of the input

    Returns
    -------
    conv2d : Sparse
        A `Sparse` matrix of shape (N, N) with N = in_shape[0]*in_shape[1]
        such that `conv2d @ input.ravel()` is the convolution of an `input` array
        of shape `in_shape` with the 2D `kernel`.
    """

    Nkx, Nky = kernel.shape
    Nix, Niy = in_shape
    Nkxodd, Nkyodd = np.mod(Nkx, 2), np.mod(Nky, 2)
    Ni = Nix * Niy

    entries, indr, indc = [], [], []

    for ix in range(Nix):
        for kx in range(-Nkx//2+Nkxodd, Nkx//2+Nkxodd):
            if 0 <= ix - kx < Nix:
                K = kernel[kx + Nkx//2, :]
                offsets = list(range(-Nky//2+Nkyodd, Nky//2+Nkyodd))
                s = (Niy, Niy)
                conv1d = diags(K, offsets, s)

                entries.append(conv1d.entries)
                ind1 = conv1d.indices[0, :]
                ind2 = conv1d.indices[1, :]
                indr.append(ind1 + (ix - kx)*Niy)
                indc.append(ind2 + ix*Niy)

    entries = npa.hstack(entries)
    indices = np.vstack((np.hstack(indr), np.hstack(indc)))

    # Sparse matrix shape
    shape = (Ni, Ni)

    return Sparse(entries, indices, shape)



