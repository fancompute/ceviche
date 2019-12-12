import autograd as ag
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import autograd.numpy as npa
import numpy as np

def make_sparse(entries, indices, N):
    """Construct a sparse csc matrix
    Args:
      entries: numpy array with shape (M,) giving values for non-zero
        matrix entries.
      indices: numpy array with shape (2, M) giving x and y indices for
        non-zero matrix entries.
      N: resulting matrix is N x N
    Returns:
      sparse, complex NxN matrix with specified values
    """  
    shape = (N, N)
    coo = sp.coo_matrix((entries, indices), shape=shape, dtype=np.complex128)
    return coo.tocsc()

def transpose_indices(indices):
    return np.flip(indices, axis=0)

""" Sparse Matrix-Vector Multiplication """

@ag.primitive
def sp_mult(entries, indices, x):
    """ Multiply a sparse matrix (A) by a dense vector
    Args:
      entries: numpy array with shape (num_non_zeros,) giving values for non-zero
        matrix entries.
      indices: numpy array with shape (2, num_non_zeros) giving x and y indices for
        non-zero matrix entries.
      x: 1d numpy array specifying the vector to multiply by.
    Returns:
      1d numpy array corresponding to the solution of A * x = b.
    """
    A = make_sparse(entries, indices, N=x.size)
    return A.dot(x)

def grad_sp_mult_entries_reverse(ans, entries, indices, x):
    def vjp(v):
        i, j = indices
        return v[i] * x[j]
    return vjp

def grad_sp_mult_x_reverse(ans, entries, indices, x):
    def vjp(v):
        indices_T = transpose_indices(indices)
        return sp_mult(entries, indices_T, v)
    return vjp

ag.extend.defvjp(sp_mult, grad_sp_mult_entries_reverse, None, grad_sp_mult_x_reverse)


""" Sparse Matrix Solution  """

@ag.primitive
def sp_solve(entries, indices, b):
    """ Solve a sparse matrix (A) with source x
    Args:
      entries: numpy array with shape (num_non_zeros,) giving values for non-zero
        matrix entries.
      indices: numpy array with shape (2, num_non_zeros) giving x and y indices for
        non-zero matrix entries.
      b: 1d numpy array specifying the source.
    Returns:
      1d numpy array corresponding to the solution of A * x = b.
    """
    A = make_sparse(entries, indices, N=x.size)
    return spl.spsolve(A, b)

# def solve_coo_adjoint(a_entries, a_indices, b):
#   return solve_coo(anp.conj(a_entries), a_indices[::-1], anp.conj(b))

# def grad_solve_coo_entries_reverse(ans, a_entries, a_indices, b):
#   def jvp(grad_ans):
#     lambda_ = solve_coo_adjoint(a_entries, a_indices, grad_ans)
#     i, j = a_indices
#     return -lambda_[i] * anp.conj(ans[j])
#   return jvp

def grad_sp_solve_entries_reverse(ans, entries, indices, b):
    def vjp(v):
        indices_T = transpose_indices(indices)
        adj = sp_solve(np.conj(entries), indices_T, np.conj(v))
        i, j = indices
        return -np.conj(adj[i]) * ans[j]
    return vjp

def grad_sp_solve_x_reverse(ans, entries, indices, b):
    def vjp(v):
        indices_T = transpose_indices(indices)
        return sp_solve(entries, indices_T, v)
    return vjp

ag.extend.defvjp(sp_solve, grad_sp_solve_entries_reverse, None, grad_sp_solve_x_reverse)





if __name__ == '__main__':

    DECIMAL = 3

    def make_rand(N):
        return np.random.random(N) - 0.5

    def make_rand_complex(N):
        return make_rand(N) + 1j * make_rand(N)

    def make_rand_indeces(N, M):
        return np.random.randint(low=0, high=N, size=(2, M))

    def make_rand_entries_indices(N, M):
        entries = make_rand_complex(M)
        indices = make_rand_indeces(N, M)
        return entries, indices

    def grad_num(fn, arg, delta=1e-6):
        N = arg.size
        grad = np.zeros((N,), dtype=np.complex128)
        f0 = fn(arg)
        for i in range(N):
            arg_i = arg.copy()
            arg_i[i] += delta
            fi = fn(arg_i)
            grad[i] += (fi - f0) / delta
            arg_i = arg.copy()
            arg_i[i] += 1j * delta
            fi = fn(arg_i)
            grad[i] += (fi - f0) / (1j * delta)
        return grad

    ## Setup

    M = 10
    N = int(np.sqrt(M))

    # these are the default values used in the test functions
    indices_const = make_rand_indeces(N, M)
    entries_const = make_rand_complex(M)
    x_const = make_rand_complex(N)
    b_const = make_rand_complex(N)

    def out_fn(y):        
        return npa.abs(npa.sum(y))

    ## Mult Entries Reverse

    def fn_mult_entries(entries):
        y = sp_mult(entries, indices_const, x_const)
        return out_fn(y)

    entries = make_rand_complex(M)

    grad = ag.jacobian(fn_mult_entries)(entries)
    grad_true = grad_num(fn_mult_entries, entries)

    np.testing.assert_almost_equal(grad, grad_true, decimal=DECIMAL)

    ## Mult x Reverse

    def fn_mult_x(x):
        y = sp_mult(entries_const, indices_const, x)
        return out_fn(y)

    x = make_rand_complex(N)

    grad = ag.jacobian(fn_mult_x)(x)
    grad_true = grad_num(fn_mult_x, x)

    np.testing.assert_almost_equal(grad, grad_true, decimal=DECIMAL)

    ## Solve x Reverse

    def fn_solve_entries(entries):
        y = sp_solve(entries, indices_const, b_const)
        return out_fn(y)

    entries = make_rand_complex(M)

    grad = ag.jacobian(fn_solve_entries)(entries)
    grad_true = grad_num(fn_solve_entries, entries)

    np.testing.assert_almost_equal(grad, grad_true, decimal=DECIMAL)

    ## Solve x Reverse

    def fn_solve_b(b):
        y = sp_solve(entries_const, indices_const, b)
        return out_fn(y)

    b = make_rand_complex(N)

    grad = ag.jacobian(fn_solve_b)(b)
    grad_true = grad_num(fn_solve_b, b)

    np.testing.assert_almost_equal(grad, grad_true, decimal=DECIMAL)





