import autograd as ag
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import autograd.numpy as npa
import numpy as np

""" Helper Functions """

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

""" Primitives for Sparse Matrix-Vector Multiplication """

@ag.primitive
def sp_mult(entries, indices, x):
    """ Multiply a sparse matrix (A) by a dense vector (x)
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

def grad_sp_mult_entries_forward(g, b, entries, indices, x):
    return sp_mult(g, indices, x)

def grad_sp_mult_x_forward(g, b, entries, indices, x):
    return sp_mult(entries, indices, g)

ag.extend.defjvp(sp_mult, grad_sp_mult_entries_forward, None, grad_sp_mult_x_forward)


""" Primitives for Sparse Matrix-Vector Solve """

@ag.primitive
def sp_solve(entries, indices, b):
    """ Solve a sparse matrix (A) with source (b)
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

def grad_sp_solve_entries_reverse(ans, entries, indices, b):
    def vjp(v):
        indices_T = transpose_indices(indices)
        adj = sp_solve(entries, indices_T, v)
        i, j = indices
        return -adj[i] * ans[j]
    return vjp

def grad_sp_solve_x_reverse(ans, entries, indices, b):
    def vjp(v):
        indices_T = transpose_indices(indices)
        return sp_solve(entries, indices_T, v)
    return vjp

ag.extend.defvjp(sp_solve, grad_sp_solve_entries_reverse, None, grad_sp_solve_x_reverse)

def grad_sp_solve_entries_forward(g, x, entries, indices, b):
    forward = sp_mult(g, indices, x)
    return sp_solve(entries, indices, -forward)

def grad_sp_solve_x_forward(g, x, entries, indices, b):
    return sp_solve(entries, indices, g)

ag.extend.defjvp(sp_solve, grad_sp_solve_entries_forward, None, grad_sp_solve_x_forward)


if __name__ == '__main__':

    """ For now, this is my test script / function for these primitives. 
        This will get organized and go in its own test file later.
    """

    import ceviche    # use the ceviche wrapper for autograd derivatives
    DECIMAL = 3       # number of decimals to check to

    def make_rand(N):
        # makes a random vector of size N with elements between -0.5 and 0.5
        return np.random.random(N) - 0.5

    def make_rand_complex(N):
        # makes a random complex-valued vector of size N with re and im parts between -0.5 and 0.5
        return make_rand(N) + 1j * make_rand(N)

    def make_rand_indeces(N, M):
        # make M random indeces into an NxN matrix
        return np.random.randint(low=0, high=N, size=(2, M))

    def make_rand_entries_indices(N, M):
        # make M random indeces and corresponding entries
        entries = make_rand_complex(M)
        indices = make_rand_indeces(N, M)
        return entries, indices

    def grad_num(fn, arg, delta=1e-6):
        # take a (complex) numerical derivative of function 'fn' with argument 'arg' with step size 'delta'
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

    N = 5        # size of matrix dimensions.  matrix shape = (N, N)
    M = N**2     # number of non-zeros (make it dense for numerical stability)

    # these are the default values used within the test functions
    indices_const = make_rand_indeces(N, M)
    entries_const = make_rand_complex(M)
    x_const = make_rand_complex(N)
    b_const = make_rand_complex(N)

    def out_fn(output_vector):
        # this function takes the output of each primitive and returns a real scalar (sort of like the objective function)
        return npa.abs(npa.sum(output_vector))

    def fn_mult_entries(entries):
        # sparse matrix multiplication (Ax = b) as a function of matrix entries 'A(entries)'
        b = sp_mult(entries, indices_const, x_const)
        return out_fn(b)

    def fn_mult_x(x):
        # sparse matrix multiplication (Ax = b) as a function of dense vector 'x'
        b = sp_mult(entries_const, indices_const, x)
        return out_fn(b)

    def fn_solve_entries(entries):
        # sparse matrix solve (x = A^{-1}b) as a function of matrix entries 'A(entries)'
        x = sp_solve(entries, indices_const, b_const)
        return out_fn(x)

    def fn_solve_b(b):
        # sparse matrix solve (x = A^{-1}b) as a function of source 'b'
        x = sp_solve(entries_const, indices_const, b)
        return out_fn(x)

    """ REVERSE MODE TESTS """

    ## Testing Gradients of 'Mult Entries Reverse-mode'

    entries = make_rand_complex(M)

    grad = ceviche.jacobian(fn_mult_entries, mode='reverse')(entries)[0]
    grad_true = grad_num(fn_mult_entries, entries)

    np.testing.assert_almost_equal(grad, grad_true, decimal=DECIMAL)

    ## Testing Gradients of 'Mult x Reverse-mode'

    x = make_rand_complex(N)

    grad = ceviche.jacobian(fn_mult_x, mode='reverse')(x)[0]
    grad_true = grad_num(fn_mult_x, x)

    np.testing.assert_almost_equal(grad, grad_true, decimal=DECIMAL)

    ## Testing Gradients of 'Solve x Reverse-mode'

    entries = make_rand_complex(M)

    grad = ceviche.jacobian(fn_solve_entries, mode='reverse')(entries)[0]
    grad_true = grad_num(fn_solve_entries, entries)

    np.testing.assert_almost_equal(grad, grad_true, decimal=DECIMAL)

    ## Testing Gradients of 'Solve x Reverse-mode'

    b = make_rand_complex(N)

    grad = ceviche.jacobian(fn_solve_b, mode='reverse')(b)[0]
    grad_true = grad_num(fn_solve_b, b)

    np.testing.assert_almost_equal(grad, grad_true, decimal=DECIMAL)


    """ FORWARD MODE TESTS """

    ## Testing Gradients of 'Mult entries Forward-mode'

    grad_for = ceviche.jacobian(fn_mult_entries, mode='forward')(entries)[0]
    grad_true = grad_num(fn_mult_entries, entries)

    np.testing.assert_almost_equal(grad_for, grad_true, decimal=DECIMAL)

    ## Testing Gradients of 'Mult x Forward-mode'

    grad_for = ceviche.jacobian(fn_mult_x, mode='forward')(x)[0]
    grad_true = grad_num(fn_mult_x, x)

    np.testing.assert_almost_equal(grad_for, grad_true, decimal=DECIMAL)

    ## Testing Gradients of 'Solve entries Forward-mode'

    grad_for = ceviche.jacobian(fn_solve_entries, mode='forward')(entries)[0]
    grad_true = grad_num(fn_solve_entries, entries)

    np.testing.assert_almost_equal(grad_for, grad_true, decimal=DECIMAL)

    ## Testing Gradients of 'Solve x Forward-mode'

    grad_for = ceviche.jacobian(fn_solve_b, mode='forward')(b)[0]
    grad_true = grad_num(fn_solve_b, b)

    np.testing.assert_almost_equal(grad_for, grad_true, decimal=DECIMAL)
