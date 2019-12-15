import numpy as np
import scipy.sparse.linalg as spl


""" This file stores the various sparse linear system solvers you can use for FDFD """

# try to import MKL but just use scipy sparse solve if not
try:
    from pyMKL import pardisoSolver
    HAS_MKL = True
    # print('using MKL for direct solvers')
except:
    HAS_MKL = False
    # print('using scipy.sparse for direct solvers.  Note: using MKL will make things significantly faster.')

# default iterative method to use
# for reference on the methods available, see:  https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
DEFAULT_ITERATIVE_METHOD = 'bicg'

# dict of iterative methods supported (name: function)
ITERATIVE_METHODS = {
    'bicg': spl.bicg,
    'bicgstab': spl.bicgstab,
    'cg': spl.cg,
    'cgs': spl.cgs,
    'gmres': spl.gmres,
    'lgmres': spl.lgmres,
    'qmr': spl.qmr,
    'gcrotmk': spl.gcrotmk
}

# convergence tolerance for iterative solvers.
ATOL = 1e-8

""" ========================== SOLVER FUNCTIONS ========================== """

def solve_linear(A, b, iterative_method=False):
    """ Master function to call the others """

    if iterative_method and iterative_method is not None:
        # if iterative solver string is supplied, use that method
        return _solve_iterative(A, b, iterative_method=iterative_method)
    elif iterative_method and iterative_method is None:
        # if iterative_method is supplied as None, use the default
        return _solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD)
    else:
        # otherwise, use a direct solver
        return _solve_direct(A, b)

def _solve_direct(A, b):
    """ Direct solver """

    if HAS_MKL:
        # prefered method using MKL. Much faster (on Mac at least)
        pSolve = pardisoSolver(A, mtype=13)
        pSolve.factor()
        x = pSolve.solve(b)
        pSolve.clear()
        return x
    else:
        # scipy solver.
        return spl.spsolve(A, b)

def _solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD):
    """ Iterative solver """

    # error checking on the method name (https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
    try:
        solver_fn = ITERATIVE_METHODS[iterative_method]
    except:
        raise ValueError("iterative method {} not found.\n supported methods are:\n {}".format(iterative_method, ITERATIVE_METHODS))

    # call the solver using scipy's API
    x, info = solver_fn(A, b, atol=ATOL)
    return x

def _solve_cuda(A, b, **kwargs):
    """ You could put some other solver here if you're feeling adventurous """
    raise NotImplementedError("Please implement something fast and exciting here!")


""" ============================ SPEED TESTS ============================= """

# to run speed tests use `python -W ignore ceviche/solvers.py` to suppress warnings

if __name__ == '__main__':

    from scipy.sparse import csr_matrix
    from scipy.sparse import random as random_sp
    from numpy.random import random as random
    import numpy as np
    from time import time
    import ceviche

    N = 200       # dimension of the x, and b vectors
    density = 0.3  # sparsity of the dense matrix
    A = csr_matrix(random_sp(N, N, density=density))
    b = np.random.random((N, 1)) - 0.5

    print('\nWITH RANDOM MATRICES:\n')
    print('\tfor N = {} and density = {}\n'.format(N, density))

    # DIRECT SOLVE
    t0 = time()
    x = _solve_direct(A, b)
    t1 = time()
    print('\tdirect solver:\n\t\ttook {} seconds\n'.format(t1 - t0))

    # ITERATIVE SOLVES

    for iterative_method in ITERATIVE_METHODS.keys():
        t0 = time()
        x = _solve_iterative(A, b, iterative_method=iterative_method)
        t1 = time()
        print('\titerative solver ({}):\n\t\ttook {} seconds'.format(iterative_method, t1 - t0))

    print('\n')

    print('WITH FDFD MATRICES:\n')

    m, n = 200, 100
    print('\tfor dimensions = {}\n'.format((m, n)))
    eps_r = np.random.random((m, n)) + 1
    b = np.random.random((m * n, )) - 0.5

    import sys
    sys.path.append('../ceviche')
    from ceviche.fdfd import fdfd_ez as fdfd
    from ceviche.constants import *

    npml = 10
    dl = 2e-8
    lambda0 = 1550e-9
    omega0 = 2 * np.pi * C_0 / lambda0

    F = fdfd(omega0, dl, eps_r, [10, 0])
    entries_a, indices_a = F.make_A(eps_r.flatten())
    A = ceviche.primitives.make_sparse(entries_a, indices_a, m*n)

    # DIRECT SOLVE
    t0 = time()
    x = _solve_direct(A, b)
    t1 = time()
    print('\tdirect solver:\n\t\ttook {} seconds\n'.format(t1 - t0))

    # ITERATIVE SOLVES

    for iterative_method in ITERATIVE_METHODS.keys():
        t0 = time()
        x = _solve_iterative(A, b, iterative_method=iterative_method)
        t1 = time()
        print('\titerative solver ({}):\n\t\ttook {} seconds'.format(iterative_method, t1 - t0))
