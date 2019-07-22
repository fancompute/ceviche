import autograd.numpy as np
import scipy.sparse.linalg as spl
import scipy.optimize as opt
import logging

import numpy as npo    #numpy original

from numpy.linalg import norm
from .utils import spdot

# try to import MKL but just use scipy sparse solve if not
try:
    from pyMKL import pardisoSolver
    using_mkl = True
except:
    using_mkl = False

# for reference https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html


""" ========================== MASTER FUNCTIONS ========================== """
# when importing solvers in other packages use this function


def sparse_solve(A, b, iterative=False, nonlinear=False, method='bicg'):
    """ Solve sparse linear system Ax=b for x.
        if iterative=True, can choose method using `method` kwarg.
    """
    if nonlinear:
        return _solve_nonlinear(A, b, iterative=iterative, method=method)
    else:
        return _solve_linear(A, b, iterative=iterative, method=method)

""" ========================== SOLVER FUNCTIONS ========================== """

def _solve_linear(A, b, iterative=False, method='bicg'):
    if iterative:
        return _solve_iterative(A, b, method=method)
    else:
        return _solve_direct(A, b)

def _relative_residual(A, x, b):
    """ computes relative residual: ||Ax - b|| / ||b|| """
    res = norm(A.dot(x) - b)
    return res / norm(b)

def _solve_nonlinear(A, b, iterative=False, method='bicg', verbose=False, atol=1e-10, max_iters=100):
    """ Solve Ax=b for x where A is a function of x using direct substitution """

    vec_0 = np.zeros(b.shape)   # no field
    A_0 = A(vec_0)              # system matrix with no field
    x_i = sparse_solve(A_0, b, iterative=iterative, method=method)  # linear field

    for i in range(max_iters):

        A_i = A(x_i)
        rel_res = _relative_residual(A_i, x_i, b)

        if verbose:
            print('i = {}, relative residual = {}'.format(i, rel_res))

        if rel_res < atol:
            break
        
        x_i = sparse_solve(A_i, b, iterative=iterative, method=method)

    return x_i


def _solve_direct(A, b):
    """ Direct solver """
    if using_mkl:
        pSolve = pardisoSolver(A, mtype=13)
        pSolve.factor()
        x = pSolve.solve(b)
        pSolve.clear()
        return x
    else:
        return spl.spsolve(A, b)

# dict of iterative methods supported (name: function)
ITERATIVE_METHODS = {
    'bicg': spl.bicg,
    'bicgstab': spl.bicgstab,
    'cg': spl.cg,
    'cgs': spl.cgs,
    'gmres': spl.gmres,
    'lgmres': spl.lgmres,
    # 'minres': spl.minres,  # requires a symmetric matrix
    'qmr': spl.qmr,
    'gcrotmk': spl.gcrotmk
}

ATOL = 1e-8

def _solve_iterative(A, b, method='bicg'):
    """ Iterative solver """
    try:
        solver_fn = ITERATIVE_METHODS[method]
    except:
        raise ValueError("iterative method {} not found.\n supported methods are:\n {}".format(method, ITERATIVE_METHODS))

    x, info = solver_fn(A, b, atol=ATOL)

    if info > 0:
        raise ValueError("tried {} iterations and did not converge".format(info))
    elif info < 0:
        raise ValueError("iterative solver threw error")

    return x

""" ============================ SPEED TESTS ============================= """

# to run speed tests use `python -W ignore ceviche/solvers.py` to suppress warnings

if __name__ == '__main__':

    from scipy.sparse import csr_matrix
    from scipy.sparse import random as random_sp
    from numpy.random import random as random
    import numpy as np
    from time import time

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

    for method in ITERATIVE_METHODS.keys():
        t0 = time()
        x = _solve_iterative(A, b, method=method)
        t1 = time()
        print('\titerative solver ({}):\n\t\ttook {} seconds'.format(method, t1 - t0))

    print('\n')

    print('WITH FDFD MATRICES:\n')

    m, n = 400, 100
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
    A = F.A

    # DIRECT SOLVE
    t0 = time()
    x = _solve_direct(A, b)
    t1 = time()
    print('\tdirect solver:\n\t\ttook {} seconds\n'.format(t1 - t0))

    # ITERATIVE SOLVES

    for method in ITERATIVE_METHODS.keys():
        t0 = time()
        x = _solve_iterative(A, b, method=method)
        t1 = time()
        print('\titerative solver ({}):\n\t\ttook {} seconds'.format(method, t1 - t0))
