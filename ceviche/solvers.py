import scipy.sparse.linalg as spl

# for reference https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
# run this file with `python -W ignore ceviche/solvers.py` to suppress warnings

def solve_direct(A, b):
    """ Solve Ax=b for x using a direct solver """
    return spl.spsolve(A, b)

def solve_nonlinear(A, b):
    """ Solve Ax=b for x where A is a function of x """
    raise NotImplementedError("Implement this")


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

def solve_iterative(A, b, method='bicg'):
    """ Solve Ax=b for x using an iterative solver """

    try:
        solver_fn = ITERATIVE_METHODS[method]
    except:
        raise ValueError("iterative method {} not found.\n supported methods are:\n {}".format(method, ITERATIVE_METHODS))

    return solver_fn(A, b)

# when importing solvers in other packages, this are the name: function dictionary
SOLVER_MAP = {
    'direct': solve_direct,
    'nonlinear': solve_nonlinear,
    'iterative': solve_direct
}

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
    x = solve_direct(A, b)
    t1 = time()
    print('\tdirect solver:\n\ttook {} seconds\n'.format(t1 - t0))

    # ITERATIVE SOLVES

    for method in ITERATIVE_METHODS.keys():
        t0 = time()
        x = solve_iterative(A, b, method=method)
        t1 = time()
        print('\t\titerative ({}) solver:\n\ttook {} seconds'.format(method, t1 - t0))

    print('\n')

    print('WITH FDFD MATRICES:\n')

    m, n = 100, 100
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
    x = solve_direct(A, b)
    t1 = time()
    print('\tdirect solver:\n\ttook {} seconds\n'.format(t1 - t0))

    # ITERATIVE SOLVES

    for method in ITERATIVE_METHODS.keys():
        t0 = time()
        x = solve_iterative(A, b, method=method)
        t1 = time()
        print('\t\titerative ({}) solver:\n\ttook {} seconds'.format(method, t1 - t0))
