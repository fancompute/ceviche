import scipy.sparse.linalg as spl

# for reference https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

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
    'minres': spl.minres,
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
    from time import time

    N = 100       # dimension of the x, and b vectors
    density = 0.3  # sparsity of the dense matrix
    A = csr_matrix(random_sp(N, N, density=density))
    b = random((N,))

    print('for N = {} and density = {}'.format(N, density))

    # DIRECT SOLVE
    t0 = time()
    x = solve_direct(A, b)
    t1 = time()
    print('direct solver:\n\ttook {} seconds'.format(t1 - t0))

    # ITERATIVE SOLVES

    for method in ITERATIVE_METHODS.keys():
        t0 = time()
        x = solve_iterative(A, b, method=method)
        t1 = time()
        print('iterative ({}) solver:\n\ttook {} seconds'.format(method, t1 - t0))



