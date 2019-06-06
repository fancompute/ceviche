import scipy.sparse.linalg as spl

SOLVER_MAP = {
    'direct': solve_direct,
    'nonlinear': solve_nonlinear,
    'iterative': solve_direct
}

def solve_direct(A, b):
    return spl.spsolve(A, b)

def solve_nonlinear(A, b):
    """ Where A is a function of x """
    raise NotImplementedError("Implement this")

def solve_iterative(A, b):
    raise NotImplementedError("Implement this")
