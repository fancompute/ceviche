import numpy as np
import scipy.sparse as sp
import copy

""" Just some utilities for easier testing and debugging"""

def make_sparse(N, random=True, density=1):
    """ Makes a sparse NxN matrix. """
    if not random:
        np.random.seed(0)
    D = sp.random(N, N, density=density) + 1j * sp.random(N, N, density=density)
    return D

def grad_num(fn, arg, step_size=1e-7):
    """ numerically differentiate `fn` w.r.t. its argument `arg` """
    N = arg.size
    gradient = np.zeros((N,))
    f_old = fn(arg)
    for i in range(N):
        arg_new = copy.copy(arg)
        arg_new[i] += step_size
        f_new_i = fn(arg_new)
        gradient[i] = (f_new_i - f_old) / step_size
    return gradient
