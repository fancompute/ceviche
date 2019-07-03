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
    """ numerically differentiate `fn` w.r.t. its argument `arg` 
    `arg` can be a numpy array of arbitrary shape
    `step_size` can be a number or an array of the same shape as `arg` """

    N = arg.size
    shape = arg.shape
    gradient = np.zeros((N,))
    f_old = fn(arg)

    if type(step_size) == float:
        step = step_size*np.ones((N))
    else:
        step = step_size.ravel()

    for i in range(N):
        arg_new = copy.copy(arg.ravel())
        arg_new[i] += step[i]
        f_new_i = fn(arg_new.reshape(shape))
        gradient[i] = (f_new_i - f_old) / step[i]

    return gradient.reshape(shape)

def circ2eps(x, y, r, eps_c, eps_b, dL):
    """ Define eps_r through circle parameters """
    shape = eps_b.shape   # shape of domain (in num. grids)
    Nx, Ny = (shape[0], shape[1])

    # x and y coordinate arrays
    x_coord = np.linspace(-Nx/2*dL, Nx/2*dL, Nx)
    y_coord = np.linspace(-Ny/2*dL, Ny/2*dL, Ny)

    # x and y mesh
    xs, ys = np.meshgrid(x_coord, y_coord, indexing='ij')

    eps_r = copy.copy(eps_b)
    for ih in range(x.shape[0]):
        mask = (xs - x[ih])**2 + (ys - y[ih])**2 < r[ih]**2
        eps_r[mask] = eps_c[ih]

    return eps_r


def grid_coords(array, dL):
    # Takes an array and returns the coordinates of the x and y points

    shape = Nx, Ny = array.shape   # shape of domain (in num. grids)

    # x and y coordinate arrays
    x_coord = np.linspace(-Nx/2*dL, Nx/2*dL, Nx)
    y_coord = np.linspace(-Ny/2*dL, Ny/2*dL, Ny)

    # x and y mesh
    xs, ys = np.meshgrid(x_coord, y_coord, indexing='ij')

    return xs, ys