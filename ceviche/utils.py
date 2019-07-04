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


def vjp_maker_num(fn, arg_inds, steps):
    """ Makes a vjp_maker for the numerical derivative of a function `fn`
    w.r.t. argument at position `arg_ind` using step sizes `steps` """

    def vjp_single_arg(ia):
        arg_ind = arg_inds[ia]
        step = steps[ia]

        def vjp_maker(fn_out, *args):
            shape = args[arg_ind].shape
            num_p = args[arg_ind].size
            step = steps[ia]

            def vjp(v):

                vjp_num = np.zeros(num_p)
                for ip in range(num_p):
                    args_new = list(args)
                    args_rav = args[arg_ind].flatten()
                    args_rav[ip] += step
                    args_new[arg_ind] = args_rav.reshape(shape)
                    dfn_darg = (fn(*args_new) - fn_out)/step
                    vjp_num[ip] = np.sum(v * dfn_darg)

                return vjp_num

            return vjp

        return vjp_maker

    vjp_makers = []
    for ia in range(len(arg_inds)):
        vjp_makers.append(vjp_single_arg(ia=ia))

    return tuple(vjp_makers)