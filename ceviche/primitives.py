import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from autograd.extend import primitive, defvjp
from ceviche.constants import *
from ceviche.utils import circ2eps

""" This file is the meat and bones of the FDFD.
    It defines the basic operations needed for FDFD and also their derivatives
    in a form that autograd can understand.
    This allows you to use fdfd classes in autograd functions.
    Look but don't touch!
"""


"""============================ BASIC PRIMITIVES ==========================="""

@primitive
def spdot(A, x):
    """ Dot product of sparse matrix A and dense matrix x (Ax = b) """
    return A.dot(x)

def vjp_maker_spdot(b, A, x):
    """ Gives vjp for b = spdot(A, x) """
    def vjp(v):
        return spdot(A.T, v)
    return vjp

defvjp(spdot, None, vjp_maker_spdot)


@primitive
def spsolve(A, b):
    """ Solve Ax = b for x, where A is sparse matrix, x and b are dense matrices"""
    return spl.spsolve(A, b)

def vjp_maker_spsolve_A(x, A, b):
    """ Gives vjp for x = spsolve(A, b) """
    def vjp(v):
        x_aj = spsolve(A.T, -v)
        return np.outer(x, x_aj)
    return vjp

def vjp_maker_spsolve_b(x, A, b):
    """ Gives vjp for x = spsolve(A, b) """
    def vjp(v):
        return spsolve(A.T, v)
    return vjp

defvjp(spsolve, vjp_maker_spsolve_A, vjp_maker_spsolve_b)


"""========================= SYSTEM MATRIX CREATION ========================"""

def make_A_Hz(info_dict, eps_arr):
    """ constructs the system matrix for `Hz` polarization """

    diag = 1 / EPSILON_0 * sp.spdiags(1/eps_arr, [0], eps_arr.size, eps_arr.size)
    A = spdot(info_dict['Dxf'], spdot(info_dict['Dxb'].T, diag).T) \
      + spdot(info_dict['Dyf'], spdot(info_dict['Dyb'].T, diag).T) \
      + info_dict['omega']**2 * MU_0 * sp.eye(eps_arr.size)
    return A

def make_A_Ez(info_dict, eps_arr):
    """ constructs the system matrix for `Ez` polarization """

    diag = EPSILON_0 * sp.spdiags(eps_arr, [0], eps_arr.size, eps_arr.size)
    A = 1 / MU_0 * info_dict['Dxf'].dot(info_dict['Dxb']) \
      + 1 / MU_0 * info_dict['Dyf'].dot(info_dict['Dyb']) \
      + info_dict['omega']**2 * diag
    return A

"""====================== FIELD CONVERSION PRIMITIVIES ====================="""

# @primitive
def Ez_to_Hx(Ez, info_dict):
    """ Returns magnetic field `Hx` from electric field `Ez` """
    Hx = - spdot(info_dict['Dyb'], Ez) / MU_0
    return Hx

def Ez_to_Hy(Ez, info_dict):
    """ Returns magnetic field `Hy` from electric field `Ez` """
    Hy =  spdot(info_dict['Dxb'], Ez) / MU_0
    return Hy

def E_to_H(Ez, info_dict):
    """ More convenient function to return both Hx and Hy from Ez """
    Hx = Ez_to_Hx(Ez, info_dict)
    Hy = Ez_to_Hy(Ez, info_dict)
    return Hx, Hy

def Hz_to_Ex(Hz, info_dict, eps_arr, adjoint=False):
    """ Returns electric field `Ex` from magnetic field `Hz` """
    if adjoint:
        Ex =  spdot(info_dict['Dyf'].T, Hz) / eps_arr / EPSILON_0
    else:
        Ex = -spdot(info_dict['Dyb'], Hz) / eps_arr / EPSILON_0
    return Ex

def Hz_to_Ey(Hz, info_dict, eps_arr, adjoint=False):
    """ Returns electric field `Ey` from magnetic field `Hz` """
    if adjoint:
        Ey = -spdot(info_dict['Dxf'].T, Hz) / eps_arr / EPSILON_0
    else:
        Ey =  spdot(info_dict['Dxb'], Hz) / eps_arr / EPSILON_0
    return Ey

def H_to_E(Hz, info_dict, eps_arr, adjoint=False):
    """ More convenient function to return both Ex and Ey from Hz """
    Ex = Hz_to_Ex(Hz, info_dict, eps_arr, adjoint=adjoint)
    Ey = Hz_to_Ey(Hz, info_dict, eps_arr, adjoint=adjoint)
    return Ex, Ey

"""========================= FIELD SOLVE PRIMITIVES ========================"""

@primitive
def solve_Ez(info_dict, eps_arr, source):
    """ solve `Ez = A^-1 b` where A is constructed from the FDFD `info_dict`
        and 'eps_arr' is a (1D) array of the relative permittivity
    """
    A = make_A_Ez(info_dict, eps_arr)
    b = 1j * info_dict['omega'] * source
    Ez = spl.spsolve(A, b)
    return Ez

# define the gradient of solve_Ez w.r.t. eps_arr (in Hz)
def vjp_maker_solve_Ez(Ez, info_dict, eps_arr, source):
    """ Returns a function of the error signal (v) that computes the vector-jacobian product.
          takes in the output of solve_Ez (Hz) and solve_Ez's other arguments. 
    """
    
    # construct the system matrix again
    A = make_A_Ez(info_dict, eps_arr)

    # vector-jacobian product function to return
    def vjp(v):

        # solve the adjoint problem and get those electric fields (note D info_dict are different and transposed)
        Ez_aj = spl.spsolve(A.T, -v)

        # because we care about the diagonal elements, just element-wise multiply E and E_adj
        return EPSILON_0 * info_dict['omega']**2 * np.real(Ez_aj * Ez)

    # return this function for autograd to link-later
    return vjp

def vjp_maker_solve_Ez_source(Ez, info_dict, eps_arr, source):
    """ Gives vjp for solve_Ez with respect to source """    

    A = make_A_Ez(info_dict, eps_arr)

    def vjp(v):
        return 1j * info_dict['omega'] * spl.spsolve(A.T, v)

    return vjp


defvjp(solve_Ez, None, vjp_maker_solve_Ez, vjp_maker_solve_Ez_source)


@primitive
def solve_Hz(info_dict, eps_arr, source):
    """ solve `Hz = A^-1 b` where A is constructed from the FDFD `info_dict`
        and 'eps_arr' is a (1D) array of the relative permittivity
    """
    A = make_A_Hz(info_dict, eps_arr)
    b = 1j * info_dict['omega'] * source    
    Hz = spl.spsolve(A, b)
    return Hz

# define the gradient of solve_Hz w.r.t. eps_arr (in Hz)
def vjp_maker_solve_Hz(Hz, info_dict, eps_arr, source):
    """ Returns a function of the error signal (v) that computes the vector-jacobian product.
          takes in the output of solve_Hz (Hz) and solve_Hz's other arguments. 
    """

    # get the forward electric fields
    Ex, Ey = H_to_E(Hz, info_dict, eps_arr, adjoint=False)

    # construct the system matrix again
    A = make_A_Hz(info_dict, eps_arr)

    # vector-jacobian product function to return
    def vjp(v):

        # solve the adjoint problem and get those electric fields (note D info_dict are different and transposed)
        Hz_aj = spl.spsolve(A.T, -v)
        Ex_aj, Ey_aj = H_to_E(Hz_aj, info_dict, eps_arr, adjoint=True)

        # because we care about the diagonal elements, just element-wise multiply E and E_adj
        return EPSILON_0 * np.real(Ex_aj * Ex + Ey_aj * Ey)

    # return this function for autograd to link-later
    return vjp

def vjp_maker_solve_Hz_source(Hz, info_dict, eps_arr, b):
    """ Gives vjp for solve_Hz with respect to source """    

    A = make_A_Hz(info_dict, eps_arr)

    def vjp(v):
        return 1j * info_dict['omega'] * spl.spsolve(A.T, v)

    return vjp

defvjp(solve_Hz, None, vjp_maker_solve_Hz, vjp_maker_solve_Hz_source)


"""============================ SHAPE PARAMETRIZATION ==========================="""

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