import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from autograd.extend import primitive, defvjp
from ceviche.constants import *

""" This file is the meat and bones of the FDFD.
    It defines the basic operations needed for FDFD and also their derivatives
    in a form that autograd can understand.
    This allows you to use fdfd classes in autograd functions.
    Look but don't touch!
"""

"""========================= SYSTEM MATRIX CREATION ========================"""

def make_A_Hz(info_dict, eps_arr):
    """ constructs the system matrix for `Hz` polarization """

    diag = 1 / EPSILON_0 * sp.spdiags(1/eps_arr, [0], eps_arr.size, eps_arr.size)
    A = info_dict['Dxf'].dot(diag).dot(info_dict['Dxb']) \
      + info_dict['Dyf'].dot(diag).dot(info_dict['Dyb']) \
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

@primitive
def Ez_to_Hx(Ez, info_dict):
    """ Returns magnetic field `Hx` from electric field `Ez` """
    Hx = - info_dict['Dyb'].dot(Ez) / MU_0
    return Hx

def vjp_maker_Ez_to_Hx(Ez, Hx, info_dict):
    """ Gives vjp for dHx/dEz """
    def vjp(v):
        return -(info_dict['Dyb'].T).dot(v) / MU_0
    return vjp

@primitive
def Ez_to_Hy(Ez, info_dict):
    """ Returns magnetic field `Hy` from electric field `Ez` """
    Hy =  info_dict['Dxb'].dot(Ez) / MU_0
    return Hy

def vjp_maker_Ez_to_Hy(Hy, Ez, info_dict):
    """ Gives vjp for dHy/dEz """
    def vjp(v):
        return (info_dict['Dxb'].T).dot(v) / MU_0
    return vjp

def E_to_H(Ez, info_dict):
    """ More convenient function to return both Hx and Hy from Ez """
    Hx = Ez_to_Hx(Ez, info_dict)
    Hy = Ez_to_Hy(Ez, info_dict)
    return Hx, Hy

@primitive
def Hz_to_Ex(Hz, info_dict, eps_arr, adjoint=False):
    """ Returns electric field `Ex` from magnetic field `Hz` """
    if adjoint:
        Ex = (info_dict['Dyf'].T).dot(Hz) / eps_arr / EPSILON_0
    else:
        Ex = -info_dict['Dyb'].dot(Hz) / eps_arr / EPSILON_0
    return Ex

def vjp_maker_Hz_to_Ex_Hz(Ex, Hz, info_dict, eps_arr, adjoint=False):
    """ Gives vjp for dEx/dHz """
    def vjp(v):
        return -(info_dict['Dyb'].T).dot(v / eps_arr / EPSILON_0)
    return vjp

def vjp_maker_Hz_to_Ex_eps_arr(Ex, Hz, info_dict, eps_arr, adjoint=False):
    """ Gives vjp for dEx/deps_arr """
    def vjp(v):
        return np.real(-v * Ex / eps_arr)
    return vjp

@primitive
def Hz_to_Ey(Hz, info_dict, eps_arr, adjoint=False):
    """ Returns electric field `Ey` from magnetic field `Hz` """
    if adjoint:
        Ey = -(info_dict['Dxf'].T).dot(Hz) / eps_arr / EPSILON_0
    else:
        Ey = info_dict['Dxb'].dot(Hz) / eps_arr / EPSILON_0
    return Ey

def vjp_maker_Hz_to_Ey_Hz(Ey, Hz, info_dict, eps_arr, adjoint=False):
    """ Gives vjp for dEy/dHz """
    def vjp(v):
        return (info_dict['Dxb'].T).dot(v / eps_arr / EPSILON_0)
    return vjp

def vjp_maker_Hz_to_Ey_eps_arr(Ey, Hz, info_dict, eps_arr, adjoint=False):
    """ Gives vjp for dEy/deps_arr """
    def vjp(v):
        return np.real(-v * Ey / eps_arr)
    return vjp

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

"""=================== LINKING PRIMITIVIES TO DERIVATIVES =================="""

def link_vjps():
    """ This links the vjp_maker functions to their primitives """
    defvjp(solve_Ez, None, vjp_maker_solve_Ez, vjp_maker_solve_Ez_source)
    defvjp(Ez_to_Hx, vjp_maker_Ez_to_Hx, vjp_maker_Ez_to_Hx, None)
    defvjp(Ez_to_Hy, vjp_maker_Ez_to_Hy, vjp_maker_Ez_to_Hy, None)
    defvjp(solve_Hz, None, vjp_maker_solve_Hz, vjp_maker_solve_Hz_source)
    defvjp(Hz_to_Ex, vjp_maker_Hz_to_Ex_Hz, None, vjp_maker_Hz_to_Ex_eps_arr, None)
    defvjp(Hz_to_Ey, vjp_maker_Hz_to_Ey_Hz, None, vjp_maker_Hz_to_Ey_eps_arr, None)

# run this function on import ^
link_vjps()
