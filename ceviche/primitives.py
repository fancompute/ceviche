import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from autograd.extend import primitive, defvjp, defjvp
from ceviche.constants import *

""" This file is the meat and bones of the FDFD.
    It defines the basic operations needed for FDFD and also their derivatives
    in a form that autograd can understand.
    This allows you to use fdfd classes in autograd functions.
    Look but don't touch!

    NOTES for the curious (since this information isnt in autograd documentation...)

        To define a function as being trackable by autograd, need to add the 
        @primitive decorator

    REVERSE MODE
        'vjp' defines the vector-jacobian product for reverse mode (adjoint)
        a vjp_maker function takes as arguments
            1. the output of the @primitive
            2. the rest of the original arguments in the @primitive
        and returns
            a *function* of the backprop vector (v) that defines the operation
            (d{function} / d{argument_i})^T @ v

    FORWARD MODE:
        'jvp' defines the jacobian-vector product for forward mode (FMD)
        a jvp_maker function takes as arguments
            1. the forward propagating vector (g)
            2. the rest of the original arguments in the @primitive
        and returns
            (d{function} / d{argument_i}) @ g

    After this, you need to link the @primitive to its vjp/jvp using
    defvjp(function, arg1's vjp, arg2's vjp, ...)
    defjvp(function, arg1's jvp, arg2's jvp, ...)
"""


"""========================= SPARSE DOT PRODUCT ==========================="""

@primitive
def spdot(A, x):
    """ Dot product of sparse matrix A and dense matrix x (Ax = b) """
    return A.dot(x)

def vjp_maker_spdot(b, A, x):
    """ Gives vjp for b = spdot(A, x) w.r.t. x"""
    def vjp(v):
        return spdot(A.T, v)
    return vjp

def jvp_spdot(g, b, A, x):
    """ Gives jvp for b = spdot(A, x) w.r.t. x"""
    return spdot(A, g)

defvjp(spdot, None, vjp_maker_spdot)
defjvp(spdot, None, jvp_spdot)

"""===================== SPARSE LINEAR SYSTEM SOLVE ======================="""
# note: this isn't used in the implementation of ceviche.  Just here for later.

@primitive
def spsolve(A, b):
    """ Solve Ax = b for x, where A is sparse matrix, x and b are dense matrices"""
    return spl.spsolve(A, b)

def vjp_maker_spsolve_A(x, A, b):
    """ Gives vjp for x = spsolve(A, b) w.r.t. A """
    def vjp(v):
        x_aj = spsolve(A.T, -v)
        return np.outer(x, x_aj)
    return vjp

def vjp_maker_spsolve_b(x, A, b):
    """ Gives vjp for x = spsolve(A, b) w.r.t. b """
    def vjp(v):
        return spsolve(A.T, v)
    return vjp

def jvp_spsolve_A(g, x, A, b):
    """ Gives vjp for x = spsolve(A, b) w.r.t. A """
    return spsolve(A, x * -g)

def jvp_spsolve_b(g, x, A, b):
    """ Gives vjp for x = spsolve(A, b) w.r.t. b """
    return spsolve(A, g)

defvjp(spsolve, vjp_maker_spsolve_A, vjp_maker_spsolve_b)
defjvp(spsolve, jvp_spsolve_A, jvp_spsolve_b)

"""======================== SYSTEM MATRIX CREATION ========================"""

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

"""========================== FIELD CONVERSIONS ==========================="""

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
    # note: adjoint switch is because backprop thru this fn. has different form
    if adjoint:
        Ex =  spdot(info_dict['Dyf'].T, Hz) / eps_arr / EPSILON_0
    else:
        Ex = -spdot(info_dict['Dyb'],   Hz) / eps_arr / EPSILON_0
    return Ex

def Hz_to_Ey(Hz, info_dict, eps_arr, adjoint=False):
    """ Returns electric field `Ey` from magnetic field `Hz` """
    if adjoint:
        Ey = -spdot(info_dict['Dxf'].T, Hz) / eps_arr / EPSILON_0
    else:
        Ey =  spdot(info_dict['Dxb'],   Hz) / eps_arr / EPSILON_0
    return Ey

def H_to_E(Hz, info_dict, eps_arr, adjoint=False):
    """ More convenient function to return both Ex and Ey from Hz """
    Ex = Hz_to_Ex(Hz, info_dict, eps_arr, adjoint=adjoint)
    Ey = Hz_to_Ey(Hz, info_dict, eps_arr, adjoint=adjoint)
    return Ex, Ey

"""======================== SOLVING FOR THE FIELDS ========================"""

@primitive
def solve_Ez(info_dict, eps_arr, source):
    """ solve `Ez = A^-1 b` where A is constructed from the FDFD `info_dict`
        and 'eps_arr' is a (1D) array of the relative permittivity
    """
    A = make_A_Ez(info_dict, eps_arr)
    b = 1j * info_dict['omega'] * source
    Ez = spl.spsolve(A, b)
    return Ez

# define the gradient of solve_Ez w.r.t. eps_arr (in Ez)
def vjp_maker_solve_Ez(Ez, info_dict, eps_arr, source):
    """ Gives vjp for solve_Ez with respect to eps_arr """    
    # construct the system matrix again
    A = make_A_Ez(info_dict, eps_arr)
    # vector-jacobian product function to return
    def vjp(v):
        # solve the adjoint problem and get those electric fields (note D info_dict are different and transposed)
        Ez_aj = spl.spsolve(A.T, -v)
        # because we care about the diagonal elements, just element-wise multiply E and E_adj
        # note: need np.real() for adjoint returns w.r.t. real quantities but not in forward mode
        return EPSILON_0 * info_dict['omega']**2 * np.real(Ez_aj * Ez)
    return vjp

def vjp_maker_solve_Ez_source(Ez, info_dict, eps_arr, source):
    """ Gives vjp for solve_Ez with respect to source """    
    A = make_A_Ez(info_dict, eps_arr)
    def vjp(v):
        return 1j * info_dict['omega'] * spl.spsolve(A.T, v)
    return vjp

# define the gradient of solve_Ez w.r.t. eps_arr (in Ez)
def jvp_solve_Ez(g, Ez, info_dict, eps_arr, source):
    """ Gives jvp for solve_Ez with respect to eps_arr """    
    # construct the system matrix again and the RHS of the gradient expersion
    A = make_A_Ez(info_dict, eps_arr)
    u = Ez * -g
    # solve the adjoint problem and get those electric fields (note D info_dict are different and transposed)
    Ez_for = spl.spsolve(A, u)
    # because we care about the diagonal elements, just element-wise multiply E and E_adj
    return EPSILON_0 * info_dict['omega']**2 * Ez_for

def jvp_solve_Ez_source(g, Ez, info_dict, eps_arr, source):
    """ Gives jvp for solve_Ez with respect to source """  
    A = make_A_Ez(info_dict, eps_arr)      
    return 1j * info_dict['omega'] * spsolve(A, g)

defvjp(solve_Ez, None, vjp_maker_solve_Ez, vjp_maker_solve_Ez_source)
defjvp(solve_Ez, None, jvp_solve_Ez, jvp_solve_Ez_source)

@primitive
def solve_Hz(info_dict, eps_arr, source):
    """ solve `Hz = A^-1 b` where A is constructed from the FDFD `info_dict`
        and 'eps_arr' is a (1D) array of the relative permittivity
    """
    A = make_A_Hz(info_dict, eps_arr)
    b = 1j * info_dict['omega'] * source    
    Hz = spl.spsolve(A, b)
    return Hz

def vjp_maker_solve_Hz(Hz, info_dict, eps_arr, source):
    """ Gives vjp for solve_Hz with respect to eps_arr """    
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

def vjp_maker_solve_Hz_source(Hz, info_dict, eps_arr, source):
    """ Gives vjp for solve_Hz with respect to source """    
    A = make_A_Hz(info_dict, eps_arr)
    def vjp(v):
        return 1j * info_dict['omega'] * spl.spsolve(A.T, v)
    return vjp

# define the gradient of solve_Hz w.r.t. eps_arr (in Hz)
def jvp_solve_Hz(g, Hz, info_dict, eps_arr, source):
    """ Gives jvp for solve_Hz with respect to eps_arr """    
    # construct the system matrix again and the RHS of the gradient expersion
    A = make_A_Hz(info_dict, eps_arr)
    ux = spdot(info_dict['Dxb'], Hz)
    uy = spdot(info_dict['Dyb'], Hz)
    diag = sp.spdiags(1 / eps_arr, [0], eps_arr.size, eps_arr.size)
    # the g gets multiplied in at the middle of the expression
    ux = ux * diag * g * diag
    uy = uy * diag * g * diag
    ux = spdot(info_dict['Dxf'], ux)
    uy = spdot(info_dict['Dyf'], uy)
    # add the x and y components and multiply by A_inv on the left
    u = (ux + uy)
    Hz_for = spl.spsolve(A, u)
    return 1 / EPSILON_0 * Hz_for

def jvp_solve_Hz_source(g, Hz, info_dict, eps_arr, source):
    """ Gives jvp for solve_Hz with respect to source """    
    A = make_A_Hz(info_dict, eps_arr)      
    return 1j * info_dict['omega'] * spsolve(A, g)

defvjp(solve_Hz, None, vjp_maker_solve_Hz, vjp_maker_solve_Hz_source)
defjvp(solve_Hz, None, jvp_solve_Hz, jvp_solve_Hz_source)
