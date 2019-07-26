import autograd.numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import copy
import warnings

from autograd.extend import primitive, defvjp, defjvp

from .constants import *
from .solvers import sparse_solve, DEFAULT_SOLVER
from .utils import spdot, block_4
from .jacobians import jacobian

class fdfd():
    """ Base class for FDFD simulation """

    def __init__(self, omega, dL, eps_r, npml):
        """ initialize with a given structure and source """

        self.omega = omega
        self.dL = dL
        self.npml = npml

        self.info_dict = {'omega': self.omega}

        self.eps_r = eps_r
        self.setup_derivatives()
        self.A = self.make_A(self.eps_vec)

    def setup_derivatives(self):

        # Creates all of the operators needed for later
        info_dict = compute_derivative_matrices(self.omega, self.shape, self.npml, self.dL)
        self.Dxf, self.Dxb, self.Dyf, self.Dyb = info_dict

        # save to a dictionary for convenience passing to primitives
        self.info_dict['Dxf'] = self.Dxf
        self.info_dict['Dxb'] = self.Dxb
        self.info_dict['Dyf'] = self.Dyf
        self.info_dict['Dyb'] = self.Dyb

    @staticmethod
    def get_shape(arr_or_fn):
        """ Returns the shape of arr_or_fn (which can be array or function returning array) """
        if callable(arr_or_fn):
            return arr_or_fn(0).shape
        else:
            return arr_or_fn.shape

    @property
    def eps_r(self):
        """ Returns the relative permittivity grid """
        return self._eps_r

    @eps_r.setter
    def eps_r(self, new_eps):
        raise NotImplementedError("need to implement setter method for eps_r")

    def make_A(self, eps_r):
        raise NotImplementedError("need to make a make_A() method")

    def solve_fn(self, eps_vec, source_vec, iterative=False, method=DEFAULT_SOLVER):
        raise NotImplementedError("need to implement a solve function")

    def z_to_xy(self, Fz_vec, eps_vec):
        raise NotImplementedError("need to implement a z -> {x, y} field conversion function")

    def solve(self, source, iterative=False, method=DEFAULT_SOLVER):
        """ Generic solve function """

        # make source a vector
        source_vec = source.flatten()

        # solve the z component of the fields
        Fz_vec = self.solve_fn(self.eps_vec, source_vec, iterative=iterative, method=method)

        # get the x and y vecays, put into tuple
        Fx_vec, Fy_vec = self.z_to_xy(Fz_vec, self.eps_vec)
        field_vectors = (Fx_vec, Fy_vec, Fz_vec)

        # convert all fields to grid and return tuple of them
        Fs = map(self._vec_to_grid, field_vectors)
        return tuple(Fs)

    def _vec_to_grid(self, vec):
        return np.reshape(vec, self.shape)

""" base classes for linear and nonlinear fdfd problems """

class fdfd_linear(fdfd):

    def __init__(self, omega, L0, eps_r, npml):
        super().__init__(omega, L0, eps_r, npml)

    @fdfd.eps_r.setter
    def eps_r(self, new_eps):
        """ Defines some attributes when eps_r is set. """
        assert not callable(new_eps), "for linear problems, eps_r must be a static array"        
        self.shape = self.Nx, self.Ny = new_eps.shape
        self.eps_vec = new_eps.flatten()
        self._eps_r = new_eps

class fdfd_nonlinear(fdfd):

    def __init__(self, omega, L0, eps_r, npml):
        super().__init__(omega, L0, eps_r, npml)

    @fdfd.eps_r.setter
    def eps_r(self, new_eps):
        """ Defines some attributes when eps_r is set. """
        assert callable(new_eps), "for nonlinear problems, eps_r must be a static array"        
        self.shape = self.Nx, self.Ny = new_eps(0).shape
        try:
            # try autogradding the epsilon function with field of 0
            # this comes with a warning, ignore it
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jacobian(new_eps)(np.zeros(self.shape))
        except:
            raise ValueError("The supplied eps_r function is NOT autograd-compatable, make sure you defined it using autograd.numpy functions!")
        self.eps_vec = lambda Ez: new_eps(Ez.reshape(self.shape)).flatten()
        self.__eps_r = new_eps

""" These are the fdfd classes that you'll actually want to use """

class fdfd_hz(fdfd_linear):
    """ FDFD class for linear Hz polarization """

    def __init__(self, omega, L0, eps_r, npml):
        super().__init__(omega, L0, eps_r, npml)

    def make_A(self, eps_vec):
        return make_A_Hz(self.info_dict, eps_vec)

    def solve_fn(self, eps_vec, source_vec, iterative=False, method=DEFAULT_SOLVER):
        return solve_Hz(self.info_dict, eps_vec, source_vec, iterative=iterative, method=method)

    def z_to_xy(self, Fz_vec, eps_vec):
        return H_to_E(Fz_vec, self.info_dict, eps_vec)

class fdfd_ez(fdfd_linear):
    """ FDFD class for linear Ez polarization """

    def __init__(self, omega, L0, eps_r, npml):
        assert not callable(eps_r), "for linear problems, eps_r must be a static array"
        super().__init__(omega, L0, eps_r, npml)

    def make_A(self, eps_vec):
        return make_A_Ez(self.info_dict, eps_vec)

    def solve_fn(self, eps_vec, source_vec, iterative=False, method=DEFAULT_SOLVER):
        return solve_Ez(self.info_dict, eps_vec, source_vec, iterative=False, method=DEFAULT_SOLVER)

    def z_to_xy(self, Fz_vec, eps_vec):
        return E_to_H(Fz_vec, self.info_dict, None)

class fdfd_ez_nl(fdfd_nonlinear):
    """ FDFD class for nonlinear Ez polarization """

    def __init__(self, omega, L0, eps_fn, npml):
        assert callable(eps_fn), "for nonlinear problems, eps_r must be a function of Ez"
        super().__init__(omega, L0, eps_fn, npml)

    def make_A(self, eps_fn):
        return lambda Ez: make_A_Ez(self.info_dict, eps_vec(Ez))

    def solve_fn(self, eps_fn, source_vec, iterative=False, method=DEFAULT_SOLVER):
        return solve_Ez_nl(self.info_dict, eps_fn, source_vec, iterative=False, method=DEFAULT_SOLVER)

    def z_to_xy(self, Fz_vec, eps_fn):
        return E_to_H(Fz_vec, self.info_dict, None)


""" This section is the meat and bones of the FDFD.
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

"""======================== SYSTEM MATRIX CREATION ========================"""

def make_A_Hz(info_dict, eps_vec):
    """ constructs the system matrix for `Hz` polarization """
    diag = 1 / EPSILON_0 * sp.spdiags(1/eps_vec, [0], eps_vec.size, eps_vec.size)
    A = spdot(info_dict['Dxf'], spdot(info_dict['Dxb'].T, diag).T) \
      + spdot(info_dict['Dyf'], spdot(info_dict['Dyb'].T, diag).T) \
      + info_dict['omega']**2 * MU_0 * sp.eye(eps_vec.size)
    return A

# note: this primitive not actually needed. but saving for later
# @primitive
def make_A_Ez(info_dict, eps_vec):
    """ constructs the system matrix for `Ez` polarization """
    diag = EPSILON_0 * sp.spdiags(eps_vec, [0], eps_vec.size, eps_vec.size)
    A = 1 / MU_0 * info_dict['Dxf'].dot(info_dict['Dxb']) \
      + 1 / MU_0 * info_dict['Dyf'].dot(info_dict['Dyb']) \
      + info_dict['omega']**2 * diag
    return A

# def vjp_maker_make_A_Ez(A, info_dict, eps_vec):
#     def vjp(v):
#         return EPSILON_0 * info_dict['omega']**2 * v
#     return vjp

# defvjp(make_A_Ez, None, vjp_maker_make_A_Ez)

"""========================== FIELD CONVERSIONS ==========================="""

def Ez_to_Hx(Ez, info_dict):
    """ Returns magnetic field `Hx` from electric field `Ez` """
    Hx = - spdot(info_dict['Dyb'], Ez) / MU_0
    return Hx

def Ez_to_Hy(Ez, info_dict):
    """ Returns magnetic field `Hy` from electric field `Ez` """
    Hy =  spdot(info_dict['Dxb'], Ez) / MU_0
    return Hy

def E_to_H(Ez, info_dict, eps_vec=None):
    """ More convenient function to return both Hx and Hy from Ez """
    Hx = Ez_to_Hx(Ez, info_dict)
    Hy = Ez_to_Hy(Ez, info_dict)
    return Hx, Hy

def Hz_to_Ex(Hz, info_dict, eps_vec, adjoint=False):
    """ Returns electric field `Ex` from magnetic field `Hz` """
    # note: adjoint switch is because backprop thru this fn. has different form
    if adjoint:
        Ex =  spdot(info_dict['Dyf'].T, Hz) / eps_vec / EPSILON_0
    else:
        Ex = -spdot(info_dict['Dyb'],   Hz) / eps_vec / EPSILON_0
    return Ex

def Hz_to_Ey(Hz, info_dict, eps_vec, adjoint=False):
    """ Returns electric field `Ey` from magnetic field `Hz` """
    if adjoint:
        Ey = -spdot(info_dict['Dxf'].T, Hz) / eps_vec / EPSILON_0
    else:
        Ey =  spdot(info_dict['Dxb'],   Hz) / eps_vec / EPSILON_0
    return Ey

def H_to_E(Hz, info_dict, eps_vec, adjoint=False):
    """ More convenient function to return both Ex and Ey from Hz """
    Ex = Hz_to_Ex(Hz, info_dict, eps_vec, adjoint=adjoint)
    Ey = Hz_to_Ey(Hz, info_dict, eps_vec, adjoint=adjoint)
    return Ex, Ey

"""======================== SOLVING FOR THE FIELDS ========================"""

# Linear Ez

@primitive
def solve_Ez(info_dict, eps_vec, source, iterative=False, method=DEFAULT_SOLVER):
    """ solve `Ez = A^-1 b` where A is constructed from the FDFD `info_dict`
        and 'eps_vec' is a (1D) vecay of the relative permittivity
    """
    A = make_A_Ez(info_dict, eps_vec)
    b = 1j * info_dict['omega'] * source
    Ez = sparse_solve(A, b, iterative=iterative, method=method)
    return Ez

# define the gradient of solve_Ez w.r.t. eps_vec (in Ez)
def vjp_maker_solve_Ez(Ez, info_dict, eps_vec, source, iterative=False, method=DEFAULT_SOLVER):
    """ Gives vjp for solve_Ez with respect to eps_vec """    
    # construct the system matrix again
    A = make_A_Ez(info_dict, eps_vec)
    # vector-jacobian product function to return
    def vjp(v):
        # solve the adjoint problem and get those electric fields (note D info_dict are different and transposed)
        Ez_aj = sparse_solve(A.T, -v, iterative=iterative, method=method)
        # because we care about the diagonal elements, just element-wise multiply E and E_adj
        # note: need np.real() for adjoint returns w.r.t. real quantities but not in forward mode
        return EPSILON_0 * info_dict['omega']**2 * np.real(Ez_aj * Ez)
    return vjp

def vjp_maker_solve_Ez_source(Ez, info_dict, eps_vec, source, iterative=False, method=DEFAULT_SOLVER):
    """ Gives vjp for solve_Ez with respect to source """    
    A = make_A_Ez(info_dict, eps_vec)
    def vjp(v):
        return 1j * info_dict['omega'] * sparse_solve(A.T, v, iterative=iterative, method=method)
    return vjp

# define the gradient of solve_Ez w.r.t. eps_vec (in Ez)
def jvp_solve_Ez(g, Ez, info_dict, eps_vec, source, iterative=False, method=DEFAULT_SOLVER):
    """ Gives jvp for solve_Ez with respect to eps_vec """    
    # construct the system matrix again and the RHS of the gradient expersion
    A = make_A_Ez(info_dict, eps_vec)
    u = Ez * -g
    # solve the adjoint problem and get those electric fields (note D info_dict are different and transposed)
    Ez_for = sparse_solve(A, u, iterative=iterative, method=method)
    # because we care about the diagonal elements, just element-wise multiply E and E_adj
    return EPSILON_0 * info_dict['omega']**2 * Ez_for

def jvp_solve_Ez_source(g, Ez, info_dict, eps_vec, source, iterative=False, method=DEFAULT_SOLVER):
    """ Gives jvp for solve_Ez with respect to source """  
    A = make_A_Ez(info_dict, eps_vec)      
    return 1j * info_dict['omega'] * sparse_solve(A, g, iterative=iterative, method=method)

defvjp(solve_Ez, None, vjp_maker_solve_Ez, vjp_maker_solve_Ez_source)
defjvp(solve_Ez, None, jvp_solve_Ez, jvp_solve_Ez_source)

# Linear Hz

@primitive
def solve_Hz(info_dict, eps_vec, source, iterative=False, method=DEFAULT_SOLVER):
    """ solve `Hz = A^-1 b` where A is constructed from the FDFD `info_dict`
        and 'eps_vec' is a (1D) vecay of the relative permittivity
    """
    A = make_A_Hz(info_dict, eps_vec)
    b = 1j * info_dict['omega'] * source    
    Hz = sparse_solve(A, b, iterative=iterative, method=method)
    return Hz

def vjp_maker_solve_Hz(Hz, info_dict, eps_vec, source, iterative=False, method=DEFAULT_SOLVER):
    """ Gives vjp for solve_Hz with respect to eps_vec """    
    # get the forward electric fields
    Ex, Ey = H_to_E(Hz, info_dict, eps_vec, adjoint=False)
    # construct the system matrix again
    A = make_A_Hz(info_dict, eps_vec)
    # vector-jacobian product function to return
    def vjp(v):
        # solve the adjoint problem and get those electric fields (note D info_dict are different and transposed)
        Hz_aj = sparse_solve(A.T, -v, iterative=iterative, method=method)
        Ex_aj, Ey_aj = H_to_E(Hz_aj, info_dict, eps_vec, adjoint=True)
        # because we care about the diagonal elements, just element-wise multiply E and E_adj
        return EPSILON_0 * np.real(Ex_aj * Ex + Ey_aj * Ey)
    # return this function for autograd to link-later
    return vjp

def vjp_maker_solve_Hz_source(Hz, info_dict, eps_vec, source, iterative=False, method=DEFAULT_SOLVER):
    """ Gives vjp for solve_Hz with respect to source """    
    A = make_A_Hz(info_dict, eps_vec)
    def vjp(v):
        return 1j * info_dict['omega'] * sparse_solve(A.T, v, iterative=iterative, method=method)
    return vjp

# define the gradient of solve_Hz w.r.t. eps_vec (in Hz)
def jvp_solve_Hz(g, Hz, info_dict, eps_vec, source, iterative=False, method=DEFAULT_SOLVER):
    """ Gives jvp for solve_Hz with respect to eps_vec """    
    # construct the system matrix again and the RHS of the gradient expersion
    A = make_A_Hz(info_dict, eps_vec)
    ux = spdot(info_dict['Dxb'], Hz)
    uy = spdot(info_dict['Dyb'], Hz)
    diag = sp.spdiags(1 / eps_vec, [0], eps_vec.size, eps_vec.size)
    # the g gets multiplied in at the middle of the expression
    ux = ux * diag * g * diag
    uy = uy * diag * g * diag
    ux = spdot(info_dict['Dxf'], ux)
    uy = spdot(info_dict['Dyf'], uy)
    # add the x and y components and multiply by A_inv on the left
    u = (ux + uy)
    Hz_for = sparse_solve(A, u, iterative=iterative, method=method)
    return 1 / EPSILON_0 * Hz_for

def jvp_solve_Hz_source(g, Hz, info_dict, eps_vec, source, iterative=False, method=DEFAULT_SOLVER):
    """ Gives jvp for solve_Hz with respect to source """    
    A = make_A_Hz(info_dict, eps_vec)      
    return 1j * info_dict['omega'] * sparse_solve(A, g, iterative=iterative, method=method)

defvjp(solve_Hz, None, vjp_maker_solve_Hz, vjp_maker_solve_Hz_source)
defjvp(solve_Hz, None, jvp_solve_Hz, jvp_solve_Hz_source)

# Nonlinear Ez

"""=========================== SPECIAL SOLVE =========================="""

from numpy.linalg import norm
from .utils import get_value

@primitive
def special_solve(info_dict, eps, b, iterative=False, method=DEFAULT_SOLVER):
    A = make_A_Ez(info_dict, eps)
    return sparse_solve(A, b, iterative=iterative, method=method)

def special_solve_T(info_dict, eps, b, iterative=False, method=DEFAULT_SOLVER):
    A = make_A_Ez(info_dict, eps)
    return sparse_solve(A.T, b, iterative=iterative, method=method)

def vjp_special_solve(x, info_dict, eps, b, iterative=False, method=DEFAULT_SOLVER):
    def vjp(v):
        x_aj = special_solve_T(info_dict, eps, -v, iterative=iterative, method=method)
        return info_dict['omega']**2 * EPSILON_0 * x * x_aj
    return vjp

defvjp(special_solve, None, vjp_special_solve, None)

def solve_nonlinear(info_dict, eps_fn, b, iterative=False, method=DEFAULT_SOLVER, verbose=False, atol=1e-10, max_iters=10):
    """ Solve Ax=b for x where A is a function of x using direct substitution """

    def relative_residual(eps, x, b):
        """ computes relative residual: ||Ax - b|| / ||b|| """
        A = make_A_Ez(info_dict, eps)
        res = norm(A.dot(x) - b)
        return res / norm(b)

    #note, uncommenting first one 'unhooks' this function from autograd, uses vjp of solve_Ez_nl
    # eps_fn_static = lambda E: get_value(eps_fn(E))
    #note, uncommenting second one uses vjps defined for special solve
    eps_fn_static = lambda E: eps_fn(E)

    vec_0 = np.zeros(b.shape)
    eps_0 = eps_fn_static(vec_0)

    E_i = special_solve(info_dict, eps_0, b)

    for i in range(max_iters):

        eps_i = eps_fn_static(E_i)
        rel_res = relative_residual(get_value(eps_i), get_value(E_i), b)

        if verbose:
            print('i = {}, relative residual = {}'.format(i, rel_res))

        if rel_res < atol:
            break
        
        E_i = special_solve(info_dict, eps_i, b)

    return E_i

@primitive
def solve_Ez_nl(info_dict, eps_fn, source, iterative=False, method=DEFAULT_SOLVER):

    b = 1j * info_dict['omega'] * source
    Ez = solve_nonlinear(info_dict, eps_fn, b)
    return Ez

# To do: write our simpler adjoint formalism for converged solutions here

def vjp_maker_solve_Ez_nl_eps(Ez, info_dict, eps_fn, source, iterative=False, method=DEFAULT_SOLVER):

    print('why does this vjp never get called??')
    # eps_eval = get_value(eps_fn(Ez))
    # A = make_A_Ez(info_dict, eps_eval)
    # zero = sp.csr_matrix(info_dict['shape'], dtype=np.complex128)
    # A_block = block_4(A, zero, zero, A)
    # b = 1j * info_dict['omega'] * source
    # f = spdot(A, Ez) - b

    def vjp(v):
        # just some random function for testing
        return v

    return vjp

def vjp_maker_solve_Ez_nl_b(Ez, info_dict, eps_fn, source, iterative=False, method=DEFAULT_SOLVER):
    def vjp(v):
        # just some random function for testing
        return v
    return vjp

defvjp(solve_Ez_nl, None, vjp_maker_solve_Ez_nl_eps, vjp_maker_solve_Ez_nl_b)

"""=========================== HELPER FUNCTIONS ==========================="""

def compute_derivative_matrices(omega, shape, npml, dL):

    # make the S-matrices for PML
    (Sxf, Sxb, Syf, Syb) = S_create(omega, shape, npml, dL)

    # Construct derivate matrices without PML
    Dxf_0 = createDws('x', 'f', dL, shape)
    Dxb_0 = createDws('x', 'b', dL, shape)
    Dyf_0 = createDws('y', 'f', dL, shape)
    Dyb_0 = createDws('y', 'b', dL, shape)

    # apply PML to derivative matrices
    Dxf = Sxf.dot(Dxf_0)
    Dxb = Sxb.dot(Dxb_0)
    Dyf = Syf.dot(Dyf_0)
    Dyb = Syb.dot(Dyb_0)

    return Dxf, Dxb, Dyf, Dyb


def S_create(omega, shape, npml, dL):
    # creates S matrices for the PML creation

    Nx, Ny = shape
    N = Nx * Ny
    x_range = [0, float(dL * Nx)]
    y_range = [0, float(dL * Ny)]

    Nx_pml, Ny_pml = npml    

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor('f', omega, dL, Nx, Nx_pml)
    s_vector_x_b = create_sfactor('b', omega, dL, Nx, Nx_pml)
    s_vector_y_f = create_sfactor('f', omega, dL, Ny, Ny_pml)
    s_vector_y_b = create_sfactor('b', omega, dL, Ny, Ny_pml)

    # Fill the 2D space with layers of appropriate s-factors
    Sx_f_2D = np.zeros(shape, dtype=np.complex128)
    Sx_b_2D = np.zeros(shape, dtype=np.complex128)
    Sy_f_2D = np.zeros(shape, dtype=np.complex128)
    Sy_b_2D = np.zeros(shape, dtype=np.complex128)

    for i in range(0, Ny):
        Sx_f_2D[:, i] = 1 / s_vector_x_f
        Sx_b_2D[:, i] = 1 / s_vector_x_b

    for i in range(0, Nx):
        Sy_f_2D[i, :] = 1 / s_vector_y_f
        Sy_b_2D[i, :] = 1 / s_vector_y_b

    # Reshape the 2D s-factors into a 1D s-vecay
    Sx_f_vec = Sx_f_2D.reshape((-1,))
    Sx_b_vec = Sx_b_2D.reshape((-1,))
    Sy_f_vec = Sy_f_2D.reshape((-1,))
    Sy_b_vec = Sy_b_2D.reshape((-1,))

    # Construct the 1D total s-vecay into a diagonal matrix
    Sx_f = sp.spdiags(Sx_f_vec, 0, N, N)
    Sx_b = sp.spdiags(Sx_b_vec, 0, N, N)
    Sy_f = sp.spdiags(Sy_f_vec, 0, N, N)
    Sy_b = sp.spdiags(Sy_b_vec, 0, N, N)

    return Sx_f, Sx_b, Sy_f, Sy_b


def createDws(w, s, dL, shape):
    """ creates the derivative matrices
            NOTE: python uses C ordering rather than Fortran ordering. Therefore the
            derivative operators are constructed slightly differently than in MATLAB
    """

    Nx, Ny = shape

    if w is 'x':
        if Nx > 1:
            if s is 'f':
                dxf = sp.diags([-1, 1, 1], [0, 1, -Nx+1], shape=(Nx, Nx))
                Dws = 1 / dL * sp.kron(dxf, sp.eye(Ny))
            else:
                dxb = sp.diags([1, -1, -1], [0, -1, Nx-1], shape=(Nx, Nx))
                Dws = 1 / dL * sp.kron(dxb, sp.eye(Ny))
        else:
            Dws = sp.eye(Ny)            
    if w is 'y':
        if Ny > 1:
            if s is 'f':
                dyf = sp.diags([-1, 1, 1], [0, 1, -Ny+1], shape=(Ny, Ny))
                Dws = 1 / dL * sp.kron(sp.eye(Nx), dyf)
            else:
                dyb = sp.diags([1, -1, -1], [0, -1, Ny-1], shape=(Ny, Ny))
                Dws = 1 / dL * sp.kron(sp.eye(Nx), dyb)
        else:
            Dws = sp.eye(Nx)
    return Dws


def sig_w(l, dw, m=3, lnR=-30):
    # helper for S()
    sig_max = -(m + 1) * lnR / (2 * ETA_0 * dw)
    return sig_max * (l / dw)**m


def S(l, dw, omega):
    # helper for create_sfactor()
    return 1 - 1j * sig_w(l, dw) / (omega * EPSILON_0)


def create_sfactor(s, omega, dL, N, N_pml):
    # used to help construct the S matrices for the PML creation

    sfactor_vecay = np.ones(N, dtype=np.complex128)
    if N_pml < 1:
        return sfactor_vecay

    dw = N_pml * dL

    for i in range(N):
        if s is 'f':
            if i <= N_pml:
                sfactor_vecay[i] = S(dL * (N_pml - i + 0.5), dw, omega)
            elif i > N - N_pml:
                sfactor_vecay[i] = S(dL * (i - (N - N_pml) - 0.5), dw, omega)
        if s is 'b':
            if i <= N_pml:
                sfactor_vecay[i] = S(dL * (N_pml - i + 1), dw, omega)
            elif i > N - N_pml:
                sfactor_vecay[i] = S(dL * (i - (N - N_pml) - 1), dw, omega)
    return sfactor_vecay
