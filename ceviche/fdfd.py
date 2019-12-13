import autograd.numpy as npa
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import copy
import warnings

from autograd.extend import primitive, defvjp, defjvp

from .constants import *
from .jacobians import jacobian
from .primitives import sp_solve, sp_mult, get_entries_indices

AVG = False  # whether to do grid averaging (under development, gradients dont match exactly yet.)

class fdfd():
    """ Base class for FDFD simulation """

    def __init__(self, omega, dL, eps_r, npml, bloch_x=0.0, bloch_y=0.0):
        """ initialize with a given structure and source 
                omega: angular frequency (rad/s)
                dL: grid cell size (m)
                eps_r: array containing relative permittivity
                npml: list of number of PML grid cells in [x, y]
                bloch_{x,y} phase difference across {x,y} boundaries for bloch periodic boundary conditions (default = 0 = periodic)
        """

        self.omega = omega
        self.dL = dL
        self.npml = npml

        self.bloch_x = bloch_x
        self.bloch_y = bloch_y

        self.eps_r = eps_r
        self.shape = eps_r.shape

        self.setup_derivatives()

    def setup_derivatives(self):

        # Creates all of the operators needed for later
        derivs = compute_derivative_matrices(self.omega, self.shape, self.npml, self.dL, bloch_x=self.bloch_x, bloch_y=self.bloch_y)
        self.Dxf, self.Dxb, self.Dyf, self.Dyb = derivs
        self.Dxf_entries, self.Dxf_indices = get_entries_indices(self.Dxf)
        self.Dxb_entries, self.Dxb_indices = get_entries_indices(self.Dxb)
        self.Dyf_entries, self.Dyf_indices = get_entries_indices(self.Dyf)
        self.Dyb_entries, self.Dyb_indices = get_entries_indices(self.Dyb)

    @property
    def eps_r(self):
        """ Returns the relative permittivity grid """
        return self._eps_r

    @eps_r.setter
    def eps_r(self, new_eps):
        """ Defines some attributes when eps_r is set. """
        self.shape = new_eps.shape
        self._eps_r = new_eps

    def make_A(self, eps_r):
        raise NotImplementedError("need to make a make_A() method")

    def z_to_xy(self, Fz_vec, eps_vec):
        raise NotImplementedError("need to implement a z -> {x, y} field conversion function")

    def solve(self, source):
        """ Generic solve function """

        # flatten the permittivity and source grids
        source_vec = source.flatten()
        eps_vec = self.eps_r.flatten()

        # create the A matrix for this polarization
        A_entries, A_indices = self.make_A(eps_vec)

        # solve the z component of the fields using A and the source
        Fz_vec = sp_solve(A_entries, A_indices, source_vec)

        # get the x and y vectors and put all components into tuple
        Fx_vec, Fy_vec = self.z_to_xy(Fz_vec, eps_vec)

        # put all field components into a tuple, convert to grid shape and return them all
        field_vectors = (Fx_vec, Fy_vec, Fz_vec)
        Fs = map(self._vec_to_grid, field_vectors)
        return tuple(Fs)

    def _vec_to_grid(self, vec):
        # converts a vector quantity into an array of the shape of the FDFD simulation
        return npa.reshape(vec, self.shape)


""" These are the fdfd classes that you'll actually want to use """

class fdfd_ez(fdfd):
    """ FDFD class for linear Ez polarization """

    def __init__(self, omega, L0, eps_r, npml, bloch_x=0.0, bloch_y=0.0):
        super().__init__(omega, L0, eps_r, npml, bloch_x=bloch_x, bloch_y=bloch_y)

    def make_A(self, eps_vec):

        N = eps_vec.size

        C = 1 / MU_0 * self.Dxf.dot(self.Dxb) \
          + 1 / MU_0 * self.Dyf.dot(self.Dyb)
        c_entries, c_indices = get_entries_indices(C)

        # indices into the diagonal of a sparse matrix
        diag_entries = EPSILON_0 * self.omega**2 * eps_vec
        diag_indices = npa.vstack((npa.arange(N), npa.arange(N)))

        A_entries = npa.hstack((diag_entries, c_entries))
        A_indices = npa.hstack((diag_indices, c_indices))

        return A_entries, A_indices

    def _Ez_to_Hx(self, Ez_vec):
        """ Returns magnetic field `Hx` from electric field `Ez` """
        Hx = 1 / MU_0 * sp_mult(self.Dyb_entries, self.Dyf_indices, Ez_vec)
        return Hx

    def _Ez_to_Hy(self, Ez_vec):
        """ Returns magnetic field `Hy` from electric field `Ez` """
        Hy = 1 / MU_0 * sp_mult(self.Dxb_entries, self.Dxf_indices, Ez_vec)
        return Hy

    def _E_to_H(self, Ez_vec):
        """ More convenient function to return both Hx and Hy from Ez """
        Hx_vec = self._Ez_to_Hx(Ez_vec)
        Hy_vec = self._Ez_to_Hy(Ez_vec)
        return Hx_vec, Hy_vec

    def z_to_xy(self, Ez_vec, eps_vec):
        return self._E_to_H(Ez_vec)

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

# # NEED TO DO GRID AVERAGING HERE NEXT!
# def make_A_Hz(info_dict, eps_vec_zz):
#     """ constructs the system matrix for `Hz` polarization """
#     N = eps_vec_zz.size
#     eps_vec_xx, eps_vec_yy = vec_zz_to_xy(info_dict, eps_vec_zz, grid_averaging=AVG)
#     diag_xx = 1 / EPSILON_0 * sp.spdiags(1/eps_vec_xx, [0], N, N)
#     diag_yy = 1 / EPSILON_0 * sp.spdiags(1/eps_vec_yy, [0], N, N)
#     A = spdot(info_dict['Dxf'], spdot(info_dict['Dxb'].T, diag_xx).T) \
#       + spdot(info_dict['Dyf'], spdot(info_dict['Dyb'].T, diag_yy).T) \
#       + info_dict['omega']**2 * MU_0 * sp.eye(N)
#     return A

# def make_A_Ez(info_dict, eps_vec_zz):
#     """ constructs the system matrix for `Ez` polarization """
#     N = eps_vec_zz.size
#     diag_zz = EPSILON_0 * sp.spdiags(eps_vec_zz, [0], N, N)
#     A = 1 / MU_0 * info_dict['Dxf'].dot(info_dict['Dxb']) \
#       + 1 / MU_0 * info_dict['Dyf'].dot(info_dict['Dyb']) \
#       + info_dict['omega']**2 * diag_zz
#     return A

# """========================== FIELD CONVERSIONS ==========================="""


# def Hz_to_Ex(Hz, info_dict, eps_vec_zz, adjoint=False):
#     """ Returns electric field `Ex` from magnetic field `Hz` """
#     # note: adjoint switch is because backprop thru this fn. has different form
#     eps_vec_xx, eps_vec_yy = vec_zz_to_xy(info_dict, eps_vec_zz, grid_averaging=AVG) 
#     if adjoint:
#         Ex =  spdot(info_dict['Dyf'].T, Hz) / eps_vec_zz / EPSILON_0
#     else:
#         Ex = -spdot(info_dict['Dyb'],   Hz) / eps_vec_xx / EPSILON_0
#     return Ex

# def Hz_to_Ey(Hz, info_dict, eps_vec_zz, adjoint=False):
#     """ Returns electric field `Ey` from magnetic field `Hz` """
#     eps_vec_xx, eps_vec_yy = vec_zz_to_xy(info_dict, eps_vec_zz, grid_averaging=AVG)
#     if adjoint:
#         Ey = -spdot(info_dict['Dxf'].T, Hz) / eps_vec_zz / EPSILON_0
#     else:        
#         Ey =  spdot(info_dict['Dxb'],   Hz) / eps_vec_yy / EPSILON_0
#     return Ey

# def H_to_E(Hz, info_dict, eps_vec_zz, adjoint=False):
#     """ More convenient function to return both Ex and Ey from Hz """
#     eps_vec_xx, eps_vec_yy = vec_zz_to_xy(info_dict, eps_vec_zz, grid_averaging=AVG)    
#     Ex = Hz_to_Ex(Hz, info_dict, eps_vec_zz, adjoint=adjoint)
#     Ey = Hz_to_Ey(Hz, info_dict, eps_vec_zz, adjoint=adjoint)
#     return Ex, Ey

# """======================== SOLVING FOR THE FIELDS ========================"""

# # Linear Ez

# def solve_Ez(info_dict, eps_vec_zz, source, iterative=False, method=DEFAULT_SOLVER):
#     """ solve `Ez = A^-1 b` where A is constructed from the FDFD `info_dict`
#         and 'eps_vec' is a (1D) vecay of the relative permittivity
#     """
#     A = make_A_Ez(info_dict, eps_vec_zz)
#     b = 1j * info_dict['omega'] * source
#     Ez = sparse_solve(A, b, iterative=iterative, method=method)
#     return Ez

# def solve_Hz(info_dict, eps_vec_zz, source, iterative=False, method=DEFAULT_SOLVER):
#     """ solve `Hz = A^-1 b` where A is constructed from the FDFD `info_dict`
#         and 'eps_vec' is a (1D) vecay of the relative permittivity
#     """

#     A = make_A_Hz(info_dict, eps_vec_zz)
#     b = 1j * info_dict['omega'] * source    
#     Hz = sparse_solve(A, b, iterative=iterative, method=method)
#     return Hz


# """=========================== SPECIAL SOLVE =========================="""

# from numpy.linalg import norm
# from .utils import get_value

# @primitive
# def special_solve(info_dict, eps, b, iterative=False, method=DEFAULT_SOLVER):
#     A = make_A_Ez(info_dict, eps)
#     return sparse_solve(A, b, iterative=iterative, method=method)

# def special_solve_T(info_dict, eps, b, iterative=False, method=DEFAULT_SOLVER):
#     A = make_A_Ez(info_dict, eps)
#     return sparse_solve(A.T, b, iterative=iterative, method=method)

# def vjp_special_solve(x, info_dict, eps, b, iterative=False, method=DEFAULT_SOLVER):
#     def vjp(v):
#         x_aj = special_solve_T(info_dict, eps, -v, iterative=iterative, method=method)
#         return info_dict['omega']**2 * EPSILON_0 * x * x_aj
#     return vjp

# defvjp(special_solve, None, vjp_special_solve, None)

# def solve_nonlinear(info_dict, eps_fn, b, iterative=False, method=DEFAULT_SOLVER, verbose=False, atol=1e-10, max_iters=10):
#     """ Solve Ax=b for x where A is a function of x using direct substitution """

#     def relative_residual(eps, x, b):
#         """ computes relative residual: ||Ax - b|| / ||b|| """
#         A = make_A_Ez(info_dict, eps)
#         res = norm(A.dot(x) - b)
#         return res / norm(b)

#     #note, uncommenting first one 'unhooks' this function from autograd, uses vjp of solve_Ez_nl
#     # eps_fn_static = lambda E: get_value(eps_fn(E))
#     #note, uncommenting second one uses vjps defined for special solve
#     eps_fn_static = lambda E: eps_fn(E)

#     vec_0 = npa.zeros(b.shape)
#     eps_0 = eps_fn_static(vec_0)

#     E_i = special_solve(info_dict, eps_0, b)

#     for i in range(max_iters):

#         eps_i = eps_fn_static(E_i)
#         rel_res = relative_residual(get_value(eps_i), get_value(E_i), b)

#         if verbose:
#             print('i = {}, relative residual = {}'.format(i, rel_res))

#         if rel_res < atol:
#             break
        
#         E_i = special_solve(info_dict, eps_i, b)

#     return E_i

# @primitive
# def solve_Ez_nl(info_dict, eps_fn, source, iterative=False, method=DEFAULT_SOLVER):

#     b = 1j * info_dict['omega'] * source
#     Ez = solve_nonlinear(info_dict, eps_fn, b)
#     return Ez

# # To do: write our simpler adjoint formalism for converged solutions here

# def vjp_maker_solve_Ez_nl_eps(Ez, info_dict, eps_fn, source, iterative=False, method=DEFAULT_SOLVER):

#     print('why does this vjp never get called??')
#     # eps_eval = get_value(eps_fn(Ez))
#     # A = make_A_Ez(info_dict, eps_eval)
#     # zero = sp.csr_matrix(info_dict['shape'], dtype=npa.complex128)
#     # A_block = block_4(A, zero, zero, A)
#     # b = 1j * info_dict['omega'] * source
#     # f = spdot(A, Ez) - b

#     def vjp(v):
#         # just some random function for testing
#         return v

#     return vjp

# def vjp_maker_solve_Ez_nl_b(Ez, info_dict, eps_fn, source, iterative=False, method=DEFAULT_SOLVER):
#     def vjp(v):
#         # just some random function for testing
#         return v
#     return vjp

# defvjp(solve_Ez_nl, None, vjp_maker_solve_Ez_nl_eps, vjp_maker_solve_Ez_nl_b)


"""============================= SOURCE / TFSF ============================"""

def b_TFSF(fdfd, inside_mask, theta):
    """ Returns a source vector for FDFD that will implement TFSF 
            A: the FDFD system matrix
            inside_mask: a binary mask (vector) specifying the inside of the TFSF region
            theta: [0, 2pi] the angle of the source relative to y=0+

                      y ^
                        |
                        |
                  <-----|----- > x
                        |\
                        | \                                     
                        v |\
                          theta             
    see slide 32 of https://empossible.net/wp-content/uploads/2019/08/Lecture-4d-FDFD-Formulation.pdf                                                       
    """

    lambda0 = 2 * npa.pi * C_0 / fdfd.omega
    f_src = compute_f(theta, lambda0, fdfd.dL,  inside.shape)

    Q = compute_Q(inside_mask) / fdfd.omega # why this omega??
    A = fdfd.make_A(fdfd.eps_r.copy().flatten())

    quack = (Q.dot(A) - A.dot(Q))

    return quack.dot(f_src)

def compute_Q(inside_mask):
    """ Compute the matrix used in PDF to get source """

    # convert masks to vectors and get outside portion
    inside_vec = inside_mask.flatten()
    outside_vec = 1 - inside_vec
    N = outside_vec.size

    # make a sparse diagonal matrix and return
    Q = sp.diags([outside_vec], [0], shape=(N, N))
    return Q


def compute_f(theta, lambda0, dL, shape):
    """ Compute the 'vacuum' field vector """

    # get plane wave k vector components (in units of grid cells)
    k0 = 2 * npa.pi / lambda0 * dL
    kx =  k0 * npa.sin(theta)
    ky = -k0 * npa.cos(theta)  # negative because downwards

    # array to write into
    f_src = npa.zeros(shape, dtype=npa.complex128)

    # get coordinates
    Nx, Ny = shape
    xpoints = npa.arange(Nx)
    ypoints = npa.arange(Ny)
    xv, yv = npa.meshgrid(xpoints, ypoints, indexing='ij')

    # compute values and insert into array
    x_PW = npa.exp(1j * xpoints * kx)[:, None]
    y_PW = npa.exp(1j * ypoints * ky)[:, None]

    f_src[xv, yv] = npa.outer(x_PW, y_PW)

    return f_src.flatten()

"""=========================== HELPER FUNCTIONS ==========================="""

def compute_derivative_matrices(omega, shape, npml, dL, bloch_x=0.0, bloch_y=0.0):

    # make the S-matrices for PML
    (Sxf, Sxb, Syf, Syb) = S_create(omega, shape, npml, dL)

    # Construct derivate matrices without PML
    Dxf_0 = createDws('x', 'f', dL, shape, bloch_x=bloch_x, bloch_y=bloch_y)
    Dxb_0 = createDws('x', 'b', dL, shape, bloch_x=bloch_x, bloch_y=bloch_y)
    Dyf_0 = createDws('y', 'f', dL, shape, bloch_x=bloch_x, bloch_y=bloch_y)
    Dyb_0 = createDws('y', 'b', dL, shape, bloch_x=bloch_x, bloch_y=bloch_y)

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
    Sx_f_2D = npa.zeros(shape, dtype=npa.complex128)
    Sx_b_2D = npa.zeros(shape, dtype=npa.complex128)
    Sy_f_2D = npa.zeros(shape, dtype=npa.complex128)
    Sy_b_2D = npa.zeros(shape, dtype=npa.complex128)

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


def createDws(w, s, dL, shape, bloch_x=0.0, bloch_y=0.0):
    """ creates the derivative matrices
            NOTE: python uses C ordering rather than Fortran ordering. Therefore the
            derivative operators are constructed slightly differently than in MATLAB
    """

    Nx, Ny = shape

    if w is 'x':
        if Nx > 1:
            phasor_x = npa.exp(1j * bloch_x)   
            if s is 'f':
                # dxf = sp.diags([-1, 1, 1], [0, 1, -Nx+1], shape=(Nx, Nx))
                dxf = sp.diags([-1, 1, phasor_x], [0, 1, -Nx+1], shape=(Nx, Nx), dtype=npa.complex128)
                Dws = 1 / dL * sp.kron(dxf, sp.eye(Ny))
            else:
                # dxb = sp.diags([1, -1, -1], [0, -1, Nx-1], shape=(Nx, Nx))
                dxb = sp.diags([1, -1, -npa.conj(phasor_x)], [0, -1, Nx-1], shape=(Nx, Nx), dtype=npa.complex128)
                Dws = 1 / dL * sp.kron(dxb, sp.eye(Ny))
        else:
            Dws = sp.eye(Ny)
    if w is 'y':
        if Ny > 1:
            phasor_y = npa.exp(1j * bloch_y)               
            if s is 'f':
                dyf = sp.diags([-1, 1, phasor_y], [0, 1, -Ny+1], shape=(Ny, Ny))
                Dws = 1 / dL * sp.kron(sp.eye(Nx), dyf)
            else:
                dyb = sp.diags([1, -1, -npa.conj(phasor_y)], [0, -1, Ny-1], shape=(Ny, Ny))
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

    sfactor_vecay = npa.ones(N, dtype=npa.complex128)
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

