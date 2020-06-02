import autograd.numpy as npa
import scipy.sparse as sp
import matplotlib as mpl
mpl.rcParams['figure.dpi']=100
import matplotlib.pylab as plt

from .constants import *
from .primitives import sp_solve, sp_mult, spsp_mult
from .derivatives import compute_derivative_matrices, compute_derivative_matrices_3D
from .utils import get_entries_indices, make_sparse

# notataion is similar to that used in: http://www.jpier.org/PIERB/pierb36/11.11092006.pdf

class fdfd():
    """ Base class for FDFD simulation """

    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None):
        """ initialize with a given structure and source
                omega: angular frequency (rad/s)
                dL: grid cell size (m)
                eps_r: array containing relative permittivity
                npml: list of number of PML grid cells in [x, y]
                bloch_{x,y} phase difference across {x,y} boundaries for bloch periodic boundary conditions (default = 0 = periodic)
        """
        self.is_3D = False
        self.omega = omega
        self.dL = dL
        self.npml = npml
        self._setup_bloch_phases(bloch_phases)
        self.eps_r = eps_r
        self._setup_derivatives()

    """ what happens when you reassign the permittivity of the fdfd object """

    @property
    def eps_r(self):
        """ Returns the relative permittivity grid """
        return self._eps_r

    @eps_r.setter
    def eps_r(self, new_eps):
        """ Defines some attributes when eps_r is set. """
        if self.is_3D is True:
            self._save_shape_3D(new_eps)
        else:
            self._save_shape(new_eps)
        self._eps_r = new_eps

    """ classes inherited from fdfd() must implement their own versions of these functions for `fdfd.solve()` to work """

    def _make_A(self, eps_r):
        """ This method constucts the entries and indices into the system matrix """
        raise NotImplementedError("need to make a _make_A() method")

    def _solve_fn(self, entries_a, indices_a, source_vec):
        """ This method takes the system matrix and source and returns the x, y, and z field components """
        raise NotImplementedError("need to implement function to solve for field components")

    """ You call this to function to solve for the electromagnetic fields """

    def solve(self, source_z):
        """ Outward facing function (what gets called by user) that takes a source grid and returns the field components """

        # flatten the permittivity and source grid
        source_vec = self._grid_to_vec(source_z)
        eps_vec = self._grid_to_vec(self.eps_r)

        # create the A matrix for this polarization
        entries_a, indices_a = self._make_A(eps_vec)

        # solve field componets usng A and the source
        Fx_vec, Fy_vec, Fz_vec = self._solve_fn(eps_vec, entries_a, indices_a, source_vec)
        # put all field components into a tuple, convert to grid shape and return them all
        Fx = self._vec_to_grid(Fx_vec)
        Fy = self._vec_to_grid(Fy_vec)
        Fz = self._vec_to_grid(Fz_vec)

        return Fx, Fy, Fz

    """ Utility functions for FDFD object """

    def _setup_derivatives(self):
        """ Makes the sparse derivative matrices and does some processing for ease of use """

        # Creates all of the operators needed for later
        derivs = compute_derivative_matrices(self.omega, self.shape, self.npml, self.dL, bloch_x=self.bloch_x, bloch_y=self.bloch_y)

        # stores the raw sparse matrices
        self.Dxf, self.Dxb, self.Dyf, self.Dyb = derivs

        # store the entries and elements
        self.entries_Dxf, self.indices_Dxf = get_entries_indices(self.Dxf)
        self.entries_Dxb, self.indices_Dxb = get_entries_indices(self.Dxb)
        self.entries_Dyf, self.indices_Dyf = get_entries_indices(self.Dyf)
        self.entries_Dyb, self.indices_Dyb = get_entries_indices(self.Dyb)

        # stores some convenience functions for multiplying derivative matrices by a vector `vec`
        self.sp_mult_Dxf = lambda vec: sp_mult(self.entries_Dxf, self.indices_Dxf, vec)
        self.sp_mult_Dxb = lambda vec: sp_mult(self.entries_Dxb, self.indices_Dxb, vec)
        self.sp_mult_Dyf = lambda vec: sp_mult(self.entries_Dyf, self.indices_Dyf, vec)
        self.sp_mult_Dyb = lambda vec: sp_mult(self.entries_Dyb, self.indices_Dyb, vec)

    def _setup_bloch_phases(self, bloch_phases):
        """ Saves the x y and z bloch phases based on list of them 'bloch_phases' """

        self.bloch_x = 0.0
        self.bloch_y = 0.0
        self.bloch_z = 0.0
        if bloch_phases is not None:
            self.bloch_x = bloch_phases[0]
            if len(bloch_phases) > 1:
                self.bloch_y = bloch_phases[1]
            if len(bloch_phases) > 2:
                self.bloch_z = bloch_phases[2]

    def _vec_to_grid(self, vec):
        """ converts a vector quantity into an array of the shape of the FDFD simulation """
        return npa.reshape(vec, self.shape)

    def _grid_to_vec(self, grid):
        """ converts a grid of the shape of the FDFD simulation to a flat vector """
        return grid.flatten()

    def _save_shape(self, grid):
        """ Sores the shape and size of `grid` array to the FDFD object """
        self.shape = grid.shape
        self.Nx, self.Ny = self.shape
        self.N = self.Nx * self.Ny
        
    def _save_shape_3D(self, grid):
        """ Sores the shape and size of `grid` array to the FDFD object """
        self.shape = grid.shape
        self.Nx, self.Ny, self.Nz = self.shape
        self.N = self.Nx * self.Ny * self.Nz
        
    @staticmethod
    def _default_val(val, default_val=None):
        # not used yet
        return val if val is not None else default_val

    """ Field conversion functions for 2D.  Function names are self explanatory """

    def _Ex_Ey_to_Hz(self, Ex_vec, Ey_vec):
        return  1 / 1j / self.omega / MU_0 * (self.sp_mult_Dxb(Ey_vec) - self.sp_mult_Dyb(Ex_vec))

    def _Ez_to_Hx(self, Ez_vec):
        return -1 / 1j / self.omega / MU_0 * self.sp_mult_Dyb(Ez_vec)

    def _Ez_to_Hy(self, Ez_vec):
        return  1 / 1j / self.omega / MU_0 * self.sp_mult_Dxb(Ez_vec)

    def _Ez_to_Hx_Hy(self, Ez_vec):
        Hx_vec = self._Ez_to_Hx(Ez_vec)
        Hy_vec = self._Ez_to_Hy(Ez_vec)
        return Hx_vec, Hy_vec

    # addition of 1e-5 is for numerical stability when tracking gradients of eps_xx, and eps_yy -> 0
    def _Hz_to_Ex(self, Hz_vec, eps_vec_xx):
        return  1 / 1j / self.omega / EPSILON_0 / (eps_vec_xx + 1e-5) * self.sp_mult_Dyf(Hz_vec)

    def _Hz_to_Ey(self, Hz_vec, eps_vec_yy):
        return -1 / 1j / self.omega / EPSILON_0 / (eps_vec_yy + 1e-5) * self.sp_mult_Dxf(Hz_vec)

    def _Hx_Hy_to_Ez(self, Hx_vec, Hy_vec, eps_vec_zz):
        return  1 / 1j / self.omega / EPSILON_0 / (eps_vec_zz + 1e-5) * (self.sp_mult_Dxf(Hy_vec) - self.sp_mult_Dyf(Hx_vec))

    def _Hz_to_Ex_Ey(self, Hz_vec, eps_vec_xx, eps_vec_yy):
        Ex_vec = self._Hz_to_Ex(Hz_vec, eps_vec_xx)
        Ey_vec = self._Hz_to_Ey(Hz_vec, eps_vec_yy)
        return Ex_vec, Ey_vec

""" These are the fdfd classes that you'll actually want to use """

class fdfd_ez(fdfd):
    """ FDFD class for linear Ez polarization """
   
    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None):
        super().__init__(omega, dL, eps_r, npml, bloch_phases=bloch_phases)
        
    def _make_A(self, eps_vec):
        C = - 1 / MU_0 * self.Dxf.dot(self.Dxb) \
            - 1 / MU_0 * self.Dyf.dot(self.Dyb)
        #print('C的size',C.shape)
        print('CC',C)
        entries_c, indices_c = get_entries_indices(C)
        # indices into the diagonal of a sparse matrix
        entries_diag = - EPSILON_0 * self.omega**2 * eps_vec
        #print('eps_vec的size',eps_vec.shape)
        indices_diag = npa.vstack((npa.arange(self.N), npa.arange(self.N)))
        print('EE',entries_diag)
        entries_a = npa.hstack((entries_diag, entries_c))
        indices_a = npa.hstack((indices_diag, indices_c))
        print('AA',entries_a)
        return entries_a, indices_a

    def _solve_fn(self, eps_vec, entries_a, indices_a, Jz_vec):

        b_vec = 1j * self.omega * Jz_vec
        print(b_vec)
        Ez_vec = sp_solve(entries_a, indices_a, b_vec)
        print('Ez_vec',Ez_vec)
        Hx_vec, Hy_vec = self._Ez_to_Hx_Hy(Ez_vec)
        return Hx_vec, Hy_vec, Ez_vec

class fdfd_hz(fdfd):
    """ FDFD class for linear Ez polarization """

    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None):
        super().__init__(omega, dL, eps_r, npml, bloch_phases=bloch_phases)

    def _grid_average_2d(self, eps_vec):

        eps_grid = self._vec_to_grid(eps_vec)
        eps_grid_xx = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=1, shift=1))
        eps_grid_yy = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=0, shift=1))
        eps_vec_xx = self._grid_to_vec(eps_grid_xx)
        eps_vec_yy = self._grid_to_vec(eps_grid_yy)
        eps_vec_xx = eps_vec_xx
        eps_vec_yy = eps_vec_yy
        return eps_vec_xx, eps_vec_yy

    def _make_A(self, eps_vec):

        eps_vec_xx, eps_vec_yy = self._grid_average_2d(eps_vec)
        eps_vec_xx_inv = 1 / (eps_vec_xx + 1e-5)  # the 1e-5 is for numerical stability
        eps_vec_yy_inv = 1 / (eps_vec_yy + 1e-5)  # autograd throws 'divide by zero' errors.

        indices_diag = npa.vstack((npa.arange(self.N), npa.arange(self.N)))

        entries_DxEpsy,   indices_DxEpsy   = spsp_mult(self.entries_Dxb, self.indices_Dxb, eps_vec_yy_inv, indices_diag, self.N)
        entires_DxEpsyDx, indices_DxEpsyDx = spsp_mult(entries_DxEpsy, indices_DxEpsy, self.entries_Dxf, self.indices_Dxf, self.N)

        entries_DyEpsx,   indices_DyEpsx   = spsp_mult(self.entries_Dyb, self.indices_Dyb, eps_vec_xx_inv, indices_diag, self.N)
        entires_DyEpsxDy, indices_DyEpsxDy = spsp_mult(entries_DyEpsx, indices_DyEpsx, self.entries_Dyf, self.indices_Dyf, self.N)

        entries_d = 1 / EPSILON_0 * npa.hstack((entires_DxEpsyDx, entires_DyEpsxDy))
        indices_d = npa.hstack((indices_DxEpsyDx, indices_DyEpsxDy))

        entries_diag = MU_0 * self.omega**2 * npa.ones(self.N)

        entries_a = npa.hstack((entries_d, entries_diag))
        indices_a = npa.hstack((indices_d, indices_diag))

        return entries_a, indices_a

    def _solve_fn(self, eps_vec, entries_a, indices_a, Mz_vec):

        b_vec = 1j * self.omega * Mz_vec          # needed so fields are SI units
        Hz_vec = sp_solve(entries_a, indices_a, b_vec)
        eps_vec_xx, eps_vec_yy = self._grid_average_2d(eps_vec)

        # strip out the x and y components of E and find the Hz component
        Ex_vec, Ey_vec = self._Hz_to_Ex_Ey(Hz_vec, eps_vec_xx, eps_vec_yy)

        return Ex_vec, Ey_vec, Hz_vec



class fdfd3D(fdfd):
    """ 3D FDFD class (work in progress) """

    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None):
        self.is_3D = True
        self.omega = omega
        self.dL = dL
        self.npml = npml
        self._setup_bloch_phases(bloch_phases)
        self.eps_r = eps_r
        self.shape = eps_r.shape
        print("eps_r.shape:",eps_r.shape)

    def _grid_average_3d(self, eps_vec):
        eps_grid = self._vec_to_grid(eps_vec)
        eps_grid_xx = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=2, shift=1))
        eps_grid_yy = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=1, shift=1))
        eps_grid_zz = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=0, shift=1))
        eps_vec_xx = self._grid_to_vec(eps_grid_xx)
        eps_vec_yy = self._grid_to_vec(eps_grid_yy)
        eps_vec_zz = self._grid_to_vec(eps_grid_zz)
        eps_vec_xx = eps_vec_xx
        eps_vec_yy = eps_vec_yy
        eps_vec_zz = eps_vec_zz
        vec = npa.hstack((eps_vec_xx, eps_vec_yy, eps_vec_zz))
        return vec

    def _make_A(self, eps_vec):
        curls_e, curls_h = compute_derivative_matrices_3D(self.omega, self.shape, self.npml, self.dL, self.bloch_x, self.bloch_y, self.bloch_z)
        EPS_vec = self._grid_average_3d(eps_vec)
        C = (1 / MU_0 * curls_h.dot(curls_e)) 
        entries_c, indices_c = get_entries_indices(C)
        entries_diag = - EPSILON_0 * self.omega**2 * EPS_vec
        indices_diag = npa.vstack((npa.arange(3*self.N), npa.arange(3*self.N)))
        entries_a = npa.hstack((entries_diag, entries_c))
        indices_a = npa.hstack((indices_diag, indices_c))

        return entries_a, indices_a
        
    def solve(self, source):
        """ Outward facing function (what gets called by user) that takes a source grid and returns the field components """
         
        # flatten the permittivity and source grid
        source_vec = self._grid_to_vec(source)
        print("Shape_source_vec",source_vec.shape)
        eps_vec = self._grid_to_vec(self.eps_r)

        # create the A matrix for this polarization
        entries_a, indices_a = self._make_A(eps_vec)

        # solve field componets usng A and the source
        E_vec = self._solve_fn(entries_a, indices_a, source_vec)
        return E_vec
        
    def _solve_fn(self,entries_a, indices_a, source_vec):
        
        b_vec = 1j * self.omega * source_vec

        E_vec = sp_solve(entries_a, indices_a, b_vec)

        return E_vec
      
    def Vec3Dtogrid(self,Vec):
        N = self.N
        shape = self.shape
        Vecx = npa.zeros(N)
        Vecy = npa.zeros(N)
        Vecz = npa.zeros(N)
        Vecx[:] = Vec[0:N].real
        Vecy[:] = Vec[N:2*N].real
        Vecz[:] = Vec[2*N:3*N].real
        Vec_gridx = npa.reshape(Vecx, self.shape)
        print("ShapeV:",Vec_gridx.shape)
        Vec_gridy = npa.reshape(Vecy, self.shape)
        Vec_gridz = npa.reshape(Vecz, self.shape)
        return Vec_gridx,Vec_gridy,Vec_gridz
