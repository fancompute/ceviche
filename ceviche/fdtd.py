import numpy as np
import autograd.numpy as npa

from copy import copy, deepcopy

from .constants import *
from .utils import reshape_to_ND, grid_center_to_xyz, grid_xyz_to_center
from .derivatives import curl_E, curl_H

class fdtd():

    def __init__(self, eps_r, dL, npml):
        """ Makes an FDTD object
                eps_r: the relative permittivity (array > 1)
                    if eps_r.shape = 3, it holds a single permittivity
                    if eps_r.shape = 4, the last index is the batch index (running several simulations at once)
                dL: the grid size(s) (float/int or list of 3 floats/ints for dx, dy, dz)
                npml: the number of PML grids in each dimension (list of 3 ints)
        """

        # set the grid shape
        eps_r = reshape_to_ND(eps_r, N=3)
        self.Nx, self.Ny, self.Nz = self.grid_shape = eps_r.shape

        # set the attributes
        self.dL = dL
        self.npml = npml
        self.eps_r = eps_r

    def __repr__(self):
        return "FDTD(eps_r.shape={}, dL={}, NPML={})".format(self.grid_shape, self.dL, self.npml)

    def __str__(self):
        return "FDTD object:\n\tdomain size = {}\n\tdL = {}\n\tNPML = {}".format(self.grid_shape, self.dL, self.npml)

    @property
    def dL(self):
        """ Returns the grid size """
        return self.__dL

    @dL.setter
    def dL(self, new_dL):
        """ Resets the time step when dL is set. """
        self.__dL = new_dL
        self._set_time_step()

    @property
    def npml(self):
        """ Returns the pml grid size list """
        return self.__npml

    @npml.setter
    def npml(self, new_npml):
        """ Defines some attributes when npml is set. """
        self.__npml = new_npml
        self._compute_sigmas()

    @property
    def eps_r(self):
        """ Returns the relative permittivity grid """
        return self.__eps_r

    @eps_r.setter
    def eps_r(self, new_eps):
        """ Defines some attributes when eps_r is set. """
        self.__eps_r = new_eps
        self.eps_xx, self.eps_yy, self.eps_zz = grid_center_to_xyz(self.__eps_r)
        self.eps_arr = self.__eps_r.flatten()
        self.N = self.eps_arr.size
        self.grid_shape = self.Nx, self.Ny, self.Nz = self.__eps_r.shape
        self._compute_update_parameters()
        self.initialize_fields()

    def forward(self, Jx=None, Jy=None, Jz=None):
        """ one time step of FDTD """

        self.t_index += 1

        # get curls of E
        CEx = curl_E(0, self.Ex, self.Ey, self.Ez, self.dL)
        CEy = curl_E(1, self.Ex, self.Ey, self.Ez, self.dL)
        CEz = curl_E(2, self.Ex, self.Ey, self.Ez, self.dL)

        # update the curl E integrals
        self.ICEx = self.ICEx + CEx
        self.ICEy = self.ICEy + CEy
        self.ICEz = self.ICEz + CEz

        # update the H field integrals
        self.IHx = self.IHx + self.Hx
        self.IHy = self.IHy + self.Hy
        self.IHz = self.IHz + self.Hz

        # update the H fields
        self.Hx = self.mHx1 * self.Hx + self.mHx2 * CEx + self.mHx3 * self.ICEx + self.mHx4 * self.IHx
        self.Hy = self.mHy1 * self.Hy + self.mHy2 * CEy + self.mHy3 * self.ICEy + self.mHy4 * self.IHy
        self.Hz = self.mHz1 * self.Hz + self.mHz2 * CEz + self.mHz3 * self.ICEz + self.mHz4 * self.IHz

        # update fields dict
        self.fields['Hx'] = self.Hx
        self.fields['Hy'] = self.Hy
        self.fields['Hz'] = self.Hz

        # get curls of H
        CHx = curl_H(0, self.Hx, self.Hy, self.Hz, self.dL)
        CHy = curl_H(1, self.Hx, self.Hy, self.Hz, self.dL)
        CHz = curl_H(2, self.Hx, self.Hy, self.Hz, self.dL)

        # update the curl E integrals
        self.ICHx = self.ICHx + CHx
        self.ICHy = self.ICHy + CHy
        self.ICHz = self.ICHz + CHz

        # update the D field integrals
        self.IDx = self.IDx + self.Dx
        self.IDy = self.IDy + self.Dy
        self.IDz = self.IDz + self.Dz    

        # update the D fields
        self.Dx = self.mDx1 * self.Dx + self.mDx2 * CHx + self.mDx3 * self.ICHx + self.mDx4 * self.IDx
        self.Dy = self.mDy1 * self.Dy + self.mDy2 * CHy + self.mDy3 * self.ICHy + self.mDy4 * self.IDy
        self.Dz = self.mDz1 * self.Dz + self.mDz2 * CHz + self.mDz3 * self.ICHz + self.mDz4 * self.IDz

        # add sources to the electric fields
        self.Dx += 0 if Jx is None else Jx
        self.Dy += 0 if Jy is None else Jy
        self.Dz += 0 if Jz is None else Jz

        # update field dict
        self.fields['Dx'] = self.Dx
        self.fields['Dy'] = self.Dy
        self.fields['Dz'] = self.Dz

        # update the E fields
        self.Ex = self.mEx1 * self.Dx 
        self.Ey = self.mEy1 * self.Dy
        self.Ez = self.mEz1 * self.Dz           

        # update field dict
        self.fields['Ex'] = self.Ex
        self.fields['Ey'] = self.Ey
        self.fields['Ez'] = self.Ez

        return self.fields


    def initialize_fields(self):
        """ Initializes:
              - the H, D, and E fields for updating
              - the integration terms needed to deal with PML
              - the curls of the fields
        """

        self.t_index = 0

        # magnetic fields
        self.Hx = npa.zeros(self.grid_shape)
        self.Hy = npa.zeros(self.grid_shape)
        self.Hz = npa.zeros(self.grid_shape)

        # E field curl integrals
        self.ICEx = npa.zeros(self.grid_shape)
        self.ICEy = npa.zeros(self.grid_shape)
        self.ICEz = npa.zeros(self.grid_shape)

        # H field integrals
        self.IHx = npa.zeros(self.grid_shape)
        self.IHy = npa.zeros(self.grid_shape)
        self.IHz = npa.zeros(self.grid_shape)

        # E field curls
        self.CEx = npa.zeros(self.grid_shape)
        self.CEy = npa.zeros(self.grid_shape)
        self.CEz = npa.zeros(self.grid_shape)

        # H field curl integrals
        self.ICHx = npa.zeros(self.grid_shape)
        self.ICHy = npa.zeros(self.grid_shape)
        self.ICHz = npa.zeros(self.grid_shape)

        # D field integrals
        self.IDx = npa.zeros(self.grid_shape)
        self.IDy = npa.zeros(self.grid_shape)
        self.IDz = npa.zeros(self.grid_shape)

        # H field curls
        self.CHx = npa.zeros(self.grid_shape)
        self.CHy = npa.zeros(self.grid_shape)
        self.CHz = npa.zeros(self.grid_shape)

        # electric displacement fields
        self.Dx = npa.zeros(self.grid_shape)
        self.Dy = npa.zeros(self.grid_shape)
        self.Dz = npa.zeros(self.grid_shape)

        # electric fields
        self.Ex = npa.zeros(self.grid_shape)
        self.Ey = npa.zeros(self.grid_shape)
        self.Ez = npa.zeros(self.grid_shape)

        # field dictionary to return layer
        self.fields = {'Ex': npa.zeros(self.grid_shape),
                       'Ey': npa.zeros(self.grid_shape),
                       'Ez': npa.zeros(self.grid_shape), 
                       'Dx': npa.zeros(self.grid_shape),
                       'Dy': npa.zeros(self.grid_shape),
                       'Dz': npa.zeros(self.grid_shape),
                       'Hx': npa.zeros(self.grid_shape),
                       'Hy': npa.zeros(self.grid_shape),
                       'Hz': npa.zeros(self.grid_shape)
                      }

    def _set_time_step(self, stability_factor=0.5):
        """ Set the time step based on the generalized Courant stability condition
                Delta T < 1 / C_0 / sqrt(1 / dx^2 + 1/dy^2 + 1/dz^2)
                dt = courant_condition * stability_factor, so stability factor should be < 1
        """

        dL_sum = 3 / self.dL ** 2
        dL_avg = 1 / npa.sqrt(dL_sum)
        courant_stability = dL_avg / C_0
        self.dt = courant_stability * stability_factor

    def _compute_sigmas(self):
        """ Computes sigma tensors for PML """

        # initialize sigma matrices on the full 2X grid

        grid_shape_2X = (2 * self.Nx, 2 * self.Ny, 2 * self.Nz)
        sigx2 = np.zeros(grid_shape_2X)
        sigy2 = np.zeros(grid_shape_2X)
        sigz2 = np.zeros(grid_shape_2X)

        # sigma vector in the X direction
        for nx in range(2 * self.npml[0]):
            nx1 = 2 * self.npml[0] - nx + 1
            nx2 = 2 * self.Nx - 2 * self.npml[0] + nx            
            sigx2[nx1, :, :] = (0.5 * EPSILON_0 / self.dt) * (nx / 2 / self.npml[0])**3
            sigx2[nx2, :, :] = (0.5 * EPSILON_0 / self.dt) * (nx / 2 / self.npml[0])**3

        # sigma arrays in the Y direction
        for ny in range(2 * self.npml[1]):
            ny1 = 2 * self.npml[1] - ny + 1
            ny2 = 2 * self.Ny - 2 * self.npml[1] + ny
            sigy2[:, ny1, :] = (0.5 * EPSILON_0 / self.dt) * (ny / 2 / self.npml[1])**3
            sigy2[:, ny2, :] = (0.5 * EPSILON_0 / self.dt) * (ny / 2 / self.npml[1])**3

        # sigma arrays in the Z direction
        for nz in range(2 * self.npml[2]):
            nz1 = 2 * self.npml[2] - nz + 1
            nz2 = 2 * self.Nz - 2 * self.npml[2] + nz
            sigz2[:, :, nz1] = (0.5 * EPSILON_0 / self.dt) * (nz / 2 / self.npml[2])**3
            sigz2[:, :, nz2] = (0.5 * EPSILON_0 / self.dt) * (nz / 2 / self.npml[2])**3            

        # # PML tensors for H field
        self.sigHx = sigx2[1::2,  ::2,  ::2]
        self.sigHy = sigy2[ ::2, 1::2,  ::2]
        self.sigHz = sigz2[ ::2,  ::2, 1::2]

        # # PML tensors for D field
        self.sigDx = sigx2[ ::2, 1::2, 1::2]
        self.sigDy = sigy2[1::2,  ::2, 1::2]
        self.sigDz = sigz2[1::2, 1::2,  ::2]

    def _compute_update_parameters(self, mu_r=1.0):
        """ Computes update coefficients based on values computed earlier.
                For more details, see http://emlab.utep.edu/ee5390fdtd/Lecture%2014%20--%203D%20Update%20Equations%20with%20PML.pdf
                NOTE: relative permeability set = 1 for now
        """

        # H field update coefficients
        self.mHx0 = (1 / self.dt + (self.sigHy + self.sigHz) / 2 / EPSILON_0 + self.sigHy * self.sigHz * self.dt / 4 / EPSILON_0**2)
        self.mHy0 = (1 / self.dt + (self.sigHx + self.sigHz) / 2 / EPSILON_0 + self.sigHx * self.sigHz * self.dt / 4 / EPSILON_0**2)
        self.mHz0 = (1 / self.dt + (self.sigHx + self.sigHy) / 2 / EPSILON_0 + self.sigHx * self.sigHy * self.dt / 4 / EPSILON_0**2)

        self.mHx1 = (1 / self.mHx0 * (1/self.dt - (self.sigHy + self.sigHz) / 2 / EPSILON_0 - self.sigHy * self.sigHz * self.dt / 4 / EPSILON_0**2))
        self.mHy1 = (1 / self.mHy0 * (1/self.dt - (self.sigHx + self.sigHz) / 2 / EPSILON_0 - self.sigHx * self.sigHz * self.dt / 4 / EPSILON_0**2))
        self.mHz1 = (1 / self.mHz0 * (1/self.dt - (self.sigHx + self.sigHy) / 2 / EPSILON_0 - self.sigHx * self.sigHy * self.dt / 4 / EPSILON_0**2))

        self.mHx2 = (-1 / self.mHx0 * C_0 / mu_r)
        self.mHy2 = (-1 / self.mHy0 * C_0 / mu_r)
        self.mHz2 = (-1 / self.mHz0 * C_0 / mu_r)

        self.mHx3 = (-1 / self.mHx0 * C_0 * self.dt * self.sigHx / EPSILON_0 / mu_r)
        self.mHy3 = (-1 / self.mHy0 * C_0 * self.dt * self.sigHy / EPSILON_0 / mu_r)
        self.mHz3 = (-1 / self.mHz0 * C_0 * self.dt * self.sigHz / EPSILON_0 / mu_r)

        self.mHx4 = (-1 / self.mHx0 * self.dt * self.sigHy * self.sigHz / EPSILON_0**2)
        self.mHy4 = (-1 / self.mHy0 * self.dt * self.sigHx * self.sigHz / EPSILON_0**2)
        self.mHz4 = (-1 / self.mHz0 * self.dt * self.sigHx * self.sigHy / EPSILON_0**2)

        # D field update coefficients
        self.mDx0 = (1 / self.dt + (self.sigDy + self.sigDz) / 2 / EPSILON_0 + self.sigDy * self.sigDz * self.dt / 4 / EPSILON_0**2)
        self.mDy0 = (1 / self.dt + (self.sigDx + self.sigDz) / 2 / EPSILON_0 + self.sigDx * self.sigDz * self.dt / 4 / EPSILON_0**2)
        self.mDz0 = (1 / self.dt + (self.sigDx + self.sigDy) / 2 / EPSILON_0 + self.sigDx * self.sigDy * self.dt / 4 / EPSILON_0**2)

        self.mDx1 = (1 / self.mDx0 * (1/self.dt - (self.sigDy + self.sigDz) / 2 / EPSILON_0 - self.sigDy * self.sigDz * self.dt / 4 / EPSILON_0**2))
        self.mDy1 = (1 / self.mDy0 * (1/self.dt - (self.sigDx + self.sigDz) / 2 / EPSILON_0 - self.sigDx * self.sigDz * self.dt / 4 / EPSILON_0**2))
        self.mDz1 = (1 / self.mDz0 * (1/self.dt - (self.sigDx + self.sigDy) / 2 / EPSILON_0 - self.sigDx * self.sigDy * self.dt / 4 / EPSILON_0**2))

        self.mDx2 = (1 / self.mDx0 * C_0)
        self.mDy2 = (1 / self.mDy0 * C_0)
        self.mDz2 = (1 / self.mDz0 * C_0)

        self.mDx3 = (1 / self.mDx0 * C_0 * self.dt * self.sigDx / EPSILON_0)
        self.mDy3 = (1 / self.mDy0 * C_0 * self.dt * self.sigDy / EPSILON_0)
        self.mDz3 = (1 / self.mDz0 * C_0 * self.dt * self.sigDz / EPSILON_0)

        self.mDx4 = (-1 / self.mDx0 * self.dt * self.sigDy * self.sigDz / EPSILON_0**2)
        self.mDy4 = (-1 / self.mDy0 * self.dt * self.sigDx * self.sigDz / EPSILON_0**2)
        self.mDz4 = (-1 / self.mDz0 * self.dt * self.sigDx * self.sigDy / EPSILON_0**2)

        # D -> E update coefficients
        self.mEx1 = (1 / self.eps_xx)
        self.mEy1 = (1 / self.eps_yy)
        self.mEz1 = (1 / self.eps_zz)
