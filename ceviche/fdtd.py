import autograd.numpy as np
import autograd.numpy as npa
# import numpy as np
from copy import copy, deepcopy
from ceviche.constants import *
from ceviche.derivs_fdtd import curl_E_numpy as curl_E
from ceviche.derivs_fdtd import curl_H_numpy as curl_H

class FDTD():

    def __init__(self, eps_r, dL, npml):
        """ Makes an FDTD object
                eps_r: the relative permittivity (array > 1)
                    if eps_r.shape = 3, it holds a single permittivity
                    if eps_r.shape = 4, the last index is the batch index (running several simulations at once)
                dL: the grid size(s) (float/int or list of 3 floats/ints for dx, dy, dz)
                NPML: the number of PML grids in each dimension (list of 3 ints)
                chi3: the 3rd order nonlinear susceptibility (3 dimensional array))                
        """

        self.dL = dL
        self._set_time_step()

        self.npml = npml
        self.Nx, self.Ny, self.Nz = eps_r.shape
        self._compute_sigmas()

        self.eps_r = eps_r

        # source lists
        self.sources = {'Jx': [], 'Jy': [], 'Jz': []}

    def __repr__(self):
        return "FDTD(eps_r.shape={}, dL={}, NPML={})".format(self.grid_shape, self.dL, self.npml)

    def __str__(self):
        if self.sources:
            source_str = '[\n\t\t   ' + '\n\t\t   '.join([str(s) for s in self.sources]) + '\n\t]'
        else:
            source_str = '[]'
        return "FDTD object:\n\tdomain size = {}\n\tdL = {}\n\tNPML = {}\n\tbatches = {}\n\tsources = {}".format(self.grid_shape[:3], self.dL, self.npml, self.n_batches, source_str)

    @property
    def eps_r(self):
        """ Returns the relative permittivity grid """
        return self.__eps_r

    @eps_r.setter
    def eps_r(self, new_eps):
        """ Defines some attributes when eps_r is set. """
        self.__eps_r = new_eps
        self.eps_xx, self.eps_yy, self.eps_zz = self._grid_center_to_xyz(self.__eps_r)
        self.eps_arr = self.__eps_r.flatten()
        self.N = self.eps_arr.size
        self.grid_shape = self.Nx, self.Ny, self.Nz = self.__eps_r.shape
        self._compute_update_parameters()
        self.initialize_fields()

    def run(self, steps):
        """ Generator that runs the FDTD forward `steps` time steps """

        for _ in range(steps):
            yield self.fields
            self.forward()

    def forward(self):
        """ one time step of FDTD """

        self.t_index += 1

        # get curls of E
        CEx = curl_E(0, self.Ex, self.Ey, self.Ez, self.dL, self.dL, self.dL)
        CEy = curl_E(1, self.Ex, self.Ey, self.Ez, self.dL, self.dL, self.dL)
        CEz = curl_E(2, self.Ex, self.Ey, self.Ez, self.dL, self.dL, self.dL)

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
        CHx = curl_H(0, self.Hx, self.Hy, self.Hz, self.dL, self.dL, self.dL)
        CHy = curl_H(1, self.Hx, self.Hy, self.Hz, self.dL, self.dL, self.dL)
        CHz = curl_H(2, self.Hx, self.Hy, self.Hz, self.dL, self.dL, self.dL)

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

        # update field dict
        self.fields['Dx'] = self.Dx
        self.fields['Dy'] = self.Dy
        self.fields['Dz'] = self.Dz

        # compute intensity
        # I = self._compute_intensity(self.Ex, self.Ey, self.Ez)

        # update the E fields
        self.Ex = self.mEx1 * self.Dx 
        self.Ey = self.mEy1 * self.Dy
        self.Ez = self.mEz1 * self.Dz           

        # add sources to the electric fields
        self._inject_sources()

        # update field dict
        self.fields['Ex'] = self.Ex
        self.fields['Ey'] = self.Ey
        self.fields['Ez'] = self.Ez

    def _inject_sources(self):
        """ Injects the current sources into the simulation as stored in self.sources
                Note: would like to avoid for loops here eventually..
        """

        for source in self.sources['Jx']:
            self.Ex = self.Ex + source.J_fn(self.t_index)
        for source in self.sources['Jy']:
            self.Ey = self.Ey + source.J_fn(self.t_index)
        for source in self.sources['Jz']:
            self.Ez = self.Ez + source.J_fn(self.t_index)            

    def add_src(self, newSource):
        """ adds a Source object to the simulation """

        # copies the old J(t) -> np.array function and adds the new source's contribution. (hacky, but works)
        if newSource.component == 'Jx':
            self.sources['Jx'].append(newSource)
        elif newSource.component == 'Jy':
            self.sources['Jy'].append(newSource)
        elif newSource.component == 'Jz':
            self.sources['Jz'].append(newSource)

    def _set_time_step(self, stability_factor=0.5):
        """ Set the time step based on the generalized Courant stability condition
                Delta T < 1 / C_0 / sqrt(1 / dx^2 + 1/dy^2 + 1/dz^2)
                dt = courant_condition * stability_factor, so stability factor should be < 1
        """

        dL_sum = 3 / self.dL ** 2
        dL_avg = 1 / npa.sqrt(dL_sum)
        courant_stability = dL_avg / C_0
        self.dt = courant_stability * stability_factor

    @staticmethod
    def _grid_center_to_xyz(Q_mid, averaging=True):
        """ Computes the interpolated value of the quantity Q_mid felt at the Ex, Ey, Ez positions of the Yee latice
            Returns these three components
        """

        # initialize
        Q_xx = copy(Q_mid)
        Q_yy = copy(Q_mid)
        Q_zz = copy(Q_mid)

        # if averaging, set the respective xx, yy, zz components to the midpoint in the Yee lattice.
        if averaging:

            # get the value from the middle of the next cell over
            Q_x_r = npa.roll(Q_mid, shift=1, axis=0)
            Q_y_r = npa.roll(Q_mid, shift=1, axis=1)
            Q_z_r = npa.roll(Q_mid, shift=1, axis=2)

            # average with the two middle values
            Q_xx = (Q_mid + Q_x_r)/2
            Q_yy = (Q_mid + Q_y_r)/2
            Q_zz = (Q_mid + Q_z_r)/2

        return Q_xx, Q_yy, Q_zz

    @staticmethod
    def _grid_xyz_to_center(Q_xx, Q_yy, Q_zz):
        """ Computes the interpolated value of the quantitys Q_xx, Q_yy, Q_zz at the center of Yee latice
            Returns these three components
        """

        # compute the averages
        Q_xx_avg = (Q_xx.astype('float') + npa.roll(Q_xx, shift=1, axis=0))/2
        Q_yy_avg = (Q_yy.astype('float') + npa.roll(Q_yy, shift=1, axis=1))/2
        Q_zz_avg = (Q_zz.astype('float') + npa.roll(Q_zz, shift=1, axis=2))/2

        return Q_xx_avg, Q_yy_avg, Q_zz_avg

    def compute_dmEs(self, region):
        """ computes the derivative of the mE{x,y,z}1 update coefficients for this FDTD object
                region: binary numpy array of same shape as self.eps_r 
                        is 1 where the parameter perturbs the permittivity
        """

        assert self.grid_shape == region.shape, "region shape must be the same as the grid shape of F, given {} and {} respectively".format(self.grid_shape, region.shape)

        # snap to yee grid
        dx, dy, dz = self._grid_center_to_xyz(region)
        dmEx1 = -dx / self.eps_xx / self.eps_xx
        dmEy1 = -dy / self.eps_yy / self.eps_yy
        dmEz1 = -dz / self.eps_zz / self.eps_zz

        return dmEx1, dmEy1, dmEz1

    def _compute_intensity(self, Ex, Ey, Ez):
        """ Returns the electric intensity averaged at the yee cell center.
        """

        # note, might want to grid xyz to center this one.
        return npa.square(Ex) + npa.square(Ey) + npa.square(Ez)

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
                       'Hz': npa.zeros(self.grid_shape),
                       'I' : npa.zeros(self.grid_shape),
                      }

if __name__ == '__main__':

    """ Simple use of ceviche, running an FDTD simulation """

    from plot import animate_field
    from sources import Gaussian, CW
    import matplotlib.pylab as plt
    from time import time

    # make a simulation with a given permittivity distribution
    eps_r = np.ones((100,100,1,1))
    eps_r[30:70, 30:70, 0, 0] = 1
    F = FDTD(eps_r, dL=1e-8, NPML=[20, 20, 0], deriv='numpy')

    source_loc1 = np.zeros(F.grid_shape)
    source_loc1[40:60, 50, 0, :] = 2
    G1 = Gaussian(mask=source_loc1, component='Jz', amp=10, sigma=20, t0=100)

    source_loc2 = np.zeros(F.grid_shape)
    source_loc2[50, 40:60, 0, 0] = 1
    G2 = Gaussian(mask=source_loc2, component='Jz', amp=-10, sigma=20, t0=100)

    source_loc3 = np.zeros(F.grid_shape)
    source_loc3[50, 50, 0, 0] = 1
    CW = CW(mask=source_loc3, component='Jz', amp=1, t0=400, freq=1/20)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.pcolormesh(np.zeros(F.grid_shape)[:, :, 0, 0],  cmap='RdBu')

    F.add_src(G1)
    F.add_src(G2)

    storage = np.zeros((3, 100, 100, 1000))
    t = time()
    # run the fields over time
    for t_index, fields in enumerate(F.run(steps=1000)):
        storage[0, :, :, t_index] = fields['Ex'][:,:,0,0]
        storage[1, :, :, t_index] = fields['Ey'][:,:,0,0]
        storage[2, :, :, t_index] = fields['Ez'][:,:,0,0]    
        # plot first simulation
        # animate_field(ax, im, fields, t_index, component='Ez', skip=10, z_index=0, batch_index=0, amp=0.2)
    print('took {} seconds'.format(time() - t))
    # F.initialize_fields()

    # # run the fields over time
    # for t_index, fields in enumerate(F.run(steps=400)):
    #     # plot second simulation
    #     animate_field(ax, im, fields, t_index, component='Ez', skip=10, z_index=0, batch_index=1, amp=0.4)
