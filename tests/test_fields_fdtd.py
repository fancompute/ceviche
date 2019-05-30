import unittest
import numpy as np
import matplotlib.pylab as plt

from ceviche import fdtd
from ceviche.sources import Gaussian

class TestFields_FDTD(unittest.TestCase):

    """ Tests the field patterns by inspection """


    def setUp(self):

        # basic simulation parameters
        self.Nx = 80
        self.Ny = 80
        self.Nz = 1

        self.omega = 2*np.pi*200e12
        self.dL = 5e-8
        self.pml = [20, 20, 0]

        # source parameters
        self.steps = 700
        self.t0 = 300
        self.sigma = 20        
        self.source_amp = 2e2
        self.source = np.zeros((self.Nx, self.Ny, self.Nz))
        self.source[self.Nx//2, self.Ny//2, self.Nz//2] = self.source_amp

        # starting relative permittivity (random for debugging)
        self.eps_r   = np.random.random((self.Nx, self.Ny, self.Nz)) + 1
        self.eps_arr = self.eps_r.flatten()

        # simulation parameters
        self.skip_rate = 20

    def test_fields_E(self):

        F = fdtd(self.eps_r, dL=self.dL, npml=self.pml)

        G1 = Gaussian(mask=self.source, component='Jz', amp=self.source_amp, sigma=self.sigma, t0=self.t0)

        F.add_src(G1)

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.pcolormesh(np.zeros((self.Nx, self.Ny)), cmap='RdBu')

        for t_index, fields in enumerate(F.run(self.steps)):

            if t_index % self.skip_rate == 0:

                max_E = np.abs(fields['Ez']).max()
                im.set_array(fields['Ez'][:, :, 0].ravel())
                im.set_clim([-1, 1])
                plt.pause(0.001)
                ax.set_title('time = {}'.format(t_index))

    def test_fields_H(self):

        F = fdtd(self.eps_r, dL=self.dL, npml=self.pml)

        G1 = Gaussian(mask=self.source, component='Jx', amp=self.source_amp, sigma=self.sigma, t0=self.t0)

        F.add_src(G1)

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.pcolormesh(np.zeros((self.Nx, self.Ny)), cmap='RdBu')

        for t_index, fields in enumerate(F.run(self.steps)):

            if t_index % self.skip_rate == 0:

                max_E = np.abs(fields['Hz']).max()
                im.set_array(fields['Hz'][:, :, 0].ravel())
                im.set_clim([-1, 1])
                plt.pause(0.001)
                ax.set_title('time = {}'.format(t_index))

if __name__ == '__main__':
    unittest.main()
