import unittest
import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append('../ceviche')

from ceviche import fdtd

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
        self.source_amp = 10
        self.source_pos = np.zeros((self.Nx, self.Ny, self.Nz))
        self.source_pos[self.Nx//2, self.Ny//2, self.Nz//2] = self.source_amp
        self.gaussian = lambda t: self.source_pos * self.source_amp * np.exp(-(t - self.t0)**2 / 2 / self.sigma**2)

        # starting relative permittivity (random for debugging)
        self.eps_r   = np.random.random((self.Nx, self.Ny, self.Nz)) + 1
        self.eps_arr = self.eps_r.flatten()

        # simulation parameters
        self.skip_rate = 20

    def test_fields_E(self):

        F = fdtd(self.eps_r, dL=self.dL, npml=self.pml)

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.pcolormesh(np.zeros((self.Nx, self.Ny)), cmap='RdBu')

        for t_index in range(self.steps):

            fields = F.forward(Jz=self.gaussian(t_index))

            if t_index % self.skip_rate == 0:

                max_E = np.abs(fields['Ez']).max()
                im.set_array(fields['Ez'][:, :, 0].ravel())
                im.set_clim([-1, 1])
                plt.pause(0.001)
                ax.set_title('time = {}'.format(t_index))

    def test_fields_H(self):

        F = fdtd(self.eps_r, dL=self.dL, npml=self.pml)

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.pcolormesh(np.zeros((self.Nx, self.Ny)), cmap='RdBu')

        for t_index in range(self.steps):

            fields = F.forward(Jx=self.gaussian(t_index))

            if t_index % self.skip_rate == 0:

                max_E = np.abs(fields['Hz']).max()
                im.set_array(fields['Hz'][:, :, 0].ravel())
                im.set_clim([-1, 1])
                plt.pause(0.001)
                ax.set_title('time = {}'.format(t_index))

if __name__ == '__main__':
    unittest.main()
