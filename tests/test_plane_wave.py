import unittest
import numpy as np
import matplotlib.pylab as plt

from ceviche import fdfd_ez, fdfd_hz
from ceviche.constants import *

class TestPlaneWave(unittest.TestCase):

    """ Tests whether a plane wave has the right wavelength """

    def setUp(self):

        # wavelength should be 1.5 um
        wavelength = 1.5e-6
        self.omega = 2 * np.pi * C_0 / wavelength
        self.dL = 1.5e-8

        print('for a wavelength of {} meters \n\
            and grid spacing of {},\n\
            we expect a wavelength of {} grid spaces'.format(wavelength, self.dL, int(wavelength/self.dL)))

        self.Nx, self.Ny = 400, 10    # grid size
        self.eps_r = np.ones((self.Nx, self.Ny))
        self.source = np.zeros((self.Nx, self.Ny))
        self.source[self.Nx//2, :] = 1e-8
        self.npml = [20, 0]

    def test_Hz(self):
        print('\ttesting Hz')

        F = fdfd_hz(self.omega, self.dL, self.eps_r, self.npml)
        Ex, Ey, Hz = F.solve(self.source)
        Hz_max = np.max(np.abs(Hz))        
        plt.imshow(np.real(Hz), cmap='RdBu', vmin=-Hz_max/2, vmax=Hz_max/2)
        plt.show()

    def test_Ez(self):
        print('\ttesting Ez')

        F = fdfd_ez(self.omega, self.dL, self.eps_r, self.npml)
        Hx, Hy, Ez = F.solve(self.source)
        Ez_max = np.max(np.abs(Ez))
        plt.imshow(np.real(Ez), cmap='RdBu', vmin=-Ez_max/2, vmax=Ez_max/2)
        plt.show()


if __name__ == '__main__':
    unittest.main()
