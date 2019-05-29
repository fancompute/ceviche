import unittest
import numpy as np
import matplotlib.pylab as plt

from ceviche.fdfd import fdfd_ez, fdfd_hz
from ceviche.constants import *

class TestFields(unittest.TestCase):

    """ Tests the field patterns """

    def setUp(self):
        wavelength = 1.5e-6
        self.omega = 2 * np.pi * C_0 / wavelength#200e12
        self.dL = 2e3                  # 1 micron
        self.Nx, self.Ny = 131, 131    # grid size
        self.eps_r = np.ones((self.Nx, self.Ny))
        # self.eps_r[40:60, 40:60] = 10
        self.source = np.zeros((self.Nx, self.Ny))
        self.source[self.Nx//2, :] = 1e-8
        self.npml = [20, 0]

    def test_Hz(self):
        print('\ttesting Hz')

        F = fdfd_hz(self.omega, self.dL, self.eps_r, self.source, self.npml)
        Ex, Ey, Hz = F.solve()
        Hz_max = np.max(np.abs(Hz))        
        plt.imshow(np.real(Hz), cmap='RdBu', vmin=-Hz_max/2, vmax=Hz_max/2)
        plt.show()

    def test_Ez(self):
        print('\ttesting Ez')

        F = fdfd_ez(self.omega, self.dL, self.eps_r, self.source, self.npml)
        Hx, Hy, Ez = F.solve()
        Ez_max = np.max(np.abs(Ez))
        plt.imshow(np.real(Ez), cmap='RdBu', vmin=-Ez_max/2, vmax=Ez_max/2)
        plt.show()

if __name__ == '__main__':
    unittest.main()





