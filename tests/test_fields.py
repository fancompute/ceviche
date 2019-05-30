import unittest
import numpy as np
import matplotlib.pylab as plt

from ceviche.fdfd import fdfd_ez, fdfd_hz

class TestFields(unittest.TestCase):

    """ Tests the field patterns by inspection """

    def setUp(self):
        self.omega = 2 * np.pi * 200e12  # 200 THz / 1.5 um
        self.dL = 5e-8                   # 50 nanometers
        self.Nx, self.Ny = 141, 141      # grid size
        self.eps_r = np.ones((self.Nx, self.Ny))
        self.source = np.zeros((self.Nx, self.Ny))
        self.source[self.Nx//2, self.Ny//2] = 1
        self.npml = [20, 20]

    def test_Hz(self):
        print('\ttesting Hz')

        F = fdfd_hz(self.omega, self.dL, self.eps_r, self.source, self.npml)
        Ex, Ey, Hz = F.solve()
        Hz_max = np.max(np.abs(Hz))        
        plt.imshow(np.real(Hz), cmap='RdBu', vmin=-Hz_max/5, vmax=Hz_max/5)
        plt.show()

    def test_Ez(self):
        print('\ttesting Ez')

        F = fdfd_ez(self.omega, self.dL, self.eps_r, self.source, self.npml)
        Hx, Hy, Ez = F.solve()
        Ez_max = np.max(np.abs(Ez))
        plt.imshow(np.real(Ez), cmap='RdBu', vmin=-Ez_max/5, vmax=Ez_max/5)
        plt.show()

if __name__ == '__main__':
    unittest.main()
