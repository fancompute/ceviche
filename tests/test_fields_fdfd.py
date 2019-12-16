import unittest
import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append('../ceviche')

from ceviche import fdfd_ez, fdfd_hz

class TestFields_FDFD(unittest.TestCase):

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

        F = fdfd_hz(self.omega, self.dL, self.eps_r, self.npml)
        Ex, Ey, Hz = F.solve(self.source)
        plot_component = Hz
        field_max = np.max(np.abs(plot_component))
        plt.imshow(np.real(plot_component), cmap='RdBu', vmin=-field_max/5, vmax=field_max/5)
        plt.show()

    def test_Ez(self):
        print('\ttesting Ez')

        F = fdfd_ez(self.omega, self.dL, self.eps_r, self.npml)
        Hx, Hy, Ez = F.solve(self.source)
        plot_component = Ez
        field_max = np.max(np.abs(plot_component))
        plt.imshow(np.real(plot_component), cmap='RdBu', vmin=-field_max/5, vmax=field_max/5)
        plt.show()

if __name__ == '__main__':
    unittest.main()
