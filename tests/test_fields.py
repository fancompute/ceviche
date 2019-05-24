import unittest
import numpy as np
import matplotlib.pylab as plt

from ceviche.fdfd import fdfd_ez, fdfd_hz

class TestFields(unittest.TestCase):

    """ Tests the field patterns """

    def setUp(self):
        self.omega = 2*np.pi*200e12
        self.dL = 1e-5                  # 1 micron
        self.Nx, self.Ny = 131, 101    # grid size
        self.eps_r = np.ones((self.Nx, self.Ny))
        # self.eps_r[40:60, 40:60] = 5
        self.source = np.zeros((self.Nx, self.Ny))
        self.source[self.Nx//2, self.Ny//2] = 1
        self.npml = [20, 20]

    def test_Hz(self):
        print('\ttesting Hz')

        F = fdfd_hz(self.omega, self.dL, self.eps_r, self.source, self.npml)
        Ex, Ey, Hz = F.solve()
        plt.imshow(np.real(Hz))
        plt.show()

    def test_Ez(self):
        print('\ttesting Ez')

        F = fdfd_ez(self.omega, self.dL, self.eps_r, self.source, self.npml)
        Hx, Hy, Ez = F.solve()
        plt.imshow(np.real(Ez))
        plt.show()

if __name__ == '__main__':
    unittest.main()
    omega = 2*np.pi*200*1e12  # 200 THz
    L0 = 1e-6                 # 1 micron
    Nx, Ny = 101, 101    # grid size
    eps_r = np.ones((Nx, Ny))
    source = np.zeros((Nx, Ny))
    source[Nx//2, Ny//2] = 1
    npml = [10, 10]    
    F = fdfd_ez(omega, L0, eps_r, source, npml)
    # plt.spy(F.A, markersize=0.1)
    # plt.show()




