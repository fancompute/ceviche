import autograd.numpy as np
import matplotlib.pylab as plt

from autograd import grad

from ceviche import fdfd_hz
from ceviche.constants import C_0

PLOT = True

# some parameters
wavelengths = [500e-9, 450e-9]#, 650e-9]

H = 2e-6  # height of slab
L = 2e-6  # width of slab

spc = 1e-6   # space between source and PML, source and structure
dL = 20e-9

npml = 20       # number of PML grids
eps_max = 15   # material index

# setup arrays
Nx = int(L / dL)
Ny = 2*npml + 2*int(spc / dL) + int(H / dL)
x_pts = dL * np.arange(Nx)

slab_region = np.zeros((Nx, Ny))
slab_region[:, npml + int(spc / dL):npml + int(spc / dL) + int(H / dL)] = 1

eps_r = np.ones((Nx, Ny))

source = np.zeros((Nx, Ny))
source[:, npml + int(spc / 2 / dL)] = 1

probe_index_y = npml + int(3 * spc / 2 / dL) + int(H / dL)

probe1 = np.zeros((Nx, Ny))
probe1[Nx//3, probe_index_y] = 1
probe2 = np.zeros((Nx, Ny))
probe2[2*Nx//3, probe_index_y] = 1

omega1 = 2 * np.pi * C_0 / wavelengths[0]
fdfd1 = fdfd_hz(omega1, dL, eps_r, source, npml=[0, npml])
Ex, Ey, Hz = fdfd1.solve()
P1 = np.sum(np.abs(Hz) * probe1)

omega2 = 2 * np.pi * C_0 / wavelengths[1]
fdfd2 = fdfd_hz(omega2, dL, eps_r, source, npml=[0, npml])
Ex, Ey, Hz = fdfd2.solve()
P2 = np.sum(np.abs(Hz) * probe2)

if PLOT:
    plt.imshow((probe1 + probe2 + source + slab_region).T)
    plt.show()

def plot_field(fdfd):
    Ex, Ey, Hz = fdfd.solve()
    plt.imshow(np.real(Hz))
    plt.show()

def plot_field_ag(fdfd):
    Ex, Ey, Hz = fdfd.solve()
    plt.imshow(np.abs(Hz._value))
    plt.show()

plot_field(fdfd1)
plot_field(fdfd2)

def objective(eps_arr):

    eps_r = eps_arr.reshape((Nx, Ny))

    fdfd1.eps_r = eps_r
    Ex, Ey, Hz1 = fdfd1.solve()
    power1_1 = np.abs(np.sum(probe1 * Hz1))
    power1_2 = np.abs(np.sum(probe2 * Hz1))

    fdfd2.eps_r = eps_r
    Ex, Ey, Hz2 = fdfd2.solve()
    power2_1 = np.abs(np.sum(probe1 * Hz2))
    power2_2 = np.abs(np.sum(probe2 * Hz2))

    return -power1_1 / P1 - power2_2 / P2# + power1_2 / P1 + power2_1 / P2

# define the gradient for autograd
grad_J = grad(objective)

# optimization loop
NIter = 100
from scipy.optimize import minimize
bounds = [(1, eps_max) if slab_region.flatten()[i] == 1 else (1,1) for i in range(eps_r.size)]
minimize(objective, eps_r, args=(), method='L-BFGS-B', jac=grad_J,
    bounds=bounds, tol=None, callback=None,
    options={'disp': True, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': NIter, 'iprint': -1, 'maxls': 20})



plot_field_ag(fdfd1)
plot_field_ag(fdfd2)
