import unittest
import numpy as np
import matplotlib.pylab as plt

# import the FDFD from autotrack/
from ceviche.fdfd import fdfd_hz

# get the automatic differentiation
import autograd.numpy as npa
from autograd import grad

# whether to plot things below
PLOT = False

# make parameters
omega = 2*np.pi*200e12
L0 = 1e-5
eps_max = 2
npml = 10
spc = 10
L = 50
Nx, Ny = 2*npml + 4*spc + L, 2*npml + 4*spc + L
eps_r = np.ones((Nx, Ny))

# make source
source = np.zeros((Nx, Ny))
source[npml+spc, Ny//2] = 1

# make design region
box_region = np.zeros((Nx, Ny))
box_region[npml+2*spc:npml+2*spc+L, npml+2*spc:npml+2*spc+L] = 1

# make the accelration probe
probe = np.zeros((Nx, Ny), dtype=np.complex128)
probe[-npml-spc, Ny//2] = 1

# plot the probe through channel
if PLOT:
    plt.imshow(np.abs(probe + box_region + source).T)
    plt.show()

# vacuum test, get normalization
F = fdfd_hz(omega, L0, eps_r, source, [npml, npml])
Ex, Ey, Hz = F.solve()
E_mag = np.sqrt(np.square(np.abs(Ex)) + np.square(np.abs(Ey)))
H_mag = np.abs(Hz)
I_E0 = np.abs(np.square(np.sum(E_mag * probe)))
I_H0 = np.abs(np.square(np.sum(H_mag * probe)))

print('I_H0 = {} V/m'.format(I_H0))

# plot the vacuum fields
if PLOT:
    plt.imshow(np.real(Hz), cmap='RdBu')
    plt.title('Hz / E0 (<-)')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()

# maximum electric field magnitude in the domain
def Emax(Ex, Ey):
    E_mag = npa.sqrt(npa.square(npa.abs(Ex)) + npa.square(npa.abs(Ey)))
    return npa.max(E_mag)

# average electric field magnitude in the domain
def Eavg(Ex, Ey):
    E_mag = npa.sqrt(npa.square(npa.abs(Ex)) + npa.square(npa.abs(Ey)))
    return npa.mean(E_mag)

# defines the acceleration gradient as a function of the relative permittivity grid
def intensity(eps_arr):

    eps_r = eps_arr.reshape((Nx, Ny))
    # set the permittivity of the FDFD and solve the fields
    F.eps_r = eps_r
    Ex, Ey, Hz = F.solve()

    # compute the gradient and normalize if you want
    I = npa.sum(npa.square(npa.abs(Hz * probe)))
    return -I / I_H0

# define the gradient for autograd
grad_I = grad(intensity)

# optimization loop
# NIter = 40
# step_size = 1e0
# for i in range(NIter):
#     I = intensity(eps_r)
#     print('on iter {} / {}, intensity gradient = {}'.format(i, NIter, I))
#     dg_deps = grad_I(eps_r)
#     eps_r = eps_r + step_size * box_region * dg_deps
#     eps_r[eps_r < 1] = 1
#     eps_r[eps_r > eps_max] = eps_max

from scipy.optimize import minimize
bounds = [(1, eps_max) if box_region.flatten()[i] == 1 else (1,1) for i in range(eps_r.size)]
minimize(intensity, eps_r, args=(), method='L-BFGS-B', jac=grad_I,
    bounds=bounds, tol=None, callback=None,
    options={'disp': True, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 150, 'iprint': -1, 'maxls': 20})

# plot the final permittivity
plt.imshow(F.eps_r._value.T, cmap='nipy_spectral')
plt.colorbar()
plt.show()

Ex, Ey, Hz = F.solve()
plt.imshow(np.real(Hz._value).T, cmap='RdBu')
plt.title('H_z / E0 (<-)')
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()
plt.show()

