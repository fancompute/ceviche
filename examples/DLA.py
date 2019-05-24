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
omega = 2*np.pi*1e2
L0 = 1
Nx, Ny = 400, 40
eps_max = 5
eps_r = np.ones((Nx, Ny))
source = np.zeros((Nx, Ny))
source[30, :] = 10
npml = [20, 0]
spc = 100
gap = 20

# make design region
design_region = np.zeros((Nx, Ny))
design_region[spc:Nx//2-gap//2, :] = 1
design_region[Nx//2+gap//2:Nx-spc, :] = 1

# make the accelration probe
eta = np.zeros((Nx, Ny), dtype=np.complex128)
channel_ys = np.arange(Ny)
eta[Nx//2, :] = np.exp(1j * 2 * np.pi * channel_ys / Ny)

# plot the probe through channel
if PLOT:
    plt.plot(np.real(eta[Nx//2,:]), label='RE\{eta\}')
    plt.xlabel('position along channel (y)')
    plt.ylabel('eta (y)')
    plt.show()

# vacuum test, get normalization
F = fdfd_hz(omega, L0, eps_r, source, npml)
Ex, Ey, Hz = F.solve()
E_mag = np.sqrt(np.square(np.abs(Ex)) + np.square(np.abs(Ey)))
E0 = np.max(E_mag[spc:Nx-spc, :])
print('E0 = {} V/m'.format(E0))

# plot the vacuum fields
if PLOT:
    plt.imshow(np.real(Ey) / E0, cmap='RdBu')
    plt.title('E_y / E0 (<-)')
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
def accel_gradient(eps_r):

    # set the permittivity of the FDFD and solve the fields
    F.eps_r = eps_r
    Ex, Ey, Hz = F.solve()

    # compute the gradient and normalize if you want
    G = npa.sum(Ey * eta / Ny / E0)
    return np.abs(G) / Eavg(Ex, Ey)

# define the gradient for autograd
grad_g = grad(accel_gradient)

# optimization loop
NIter = 40
step_size = 1e-1
for i in range(NIter):
    g = accel_gradient(eps_r)
    print('on iter {} / {}, acceleration gradient = {}'.format(i, NIter, g))
    dg_deps = grad_g(eps_r)
    eps_r = eps_r + step_size * design_region * dg_deps
    eps_r[eps_r < 1] = 1
    eps_r[eps_r > eps_max] = eps_max

# plot the final permittivity
plt.imshow(F.eps_r._value, cmap='nipy_spectral')
plt.colorbar()
plt.show()

Ex, Ey, Hz = F.solve()
plt.imshow(np.real(Ey._value) / E0, cmap='RdBu')
plt.title('E_y / E0 (<-)')
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()
plt.show()
