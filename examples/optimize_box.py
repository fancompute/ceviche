import unittest
import numpy as np
import matplotlib.pylab as plt
import autograd.numpy as npa

import sys
sys.path.append('../ceviche')

from ceviche import fdfd_hz, jacobian
from ceviche.utils import imarr

""" Optimize intensity focusing through a box with continuous varying permittivity """

# whether to plot setup stuff
PLOT = False

# make parameters
omega = 2 * np.pi * 200e12  # lambda = 1.5 um
dL = 4e-8
eps_max = 2
npml = 10
spc = 10
L = 3e-6
Nx, Ny = 2*npml + 4*spc + int(L/dL), 2*npml + 4*spc + int(L/dL)
eps_r = np.ones((Nx, Ny))

# make source
source = np.zeros((Nx, Ny))
source[npml+spc, Ny//2] = 1

# make design region
box_region = np.zeros((Nx, Ny))
box_region[npml+2*spc:npml+2*spc+int(L/dL), npml+2*spc:npml+2*spc+int(L/dL)] = 1

# make the accelration probe
probe = np.zeros((Nx, Ny), dtype=np.complex128)
probe[-npml-spc, Ny//2] = 1

# plot the probe through channel
if PLOT:
    plt.imshow(np.abs(imarr(probe + box_region + source)))
    plt.show()

# vacuum test, get normalization
F = fdfd_hz(omega, dL, eps_r, [npml, npml])
Ex, Ey, Hz = F.solve(source)
E_mag = np.sqrt(np.square(np.abs(Ex)) + np.square(np.abs(Ey)))
H_mag = np.abs(Hz)
I_E0 = np.abs(np.square(np.sum(E_mag * probe)))
I_H0 = np.abs(np.square(np.sum(H_mag * probe)))

print('I_H0 = {}'.format(I_H0))

# plot the vacuum fields
if PLOT:
    plt.imshow(np.real(imarr(Hz)), cmap='RdBu')
    plt.title('real(Hz)')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()

# defines the intensity on the other side of the box as a function of the relative permittivity grid
def intensity(eps_arr):

    eps_r = eps_arr.reshape((Nx, Ny))
    # set the permittivity of the FDFD and solve the fields
    F.eps_r = eps_r
    Ex, Ey, Hz = F.solve(source)

    # compute the gradient and normalize if you want
    I = npa.sum(npa.square(npa.abs(Hz * probe)))
    return -I / I_H0

# define the gradient for autograd
grad_I = jacobian(intensity, mode='reverse')

# initialize the design region with some eps
eps_r[box_region == 1] = eps_max

from scipy.optimize import minimize
bounds = [(1, eps_max) if box_region.flatten()[i] == 1 else (1,1) for i in range(eps_r.size)]

minimize(intensity, eps_r, args=(), method='L-BFGS-B', jac=grad_I,
    bounds=bounds, tol=None, callback=None,
    options={'disp': True,
             'maxcor': 10,
             'ftol': 2.220446049250313e-09,
             'gtol': 1e-05,
             'eps': 1e-08,
             'maxfun': 15000,
             'maxiter': 100,
             'iprint': -1,
             'maxls': 20})

# plot the final permittivity
plt.imshow(imarr(F.eps_r), cmap='nipy_spectral')
plt.colorbar()
plt.show()

# plot the fields
Ex, Ey, Hz = F.solve(source)
plt.imshow(np.real(imarr(Hz)), cmap='RdBu')
plt.title('real(H_z)')
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()
plt.show()

