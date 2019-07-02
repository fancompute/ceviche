import unittest
import numpy as np
import matplotlib.pylab as plt

from ceviche import fdfd_hz

import autograd.numpy as npa
from autograd import grad
from ceviche.parameterizations import Circle_Shapes

# whether to plot setup stuff
PLOT = True

# make parameters
omega = 2 * np.pi * 200e12  # lambda = 1.5 um
dL = 4e-8
eps_max = 2
npml = 10
spc = 10
L = 3e-6
Nx, Ny = 2*npml + 4*spc + int(L/dL), 2*npml + 4*spc + int(L/dL)
eps_r = np.ones((Nx, Ny))

# # 9 holes
# xh = dL*np.array([-20, 0, 20])
# yh = dL*np.array([-20, 0, 20])

# 16 holes
xh = dL*np.linspace(-30, 30, 4)
yh = dL*np.linspace(-30, 30, 4)

(xhm, yhm) = np.meshgrid(xh, yh)
rhm = dL*5*np.ones((xhm.size))
ehm = np.ones((xhm.size))
params = np.array([xhm.ravel(), yhm.ravel(), rhm, ehm])

# make source
source = np.zeros((Nx, Ny))
source[npml+spc, Ny//2] = 1

# make design region and background epsilon
box_region = np.zeros((Nx, Ny))
box_region[npml+2*spc:npml+2*spc+int(L/dL), npml+2*spc:npml+2*spc+int(L/dL)] = 1
eps_background = np.ones((Nx, Ny))
eps_background[box_region == 1] = eps_max

# plot the initial permittivity for debugging
if PLOT:
    eps_init = Circle_Shapes.get_eps(params, eps_background, dL)
    plt.imshow(eps_init, cmap='gray')
    plt.colorbar()
    plt.show()

# make the accelration probe
probe = np.zeros((Nx, Ny), dtype=np.complex128)
probe[-npml-spc, Ny//2] = 1

# plot the probe through channel
if PLOT:
    plt.imshow(np.abs(probe + box_region + source).T)
    plt.show()

# vacuum test, get normalization
F = fdfd_hz(omega, dL, eps_r, [npml, npml])
Ex, Ey, Hz = F.solve(source)
E_mag = np.sqrt(np.square(np.abs(Ex)) + np.square(np.abs(Ey)))
H_mag = np.abs(Hz)
I_E0 = np.abs(np.square(np.sum(E_mag * probe)))
I_H0 = np.abs(np.square(np.sum(H_mag * probe)))

print('I_H0 = {}'.format(I_H0))

# plot the startig fields
if PLOT:
    plt.imshow(np.real(Hz).T, cmap='RdBu')
    plt.title('real(Hz)')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()

# defines the intensity on the other side of the box as a function of the relative permittivity grid
def intensity(params):

    eps_r = Circle_Shapes.get_eps(params.reshape((4, params.size//4)), eps_background, dL)
    # set the permittivity of the FDFD and solve the fields
    F.eps_r = eps_r
    Ex, Ey, Hz = F.solve(source)

    # compute the gradient and normalize if you want
    I = npa.sum(npa.square(npa.abs(Hz * probe)))
    return -I / I_H0

# define the gradient for autograd
grad_I = grad(intensity)

# from ceviche.optimizers import adam_minimize
# # bounds = [(1, eps_max) if box_region.flatten()[i] == 1 else (1,1) for i in range(eps_r.size)]
# of_list = adam_minimize(intensity, params, jac=grad_I, step_size=dL/4, Nsteps=100,
#     bounds=None, options={'disp': True})
# plt.plot(-np.array(of_list))
# plt.show()

from scipy.optimize import minimize
# bounds = [(1, eps_max) if box_region.flatten()[i] == 1 else (1,1) for i in range(eps_r.size)]
minimize(intensity, params.ravel(), args=(), method='L-BFGS-B', jac=grad_I,
    bounds=None, tol=None, callback=None,
    options={'disp': True, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 
    'eps': 1e-08, 'maxfun': 15000, 'maxiter': 20, 'iprint': -1, 'maxls': 18})


# plot the final permittivity
plt.imshow(F.eps_r._value.T, cmap='nipy_spectral')
plt.colorbar()
plt.show()

# plot the fields
Ex, Ey, Hz = F.solve(source)
plt.imshow(np.real(Hz._value).T, cmap='RdBu')
plt.title('real(H_z)')
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()
plt.show()

