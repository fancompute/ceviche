import unittest
import numpy as np
import matplotlib.pylab as plt
import autograd.numpy as npa
from autograd import grad

import sys
sys.path.append('../ceviche')

from ceviche.fdfd import fdfd_ez, fdfd_ez_nl
from ceviche.jacobians import jacobian
from ceviche.utils import imarr, get_value

""" NONLINEAR SWITCH 
    
    This is the same as the box.py example.
    However, now we have a linear + nonlinear simulation.
    The objective is to maximize the intensity felt at the probe for the LINEAR sim.
    While minimizing the intensity for the NONLINEAR simulation.

    The nonlinearity is modelled as a kerr effect.
    Where nonlinearity strength is proportional to material density.

"""

# whether to plot setup stuff
PLOT = False

# make parameters
omega = 2 * np.pi * 200e12  # lambda = 1.5 um
dL = 10e-8
eps_max = 2
npml = 10
spc = 10
L = 3e-6
Nx, Ny = 2*npml + 4*spc + int(L/dL), 2*npml + 4*spc + int(L/dL)
eps_r = np.ones((Nx, Ny))

# the nonlinear epsilon to use
chi3 = 2e8
def eps_nl(Ez):
    density = (eps_max - eps_r)/ (eps_max - 1)
    return eps_r + density * 3 * chi3 * np.square(np.abs(Ez))

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
    plt.imshow(np.abs(probe + box_region + source).T)
    plt.show()

# vacuum test, get normalization
F_lin = fdfd_ez(omega, dL, eps_r, [npml, npml])
F_nl = fdfd_ez_nl(omega, dL, eps_nl, [npml, npml])

Hx, Hy, Ez = F_lin.solve(source)
H_mag = np.sqrt(np.square(np.abs(Hx)) + np.square(np.abs(Hy)))
E_mag = np.abs(Ez)
I_H0 = np.abs(np.square(np.sum(H_mag * probe)))
I_E0 = np.abs(np.square(np.sum(E_mag * probe)))

print('I_E0 = {}'.format(I_E0))

# plot the vacuum fields
if PLOT:
    plt.imshow(np.real(Ez).T, cmap='RdBu')
    plt.title('real(Hz)')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()

mod_strength = np.max(np.abs(eps_nl(Ez) - eps_r))
print('max nonlinear change in epsilon = {}'.format(mod_strength))

# defines the intensity on the other side of the box as a function of the relative permittivity grid
def intensity(eps_arr):

    # reshape the design variables
    eps_r = eps_arr.reshape((Nx, Ny))

    # linear simulation
    F_lin.eps_r = eps_r
    _, _, Ez    = F_lin.solve(source)

    # nonlinear simulation
    density = (eps_max - eps_r)/ (eps_max - 1)
    F_nl.eps_r = lambda E: eps_r + density * 3 * chi3 * npa.square(npa.abs(Ez))
    _, _, Ez_nl = F_nl.solve(source)    

    # compute the intesnities of both simulations
    I_lin = npa.sum(npa.square(npa.abs(Ez * probe)))
    I_nl = npa.sum(npa.square(npa.abs(Ez_nl * probe)))

    # maximize linear intensity, minimize nonlinear
    return - (I_lin - I_nl) / I_E0

# define the gradient for the optmization routine
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

# # plot the final permittivity (doesnt work?)
# plt.imshow(imarr(F_lin.eps_r), cmap='nipy_spectral')
# plt.colorbar()
# plt.show()

f, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True)

# plot the linear fields
_, _, Ez = F_lin.solve(source)
ax1.imshow(np.abs(imarr(Ez)), cmap='RdBu')
ax1.set_title('linear |Ez|')
ax1.set_xlabel('y')
ax1.set_ylabel('x')
# plt.colorbar()

# plot the nonlinear fields
_, _, Ez_nl = F_nl.solve(source)
ax2.imshow(np.abs(imarr(Ez_nl)), cmap='RdBu')
ax2.set_title('nonlinear |Ez|')
ax2.set_xlabel('y')
ax2.set_ylabel('x')
# plt.colorbar()

# plot the nonlinear fields
_, _, Ez_nl = F_nl.solve(source)
ax3.imshow(np.abs(imarr(Ez) - imarr(Ez_nl)), cmap='RdBu')
ax3.set_title('difference |Ez - Ez_nl|')
ax3.set_xlabel('y')
ax3.set_ylabel('x')
# plt.colorbar()

plt.show()
