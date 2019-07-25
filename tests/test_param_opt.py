import autograd.numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import copy

from autograd.extend import primitive, defvjp
from autograd import grad

from ceviche.utils import grad_num
# from ceviche.primitives import *
from ceviche.fdfd import fdfd_hz, fdfd_ez

""" Runs the same optimization for a bunch of different parameterizations 
    Plots the final results 
"""

def get_normalization(F, source, probe):

    Ex, Ey, Hz = F.solve(source)

    E_mag = np.square(np.abs(Ex)) + np.square(np.abs(Ey))
    H_mag = np.abs(Hz)

    I_E0 = np.abs(np.sum(E_mag * probe))
    I_H0 = np.abs(np.square(np.sum(H_mag * probe)))

    return I_E0, I_H0


# defines the intensity on the other side of the box as a function of the relative permittivity grid
def intensity(eps_arr):

    eps_r = eps_arr.reshape((Nx, Ny))
    # set the permittivity of the FDFD and solve the fields
    F.eps_r = eps_r
    Ex, Ey, Hz = F.solve(source)

    # compute the gradient and normalize if you want
    I = np.sum(np.square(np.abs(Hz * probe)))
    I_E0, I_H0 = get_normalization(F, source, probe)

    return -I / I_H0


if __name__ == '__main__':

        # make parameters
    omega = 2 * np.pi * 200e12
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

    probe = np.zeros((Nx, Ny), dtype=np.complex128)
    probe[-npml-spc, Ny//2] = 1

    F = fdfd_hz(omega, dL, eps_r, [npml, npml])

    # define the gradient for autograd
    grad_I = grad(intensity)

    from scipy.optimize import minimize
    bounds = [(1, eps_max) if box_region.flatten()[i] == 1 else (1,1) for i in range(eps_r.size)]
    minimize(intensity, eps_r, args=(), method='L-BFGS-B', jac=grad_I,
        bounds=bounds, tol=None, callback=None,
        options={'disp': True, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 150, 'iprint': -1, 'maxls': 20})
