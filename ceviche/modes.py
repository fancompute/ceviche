import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from copy import deepcopy

from ceviche.constants import *
from ceviche.fdfd import compute_derivative_matrices

def get_modes(eps_cross, omega, dL, npml, m=1, filtering=True):
    """ Solve for the modes of a waveguide cross section 
        ARGUMENTS
            eps_cross: the permittivity profile of the waveguide
            omega:     angular frequency of the modes
            dL:        grid size of the cross section
            npml:      number of PML points on each side of the cross section
            m:         number of modes to solve for
            filtering:    whether to filter out evanescent modes
        RETURNS
            vals:      array of effective indeces of the modes
            vectors:   array containing the corresponding mode profiles
    """

    k0 = omega / C_0

    N = eps_cross.size

    matrices = compute_derivative_matrices(omega, (N, 1), [npml, 0], dL=dL)

    Dxf, Dxb, Dyf, Dyb = matrices

    diag_eps_r = sp.spdiags(eps_cross.flatten(), [0], N, N)
    A = diag_eps_r + Dxf.dot(Dxb) * (1 / k0) ** 2

    n_max = np.sqrt(np.max(eps_cross))
    vals, vecs = solver_eigs(A, m, guess_value=4*n_max)

    if filtering:
        vals, vecs = filter_modes(vals, vecs)

    if vals.size == 0:
        raise BaseException("Could not find any eigenmodes for this waveguide")

    vecs = normalize_modes(vecs)

    return vals, vecs


def solver_eigs(A, Neigs, guess_value=1.0):
    """ solves for `Neigs` eigenmodes of A 
            A:            sparse linear operator describing modes
            Neigs:        number of eigenmodes to return
            guess_value:  estimate for the eigenvalues
        For more info, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html
    """

    values, vectors = spl.eigs(A, k=Neigs, sigma=guess_value, v0=None, which='LM')

    return values, vectors


def filter_modes(values, vectors, n_eff_low=1.0, evan_cutoff=1e-12):
    """ Filters out evanescent modes
        ARGUMENTS
            values: array of effective index values
            vectors: array of mode profiles
            n_eff_low: the lowest real(n_eff) that is acceptable
            evan_cutoff: the highest |imag(n_eff)| that is acceptable
        RETURNS
            vals:      array of filtered effective indeces of the modes
            vectors:   array containing the corresponding, filtered mode profiles
    """

    # conditions for keeping the mode
    positive_neff = (np.real(values) > n_eff_low)
    propagating = (np.abs(np.imag(values)) <= evan_cutoff)

    # logical and them together
    both_conditions = np.logical_and(positive_neff, propagating)

    # get the indeces you want to keep
    keep_indeces = np.where(both_conditions)[0]

    # filter and return arrays
    return values[keep_indeces], vectors[:, keep_indeces]


def normalize_modes(vectors):
    """ Normalize each `vec` in `vectors` such that `sum(|vec|^2)=1` 
            vectors: array with shape (n_points, n_vectors)
        NOTE: eigs already normalizes for you, so you technically dont need this function
    """

    powers = np.sum(np.square(np.abs(vectors)), axis=0)

    return vectors / np.sqrt(powers)


if __name__ == '__main__':

    """ Test on a simple ridge waveguide """
        
    from ceviche.fdfd import fdfd_ez as fdfd
    import matplotlib.pylab as plt

    lambda0 = 1.550e-6                         # free space wavelength (m)
    dL = lambda0 / 100
    npml = int(lambda0 / dL)                              # number of grid points in PML
    omega_0 = 2 * np.pi * C_0 / lambda0        # angular frequency (rad/s)

    Lx = lambda0 * 10                          # length in horizontal direction (m)

    Nx = int(Lx/dL)

    wg_perm = 4

    wg_width = lambda0 
    wg_points = np.arange(Nx//2 - int(wg_width/dL/2), Nx//2 + int(wg_width/dL/2))

    eps_wg = np.ones((Nx,))
    eps_wg[wg_points] = wg_perm

    vals, vecs = get_modes(eps_wg, omega_0, dL, npml=10, m=10)

    plt.plot(np.linspace(-Lx, Lx, Nx) / 2 / lambda0, np.abs(vecs))
    plt.xlabel('x position ($\lambda_0$')
    plt.ylabel('mode profile (normalized)')
    plt.show()

