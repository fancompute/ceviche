import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from copy import deepcopy

from ceviche.constants import *
from ceviche.fdfd import compute_derivative_matrices#, Ez_to_Hx_Hy

def get_modes(eps_cross, omega, dL, npml, m=1, filtering=True):
    """ Solve for the modes of a waveguide cross section
        ARGUMENTS
            eps_cross: the permittivity profile of the waveguide
            omega:     angular frequency of the modes
            dL:        grid size of the cross section
            npml:      number of PML points on each side of the cross section
            m:         number of modes to solve for
            filtering: whether to filter out evanescent modes
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
    vals, vecs = solver_eigs(A, m, guess_value=n_max**2)

    if filtering:
        filter_re = lambda vals: np.real(vals) > 0.0
        # filter_im = lambda vals: np.abs(np.imag(vals)) <= 1e-12
        filters = [filter_re]
        vals, vecs = filter_modes(vals, vecs, filters=filters)

    if vals.size == 0:
        raise BaseException("Could not find any eigenmodes for this waveguide")

    vecs = normalize_modes(vecs)

    return vals, vecs


def insert_mode(omega, dx, x, y, epsr, target=None, npml=0, m=1, filtering=False):
    """Solve for the modes in a cross section of epsr at the location defined by 'x' and 'y'

    The mode is inserted into the 'target' array if it is suppled, if the target array is not
    supplied, then a target array is created with the same shape as epsr, and the mode is
    inserted into it.
    """
    if target is None:
        target = np.zeros(epsr.shape, dtype=complex)

    epsr_cross = epsr[x, y]
    _, mode_field = get_modes(epsr_cross, omega, dx, npml, m=m, filtering=filtering)
    target[x, y] = np.atleast_2d(mode_field)[:,m-1].squeeze()

    return target


def solver_eigs(A, Neigs, guess_value=1.0):
    """ solves for `Neigs` eigenmodes of A
            A:            sparse linear operator describing modes
            Neigs:        number of eigenmodes to return
            guess_value:  estimate for the eigenvalues
        For more info, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html
    """

    values, vectors = spl.eigs(A, k=Neigs, sigma=guess_value, v0=None, which='LM')

    return values, vectors


def filter_modes(values, vectors, filters=None):
    """ Generic Filtering Function
        ARGUMENTS
            values: array of effective index values
            vectors: array of mode profiles
            filters: list of functions of `values` that return True for modes satisfying the desired filter condition
        RETURNS
            vals:      array of filtered effective indeces of the modes
            vectors:   array containing the corresponding, filtered mode profiles
    """

    # if no filters, just return
    if filters is None:
        return values, vectors

    # elements to keep, all for starts
    keep_elements = np.ones(values.shape)

    for f in filters:
        keep_f = f(values)
        keep_elements = np.logical_and(keep_elements, keep_f)

    # get the indeces you want to keep
    keep_indeces = np.where(keep_elements)[0]

    # filter and return arrays
    return values[keep_indeces], vectors[:, keep_indeces]


def normalize_modes(vectors):
    """ Normalize each `vec` in `vectors` such that `sum(|vec|^2)=1`
            vectors: array with shape (n_points, n_vectors)
        NOTE: eigs already normalizes for you, so you technically dont need this function
    """

    powers = np.sum(np.square(np.abs(vectors)), axis=0)

    return vectors / np.sqrt(powers)

def Ez_to_H(Ez, omega, dL, npml):
    """ Converts the Ez output of mode solver to Hx and Hy components
    """

    N = Ez.size
    matrices = compute_derivative_matrices(omega, (N, 1), [npml, 0], dL=dL)
    Dxf, Dxb, Dyf, Dyb = matrices

    # save to a dictionary for convenience passing to primitives
    info_dict = {}
    info_dict['Dxf'] = Dxf
    info_dict['Dxb'] = Dxb
    info_dict['Dyf'] = Dyf
    info_dict['Dyb'] = Dyb

    Hx, Hy = Ez_to_Hx_Hy(Ez)

    return Hx, Hy

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
    plt.xlabel('x position ($\lambda_0$)')
    plt.ylabel('mode profile (normalized)')
    plt.show()

