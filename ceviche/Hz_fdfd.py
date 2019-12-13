import autograd.numpy as npa
import scipy.sparse as sp

from ceviche.primitives import sp_mult, sp_solve, make_rand_entries_indices, make_rand, make_rand_complex, grad_num
from ceviche.jacobians import jacobian
from ceviche.fdfd import fdfd_hz

if __name__ == '__main__':

    """ Here's an example simulating Ez polarization A = C + diag(epsilon)
        Note: to do Hz, we'll need to implement primitives for sparse matrix -> sparse matrix multiplication
            A = D1 * diag(1/epsilon) * D2 + ...
    """

    Nx, Ny = 100, 100
    N = Nx * Ny

    # current source for this problem
    source = make_rand_complex(N).reshape((Nx, Ny))

    def photonics_problem(parameters):

        # construct your permittivity from the parameters
        eps_vec = 1 + npa.square(parameters)
        eps_r = eps_vec.reshape((Nx, Ny))

        F = fdfd_hz(omega=1e14, L0=1e-6, eps_r=eps_r, npml=[0, 0])
        Ez, Hx, Hy = F.solve(source)

        return npa.abs(npa.sum(Ez)) + npa.abs(npa.sum(Hx)) + npa.abs(npa.sum(Hy))

    # parameters, which will give rise to the permittivity
    p0 = make_rand_complex(N)

    print('objective function    = ', photonics_problem(p0))

    # grad_true = grad_num(photonics_problem, p0)
    # print('numerical gradient    = ', grad_true)

    # grad_for = jacobian(photonics_problem, mode='forward')(p0)[0]
    # print('forward-mode gradient = ', grad_for)

    grad_rev = jacobian(photonics_problem, mode='reverse')(p0)[0]
    # print('reverse-mode gradient = ', grad_rev)


