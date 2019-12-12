import autograd.numpy as npa
import scipy.sparse as sp

from ceviche.primitives import sp_mult, sp_solve, make_rand_entries_indices, make_rand, make_rand_complex, grad_num
from ceviche.jacobians import jacobian

if __name__ == '__main__':

    """ Here's an example simulating Ez polarization A = C + diag(epsilon)
        Note: to do Hz, we'll need to implement primitives for sparse matrix -> sparse matrix multiplication
            A = D1 * diag(1/epsilon) * D2 + ...
    """

    N = 10
    M = N**2

    # some constant matrices (like derivative matrices)
    D_entries, D_indicies = make_rand_entries_indices(N, M)

    source = make_rand_complex(N)

    def photonics_problem(parameters):

        # construct your permittivity from the parameters
        epsilon = 1 + npa.square(parameters)

        # indices into the diagonal of a sparse matrix
        diag_indices = npa.vstack((npa.arange(N), npa.arange(N)))

        A_indices = npa.hstack((diag_indices, D_indicies))
        A_entries = npa.hstack((1 / epsilon, D_entries))

        E_field = sp_solve(A_entries, A_indices, source)

        return npa.abs(npa.sum(E_field))

    # parameters
    p0 = make_rand_complex(N)

    print('objective function    = ', photonics_problem(p0))

    grad_true = grad_num(photonics_problem, p0)
    print('numerical gradient    = ', grad_true)

    grad_for = jacobian(photonics_problem, mode='forward')(p0)[0]
    print('forward-mode gradient = ', grad_for)

    grad_rev = jacobian(photonics_problem, mode='reverse')(p0)[0]
    print('reverse-mode gradient = ', grad_rev)


