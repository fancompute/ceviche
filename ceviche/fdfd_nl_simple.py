import autograd
import autograd.numpy as np
from autograd.extend import primitive, defvjp
from autograd.numpy.linalg import solve, norm

import sys
sys.path.append('../ceviche')
from ceviche.utils import get_value
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import spsolve

N = 10
B = np.random.random((N,N))
b = np.random.random((N, ))
zero = np.zeros((N,))

sparse = True

@primitive
def special_solve(eps, b):
	A = make_A(eps)
	if sparse:
		return spsolve(A, b)
	else:
		return solve(A, b)

def special_solve_T(eps, b):
	A = make_A(eps)
	if sparse:
		return spsolve(A.T, b)
	else:
		return solve(A.T, b)

def vjp_special_solve(x, eps, b):
	def vjp(v):
		x_aj = special_solve_T(eps, -v)
		return x * x_aj
	return vjp

defvjp(special_solve, vjp_special_solve)


""" MAKE A """

@primitive
def make_A(eps):
	if sparse:
		return csr_matrix(B) + spdiags(eps, [0], eps.size, eps.size)
	else:
		return B + np.diag(eps)

def vjp_maker_make_A(A, eps):	
	def vjp(v):
		return v
	return vjp

defvjp(make_A, vjp_maker_make_A)

""" SOLVE E """

def solve_E(eps_fn):
	E = solve_nl(eps_fn)
	return E

def solve_nl(eps_fn):

	E_i = np.zeros((N,))
	for i in range(10):
		eps_i = eps_fn(E_i)
		E_i = special_solve(eps_i, b)

	return E_i

if __name__ == '__main__':

	def objective(mask):

		# eps_fn = lambda E: 1 - c * E
		eps_fn = lambda E: mask * np.ones((N, )) + np.square(E)

		E_nl = solve_E(eps_fn)
		return np.sum(E_nl)

	c0 = 1.0 * np.ones((N, ))
	delta_c = 1e-6
	o1 = objective(c0)
	print('objective = ', objective(c0))
	do_dc = autograd.grad(objective)
	print('gradient = ', do_dc(c0))
	print('gradient (sum) = ', np.sum(do_dc(c0)))
	o2 = objective(c0 + delta_c)
	print('numerical = ', (o2 - o1) / delta_c)