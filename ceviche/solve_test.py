import numpy as np
import scipy.sparse.linalg as spl
from scipy.sparse import csr_matrix, diags
from scipy.sparse import random as random_sp
from numpy.random import random as random
import numpy as np
import scipy.optimize as opt

N = 200       # dimension of the x, and b vectors
density = 0.3  # sparsity of the dense matrix
A_lin = csr_matrix(random_sp(N, N, density=density))
A_nl = csr_matrix(random_sp(N, N, density=density))

A = lambda x: A_lin + 0.01 * diags(np.square(x), 0, shape=(N, N))
b = np.random.random((N, )) - 0.5

vec_0 = np.zeros(b.shape)
x_0 = spl.spsolve(A(vec_0), b) + 0.2

def F_min(x):
    res = A(x).dot(x) - b
    return np.square(np.abs(res))

res = opt.newton_krylov(F_min, x_0, method='gmres', verbose=True)
# return opt.minimize(F_min, x_0, method='CG', options={'disp':True})
