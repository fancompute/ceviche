import autograd
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import autograd.numpy as anp

def _grad_undefined(_, *args):
  raise TypeError('gradient undefined for this input argument')


""" SPARSE DOT PRODUCT """

@autograd.primitive
def mult_coo(a_entries, a_indices, x):
  """Multiply a sparse matrix by a dense vector

  Args:
    a_entries: numpy array with shape (num_zeros,) giving values for non-zero
      matrix entries.
    a_indices: numpy array with shape (2, num_zeros) giving x and y indices for
      non-zero matrix entries.
    x: 1d numpy array specifying the vector to multiply by.

  Returns:
    1d numpy array corresponding to the solution of a*x=b.
  """  
  a = sp.coo_matrix((a_entries, a_indices), shape=(x.size,)*2).tocsc()
  return a.dot(x)

def grad_solve_coo_entries_reverse(ans, a_entries, a_indices, x):
  def jvp(grad_ans):
    i, j = a_indices
    return anp.conj(grad_ans[i] * x[j])
  return jvp

def grad_solve_coo_entries_forward(grad_input, ans, a_entries, a_indices, x):
  i, j = a_indices
  return grad_input[i] * b[j]

autograd.extend.defvjp(
    mult_coo, grad_solve_coo_entries_reverse, _grad_undefined, _grad_undefined)
   
autograd.extend.defjvp(
    mult_coo, grad_solve_coo_entries_forward, _grad_undefined, _grad_undefined)


""" SPARSE LINEAR SOLVE """

@autograd.primitive
def solve_coo(a_entries, a_indices, b):
  """Solve a sparse system of linear equations.

  Args:
    a_entries: numpy array with shape (num_zeros,) giving values for non-zero
      matrix entries.
    a_indices: numpy array with shape (2, num_zeros) giving x and y indices for
      non-zero matrix entries.
    b: 1d numpy array specifying the right hand side of the equation.

  Returns:
    1d numpy array corresponding to the solution of a*x=b.
  """
  a = sp.coo_matrix((a_entries, a_indices), shape=(b.size,)*2).tocsc()
  return spl.spsolve(a, b)

# see autograd's np.linalg.solve:
# https://github.com/HIPS/autograd/blob/96a03f44da43cd7044c61ac945c483955deba957/autograd/numpy/linalg.py#L40

def solve_coo_adjoint(a_entries, a_indices, b):
  return solve_coo(anp.conj(a_entries), a_indices[::-1], anp.conj(b))

def grad_solve_coo_entries(ans, a_entries, a_indices, b):
  def jvp(grad_ans):
    lambda_ = solve_coo_adjoint(a_entries, a_indices, grad_ans)
    i, j = a_indices
    return -lambda_[i] * anp.conj(ans[j])
  return jvp

def grad_solve_coo_entries_FMD(input_grad, ans, a_entries, a_indices, b):
  one_entries = np.ones(a_entries.shape)
  inner_field = 1

def grad_solve_coo_b(ans, a_entries, a_indices, b):
  def jvp(grad_ans):
    return solve_coo_adjoint(a_entries, a_indices, grad_ans)
  return jvp

autograd.extend.defvjp(
    solve_coo, grad_solve_coo_entries, _grad_undefined, grad_solve_coo_b)
   
autograd.extend.defjvp(
    solve_coo, grad_solve_coo_entries_FMD, _grad_undefined, _grad_undefined)

def rand_compl(N):
    return anp.random.random((N,))# + 1j * anp.random.random((N,))

if __name__ == '__main__':

    from ceviche.jacobians import jacobian

    N = 5
    M = int(N * N * 1.0)

    b = rand_compl(N)

    entries_der = rand_compl(N)
    indices_der = anp.vstack([anp.arange(N), anp.arange(N)])

    background_entries = rand_compl(M)
    background_indices = anp.random.randint(low=0, high=N, size=(2, M))

    def get_fields(e):
        e_r = anp.roll(e, 1)
        entries = anp.hstack([background_entries, e_r])
        indices = anp.hstack([background_indices, indices_der])
        # x = solve_coo(entries, indices, b)
        x = mult_coo(entries, indices, b)
        return anp.abs(x)

    x0 = get_fields(entries_der)
    print('xo = ', x0)
    dx_de = jacobian(get_fields, mode='forward')(entries_der)
    print('dx_de = ', dx_de)

    grad_num = anp.zeros((N,N), dtype=anp.complex128)
    DE = 1e-8
    for i in range(N):
        entries_new = entries_der.copy()
        entries_new[i] += DE
        x_new = get_fields(entries_new)
        der_i_real = (x_new - x0) / DE

        entries_new = entries_der.copy()
        entries_new[i] += 1j * DE
        x_new = get_fields(entries_new)
        der_i_imag = (x_new - x0) / DE

        grad_num[:,i] = der_i_real + 1j * der_i_imag
    
    print(grad_num)



