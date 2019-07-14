import autograd.numpy as np

from autograd.core import make_vjp, make_jvp
from autograd.wrap_util import unary_to_nary
from autograd.extend import vspace

from ceviche.utils import get_value, float_2_array


def _get_shape(x):
    if isinstance(x, float):
        return (1,)
    else:
        return vspace(x).shape    

def jac_shape(x, ans):
    shape_in = _get_shape(x)
    shape_out = _get_shape(ans)
    return shape_in + shape_out

def jac_shape(x, ans):
    m = float_2_array(x).size
    n = float_2_array(ans).size
    return (m, n)

@unary_to_nary
def jacobian_reverse(fun, x):
    """ Compute jacobian of fun with respect to x using reverse mode differentiation"""
    vjp, ans = make_vjp(fun, x)
    grads = map(vjp, vspace(ans).standard_basis())
    return np.reshape(np.stack(grads), jac_shape(x, ans))

@unary_to_nary
def jacobian_forward(fun, x):
    """ Compute jacobian of fun with respect to x using forward mode differentiation"""
    jvp = make_jvp(fun, x)
    ans = fun(x)
    grads = map(lambda b: jvp(b)[1], vspace(x).standard_basis())
    return np.reshape(np.stack(grads), jac_shape(x, ans))

@unary_to_nary
def jacobian_numerical(fn, x, step_size=1e-7):
    """ numerically differentiate `fn` w.r.t. its argument `x` 
    `x` can be a numpy array of arbitrary shape
    `step_size` can be a number or an array of the same shape as `x` """

    in_array = float_2_array(x).flatten()
    out_array = float_2_array(fn(x)).flatten()

    m = in_array.size
    n = out_array.size
    shape = (m, n)
    jacobian = np.zeros(shape)

    for i in range(m):
        input_i = in_array.copy()
        input_i[i] += step_size
        arg_i = input_i.reshape(in_array.shape)
        output_i = fn(arg_i).flatten()
        grad_i = (output_i - out_array) / step_size
        jacobian[i, :] = get_value(grad_i)

    return jacobian


def jacobian(fun, argnum=0, mode='reverse', step_size=1e-6):
    """ Computes jacobian of `fun` with respect to argument number `argnum` using automatic differentiation """
    if mode == 'reverse':
        return jacobian_reverse(fun, argnum)
    elif mode == 'forward':
        return jacobian_forward(fun, argnum)
    elif mode == 'numerical':
        return jacobian_numerical(fun, argnum, step_size=step_size)
    else:
        raise ValueError("'mode' kwarg must be either 'reverse' or 'forward' or 'numerical', given {}".format(mode))


if __name__ == '__main__':

    N = 2
    A = np.random.random((N,))
    print('A = \n', A)

    def fn(x, b):
        return np.sum(A.T @ x - A.T @ b)

    x0 = np.random.random((N,))
    b0 = np.random.random((N,))    
    print('Jac_rev = \n', jacobian(fn, argnum=0, mode='reverse')(x0, b0))
    print('Jac_for = \n', jacobian(fn, argnum=0, mode='forward')(x0, b0))

    print('A^T = \n', A.T)
    print('Jac_rev = \n', jacobian(fn, argnum=1, mode='reverse')(x0, b0))
    print('Jac_for = \n', jacobian(fn, argnum=1, mode='forward')(x0, b0))
