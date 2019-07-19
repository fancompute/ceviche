import autograd.numpy as np

from autograd.core import make_vjp, make_jvp
from autograd.wrap_util import unary_to_nary
from autograd.extend import vspace

from .utils import get_value, get_shape, get_value_arr, float_2_array


def _jac_shape(x, ans):
    """ computes the shape of the jacobian where function has input x and output ans """
    m = float_2_array(x).size
    n = float_2_array(ans).size
    return (m, n)


@unary_to_nary
def jacobian_reverse(fun, x):
    """ Compute jacobian of fun with respect to x using reverse mode differentiation"""
    vjp, ans = make_vjp(fun, x)
    grads = map(vjp, vspace(ans).standard_basis())
    m, n = _jac_shape(x, ans)
    return np.reshape(np.stack(grads), (n, m))


@unary_to_nary
def jacobian_forward(fun, x):
    """ Compute jacobian of fun with respect to x using forward mode differentiation"""
    jvp = make_jvp(fun, x)
    # ans = fun(x)
    val_grad = map(lambda b: jvp(b), vspace(x).standard_basis())
    vals, grads = zip(*val_grad)
    ans = np.zeros((list(vals)[0].size,))  # fake answer so that dont have to compute it twice
    m, n = _jac_shape(x, ans)
    return np.reshape(np.stack(grads), (m, n)).T


@unary_to_nary
def jacobian_numerical(fn, x, step_size=1e-7):
    """ numerically differentiate `fn` w.r.t. its argument `x` """
    in_array = float_2_array(x).flatten()
    out_array = float_2_array(fn(x)).flatten()

    m = in_array.size
    n = out_array.size
    shape = (n, m)
    jacobian = np.zeros(shape)

    for i in range(m):
        input_i = in_array.copy()
        input_i[i] += step_size
        arg_i = input_i.reshape(in_array.shape)
        output_i = fn(arg_i).flatten()
        grad_i = (output_i - out_array) / step_size
        jacobian[:, i] = get_value_arr(get_value(grad_i))  # need to convert both the grad_i array and its contents to actual data.

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

    N = 3
    M = 2
    A = np.random.random((N,M))
    B = np.random.random((N,M))
    print('A = \n', A)

    def fn(x, b):
        return A @ x + B @ b

    x0 = np.random.random((M,))
    b0 = np.random.random((M,))    
    print('Jac_rev = \n', jacobian(fn, argnum=0, mode='reverse')(x0, b0))
    print('Jac_for = \n', jacobian(fn, argnum=0, mode='forward')(x0, b0))
    print('Jac_num = \n', jacobian(fn, argnum=0, mode='numerical')(x0, b0))

    print('B = \n', B)
    print('Jac_rev = \n', jacobian(fn, argnum=1, mode='reverse')(x0, b0))
    print('Jac_for = \n', jacobian(fn, argnum=1, mode='forward')(x0, b0))
    print('Jac_num = \n', jacobian(fn, argnum=1, mode='numerical')(x0, b0))

