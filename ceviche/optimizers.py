import numpy as np

def minimize(objective, params, jac, method='LBFGS', options=None):
    """ Lets write a master function here that someone can call and replace methods. """
    pass


def adam_minimize(objective, params, jac, step_size=1e-2, Nsteps=100, bounds=None, options={}):
    """Performs Nsteps steps of ADAM minimization of function `objective` with gradient `jac`.
    The `bounds` are set abruptly by rejecting an update step out of bounds."""
    of_list = []

    opt_keys = options.keys()

    if 'beta1' in opt_keys:
        beta1 = options['beta1']
    else:
        beta1 = 0.9

    if 'beta2' in opt_keys:
        beta2 = options['beta2']
    else:
        beta2 = 0.999

    for iteration in range(Nsteps):

        of = objective(params)
        of_list.append(of) 
        grad = jac(params).ravel()

        if 'disp' in opt_keys:
            if options['disp'] == True:
                print("At iteration %d objective value is %f" %(iteration, of))

        if iteration == 0:
            mopt = np.zeros(grad.shape)
            vopt = np.zeros(grad.shape)

        (grad_adam, mopt, vopt) = step_adam(grad, mopt, vopt, iteration, beta1, beta2)

        params -= step_size*grad_adam # Note: minus cause we minimize
        if bounds:
            params[params < bounds[0]] = bounds[0]
            params[params > bounds[1]] = bounds[1]

    return (of_list, params)


def step_adam(gradient, mopt_old, vopt_old, iteration, beta1, beta2, epsilon=1e-8):
    """ Performs one step of adam optimization"""

    mopt = beta1 * mopt_old + (1 - beta1) * gradient
    mopt_t = mopt / (1 - beta1**(iteration + 1))
    vopt = beta2 * vopt_old + (1 - beta2) * (np.square(gradient))
    vopt_t = vopt / (1 - beta2**(iteration + 1))
    grad_adam = mopt_t / (np.sqrt(vopt_t) + epsilon)

    return (grad_adam, mopt, vopt)