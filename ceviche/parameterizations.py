import autograd.numpy as np

def param_continuous(params, design_region):
    
    eps_max = 5

    shape = design_region.shape
    eps_new = np.ones(shape)

    eps_new += 1 + (eps_max - 1) * design_region * params.reshape(shape)

    return eps_new