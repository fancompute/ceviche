import autograd.numpy as np

def param_continuous(params, design_region):
    
    eps_max = 5

    shape = design_region.shape
    eps_old = np.ones(shape)

    # import pdb; pdb.set_trace()
    print(params)
    # note.. this is tricky with autograd.
    eps_new[design_region == 1] = 1 + (eps_max - 1) * params

    return eps_new