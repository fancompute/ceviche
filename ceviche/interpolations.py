
def interp(field_array, coords, style='gaussian'):
    if style == 'gaussian':
        return gaussian(field_array, x0)
    else:
        raise ValueError("`style` kwarg must be one of {'gaussian'}.")

def gaussian(x0, std=1):
    """ defines the interpolation scheme """

    # gaussian interpolation of the point x0
    gauss = npa.exp(-npa.square(grid_x - x0)/(2*std**2))
    # returns (normalized) weights into each of the indeces
    return gauss / npa.sum(gauss)