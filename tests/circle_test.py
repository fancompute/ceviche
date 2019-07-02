import autograd.numpy as np
from autograd import grad, multigrad_dict
import matplotlib.pylab as plt

def sigmoid(x, strength=.2):
    # used to anti-alias the circle, higher strength means sharper boundary
    return np.exp(x * strength) / (1 + np.exp(x * strength))

def circle(xs, ys, x, y, r):
    # defines an anti aliased circle    
    dist_from_edge = (xs - x)**2 + (ys - y)**2 - r**2
    return sigmoid(-dist_from_edge)

def write_circle(array, xs, ys, x, y, r, value):
    # adds a circle with value (value) to array
    mask = circle(xs, ys, x, y, r)
    array += value * mask
    return array

def grid_coords(array, dL):
    # Takes an array and returns the coordinates of the x and y points

    shape = Nx, Ny = array.shape   # shape of domain (in num. grids)

    # x and y coordinate arrays
    x_coord = np.linspace(-Nx/2*dL, Nx/2*dL, Nx)
    y_coord = np.linspace(-Ny/2*dL, Ny/2*dL, Ny)

    # x and y mesh
    xs, ys = np.meshgrid(x_coord, y_coord, indexing='ij')

    return xs, ys

# problem size
Nx, Ny = 100, 100

# objective function (being differentiated)
def objective(x, y, r, value):
    background = np.ones((Nx, Ny), dtype=np.float64)
    xs, ys = grid_coords(background, dL=1.0)
    background = write_circle(background, xs, ys, x, y, r, value)
    return np.sum(background)

# starting conditions
x0 = 1.0
y0 = 2.0
r0 = 20.5
val0 = 2.0

# compute objective function and compare to analytical
obj0 = objective(x0, y0, r0, val0)
area_circle = np.pi * r0**2
obj_analyitical = np.pi*r0**2*(val0-1) + Nx*Ny  # area * (val-1) + background sum (Nx * Ny)
print('analytical objective({}, {}, {}, {}) = {}'.format(x0, y0, r0, val0, area_circle*(val0-1) + Nx*Ny))
print('objective({}, {}, {}, {}) = {}\n'.format(x0, y0, r0, val0, obj0))

# gradient according to autograd
grad_dict = multigrad_dict(objective)
grad_dict_eval = grad_dict(x0, y0, r0, val0)

# derivatives based on the analytical problem being solved
analytical_grads = {
    'x': 0.0,
    'y': 0.0,
    'r': (val0 - 1) * 2 * np.pi * r0,   # d/dr   of Nx * Ny + (val - 1) * pi * r^2
    'value': np.pi * r0**2              # d/dval of Nx * Ny + (val - 1) * pi * r^2
}

# finite difference derivatives
eps = 1e-5
numerical_grads = {
    'x': (objective(x0+eps, y0, r0, val0) - obj0)/eps,
    'y': (objective(x0, y0+eps, r0, val0) - obj0)/eps,
    'r': (objective(x0, y0, r0+eps, val0) - obj0)/eps,
    'value': (objective(x0, y0, r0, val0+eps) - obj0)/eps
}

# print comparisons
for key in ('x','y','r','value'):
    print('autograd grad   w.r.t. {} = {}'.format(key, grad_dict_eval[key]))
    print('analytical grad w.r.t. {} = {}'.format(key, analytical_grads[key]))
    print('numerical grad  w.r.t. {} = {}\n'.format(key, numerical_grads[key]))


