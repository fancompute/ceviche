import autograd.numpy as np
import matplotlib.pylab as plt

from scipy.optimize import minimize
from autograd import grad

from ceviche import fdfd_hz
from ceviche.constants import C_0

PLOT = True

""" define parameters """

wavelengths = [450e-9, 550e-9, 650e-9]  # in meters
Nw = len(wavelengths)                   # number of wavelengths

H = 1e-6           # height of slab
L = 5e-6           # width of slab

spc = 2e-6         # space between source and PML, source and structure
dL = 20e-9         # size (meters) of each grid cell in FDFD

npml = 10          # number of PML grids
eps_max = 4       # material index

""" set up arrays for this problem (note, need to convert from meters to grid cells using dL) """

Nx = int(L / dL)                              # number of grid cells in x
Ny = 2*npml + 2*int(spc / dL) + int(H / dL)   # number of grid cells in y

# defines where the slab / design region will be (where this array = 1)
design_region = np.zeros((Nx, Ny))
design_region[:, npml + int(spc / dL):npml + int(spc / dL) + int(H / dL)] = 1

# the starting relative permittivity
eps_r = np.ones((Nx, Ny))

# defines where the source will be (plane wave)
source = np.zeros((Nx, Ny))
source[:, npml + int(spc / 2 / dL)] = 1

# defines the y_index of the measurement points
probe_index_y = npml + int(3 * spc / 2 / dL) + int(H / dL)

# make a list of probe locations, evenly spaced in x and at probe_index_y
probes = []
space_x = Nx / float(Nw)                  # even spacing between probes
for i in range(Nw):
    index_i = int(space_x * (1/2 + i))    # the x index of the ith probe
    probe_i = np.zeros((Nx, Ny))          # ith probe
    probe_i[index_i, probe_index_y] = 1   # define at this point
    probes.append(probe_i)                # add to list

# make a list of FDFDs at each wavelength and their power normalizations
fdfds = []
powers = []
for i, lam in enumerate(wavelengths):
    omega_i = 2 * np.pi * C_0 / lam      # the angular frequency
    fdfd_i = fdfd_hz(omega_i, dL, eps_r, source, npml=[0, npml])   # make an FDFD simulation
    fdfds.append(fdfd_i)                                           # add it to the list
    Ex, Ey, Hz = fdfd_i.solve()                                    # solve the fields
    powers.append(np.sum(np.square(np.abs(Hz) * probes[i])))       # compute the power at its probe, add to list

# plot the domain (design region + probes + source)
if PLOT:
    plt.imshow((sum(probes) + source + design_region).T, cmap='Greys')
    plt.title('domain')
    plt.xlabel('x')
    plt.ylabel('y')   
    plt.colorbar()     
    plt.show()

# plots the real part of Hz
def plot_field(fdfd):
    Ex, Ey, Hz = fdfd.solve()
    plt.imshow(np.real(Hz.T), cmap='RdBu')
    plt.title('real(Hz)')
    plt.xlabel('x')
    plt.ylabel('y')   
    plt.colorbar() 
    plt.show()

# plots the real part of Hz for the autograd object (just needs to be called instead of the `plot_field` after the optimization)
def plot_field_ag(fdfd):
    Ex, Ey, Hz = fdfd.solve()
    plt.imshow(np.real(Hz._value.T), cmap='RdBu')
    plt.title('real(Hz)')
    plt.xlabel('x')
    plt.ylabel('y')    
    plt.colorbar() 
    plt.show()

# plot each of the starting fields
if PLOT:
    for i in range(Nw):        
        plot_field(fdfds[i])

""" Objective function """

# the objective function (to minimize) as a function of the (flattened) permittivity array
def objective(eps_arr):

    # reshape flattened array into the right grid shape
    eps_r = eps_arr.reshape((Nx, Ny))

    # running sum of the power at each probe for each wavelength
    power_total = 0.0

    # looop through wavelengths, fdfds, probes
    for i in range(Nw):
        fdfds[i].eps_r = eps_r           # set the relative permittivity of the FDFD
        Ex, Ey, Hz = fdfds[i].solve()    # solve the fields
        power = np.abs(np.square(np.sum(probes[i] * Hz))) / powers[i]   # compute the power at the probe
        power_total += power             # add to sum

    return - power_total / Nw            # return negative (for maximizing) and normalize by number of wavelengths (starting objective = 1)

# define the gradient for autograd
grad_J = grad(objective)

"""  optimization loop """

NIter = 10000    # max number of iterations

# set the material bounds to (1, eps_max) in design region, (1,1) outside
bounds = [(1, eps_max) if design_region.flatten()[i] == 1 else (1,1) for i in range(eps_r.size)]  

# call a constrained optimization function from Scipy (progress prints to terminal)
minimize(objective, eps_r, args=(), method='L-BFGS-B', jac=grad_J,
    bounds=bounds, tol=None, callback=None,
    options={'disp': True, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': NIter, 'iprint': -1, 'maxls': 20})

# plot the fields of the final FDFDs
if PLOT:
    for i in range(Nw):        
        plot_field_ag(fdfds[i])

# plot the final permittivity
plt.imshow(fdfds[0].eps_r._value.T, cmap='Greys')
plt.title('final relative permittivity')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()
