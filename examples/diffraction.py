import autograd.numpy as np
import matplotlib.pylab as plt

from autograd import grad

from ceviche.fdfd import fdfd_ez as fdfd_hz
from ceviche.constants import C_0

# some parameters
wavelengths = [10e-9]#, 10e-9]#, 550e-9]#, 650e-9]
angles = [0]#, 10e-9]#, 0]#, 30, 30]

omegas = [2 * np.pi * C_0 / wavelength for wavelength in wavelengths]

H = 3e-6  # height of slab
L = 5e-6  # width of slab

spc = 3e-6   # space between source and PML, source and structure
dL = 1e-7

npml = 10       # number of PML grids
mat_index = 5   # material index

# setup arrays
Nx = int(L / dL)
Ny = 2*npml + 2*int(spc / dL) + int(H / dL)
x_pts = dL * np.arange(Nx)

slab_region = np.zeros((Nx, Ny))
slab_region[:, npml + int(spc / dL):npml + int(spc / dL) + int(H / dL)] = 1

eps_r = np.ones((Nx, Ny))
# eps_r[slab_region == 1] = mat_index**2

source = np.zeros((Nx, Ny))
source[:, npml + int(spc / 2 / dL)] = 1

# plt.imshow((eps_r + source + probe).T, cmap='gist_earth_r')
# plt.title('domain')
# plt.show()

def k0_prime(angle, wavelength):
    angle_rad = angle / 180*np.pi
    k0 = 2*np.pi / wavelength
    return k0 * np.sin(angle_rad)

def probe_x(x, angle, wavelength):
    k0_p = k0_prime(angle, wavelength)
    return np.exp(1j * k0_p * x)

def get_probes(angles, wavelengths, probe_index_y):
    probes = []
    for angle, wavelength in zip(angles, wavelengths):
        new_probe = np.zeros((Nx, Ny), dtype=np.complex128)
        probe_vals = probe_x(x_pts, angle, wavelength)
        new_probe[:, probe_index_y] = probe_vals
        probes.append(new_probe)
    return probes

def get_probes(angles, wavelengths, probe_index_y):
    probes = []
    for angle, wavelength in zip(angles, wavelengths):
        new_probe = np.zeros((Nx, Ny), dtype=np.complex128)
        probe_vals = probe_x(x_pts, angle, wavelength)
        new_probe[:, probe_index_y] = probe_vals
        probes.append(new_probe)
    return probes

probe_index_y = npml + int(3 * spc / 2 / dL) + int(H / dL)

probe1 = np.zeros((Nx, Ny))
probe1[Nx//3, probe_index_y] = 1
probe2 = np.zeros((Nx, Ny))
probe2[2*Nx//3, probe_index_y] = 1

probes = [probe1, probe2]

fdfds = []
for angle, wavelength in zip(angles, wavelengths):
    omega = 2 * np.pi * C_0 / wavelength
    f_new = fdfd_hz(omega, dL, eps_r, source, npml=[0, npml])
    fdfds.append(f_new)

def plot_field(fdfd):
    Ex, Ey, Hz = fdfd.solve()
    plt.imshow(np.abs(Hz._value))
    plt.show()

def objective(eps_r):

    # set the permittivities to the new values
    J = 0.0
    for f, p in zip(fdfds, probes):
        f.eps_r = eps_r
        Ex, Ey, Hz = f.solve()
        diff_power = np.abs(np.sum(p * Hz))
        J += diff_power

    return J

# define the gradient for autograd
grad_J = grad(objective)

# optimization loop
NIter = 100
step_size = 1e5
for i in range(NIter):
    J = objective(eps_r)
    print('on iter {} / {}, objective = {}'.format(i, NIter, J))    
    dJ_deps = grad_J(eps_r)
    # import pdb; pdb.set_trac()
#    plt.imshow(dJ_deps)
 #   plt.show()    
    eps_r = eps_r + step_size * slab_region * dJ_deps
    eps_r[eps_r < 1] = 1
    eps_r[eps_r > mat_index**2] = mat_index**2


plot_field(fdfds[0])


