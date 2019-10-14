import argparse

import numpy as np
import autograd.numpy as npa
import matplotlib.pylab as plt

from autograd.scipy.signal import convolve as conv
from skimage.draw import circle

from ceviche import fdfd_ez, fdfd_ez_nl, jacobian
from ceviche.optimizers import adam_optimize
from ceviche.utils import imarr, get_value
from ceviche.modes import insert_mode

import collections
# Create a container for our slice coords to be used for sources and probes
Slice = collections.namedtuple('Slice', 'x y')

def init_domain(Nx, Ny, Npml, space=10, wg_width=10, space_slice=5):
    """Initializes the domain and design region

    space       : The space between the PML and the structure
    wg_width    : The feed and probe waveguide width
    space_slice : The added space for the probe and source slices
    """
    rho = np.zeros((Nx, Ny))  
    design_region = np.zeros((Nx, Ny))
    
    # Input probe slice
    input_slice = Slice(x=np.array(Npml+1), y=np.arange(int(Ny/2-wg_width/2-space_slice), int(Ny/2+wg_width/2+space_slice)))
    
    output_slices = []
    # Input waveguide
    rho[0:int(Npml+space),int(Ny/2-wg_width/2):int(Ny/2+wg_width/2)] = 1
    
    # Output waveguide 1
    rho[int(Nx-Npml-space)::,int(Npml+space):int(Npml+space+wg_width)] = 1
    output_slices.append(Slice(x=np.array(Nx-Npml-1), y=np.arange(int(Npml+space-space_slice), int(Npml+space+wg_width+space_slice))))
    
    # Output waveguide 2
    rho[int(Nx-Npml-space)::,int(Ny/2-wg_width/2):int(Ny/2+wg_width/2)] = 1
    output_slices.append(Slice(x=np.array(Nx-Npml-1), y=np.arange(int(Ny/2-wg_width/2-space_slice), int(Ny/2+wg_width/2+space_slice))))
    
    # Output waveguide 3
    rho[int(Nx-Npml-space)::,int(Ny-Npml-space-wg_width):int(Ny-Npml-space)] = 1
    output_slices.append(Slice(x=np.array(Nx-Npml-1), y=np.arange(int(Ny-Npml-space-wg_width-space_slice), int(Ny-Npml-space+space_slice))))
    
    design_region[Npml+space:Nx-Npml-space, Npml+space:Ny-Npml-space] = 1
    rho[Npml+space:Nx-Npml-space, Npml+space:Ny-Npml-space] = 0.5

    return rho, design_region, input_slice, output_slices

def operator_proj(rho, eta=0.5, beta=100):
    """Density projection
    """
    return npa.divide(npa.tanh(beta * eta) + npa.tanh(beta * (rho - eta)), npa.tanh(beta * eta) + npa.tanh(beta * (1 - eta)))

def operator_blur(rho, radius=2):
    """Blur operator implemented via two-dimensional convolution
    """
    rr, cc = circle(radius, radius, radius+1)
    kernel = np.zeros((2*radius+1, 2*radius+1), dtype=np.float)
    kernel[rr, cc] = 1
    kernel=kernel/kernel.sum()
    # For whatever reason HIPS autograd doesn't support 'same' mode, so we need to manually crop the output
    return conv(rho, kernel, mode='full')[radius:-radius,radius:-radius]

def make_rho(rho, design_region, radius=2):
    """Helper function for applying the blue to only the design region
    """
    lpf_rho = operator_blur(rho, radius=radius) * design_region
    bg_rho = rho * (design_region==0).astype(np.float)
    return bg_rho + lpf_rho

parser = argparse.ArgumentParser()
parser.add_argument('-Nsteps', default=20, type=int)
parser.add_argument('-step_size', default=1e-2, type=float)
parser.add_argument('-omega', default=2*np.pi*200e12, type=float)
parser.add_argument('-dx', default=50e-9, type=float)
parser.add_argument('-Nx', default=100, type=int)
parser.add_argument('-Ny', default=130, type=int)
parser.add_argument('-Npml', default=20, type=int)
parser.add_argument('-epsr_min', default=1.0, type=float)
parser.add_argument('-epsr_max', default=12.0, type=float)
parser.add_argument('-blur_radius', default=3, type=int)
parser.add_argument('-beta', default=100.0, type=float)
parser.add_argument('-eta', default=0.5, type=float)
parser.add_argument('-space', default=10, type=int)
parser.add_argument('-wg_width', default=10, type=int)
parser.add_argument('-space_slice', default=7, type=int)
parser.add_argument('-objective_port_ind', default=0, type=int)
args = parser.parse_args()

Nsteps = args.Nsteps
step_size = args.step_size
omega = args.omega
dx = args.dx
Nx = args.Nx
Ny = args.Ny
Npml = args.Npml
epsr_min = args.epsr_min
epsr_max = args.epsr_max
blur_radius = args.blur_radius
beta = args.beta
eta = args.eta
space = args.space
wg_width = args.wg_width
space_slice = args.space_slice
objective_port_ind = args.objective_port_ind

# Setup initial structure
rho_init, design_region, input_slice, output_slices = init_domain(Nx, Ny, Npml, space=space, wg_width=wg_width, space_slice=space_slice)
epsr = epsr_min + (epsr_max-epsr_min) * make_rho(rho_init, design_region, radius=blur_radius)

# Setup source
source = insert_mode(omega, dx, input_slice.x, input_slice.y, epsr)

# Setup probes
probes = []
for output_slice in output_slices:
    probe = insert_mode(omega, dx, output_slice.x, output_slice.y, epsr)
    probes.append(probe)

# Simulate initial device
simulation = fdfd_ez(omega, dx, epsr, [Npml, Npml])
Hx, Hy, Ez = simulation.solve(source)
fig, ax =plt.subplots(1, 2, constrained_layout=True, figsize=(6,3))
ceviche.viz.real(Ez, outline=epsr, ax=ax[0], cbar=False)
ax[0].plot(input_slice.x*np.ones(len(input_slice.y)), input_slice.y, 'g-')
for output_slice in output_slices:
    ax[0].plot(output_slice.x*np.ones(len(output_slice.y)), output_slice.y, 'r-')
ceviche.viz.abs(epsr, ax=ax[1], cmap='Greys');
plt.show()

# Define optimization objective
def measure_modes(Ez, probe_ind=0):
    return npa.abs(npa.sum(npa.conj(Ez)*probes[probe_ind]))

def objective(rho, probe_ind=objective_port_ind):
    rho = rho.reshape((Nx, Ny))
    _rho = make_rho(rho, design_region, radius=blur_radius)
    epsr = epsr_min + (epsr_max-epsr_min)*operator_proj(_rho, beta=beta, eta=eta)
    simulation.eps_r = epsr
    _, _, Ez = simulation.solve(source)
    return measure_modes(Ez, probe_ind=probe_ind)

# Run optimization
objective_jac = jacobian(objective, mode='reverse')
(rho_optimum, loss) = adam_optimize(objective, rho_init.flatten(), objective_jac, Nsteps=Nsteps, direction='max', mask=design_region.flatten(), step_size=step_size)
rho_optimum = rho_optimum.reshape((Nx, Ny))

# Simulate optimal device
epsr = epsr_min + (epsr_max-epsr_min)*operator_proj(make_rho(rho_optimum, design_region, radius=blur_radius), beta=beta, eta=eta)
simulation = fdfd_ez(omega, dx, epsr, [Npml, Npml])
Hx, Hy, Ez = simulation.solve(source)
fig, ax =plt.subplots(1, 2, constrained_layout=True, figsize=(6,3))
ceviche.viz.real(Ez, outline=epsr, ax=ax[0], cbar=False)
ax[0].plot(input_slice.x*np.ones(len(input_slice.y)), input_slice.y, 'g-')
for output_slice in output_slices:
    ax[0].plot(output_slice.x*np.ones(len(output_slice.y)), output_slice.y, 'r-')
ceviche.viz.abs(epsr, ax=ax[1], cmap='Greys');
plt.show()
