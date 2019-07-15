from __future__ import absolute_import
import sys
sys.path.append('../ceviche')

import autograd.numpy as npa
import numpy as np
from ceviche.fdtd import fdtd
from ceviche.jacobians import jacobian
from autograd.extend import defjvp
from scipy.linalg import dft
import matplotlib.pylab as plt

from autograd.extend import primitive, defjvp

@primitive
def my_fft(x):    
    """ 
    Wrapper for numpy's FFT, so I can add a primitive to it
        FFT(x) is like a DFT matrix (D) dot with x
    """
    return np.fft.fft(x)
# my_fft = primitive(np.fft.fft)

def fft_grad(g, ans, x):
    """ 
    Define the jacobian-vector product of my_fft(x)
        The gradient of FFT times g is the vjp
        ans = fft(x) = D @ x
        jvp(fft(x))(g) = d{fft}/d{x} @ g
                       = D @ g
        Therefore, it looks like the FFT of g
    """
    return np.fft.fft(g)

defjvp(my_fft, fft_grad)

###

Nx = 50
Ny = 50
Nz = 1


npml = 10

omega = 2*np.pi*200e12
dL = 5e-8
pml = [npml, npml, 0]

# source parameters
sigma = 10e-15
total_time = 0.5e-12
t0 = sigma * 10

source_amp = 1
source_pos = np.zeros((Nx, Ny, Nz))
source_pos[npml+10, Ny//2, Nz//2] = source_amp

# starting relative permittivity (random for debugging)
eps_r   = np.random.random((Nx, Ny, Nz)) + 1
F = fdtd(eps_r, dL=dL, npml=pml)
dt = F.dt

steps = int(total_time / dt)
print('{} time steps'.format(steps))

gaussian = lambda t: source_pos * source_amp * np.exp(-(t - t0 / dt)**2 / 2 / (sigma / dt)**2)
source = lambda t: source_pos * gaussian(t) * np.cos(omega * t * dt)

# plt.plot(dt * np.arange(steps), np.sum(source(np.arange(steps)), axis=(0,1)))
# plt.show()

measure_pos = np.zeros((Nx, Ny, Nz))
measure_pos[-npml-10, Ny//2, Nz//2] = 1

def objective(eps_space):
    F.eps_r *= eps_space
    measured = []
    for t_index in range(steps):
        fields = F.forward(Jz=source(t_index))
        measured.append(npa.sum(fields['Ez'] * measure_pos))
    measured_f = my_fft(npa.array(measured))
    # measured_f = dft(steps) @ npa.array(measured)
    spectral_power = npa.square(npa.abs(measured_f))
    return spectral_power

eps_space = 1.0
spectral_power = objective(eps_space)
jac_power = jacobian(objective, mode='forward')(eps_space)

n_disp = 140

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
delta_f = 1 / steps / dt
freq_x = np.arange(n_disp) * delta_f
ax1.plot(freq_x, spectral_power[:n_disp], 'k-')
ax2.plot(freq_x, jac_power[:n_disp,0], 'g-', label='FMD')
ax1.set_ylabel('spectral power', color='k')
ax2.set_ylabel('dP/depsilon', color='g')
ax2.spines['right'].set_color('g')
ax2.tick_params(axis='y', colors='g')
plt.show()