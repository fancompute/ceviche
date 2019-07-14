import sys
sys.path.append('../ceviche')

import autograd.numpy as npa
import numpy as np
from ceviche.fdtd import fdtd
from ceviche.jacobians import jacobian
from autograd.extend import defjvp
from scipy.linalg import dft
import matplotlib.pylab as plt

# def fft_grad(get_args, fft_fun, ans, x, *args, **kwargs):
#     axes, s, norm = get_args(x, *args, **kwargs)
#     check_no_repeated_axes(axes)
#     vs = vspace(x)
#     return lambda g: match_complex(x, truncate_pad(fft_fun(g, *args, **kwargs), vs.shape))

# defvjp(fft, lambda *args, **kwargs:
#         fft_grad(get_fft_args, fft, *args, **kwargs))


Nx = 50
Ny = 50
Nz = 1

npml = 10

omega = 2*np.pi*200e12
dL = 5e-8
pml = [npml, npml, 0]

# source parameters
sigma = 10e-15
total_time = 0.4e-12
t0 = sigma * 10

source_amp = 1
source_pos = np.zeros((Nx, Ny, Nz))
source_pos[npml+10, Ny//2, Nz//2] = source_amp

# starting relative permittivity (random for debugging)
eps_r   = np.random.random((Nx, Ny, Nz)) + 1
F = fdtd(eps_r, dL=dL, npml=pml)

steps = int(total_time / F.dt)
print('{} time steps'.format(steps))

gaussian = lambda t: source_pos * source_amp * np.exp(-(t - t0 / F.dt)**2 / 2 / (sigma / F.dt)**2)
source = lambda t: source_pos * gaussian(t) * np.cos(omega * t * F.dt)

plt.plot(F.dt * np.arange(steps), np.sum(source(np.arange(steps)), axis=(0,1)))
plt.show()

measure_pos = np.zeros((Nx, Ny, Nz))
measure_pos[-npml-10, Ny//2, Nz//2] = 1

def objective(eps_space):
    F.eps_r *= eps_space
    measured = []
    for t_index in range(steps):
        fields = F.forward(Jz=source(t_index))
        measured.append(npa.sum(fields['Ez'] * measure_pos))
    # measured_f = npa.fft.fft(npa.array(measured))
    # measured_f = npa.fft.fftshift(measured_f)
    measured_f = dft(steps) @ npa.array(measured)
    spectral_power = npa.square(npa.abs(measured_f))
    return spectral_power

eps_space = 1.0
spectral_power = objective(eps_space)
jac_power = jacobian(objective, mode='forward')(eps_space)

N = steps

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
delta_f = 1/steps/F.dt
freq_x = np.arange(steps) * delta_f
ax1.plot(freq_x, spectral_power[:N], 'k-')
ax2.plot(freq_x, jac_power[:N,0], 'g-')
ax1.set_ylabel('spectral power', color='k')
ax2.set_ylabel('dP/depsilon', color='g')
ax2.spines['right'].set_color('g')
ax2.tick_params(axis='y', colors='g')
plt.show()