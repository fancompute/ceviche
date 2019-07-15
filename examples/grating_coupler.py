import numpy as np
import autograd.numpy as npa
import sys
sys.path.append('../ceviche')
sys.path.append('../angler')
import matplotlib.pylab as plt

from ceviche.constants import *
from ceviche.utils import imarr, aniplot, my_fft
from ceviche.fdfd import fdfd_ez as fdfd
from ceviche.fdtd import fdtd

from angler.

""" DEFINE PARAMETERS """

npml = 10
dl = 2e-8
ff = 0.5
lambda0 = 1550e-9
omega0 = 2 * np.pi * C_0 / lambda0
k0 = 2 * np.pi / lambda0

neff_teeth = 2.846
neff_hole = 2.534

theta_deg = 20
theta = theta_deg / 360 * 2 * np.pi

spc = 1.5e-6
h0 = 220e-9
h1 = 150e-9
subs = 1e-6
num_teeth = 6

sub_index = 1.44      # SiO2
grating_index = 3.48  # Si (amorphous)
sub_eps = sub_index ** 2
grating_eps = grating_index ** 2

neff = ff * neff_teeth + (1 - ff) * neff_hole
print('->effective index of {}'.format(neff))

Lambda = lambda0 / (neff - np.sin(theta))
w = Lambda * (num_teeth + 1)
print('->grating period of {} nanometers'.format(Lambda / 1e-9))

NPML = [npml, npml]

Nx = npml + int((spc + w + 2 * spc) / dl) + npml                   # left to right
Ny = npml + int((2 * spc + subs + h0 + subs + spc) / dl) + npml    # top to bottom
print('->dimensions of {} x {}'.format(Nx, Ny))


""" DEFINE PERMITTIVITY """

eps_r = np.ones((Nx, Ny))

# define substrate
eps_r[:, npml + int(spc / dl) : npml + int((spc + 2 * subs + h0) / dl)] = sub_eps

# define grating base
eps_r[:, npml + int((spc + subs) / dl) : npml + int((spc + subs + h1) / dl)] = grating_eps
eps_base = eps_r.copy()  # make a copy to add teeth other ways later

# define grating teeth
print('->{} grating teeth within length of {} microns'.format(num_teeth, w / 1e-6))
for i in range(num_teeth):
	tooth_begin = int((spc) / dl) + i * int(Lambda / dl)
	tooth_end = tooth_begin + int((ff * Lambda) / dl)
	eps_r[tooth_begin:tooth_end, npml + int((spc + subs + h1) / dl) : npml + int((spc + subs + h0) / dl)] = grating_eps

""" DEFINE SOURCE """

source = np.zeros((Nx, Ny), dtype=np.complex128)
x_grids = np.arange(npml + int(spc / dl), npml + int((spc + w) / dl))
ky = k0 * np.sin(theta)
source_vals = np.exp(-1j * ky * dl * x_grids)
source[x_grids, -npml - int(spc / dl)] = source_vals
source 

# plt.imshow(np.real(imarr(eps_r + 3 * source)))
# plt.colorbar()
# plt.title('discrete epsilon')
# plt.show()

""" DO AN FDFD RUN TO CHECK THINGS """

# print('->solving FDFD for discrete permittivity')
# F = fdfd(omega0, dl, eps_r, NPML)
# Hx, Hy, Ez = F.solve(source)
# plt.imshow(np.abs(imarr(Ez)))
# plt.colorbar()
# plt.show()


""" MAKE A CONITNOUS EPSILON FOR PROJECTION """

# define grating teeth
teeth_density = np.zeros((Nx, Ny))
density_arr = np.square(np.sin(2 * np.pi * dl * x_grids / Lambda / 2))
teeth_density[x_grids, npml + int((spc + subs + h1) / dl) : npml + int((spc + subs + h0) / dl)] = density_arr[:, np.newaxis]

def sigmoid(x, strength=1):
	return 1 / (np.exp(-strength * x) + 1)

def projection(density, center, eps_min, eps_max, strength=15):
	sig_dens = sigmoid(density - center, strength=strength)
	eps = (eps_max - eps_min) * sig_dens
	return eps

eps_teeth_proj = projection(teeth_density, (1 - ff), sub_eps, grating_eps)
eps_total = eps_teeth_proj + eps_base

# print('->solving FDFD for continuous epsilon')
# F = fdfd(omega0, dl, eps_total, NPML)
# Hx, Hy, Ez = F.solve(source)
# plt.imshow(np.abs(imarr(Ez)))
# plt.colorbar()
# plt.show()

""" DEFINE TIME DOMAIN PROBLEM """

eps_total = eps_total.reshape((Nx, Ny, 1))
source = source.reshape((Nx, Ny, 1))

sigma = 10e-15         # pulse duration (s)
total_time = 0.3e-12   # total simulation time (s)
t0 = sigma * 7         # delay of pulse (s)

F_t = fdtd(eps_total, dl, [npml, npml, 0])
dt = F_t.dt

steps = int(total_time / dt)
print('->total of {} time steps'.format(steps))

gaussian = lambda t: np.exp(-(t - t0 / dt)**2 / 2 / (sigma / dt)**2) * np.cos(omega0 * t * dt)
source_fn = lambda t: np.real(source * np.exp(1j * omega0 * dt * t)) * gaussian(t)

plt.plot(gaussian(np.arange(steps)))
plt.show()

aniplot(F_t, source_fn, steps, component='Ez', num_panels=11)

def spectral_power(ff):
    F.eps_r *= eps_space
    measured = []
    for t_index in range(steps):
        fields = F.forward(Jz=source(t_index))
        measured.append(npa.sum(fields['Ez'] * measure_pos))
    measured_f = my_fft(npa.array(measured))
    spect_power = npa.square(npa.abs(measured_f))
    return spect_power

plt.plot(spectral_power(0.5))
plt.show()


