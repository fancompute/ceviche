import numpy as np
import autograd.numpy as npa
import sys
sys.path.append('../ceviche')
sys.path.append('../angler')
import matplotlib.pylab as plt
import argparse
from ceviche.constants import *
from ceviche.utils import imarr, aniplot, my_fft
from ceviche.fdfd import fdfd_ez as fdfd
from ceviche.fdtd import fdtd
from ceviche.jacobians import jacobian

from angler import Simulation

""" PARSE ARGUMENTS """

parser = argparse.ArgumentParser(description='Process those args!.')
parser.add_argument('--plot', dest='plot_all', default=False, action='store_true', help='plot everything')
args = parser.parse_args()
plot_all = args.plot_all
if plot_all:
    print('plotting everything...')

""" DEFINE PARAMETERS """

npml = 10
dl = 10e-8
ff = 0.5
lambda0 = 1550e-9
omega0 = 2 * np.pi * C_0 / lambda0
k0 = 2 * np.pi / lambda0

neff_teeth = 2.846
neff_hole = 2.534

theta_deg = 20
theta = theta_deg / 360 * 2 * np.pi

spc = 2e-6
h0 = 220e-9
h1 = 150e-9
subs = 1e-6
num_teeth = 7

sigma = 40e-15         # pulse duration (s)
total_time = 1.9e-12   # total simulation time (s)
t0 = sigma * 7         # delay of pulse (s)

sub_index = 1.44      # SiO2
grating_index = 3.48  # Si (amorphous)
sub_eps = sub_index ** 2
grating_eps = grating_index ** 2

neff = ff * neff_teeth + (1 - ff) * neff_hole
print('-> effective index of {}'.format(neff))

Lambda = lambda0 / (neff - np.sin(theta))
w = Lambda * (num_teeth + 1)
print('-> grating period of {} nanometers'.format(Lambda / 1e-9))

NPML = [npml, npml]

Nx = npml + int((spc + w + 2 * spc) / dl) + npml                   # left to right
Ny = npml + int((2 * spc + subs + h0 + subs + spc) / dl) + npml    # top to bottom
print('-> dimensions of {} x {}'.format(Nx, Ny))


""" DEFINE PERMITTIVITY """

eps_r = np.ones((Nx, Ny))

# define substrate
eps_r[:, npml + int(spc / dl) : npml + int((spc + 2 * subs + h0) / dl)] = sub_eps

# define grating base
eps_r[:, npml + int((spc + subs) / dl) : npml + int((spc + subs + h1) / dl)] = grating_eps
eps_base = eps_r.copy()  # make a copy to add teeth other ways later

# define grating teeth
print('-> {} grating teeth within length of {} microns'.format(num_teeth, w / 1e-6))
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
source_p = np.sum(np.square(np.abs(source)))

if plot_all:
    plt.imshow(np.real(imarr(eps_r + 3 * np.abs(source))))
    plt.colorbar()
    plt.title('discrete epsilon w/ source')
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()

""" DO AN FDFD RUN TO CHECK THINGS """

if plot_all:
    print('-> solving FDFD for discrete permittivity')
    F = fdfd(omega0, dl, eps_r, NPML)
    Hx, Hy, Ez = F.solve(source)
    plt.imshow(np.abs(imarr(Ez)))
    plt.title('|Ez| for discrete permitivity')
    plt.xlabel('x'); plt.ylabel('y')
    plt.colorbar()
    plt.show()

""" MAKE A CONITNOUS EPSILON FOR PROJECTION """

# define grating teeth
teeth_density = np.zeros((Nx, Ny))
density_arr = np.square(np.sin(2 * np.pi * dl * x_grids / Lambda / 2))
teeth_density[x_grids, npml + int((spc + subs + h1) / dl) : npml + int((spc + subs + h0) / dl)] = density_arr[:, np.newaxis]

def sigmoid(x, strength=1):
    return 1 / (npa.exp(-strength * x) + 1)

def projection(density, center, eps_min, eps_max, strength=15):
    sig_dens = sigmoid(density - center, strength=strength)
    eps = (eps_max - eps_min) * sig_dens
    return eps

eps_teeth_proj = projection(teeth_density, (1 - ff), sub_eps, grating_eps)
eps_total = eps_teeth_proj + eps_base

if plot_all:
    print('-> solving FDFD for continuous epsilon')
    F = fdfd(omega0, dl, eps_total, NPML)
    Hx, Hy, Ez = F.solve(source)
    plt.imshow(np.abs(imarr(Ez)))
    plt.title('|Ez| for continuous permitivity')
    plt.xlabel('x'); plt.ylabel('y')
    plt.colorbar()
    plt.show()


""" DEFINE MODAL SOURCE / PROBE """

print('-> setting up modal source')
sim = Simulation(omega0, eps_base, dl, [npml, npml], pol='Ez', L0=1)
center_y = npml + int((spc + subs + h1 / 2) / dl)
sim.add_mode(neff_hole, 'x', center=[npml + int(spc / dl), center_y], width=int(5 * h1 / dl), scale=1, order=1)
sim.setup_modes()
wg_mode = np.flipud(np.abs(sim.src.copy())).reshape((Nx, Ny, 1))
wg_mode_p = np.sum(np.square(np.abs(wg_mode)))

if plot_all:
    plt.imshow(np.real(imarr(wg_mode)))
    plt.title('modal source array')
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()


""" DEFINE TIME DOMAIN PROBLEM """

eps_total = eps_total.reshape((Nx, Ny, 1))
source = source.reshape((Nx, Ny, 1))

F_t = fdtd(eps_total, dl, [npml, npml, 0])
dt = F_t.dt

steps = int(total_time / dt)
times = np.arange(steps)
print('-> total of {} time steps'.format(steps))

gaussian = lambda t: np.exp(-(t - t0 / dt)**2 / 2 / (sigma / dt)**2) * np.cos(omega0 * t * dt)
# source_fn = lambda t: np.abs(source) * np.real(np.exp(1j * omega0 * dt * t)) * gaussian(t)
source_fn = lambda t: np.abs(wg_mode) * gaussian(t)
spect_in = np.square(np.abs(my_fft(gaussian(times))))
delta_f = 1 / steps / dt
freq_x = np.arange(steps) * delta_f

# compute normalization power spectrum
F_norm = fdtd(eps_base.reshape((Nx, Ny, 1)), dl, [npml, npml, 0])
measured = []
print('-> measuring spectral power in for base waveguide')
for t_index in range(steps):
    if t_index % 1000 == 0:
        print('   - done with {} of {}'.format(t_index, steps))
    fields = F_norm.forward(Jz=source_fn(t_index))
    measured.append(npa.sum(fields['Ez'] * np.flipud(wg_mode)))

# get spectral power
print('-> computing FFT')
measured_f = my_fft(npa.array(measured))
spect_in_measure = npa.square(npa.abs(measured_f)) / wg_mode_p

if plot_all:
    plt.plot(times * F_t.dt / 1e-15, gaussian(times))
    plt.title('source')
    plt.xlabel('time (femtoseconds)')
    plt.ylabel('amplitude')
    plt.show()

if plot_all or True:
    plt.plot(freq_x / 1e12, spect_in, label='direct from source')
    plt.plot(freq_x / 1e12, spect_in_measure, label='measured')
    plt.xlabel('frequency (THz)')
    plt.ylabel('power')
    plt.legend()
    plt.show()

if plot_all:
    pass # takes a long time to run
    # aniplot(F_t, source_fn, steps, component='Ez', num_panels=5)


""" SPECTRAL POWER COMPUTATION """

# reshape bits
eps_base = eps_base.reshape((Nx, Ny, 1))
teeth_density = teeth_density.reshape((Nx, Ny, 1))

def spectral_power(ff):

    # setup FDTD
    eps_teeth_proj = projection(teeth_density, (1 - ff), sub_eps, grating_eps)
    eps_total = eps_teeth_proj + eps_base
    F = fdtd(eps_total, dl, [npml, npml, 0])

    # compute fields
    measured = []
    print('-> running FDTD within objective function')
    for t_index in range(steps):
        if t_index % 1000 == 0:
            print('   - done with {} of {}'.format(t_index, steps))
        fields = F.forward(Jz=source_fn(t_index))
        measured.append(npa.sum(fields['Ez'] * source))

    # get spectral power
    print('-> computing FFT')
    measured_f = my_fft(npa.array(measured))
    spect_power = npa.square(npa.abs(measured_f)) / source_p
    return spect_power

spect = spectral_power(ff)
T = spect / spect_in

n_disp = steps // 2

if plot_all or True:
    # plot spectral power
    fig, ax1 = plt.subplots()
    delta_f = 1 / steps / dt
    freq_x = np.arange(n_disp) * delta_f
    ax1.plot(freq_x / 1e12, spect_in[:n_disp], label='input')
    ax1.plot(freq_x / 1e12, spect[:n_disp], label='measured')
    ax1.set_ylabel('spectral power')
    ax1.set_xlabel('frequency (THz)')
    ax1.legend()
    plt.show()

P_in = np.sum(spect_in[:steps//4])
P_out = np.sum(spect[:steps//4])
P_in_max = np.max(spect_in[:steps//4])
coupling_efficiency = P_out / P_in
print('calculated a coupling efficiency of {} %'.format(coupling_efficiency))


""" DIFFERENTIATION W.R.T. FILL FACTOR """

# compute jacobians
print('-> FMD-ing')
jac_FMD = jacobian(spectral_power, mode='forward')(ff)
# jac_num = jacobian(spectral_power, mode='numerical')(ff)

# plot derivatives along with spectral power
right_color = '#c23b22'
fig, ax1 = plt.subplots()
delta_f = 1 / steps / dt
freq_x = np.arange(n_disp) * delta_f
ax1.plot(freq_x / 1e12, spect_in[:n_disp] / P_in_max, label='input')
ax1.plot(freq_x / 1e12, spect[:n_disp] / P_in_max, label='output')
ax1.set_ylabel('normalized power (P)', color='k')
ax1.set_xlabel('frequency (THz)')
ax1.legend()
ax2 = ax1.twinx()
p2 = ax2.plot(freq_x / 1e12, jac_FMD[:n_disp,0] / spect_in[:n_disp] / P_in_max, color=right_color, label='FMD')
# ax2.plot(freq_x, jac_num[:n_disp,0], 'bo', label='numerical')
ax2.set_ylabel('dP/dff', color=right_color)
ax2.spines['right'].set_color(right_color)
# ax2.legend()
ax2.tick_params(axis='y', colors=right_color)
ax1.set_xlim(left=180, right=210)
ax2.set_xlim(left=180, right=210)
plt.show()


# efficiency version of above plot
right_color = '#c23b22'
fig, ax1 = plt.subplots()
delta_f = 1 / steps / dt
freq_x = np.arange(n_disp) * delta_f
# ax1.plot(freq_x / 1e12, spect_in[:n_disp] / P_in_max, label='input')
ax1.plot(freq_x / 1e12, spect[:n_disp] / P_in_max)
ax1.set_ylabel('coupling efficiency (eff) (1 / Hz)', color='k')
ax1.set_xlabel('frequency (THz)')
ax1.legend()
ax2 = ax1.twinx()
p2 = ax2.plot(freq_x / 1e12, jac_FMD[:n_disp,0] / spect_in[:n_disp], color=right_color, label='FMD')
# ax2.plot(freq_x, jac_num[:n_disp,0], 'bo', label='numerical')
ax2.set_ylabel("d_eff/dff", color=right_color)
ax2.spines['right'].set_color(right_color)
# ax2.legend()
ax2.tick_params(axis='y', colors=right_color)
ax1.set_xlim(left=180, right=210)
ax1.set_ylim(bottom=-0.1, top=1.1)
ax2.set_xlim(left=180, right=210)
ax2.set_ylim(bottom=-1.1, top=1.1)
plt.show()


""" ALL THE PLOTS """

fname_base = './examples/figs/tmp'
# setup
plt.tight_layout()
plt.imshow(np.real(imarr(eps_r + 3 * np.abs(source[:,:,0]))))
plt.colorbar()
plt.title('discrete epsilon w/ source')
plt.xlabel('x'); plt.ylabel('y')
plt.savefig(fname_base + 'setup.pdf', dpi=400)
plt.clf()

# source
plt.tight_layout()
plt.plot(times * F_t.dt / 1e-15, gaussian(times))
plt.title('source')
plt.xlabel('time (femtoseconds)')
plt.ylabel('amplitude')
plt.xlim((0, 500))
plt.savefig(fname_base + 'pulse.pdf', dpi=400)
plt.clf()

# FDFD fields
print('-> solving FDFD for continuous epsilon')
F = fdfd(omega0, dl, eps_total[:,:,0], NPML)
Hx, Hy, Ez = F.solve(source)
plt.tight_layout()
Ez2 = np.square(np.abs(imarr(Ez)))
plt.imshow(Ez2 / Ez2.max(), cmap='magma')
plt.title('|Ez|^2 (normalized)')
plt.xlabel('x'); plt.ylabel('y')
plt.colorbar()
plt.savefig(fname_base + 'Ez2.pdf', dpi=400)
plt.clf()

left_f_P = 180; right_f_P = 210

# spectral power
delta_f = 1 / steps / dt
freq_x = np.arange(n_disp) * delta_f
plt.tight_layout()
plt.plot(freq_x / 1e12, spect_in[:n_disp] / P_in_max, label='input')
plt.plot(freq_x / 1e12, spect[:n_disp] / P_in_max, label='output')
plt.ylabel('normalized power (P)', color='k')
plt.xlabel('frequency (THz)')
plt.xlim(left=left_f_P, right=right_f_P)
plt.legend()
plt.savefig(fname_base + 'powers.pdf', dpi=400)
plt.clf()

# power derivatives
red = '#c23b22'
plt.plot(freq_x / 1e12, jac_FMD[:n_disp,0] / P_in_max, color=red, label='FMD')
plt.plot(freq_x / 1e12, np.zeros((n_disp, )), color=red, linestyle='dashed', linewidth=1)
plt.xlabel('frequency (THz)')
plt.ylabel('deriv of (P) w.r.t. fill factor')
plt.xlim(left=left_f_P, right=right_f_P)
plt.ylim(bottom=-.41, top=.41)
plt.savefig(fname_base + 'd_powers.pdf', dpi=400)
plt.clf()

left_f_n = 180; right_f_n = 210

# spectral efficiencies
green = '#388c51'
plt.tight_layout()
plt.plot(freq_x / 1e12, spect[:n_disp] / spect_in[:n_disp], color=green)
# plt.plot(freq_x / 1e12, np.zeros((n_disp,)), color=green, linestyle='dashed', linewidth=1)
plt.ylabel('coupling efficiency (n)')
plt.xlabel('frequency (THz)')
plt.xlim(left=left_f_n, right=right_f_n)
plt.ylim(bottom=-0.1, top=1.1)
plt.savefig(fname_base + 'efficiencies.pdf', dpi=400)
plt.clf()

# efficiency derivatives
purple = '#51388c'
plt.plot(freq_x / 1e12, jac_FMD[:n_disp,0] / spect_in[:n_disp], color=purple, label='FMD')
plt.plot(freq_x / 1e12, np.zeros((n_disp,)), color=purple, linestyle='dashed', linewidth=1)
plt.xlabel('frequency (THz)')
plt.ylabel('deriv of (n) w.r.t. fill factor')
plt.xlim(left=left_f_n, right=right_f_n)
plt.ylim(bottom=-.71, top=.71)
plt.savefig(fname_base + 'd_efficiencies.pdf', dpi=400)
plt.clf()
