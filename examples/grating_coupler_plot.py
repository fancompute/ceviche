import numpy as np

# setup
plt.tight_layout()
plt.imshow(np.real(imarr(eps_r + 3 * np.abs(source[:,:,0]))))
plt.colorbar()
plt.title('discrete epsilon w/ source')
plt.xlabel('x'); plt.ylabel('y')
plt.savefig('./examples/figs/setup.pdf', dpi=400)
plt.clf()

# source
plt.tight_layout()
plt.plot(times * F_t.dt / 1e-15, gaussian(times))
plt.title('source')
plt.xlabel('time (femtoseconds)')
plt.ylabel('amplitude')
plt.xlim((0, 500))
plt.savefig('./examples/figs/pulse.pdf', dpi=400)
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
plt.savefig('./examples/figs/Ez2.pdf', dpi=400)
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
plt.savefig('./examples/figs/powers.pdf', dpi=400)
plt.clf()

# power derivatives
red = '#c23b22'
plt.plot(freq_x / 1e12, jac_FMD[:n_disp,0] / P_in_max, color=red, label='FMD')
plt.plot(freq_x / 1e12, np.zeros((n_disp, )), color=red, linestyle='dashed', linewidth=1)
plt.xlabel('frequency (THz)')
plt.ylabel('deriv of (P) w.r.t. fill factor')
plt.xlim(left=left_f_P, right=right_f_P)
plt.ylim(bottom=-.41, top=.41)
plt.savefig('./examples/figs/d_powers.pdf', dpi=400)
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
plt.savefig('./examples/figs/efficiencies.pdf', dpi=400)
plt.clf()

# efficiency derivatives
purple = '#51388c'
plt.plot(freq_x / 1e12, jac_FMD[:n_disp,0] / spect_in[:n_disp], color=purple, label='FMD')
plt.plot(freq_x / 1e12, np.zeros((n_disp,)), color=purple, linestyle='dashed', linewidth=1)
plt.xlabel('frequency (THz)')
plt.ylabel('deriv of (n) w.r.t. fill factor')
plt.xlim(left=left_f_n, right=right_f_n)
plt.ylim(bottom=-.71, top=.71)
plt.savefig('./examples/figs/d_efficiencies.pdf', dpi=400)
plt.clf()


