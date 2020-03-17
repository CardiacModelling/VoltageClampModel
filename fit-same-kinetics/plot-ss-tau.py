#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

savedir = './figs'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

file_dir = './out'
file_name = 'herg25oc1'
temperature = 25.0
temperature += 273.15  # in K
fit_seed = '542811797'
fit_seed_scheme3 = '717354021'

# Load parameters
hbm_mean = np.array([np.NaN, # paper 1 supplement HBM mean parameters
        9.48e-2, 8.69e+1, 2.98e-2, 4.69e+1,
        1.04e+2, 2.19e+1, 8.05e+0, 2.99e+1])  # V, s
scheme3_mean = np.loadtxt('%s/%s-scheme3-simvclinleak/%s-solution-%s.txt' %
        (file_dir, file_name, file_name, fit_seed_scheme3))
scheme3_mean = np.append(np.NaN, scheme3_mean) * 1e3 # add 'g'; mV, ms -> V, s
selectedfile = '../manualselection/manualv2selected-%s.txt' % (file_name)
selectedwell = []
with open(selectedfile, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            selectedwell.append(l.split()[0])
selectedwell = selectedwell[:]
individual_fits = []
for cell in selectedwell:
    cma_file = './fit-results/%s/%s-staircaseramp-%s-solution-%s.txt' \
            % (file_name, file_name, cell, fit_seed)
    individual_fits.append(np.loadtxt(cma_file))
p_all = np.vstack((hbm_mean, scheme3_mean, individual_fits))

# Plot
fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))
axes[0, 0].set_ylabel(r'Steady state probability', fontsize=12)
axes[1, 0].set_ylabel(r'Time constant [ms]', fontsize=14)
for i in range(2):
    axes[-1, i].set_xlabel('Voltage [mV]', fontsize=14)
v = np.linspace(-100, 60, 200)  # mV

colour_list = ['#d95f02', '#d9d9d9', '#1b9e77']

a_vhalf = []
r_vhalf = []
a_tauv = []
r_tauv = []
for i_p, p in enumerate(p_all):
    if i_p == 0:
        i_c = 0
        alpha = 1
    elif i_p == 1:
        i_c = 2
        alpha = 1
    else:
        i_c = 1
        alpha = 0.5

    # k1
    k1_hbm = p[1] * np.exp(p[2] * v * 1e-3)
    # k2
    k2_hbm = p[3] * np.exp(-1 * p[4] * v * 1e-3)
    # k3
    k3_hbm = p[5] * np.exp(p[6] * v * 1e-3)
    # k4
    k4_hbm = p[7] * np.exp(-1 * p[8] * v * 1e-3)

    # HBM
    a_inf_hbm = k1_hbm / (k1_hbm + k2_hbm)
    a_tau_hbm = 1e3 / (k1_hbm + k2_hbm)
    r_inf_hbm = k4_hbm / (k3_hbm + k4_hbm)
    r_tau_hbm = 1e3 / (k3_hbm + k4_hbm)

    # Plot
    for i in range(2):
        axes[0, i].plot(v, a_inf_hbm, ls='-', lw=1.5, c=colour_list[i_c],
                label='_nolegend_' if i_p else r'$a_\infty$', alpha=alpha,
                zorder=-2*i_p)
        axes[0, i].plot(v, r_inf_hbm, ls='--', lw=1.5, c=colour_list[i_c],
                label='_nolegend_' if i_p else r'$r_\infty$', alpha=alpha,
                zorder=-2*i_p)

    axes[1, 0].plot(v, a_tau_hbm, ls='-', lw=1.5, c=colour_list[i_c],
            alpha=alpha, label='_nolegend_' if i_p else r'$\tau_a$',
            zorder=-2*i_p)

    axes[1, 1].plot(v, r_tau_hbm, ls='--', lw=1.5, c=colour_list[i_c],
            alpha=alpha, label='_nolegend_' if i_p else r'$\tau_r$',
            zorder=-2*i_p)

    axes[0, 1].plot(v, a_inf_hbm * r_inf_hbm, ls=':', lw=2, c=colour_list[i_c],
            alpha=alpha, zorder=-2*i_p,
            label='_nolegend_' if i_p else r'$a_\infty \times r_\infty$')

    # Some values
    #print('a_\\infty V_{1/2} = %s mV' % v[np.argmin(np.abs(a_inf_hbm - 0.5))])
    #print('\\tau_a V_{max} = %s mV' % v[np.argmax(a_tau_hbm)])
    #print('max(\\tau_a) = %s ms' % np.max(a_tau_hbm))
    #print('r_\\infty V_{1/2} = %s mV' % v[np.argmin(np.abs(r_inf_hbm - 0.5))])
    #print('\\tau_r V_{max} = %s mV' % v[np.argmax(r_tau_hbm)])
    #print('max(\\tau_r) = %s ms' % np.max(r_tau_hbm))
    a_vhalf.append(v[np.argmin(np.abs(a_inf_hbm - 0.5))])
    r_vhalf.append(v[np.argmin(np.abs(r_inf_hbm - 0.5))])
    a_tauv.append(v[np.argmax(a_tau_hbm)])
    r_tauv.append(v[np.argmax(r_tau_hbm)])

plt.tight_layout(pad=0.2, w_pad=1, h_pad=0.25)

# Save fig
axes[0, 0].set_ylim([-0.05, 1.05])
axes[0, 1].set_ylim([-0.01, 0.26])
axes[0, 1].set_xlim([-65, 45])
for i in range(2):
    for j in range(2):
        axes[i, j].legend()
plt.savefig('%s/ss-tau.pdf' % (savedir), format='pdf',
        bbox_iches='tight')
plt.savefig('%s/ss-tau.png' % (savedir), bbox_iches='tight')
plt.close('all')


fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))
axes[0, 0].set_xlabel(r'$V_{1/2, a_\infty}$', fontsize=12)
axes[0, 1].set_xlabel(r'$V_{1/2, r_\infty}$', fontsize=12)
axes[1, 0].set_xlabel(r'$V$ | $\max(\tau_a)$', fontsize=12)
axes[1, 1].set_xlabel(r'$V$ | $\max(\tau_r)$', fontsize=12)
for i in range(2):
    axes[i, 0].set_ylabel('Frequency', fontsize=14)


axes[0, 0].axvline(a_vhalf[0], color=colour_list[0], label='HBM mean')
axes[0, 1].axvline(r_vhalf[0], color=colour_list[0])
axes[1, 0].axvline(a_tauv[0], color=colour_list[0])
axes[1, 1].axvline(r_tauv[0], color=colour_list[0])

axes[0, 0].axvline(a_vhalf[1], color=colour_list[2], label='VCE mean')
axes[0, 1].axvline(r_vhalf[1], color=colour_list[2])
axes[1, 0].axvline(a_tauv[1], color=colour_list[2])
axes[1, 1].axvline(r_tauv[1], color=colour_list[2])

axes[0, 0].hist(a_vhalf[2:], color=colour_list[1], alpha=0.75,
        label='Individual fits')
axes[0, 1].hist(r_vhalf[2:], color=colour_list[1], alpha=0.75)
axes[1, 0].hist(a_tauv[2:], color=colour_list[1], alpha=0.75)
axes[1, 1].hist(r_tauv[2:], color=colour_list[1], alpha=0.75)

axes[0, 0].legend()
plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0.25)
plt.savefig('%s/ss-tau-hist.pdf' % (savedir), format='pdf',
        bbox_iches='tight')
plt.savefig('%s/ss-tau-hist.png' % (savedir), bbox_iches='tight')
plt.close('all')
