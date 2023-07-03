#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('../lib/')
import os
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pints
import seaborn as sns
from matplotlib.colors import ListedColormap

import model as m; m.vhold = -40

"""
Plot model cell experiment data with amplifier settings.
"""

saveas = 'mc4auto-minimum-tau'

#sim_list = ['staircase', 'sinewave', 'ap-beattie', 'ap-lei']
sim_list = ['nav', 'ramps']
data_idx = {'staircase': 2, 'sinewave': 1, 'ap-lei': 3, 'nav': 0, 'ramps':4}
protocol_list = {
    'nav': [lambda x: (x, np.loadtxt('nav-step.txt'))],
    'ramps': ['ramps/ramps_%s.csv' % i for i in range(10)],
    'staircase': ['staircase-ramp.csv'],
    'sinewave': ['sinewave-ramp.csv'],
    'ap-beattie': ['ap-beattie.csv'],
    'ap-lei': ['ap-lei.csv']}
legend_ncol = {
    'nav': (2, 1),
    'ramps': (2, 1),
    'staircase': (2, 1),
    'sinewave': (1, 1),
    'ap-beattie': (4, 2),
    'ap-lei': (4, 2)}

wins = {
    'nav': np.array([45, 85]),
    'ramps': np.array([45, 150]),
}

try:
    which_sim = sys.argv[1]
except:
    print('Usage: python %s [str:which_sim]' % os.path.basename(__file__))
    sys.exit()

if which_sim not in sim_list:
    raise ValueError('Input data %s is not available in the predict list' \
            % which_sim)

savedir = './figs/plot-%s' % saveas
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Data setup
import util
try:
    n_group = int(sys.argv[2])
except IndexError:
    n_group = 0  # (Rs + pred)
# Check the lab record (docx) for details
USE95 = not False
if USE95:
    n_experiments = {
        0: [0, 1, 2, 3, 4, 5],
        1: [0, 6, 7, 8, 9, 10],
        2: [0, 11, 12, 13, 14, 15],
    }
    height_ratios = [1]*(len(n_experiments[n_group])-1) + [2]
else:
    n_experiments = {
        0: [0, 1, 2, 3, 4],
        1: [0, 6, 7, 8, 9],
        2: [0, 11, 12, 13, 14],
    }
    height_ratios = None

f = 'data/20230510-Cfast_auto.dat'
#f = 'data/20230510.dat'
n_sweeps = 10 if which_sim == 'ramps' else 1

# Model setup
parameters = [
    'mc.gk',
    'mc.ck',
    'mc.gm',
    'voltageclamp.C_prs',
    'mc.Cm',
    'voltageclamp.R_series',
    'voltageclamp.V_offset_eff',
    'voltageclamp.C_prs_est',
    'voltageclamp.Cm_est',
    'voltageclamp.R_series_est',
    'voltageclamp.alpha_R',
    'voltageclamp.alpha_P',
    'voltageclamp.tau_out',
]

# What we have in the circuit
p = [
    1./0.01,  # pA/mV = 1/GOhm; g_kinetics
    100.,  # pF; C_kinetics
    1./0.1,  # pA/mV = 1/GOhm; g_membrane
    7.2,#0.0001,#4.7,  # pF; Cprs
    44.,  # pF; Cm
    30e-3,  # GOhm; Rs
    0,  # mV; Voffset+
    7.2,#0.0001,#7.2,  # pF; Cprs*
    44.,  # pF; Cm*
    30.04e-3,  # GOhm; Rs*
    None,  # alpha_R
    None,  # alpha_P
    None, # tau_out: 110e-3ms
    # For alpha=95%, probably need tau_out=155e-3ms and 30% reduction in Cm*?
]
i_alpha_r, i_alpha_p = -3, -2
i_tau = -1
tau_range = [5e-3, 10e-3, 50e-3, 100e-3, 150e-3, 200e-3]

extra_log = ['voltageclamp.Vc', 'membrane.V']

models = []
for i_sweep in range(n_sweeps):
    #model = m.Model('../mmt-model-files/full2-voltage-clamp-mc4.mmt',
    models.append(
            m.Model('../mmt-model-files/minimum-voltage-clamp-mc4.mmt',
                    protocol_def=protocol_list[which_sim][i_sweep],
                    temperature=273.15 + 23.0,  # K (doesn't matter here)
                    transform=None,  # Not needed
                    readout='voltageclamp.Iout',
                    useFilterCap=False)
        )
    models[i_sweep].set_parameters(parameters)

# Figure setup
fig, axes = plt.subplots(len(n_experiments[n_group]), 2, figsize=(8, 8),
                         sharex='all', sharey='col',
                         height_ratios=height_ratios)
colour_list = sns.color_palette('coolwarm',
                                n_colors=len(tau_range)).as_hex()

# Iterate through experiments and load data/simulate/plot
for i_experiment, n_experiment in enumerate(n_experiments[n_group]):

    # Get data
    data_ccs, data_vcs, datas = [], [], []
    for i_sweep in range(n_sweeps):
        idx = [n_experiment, data_idx[which_sim], i_sweep]
        whole_data, times = util.load(f, idx, vccc=True)
        data_ccs.append( whole_data[3] * 1e3 )  # V -> mV
        data_vcs.append( whole_data[1] * 1e3 )  # V -> mV
        datas.append( (whole_data[0] + whole_data[2]) * 1e12 )  # A -> pA
    times = times * 1e3  # s -> ms
    alpha_r, alpha_p = util.mc4_experiment_alphas[n_experiment]

    p[i_alpha_r] = alpha_r
    p[i_alpha_p] = alpha_p

    # Plot
    ax0 = axes[i_experiment, 0]
    ax1 = axes[i_experiment, 1]
    ax0.set_ylabel(r'$\alpha_R=$'f'{alpha_r*100}%\n'r'$\alpha_P=$'f'{alpha_p*100}%')
    times_x = times - wins[which_sim][0]

    for i, tau in enumerate(tau_range):
        p[i_tau] = tau
        c = colour_list[i]
        # Simulate
        Iouts, Vcs, Vms = [], [], []
        for i_sweep in range(n_sweeps):
            simulation = models[i_sweep].simulate(p, times, extra_log=extra_log)
            Iouts.append( simulation['voltageclamp.Iout'] )
            Vcs.append( simulation['voltageclamp.Vc'] )
            Vms.append( simulation['membrane.V'] )
            ax0.plot(times_x, Vms[-1], alpha=1, c=c)
            ax1.plot(times_x, Iouts[-1], alpha=1, c=c)

    for i, (data_vc, data_cc) in enumerate(zip(data_vcs, data_ccs)):
        ax0.plot(times_x, data_vc, c='#bdbdbd', ls='--', label='_' if i else r'Measured $V_{cmd}$')
        #ax0.plot(times_x, Vc, ls='--', c='#7f7f7f', label='_' if i else r'Input $V_{cmd}$')
        ax0.plot(times_x, data_cc, c='C2', ls='--', label='_' if i else r'Measured $V_{m}$')
        #ax0.plot(times_x, Vm, ls='--', c='C1', label='_' if i else r'Simulated $V_{m}$')

    for i, (data) in enumerate(datas):
        ax1.plot(times_x, data, c='C2', ls='--', label='_' if i else r'Measured $I_{out}$')
        #ax1.plot(times_x, Iout, ls='--', c='C1', label='_' if i else r'Simulated $I_{out}$')

axes[0, 0].set_title('Voltage (mV)')
axes[0, 1].set_title('Current (pA)')
axes[0, 0].legend(loc='lower right', ncol=legend_ncol[which_sim][0],
        bbox_to_anchor=(1.015, 1.25), fontsize=10,
        bbox_transform=axes[0, 0].transAxes)
axes[0, 1].legend(loc='lower right', ncol=legend_ncol[which_sim][1],
        bbox_to_anchor=(1.015, 1.25), fontsize=10,
        bbox_transform=axes[0, 1].transAxes)
axes[-1, 0].set_xlabel('Time (ms)')
axes[-1, 1].set_xlabel('Time (ms)')

axes[0, 0].set_xlim(wins[which_sim] - wins[which_sim][0])
#plt.subplots_adjust(hspace=0)
plt.tight_layout()

if USE95:
    axes[-1, 0].plot([-10, 10], [1.05, 1.05], c='#7f7f7f', lw=1.5, ls='--',
            transform=axes[-1, 0].transAxes, clip_on=False)

# Colorbar
fig.subplots_adjust(top=0.85)
cbar_ax = fig.add_axes([0.1, 0.95, 0.8, 0.0325])
cmap = ListedColormap(colour_list)
cbar = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                        orientation='horizontal')
cbar.ax.get_xaxis().set_ticks([])
for j, tau in enumerate(tau_range):
    cbar.ax.text((2 * j + 1) / (2 * len(tau_range)), .5,
                 f'{int(tau*1e3)} ms',
                 ha='center', va='center', fontsize=10)
cbar.set_label(r'Simulated delay time constant $\tau_\mathrm{z}$')

plt.savefig('%s/simulate-%s-%s-group%s.pdf' % (savedir, saveas, which_sim, n_group), format='pdf')
plt.savefig('%s/simulate-%s-%s-group%s' % (savedir, saveas, which_sim, n_group), dpi=300)
#plt.show()
plt.close()
