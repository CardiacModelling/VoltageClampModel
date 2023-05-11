#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('../lib/')
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pints

import model as m; m.vhold = -40

"""
Simulation for single model cell experiment data with amplifier settings.
"""

saveas = 'mc4nocomp'

sim_list = ['staircase', 'sinewave', 'ap-beattie', 'ap-lei', 'nav', 'ramps']
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


try:
    which_sim = sys.argv[1]
except:
    print('Usage: python %s [str:which_sim]' % os.path.basename(__file__))
    sys.exit()

if which_sim not in sim_list:
    raise ValueError('Input data %s is not available in the predict list' \
            % which_sim)

savedir = './figs'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Load data
import util
#f = 'data/20230510-Cfast_auto.dat'
f = 'data/20230510.dat'
n_sweeps = 10 if which_sim == 'ramps' else 1
data_ccs, data_vcs, datas = [], [], []
for i_sweep in range(n_sweeps):
    idx = [0, data_idx[which_sim], i_sweep]  # 0 = no compensation
    whole_data, times = util.load(f, idx, vccc=True)
    data_ccs.append( whole_data[3] * 1e3 )  # V -> mV
    data_vcs.append( whole_data[1] * 1e3 )  # V -> mV
    datas.append( (whole_data[0] + whole_data[2]) * 1e12 )  # A -> pA
times = times * 1e3  # s -> ms

#out = np.array([times * 1e-3, data_vc]).T
#np.savetxt('recorded-voltage.csv', out, delimiter=',', comments='',
#        header='\"time\",\"voltage\"')


# Model
parameters = [
    'mc.gk',
    'mc.ck',
    'mc.gm',
    'voltageclamp.cprs',
    'membrane.cm',
    'voltageclamp.rseries',
    'voltageclamp.voffset_eff',
    'voltageclamp.cprs_est',
    'voltageclamp.cm_est',
    'voltageclamp.rseries_est',
    'voltageclamp.alpha_r',
    'voltageclamp.alpha_p',
]

# What we have in the circuit
p = [
    1./0.01,  # pA/mV = 1/GOhm; g_kinetics
    100.,  # pF; C_kinetics
    1./0.1,  # pA/mV = 1/GOhm; g_membrane
    4.7,  # pF; Cprs
    44.,  # pF; Cm
    30e-3,  # GOhm; Rs
    0,  # mV; Voffset+
    0.,  # pF; Cprs*
    0.,  # pF; Cm*
    0.,  # GOhm; Rs*
    0.,  # alpha_R
    0.,  # alpha_P
]

extra_log = ['voltageclamp.Vc', 'membrane.V']

Iouts, Vcs, Vms = [], [], []
for i_sweep in range(n_sweeps):
    model = m.Model('../mmt-model-files/full2-voltage-clamp-mc4.mmt',
                    protocol_def=protocol_list[which_sim][i_sweep],
                    temperature=273.15 + 23.0,  # K
                    transform=None,
                    readout='voltageclamp.Iout',
                    useFilterCap=False)
    model.set_parameters(parameters)

    # Simulate
    simulation = model.simulate(p, times, extra_log=extra_log)
    Iouts.append( simulation['voltageclamp.Iout'] )
    Vcs.append( simulation['voltageclamp.Vc'] )
    Vms.append( simulation['membrane.V'] )


# Plot
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))

for i, (data_vc, data_cc, Vc, Vm) in enumerate(zip(data_vcs, data_ccs, Vcs, Vms)):
    axes[0].plot(times, data_vc, c='#a6bddb', label='_' if i else r'Measured $V_{cmd}$')
    axes[0].plot(times, data_cc, c='#feb24c', label='_' if i else r'Measured $V_{m}$')
    axes[0].plot(times, Vc, ls='--', c='#045a8d', label='_' if i else r'Input $V_{cmd}$')
    axes[0].plot(times, Vm, ls='--', c='#bd0026', label='_' if i else r'Simulated $V_{m}$')
axes[0].set_ylabel('Voltage (mV)', fontsize=14)
#axes[0].set_xticks([])
axes[0].legend(ncol=legend_ncol[which_sim][0])

for i, (data, Iout) in enumerate(zip(datas, Iouts)):
    axes[1].plot(times, data, alpha=0.5, c='C0', label='_' if i else 'Measurement')
    axes[1].plot(times, Iout, ls='--', c='C1', label='_' if i else 'Simulation')
#axes[1].set_ylim([-800, 1200])  # TODO?
axes[1].legend(ncol=legend_ncol[which_sim][1])
axes[1].set_ylabel('Current (pA)', fontsize=14)
axes[1].set_xlabel('Time (ms)', fontsize=14)
plt.subplots_adjust(hspace=0)
plt.savefig('%s/simulate-%s-%s.pdf' % (savedir, saveas, which_sim),
        format='pdf', bbox_inches='tight')
plt.savefig('%s/simulate-%s-%s' % (savedir, saveas, which_sim), dpi=300,
        bbox_inches='tight')
plt.show()
plt.close()
