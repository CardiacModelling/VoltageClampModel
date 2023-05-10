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

import model as m; m.vhold = 0

"""
Simulation for single model cell experiment data with amplifier settings.
"""

sim_list = ['staircase', 'sinewave', 'ap-beattie', 'ap-lei', 'nav']
data_idx = {'staircase': 1, 'sinewave': 0, 'ap-beattie': 2, 'ap-lei': 3}
protocol_list = {
    'nav': lambda x: (x, np.loadtxt('../protocol-time-series/ina-steps.txt')),
    'staircase': 'staircase-ramp.csv',
    'sinewave': 'sinewave-ramp.csv',
    'ap-beattie': 'ap-beattie.csv',
    'ap-lei': 'ap-lei.csv'}
legend_ncol = {
    'nav': (2, 1),
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
'''
import util
idx = [0, data_idx[which_sim], 0]
f = 'data/20191011_mc3re_auto.dat'
whole_data, times = util.load(f, idx, vccc=True)
times = times * 1e3  # s -> ms
data_cc = whole_data[2] * 1e3  # V -> mV
data_vc = whole_data[1] * 1e3  # V -> mV
data = (whole_data[0] + whole_data[3]) * 1e12  # A -> pA
'''
times = np.arange(0, 5000, 0.1) #TODO

#out = np.array([times * 1e-3, data_vc]).T
#np.savetxt('recorded-voltage.csv', out, delimiter=',', comments='',
#        header='\"time\",\"voltage\"')

saveas = 'mc3auto'

# Model
model = m.Model('../mmt-model-files/full2-voltage-clamp-mc3.mmt',
                protocol_def=protocol_list[which_sim],
                temperature=273.15 + 23.0,  # K
                transform=None,
                readout='voltageclamp.Iout',
                useFilterCap=False)
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
]
model.set_parameters(parameters)

# Set parameters
#''' What we have in the circuit
p = [
    1./0.01,#1. / 0.1,  # pA/mV = 1/GOhm; g_kinetics
    100.,#1000.,  # pF; C_kinetics
    1./0.1,#1. / 0.5,  # pA/mV = 1/GOhm; g_membrane
    0.0001,#4.7,  # pF; Cprs
    44.,#22,  # pF; Cm
    30e-3,  # GOhm; Rs
    -1.2,  # mV; Voffset+
    0.001,#8.88,  # pF; Cprs*
    40.,#41.19,  # pF; Cm*
    33.6e-3 * 0.8,  # GOhm; Rs* (multiplied with % compensation)
]
''' What the machine thought
p = [
    1. / 0.1,  # pA/mV = 1/GOhm; g_kinetics
    1000.,  # pF; C_kinetics
    1. / 0.5,  # pA/mV = 1/GOhm; g_membrane
    8.88,  # pF; Cprs
    41.19,  # pF; Cm
    33.6e-3,  # GOhm; Rs
    -1.2,  # mV; Voffset+
    8.88,  # pF; Cprs*
    41.19,  # pF; Cm*
    33.6e-3 * 0.8,  # GOhm; Rs* (multiplied with % compensation)
]
#'''

# Simulate
extra_log = ['voltageclamp.Vc', 'membrane.V']
simulation = model.simulate(p, times, extra_log=extra_log)
Iout = simulation['voltageclamp.Iout']
Vc = simulation['voltageclamp.Vc']
Vm = simulation['membrane.V']

# Plot
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))

#axes[0].plot(times, data_vc, c='#a6bddb', label=r'Measured $V_{cmd}$')
#axes[0].plot(times, data_cc, c='#feb24c', label=r'Measured $V_{m}$')
axes[0].plot(times, Vc, ls='--', c='#045a8d', label=r'Input $V_{cmd}$')
axes[0].plot(times, Vm, ls='--', c='#bd0026', label=r'Simulated $V_{m}$')
axes[0].set_ylabel('Voltage (mV)', fontsize=14)
#axes[0].set_xticks([])
axes[0].legend(ncol=legend_ncol[which_sim][0])

#axes[1].plot(times, data, alpha=0.5, label='Measurement')
axes[1].plot(times, Iout, ls='--', label='Simulation')
#axes[1].set_ylim([-800, 1200])  # TODO?
axes[1].set_xlim([2635, 2695])
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
