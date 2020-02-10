#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('../lib/')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pints

import model as m; m.vhold = 0

"""
Prediction for single model cell experiment data
"""

predict_list = ['staircase', 'sinewave', 'ap-beattie', 'ap-lei']
data_idx = {'staircase': 1, 'sinewave': 0, 'ap-beattie': 2, 'ap-lei': 3}
protocol_list = {
    'staircase': 'staircase-ramp.csv',
    'sinewave': 'sinewave-ramp.csv',
    'ap-beattie': 'ap-beattie.csv',
    'ap-lei': 'ap-lei.csv'}
legend_ncol = {
    'staircase': (2, 1),
    'sinewave': (1, 1),
    'ap-beattie': (4, 2),
    'ap-lei': (4, 2)}


try:
    which_predict = sys.argv[1]
except:
    print('Usage: python %s [str:which_predict]' % os.path.basename(__file__))
    sys.exit()

if which_predict not in predict_list:
    raise ValueError('Input data %s is not available in the predict list' \
            % which_predict)

savedir = './figs'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Load data
import util
idx = [0, data_idx[which_predict], 0]
f = 'data/20191011_mc3re_nocomp.dat'
whole_data, times = util.load(f, idx, vccc=True)
times = times * 1e3  # s -> ms
data_cc = whole_data[2] * 1e3  # V -> mV
data_vc = whole_data[1] * 1e3  # V -> mV
data = (whole_data[0] + whole_data[3]) * 1e12  # A -> pA

#out = np.array([times * 1e-3, data_vc]).T
#np.savetxt('recorded-voltage.csv', out, delimiter=',', comments='',
#        header='\"time\",\"voltage\"')

saveas = 'mc3nocomp'

# Model
model = m.Model('../mmt-model-files/full2-voltage-clamp-mc3.mmt',
                protocol_def=protocol_list[which_predict],
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
]
model.set_parameters(parameters)
parameter_to_fix = [
    'voltageclamp.cprs_est',
    'voltageclamp.cm_est',
    'voltageclamp.rseries_est',
]
parameter_to_fix_values = [
    0.,  # pF; Cprs*
    0.0,  # pF; Cm*
    0,  # GOhm; Rs*
]
fix_p = {}
for i, j in zip(parameter_to_fix, parameter_to_fix_values):
    fix_p[i] = j
model.set_fix_parameters(fix_p)

# Load parameters
loaddir = './out'
loadas = 'mc3nocomp'
fit_seed = 542811797
p = np.loadtxt('%s/%s-solution-%s-1.txt' % (loaddir, loadas, fit_seed))
current_label = 'Fit' if which_predict == 'staircase' else 'Prediction'

# Simulate
extra_log = ['voltageclamp.Vc', 'membrane.V']
simulation = model.simulate(p, times, extra_log=extra_log)
Iout = simulation['voltageclamp.Iout']
Vc = simulation['voltageclamp.Vc']
Vm = simulation['membrane.V']

# Plot
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 4))

#axes[0].plot(times, data_vc, c='#a6bddb', label=r'Measured $V_{cmd}$')
axes[0].plot(times, data_cc, c='#feb24c', label=r'Measured $V_{m}$')
axes[0].plot(times, Vc, ls='-', c='#045a8d', label=r'Input $V_{cmd}$')
axes[0].plot(times, Vm, ls='--', c='#bd0026', label=r'Predicted $V_{m}$')
axes[0].set_ylabel('Voltage (mV)', fontsize=14)
#axes[0].set_xticks([])
axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3, bbox_transform=axes[0].transAxes)

axes[1].plot(times, data, alpha=0.5, label=r'Measurement ($I_{out}$)')
axes[1].plot(times, Iout, ls='--', label=current_label)
axes[1].set_ylim([-800, 1200])  # TODO?
axes[1].legend(loc='lower right', bbox_to_anchor=(1, 1.02), ncol=2, bbox_transform=axes[0].transAxes)
axes[1].set_ylabel('Current (pA)', fontsize=14)
axes[1].set_xlabel('Time (ms)', fontsize=14)
plt.subplots_adjust(hspace=0)
plt.savefig('%s/predict-%s-%s.pdf' % (savedir, saveas, which_predict),
        format='pdf', bbox_inches='tight')
plt.savefig('%s/predict-%s-%s' % (savedir, saveas, which_predict), dpi=300,
        bbox_inches='tight')
plt.close()
