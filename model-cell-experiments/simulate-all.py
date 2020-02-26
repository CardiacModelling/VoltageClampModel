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
Simulation for single model cell experiment data with amplifier settings.
"""

sim_list = ['staircase', 'sinewave', 'ap-beattie', 'ap-lei']
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
idx = [0, data_idx[which_sim], 0]
f = 'data/20191002_mc_nocomp.dat'
whole_data, times = util.load(f, idx, vccc=True)
times = times * 1e3  # s -> ms
data_cc = whole_data[2] * 1e3  # V -> mV
data_vc = whole_data[1] * 1e3  # V -> mV
data = (whole_data[0] + whole_data[3]) * 1e12  # A -> pA

# Chop the last bit
times_1 = times[:154000]
data_cc_1 = data_cc[:154000]
data_vc_1 = data_vc[:154000]
data_1 = data[:154000]

saveas = 'mcnocomp'

# Model
model_1 = m.Model('../mmt-model-files/full2-voltage-clamp-mc.mmt',
                protocol_def=protocol_list[which_sim],
                temperature=273.15 + 23.0,  # K
                transform=None,
                readout='voltageclamp.Iout',
                useFilterCap=False)
parameters = [
    'mc.g',
    'voltageclamp.cprs',
    'membrane.cm',
    'voltageclamp.rseries',
    'voltageclamp.voffset_eff',
    'voltageclamp.cprs_est',
    'voltageclamp.cm_est',
    'voltageclamp.rseries_est',
]
model_1.set_parameters(parameters)

# Set parameters
#''' What we have in the circuit
p = [
    1. / 0.5,  # pA/mV = 1/GOhm; g_membrane
    4.7,  # pF; Cprs
    22,  # pF; Cm
    30e-3,  # GOhm; Rs
    0,  # mV; Voffset+
    0.,  # pF; Cprs*
    0.,  # pF; Cm*
    0.,  # GOhm; Rs*
]
''' What the machine thought
p = [
    1. / 0.5,  # pA/mV = 1/GOhm; g_membrane
    7.80,  # pF; Cprs
    32.85,  # pF; Cm
    32.6e-3,  # GOhm; Rs
    0.2,  # mV; Voffset+
    0.,  # pF; Cprs*
    0.,  # pF; Cm*
    0.,  # GOhm; Rs*
]
#'''

# Simulate
extra_log = ['voltageclamp.Vc', 'membrane.V']
simulation = model_1.simulate(p, times_1, extra_log=extra_log)
Iout_1 = simulation['voltageclamp.Iout']
Vc_1 = simulation['voltageclamp.Vc']
Vm_1 = simulation['membrane.V']

# Plot 1
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 4))

axes[0].plot(times_1, data_cc_1, c='#feb24c', label=r'Measured $V_{m}$')
axes[0].plot(times_1, Vc_1, ls='-', c='#045a8d', label=r'Input $V_{cmd}$')
axes[0].plot(times_1, Vm_1, ls='--', c='#bd0026', label=r'Simulated $V_{m}$')
axes[0].set_ylabel('Voltage (mV)', fontsize=14)
#axes[0].set_xticks([])
axes[0].legend(ncol=legend_ncol[which_sim][0])

axes[1].plot(times_1, data_1, alpha=0.5, label=r'Measured $I_{out}$')
axes[1].plot(times_1, Iout_1, ls='--', label='Simulated $I_{out}$')
axes[1].set_ylim([-400, 200])  # TODO?
axes[1].legend(ncol=legend_ncol[which_sim][1])
axes[1].set_ylabel('Current (pA)', fontsize=14)
axes[1].set_xlabel('Time (ms)', fontsize=14)
plt.subplots_adjust(hspace=0)
plt.savefig('%s/simulate-all-%s-%s.pdf' % (savedir, saveas, which_sim),
        format='pdf', bbox_inches='tight')
plt.savefig('%s/simulate-all-%s-%s' % (savedir, saveas, which_sim), dpi=300,
        bbox_inches='tight')
plt.close()


###############################################################################
f = 'data/20191011_mc3re_nocomp.dat'
whole_data, times = util.load(f, idx, vccc=True)
times = times * 1e3  # s -> ms
data_cc = whole_data[2] * 1e3  # V -> mV
data_vc = whole_data[1] * 1e3  # V -> mV
data = (whole_data[0] + whole_data[3]) * 1e12  # A -> pA

# Chop the last bit
times_2 = times[:154000]
data_cc_2 = data_cc[:154000]
data_vc_2 = data_vc[:154000]
data_2 = data[:154000]

#out = np.array([times * 1e-3, data_vc]).T
#np.savetxt('recorded-voltage.csv', out, delimiter=',', comments='',
#        header='\"time\",\"voltage\"')

saveas = 'mc3nocomp'

# Model
model_2 = m.Model('../mmt-model-files/full2-voltage-clamp-mc3.mmt',
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
model_2.set_parameters(parameters)

# Set parameters
#''' What we have in the circuit
p = [
    1. / 0.1,  # pA/mV = 1/GOhm; g_kinetics
    1000.,  # pF; C_kinetics
    1. / 0.5,  # pA/mV = 1/GOhm; g_membrane
    4.7,  # pF; Cprs
    22,  # pF; Cm
    30e-3,  # GOhm; Rs
    0,  # mV; Voffset+
    0.,  # pF; Cprs*
    0.,  # pF; Cm*
    0.,  # GOhm; Rs*
]

# Simulate
extra_log = ['voltageclamp.Vc', 'membrane.V']
simulation = model_2.simulate(p, times_2, extra_log=extra_log)
Iout_2 = simulation['voltageclamp.Iout']
Vc_2 = simulation['voltageclamp.Vc']
Vm_2 = simulation['membrane.V']

# Plot
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 4))

axes[0].plot(times_2, data_cc_2, c='#feb24c', label=r'Measured $V_{m}$')
axes[0].plot(times_2, Vc_2, ls='-', c='#045a8d', label=r'Input $V_{cmd}$')
axes[0].plot(times_2, Vm_2, ls='--', c='#bd0026', label=r'Simulated $V_{m}$')
axes[0].set_ylabel('Voltage (mV)', fontsize=14)
#axes[0].set_xticks([])
axes[0].legend(ncol=legend_ncol[which_sim][0])

axes[1].plot(times_2, data_2, alpha=0.5, label=r'Measurement ($I_{out}$')
axes[1].plot(times_2, Iout_2, ls='--', label='Simulated $I_{out}$')
axes[1].set_ylim([-800, 1200])  # TODO?
axes[1].legend(ncol=legend_ncol[which_sim][1])
axes[1].set_ylabel('Current (pA)', fontsize=14)
axes[1].set_xlabel('Time (ms)', fontsize=14)
plt.subplots_adjust(hspace=0)
plt.savefig('%s/simulate-all-%s-%s.pdf' % (savedir, saveas, which_sim),
        format='pdf', bbox_inches='tight')
plt.savefig('%s/simulate-all-%s-%s' % (savedir, saveas, which_sim), dpi=300,
        bbox_inches='tight')
plt.close()

###############################################################################
f = 'data/20191002_mc_auto.dat'
whole_data, times = util.load(f, idx, vccc=True)
times = times * 1e3  # s -> ms
data_cc = whole_data[2] * 1e3  # V -> mV
data_vc = whole_data[1] * 1e3  # V -> mV
data = (whole_data[0] + whole_data[3]) * 1e12  # A -> pA

# Chop the last bit
times_3 = times[:154000]
data_cc_3 = data_cc[:154000]
data_vc_3 = data_vc[:154000]
data_3 = data[:154000]

saveas = 'mcauto'

# Model
model_3 = m.Model('../mmt-model-files/full2-voltage-clamp-mc.mmt',
                protocol_def=protocol_list[which_sim],
                temperature=273.15 + 23.0,  # K
                transform=None,
                readout='voltageclamp.Iout',
                useFilterCap=False)
parameters = [
    'mc.g',
    'voltageclamp.cprs',
    'membrane.cm',
    'voltageclamp.rseries',
    'voltageclamp.voffset_eff',
    'voltageclamp.cprs_est',
    'voltageclamp.cm_est',
    'voltageclamp.rseries_est',
]
model_3.set_parameters(parameters)

# Set parameters
#''' What we have in the circuit
p = [
    1. / 0.5,  # pA/mV = 1/GOhm; g_membrane
    4.7,  # pF; Cprs
    22,  # pF; Cm
    30e-3,  # GOhm; Rs
    0.2,  # mV; Voffset+
    7.8,  # pF; Cprs*
    32.85,  # pF; Cm*
    32.6e-3 * 0.8,  # GOhm; Rs* (multiplied with % compensation)
]

# Simulate
extra_log = ['voltageclamp.Vc', 'membrane.V']
simulation = model_3.simulate(p, times_3, extra_log=extra_log)
Iout_3 = simulation['voltageclamp.Iout']
Vc_3 = simulation['voltageclamp.Vc']
Vm_3 = simulation['membrane.V']

# Plot
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 4))

axes[0].plot(times_3, data_cc_3, c='#feb24c', label=r'Measured $V_{m}$')
axes[0].plot(times_3, Vc_3, ls='-', c='#045a8d', label=r'Input $V_{cmd}$')
axes[0].plot(times_3, Vm_3, ls='--', c='#bd0026', label=r'Simulated $V_{m}$')
axes[0].set_ylabel('Voltage (mV)', fontsize=14)
#axes[0].set_xticks([])
axes[0].legend(ncol=legend_ncol[which_sim][0])

axes[1].plot(times_3, data_3, alpha=0.5, label=r'Measured $I_{out}$')
axes[1].plot(times_3, Iout_3, ls='--', label='Simulated $I_{out}$')
axes[1].set_ylim([-400, 200])  # TODO?
axes[1].legend(ncol=legend_ncol[which_sim][1])
axes[1].set_ylabel('Current (pA)', fontsize=14)
axes[1].set_xlabel('Time (ms)', fontsize=14)
plt.subplots_adjust(hspace=0)
plt.savefig('%s/simulate-all-%s-%s.pdf' % (savedir, saveas, which_sim),
        format='pdf', bbox_inches='tight')
plt.savefig('%s/simulate-all-%s-%s' % (savedir, saveas, which_sim), dpi=300,
        bbox_inches='tight')
plt.close()

###############################################################################
f = 'data/20191011_mc3re_auto.dat'
whole_data, times = util.load(f, idx, vccc=True)
times = times * 1e3  # s -> ms
data_cc = whole_data[2] * 1e3  # V -> mV
data_vc = whole_data[1] * 1e3  # V -> mV
data = (whole_data[0] + whole_data[3]) * 1e12  # A -> pA

# Chop the last bit
times_4 = times[:154000]
data_cc_4 = data_cc[:154000]
data_vc_4 = data_vc[:154000]
data_4 = data[:154000]

saveas = 'mc3auto'

# Model
model_4 = m.Model('../mmt-model-files/full2-voltage-clamp-mc3.mmt',
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
model_4.set_parameters(parameters)

# Set parameters
#''' What we have in the circuit
p = [
    1. / 0.1,  # pA/mV = 1/GOhm; g_kinetics
    1000.,  # pF; C_kinetics
    1. / 0.5,  # pA/mV = 1/GOhm; g_membrane
    4.7,  # pF; Cprs
    22.,  # pF; Cm
    30e-3,  # GOhm; Rs
    -1.2,  # mV; Voffset+
    8.88,  # pF; Cprs*
    41.19,  # pF; Cm*
    33.6e-3 * 0.8,  # GOhm; Rs* (multiplied with % compensation)
]

# Simulate
extra_log = ['voltageclamp.Vc', 'membrane.V']
simulation = model_4.simulate(p, times_4, extra_log=extra_log)
Iout_4 = simulation['voltageclamp.Iout']
Vc_4 = simulation['voltageclamp.Vc']
Vm_4 = simulation['membrane.V']

# Plot
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 4))

axes[0].plot(times_4, data_cc_4, c='#feb24c', label=r'Measured $V_{m}$')
axes[0].plot(times_4, Vc_4, ls='-', c='#045a8d', label=r'Input $V_{cmd}$')
axes[0].plot(times_4, Vm_4, ls='--', c='#bd0026', label=r'Simulated $V_{m}$')
axes[0].set_ylabel('Voltage (mV)', fontsize=14)
#axes[0].set_xticks([])
axes[0].legend(ncol=legend_ncol[which_sim][0])

axes[1].plot(times_4, data_4, alpha=0.5, label=r'Measured $I_{out}$')
axes[1].plot(times_4, Iout_4, ls='--', label='Simulated $I_{out}$')
axes[1].set_ylim([-800, 1200])  # TODO?
axes[1].legend(ncol=legend_ncol[which_sim][1])
axes[1].set_ylabel('Current (pA)', fontsize=14)
axes[1].set_xlabel('Time (ms)', fontsize=14)
plt.subplots_adjust(hspace=0)
plt.savefig('%s/simulate-all-%s-%s.pdf' % (savedir, saveas, which_sim),
        format='pdf', bbox_inches='tight')
plt.savefig('%s/simulate-all-%s-%s' % (savedir, saveas, which_sim), dpi=300,
        bbox_inches='tight')
plt.close()

###############################################################################
###############################################################################
###############################################################################

fig = plt.figure(figsize=(10, 6))
grid = plt.GridSpec(3 + 5 + 10 + 2 + 10, 2, hspace=0.0, wspace=0.2)

ax0 = fig.add_subplot(grid[0:3, :])

ax1_1 = fig.add_subplot(grid[8:13, 0])
ax1_2 = fig.add_subplot(grid[13:18, 0])
ax1_1.set_xticklabels([])

ax2_1 = fig.add_subplot(grid[8:13, 1])
ax2_2 = fig.add_subplot(grid[13:18, 1])
ax2_1.set_xticklabels([])

ax3_1 = fig.add_subplot(grid[20:25, 0])
ax3_2 = fig.add_subplot(grid[25:30, 0])
ax3_1.set_xticklabels([])

ax4_1 = fig.add_subplot(grid[20:25, 1])
ax4_2 = fig.add_subplot(grid[25:30, 1])
ax4_1.set_xticklabels([])

t_lower = 12400#0
t_upper = 15400#3400
zoom = np.where(((times_1 > t_lower) & (times_1 < t_upper)))[0]

ax1_1.text(0.5, 1.2, 'Type I', transform=ax1_1.transAxes, size=12, weight='bold', horizontalalignment='center', verticalalignment='bottom')
ax2_1.text(0.5, 1.2, 'Type II', transform=ax2_1.transAxes, size=12, weight='bold', horizontalalignment='center', verticalalignment='bottom')
ax1_1.text(-0.3, 0, 'Uncompensated', transform=ax1_1.transAxes, rotation=90, size=12, weight='bold', horizontalalignment='right', verticalalignment='center')
ax3_1.text(-0.3, 0, 'Compensated', transform=ax3_1.transAxes, rotation=90, size=12, weight='bold', horizontalalignment='right', verticalalignment='center')

# Protocol
ax0.plot(times_1, Vc_1, c='#7f7f7f', label=r'Input $V_{cmd}$')
ax0.set_ylabel('Voltage\n(mV)', fontsize=12)
ax0.set_xlabel('Time (ms)', fontsize=12)
ax0.set_xlim((times_1[0], times_1[-1]))
ax0.axvspan(t_lower, t_upper, alpha=0.25, color='#7f7f7f')

# 1
ax1_1.plot(times_1[zoom], data_cc_1[zoom], c='#feb24c', label=r'Measured $V_{m}$')
ax1_1.plot(times_1[zoom], Vc_1[zoom], ls='-', c='#7f7f7f', label=r'Input $V_{cmd}$')
ax1_1.plot(times_1[zoom], Vm_1[zoom], ls='--', c='#bd0026', label=r'Simulated $V_{m}$')
ax1_1.set_ylabel('Voltage\n(mV)', fontsize=12)
#TODO set legend once outside
ax1_1.legend(loc='lower left', bbox_to_anchor=(-0.01, 1.02), ncol=3, bbox_transform=ax0.transAxes)

ax1_2.plot(times_1[zoom], data_1[zoom], label=r'Measured $I_{out}$')
ax1_2.plot(times_1[zoom], Iout_1[zoom], ls='--', label='Simulated $I_{out}$')
ax1_2.set_ylim([-400, 200])
ax1_2.set_ylabel('Current\n(pA)', fontsize=12)
#TODO set legend once outside
ax1_2.legend(loc='lower right', bbox_to_anchor=(1.01, 1.02), ncol=2, bbox_transform=ax0.transAxes)

ax1_1.text(-0.15, 0.95, '(A)', transform=ax1_1.transAxes, size=12, weight='bold')
ax1_1.set_xlim((t_lower, t_upper))
ax1_2.set_xlim((t_lower, t_upper))

# 2
ax2_1.plot(times_2[zoom], data_cc_2[zoom], c='#feb24c', label=r'Measured $V_{m}$')
ax2_1.plot(times_2[zoom], Vc_2[zoom], ls='-', c='#7f7f7f', label=r'Input $V_{cmd}$')
ax2_1.plot(times_2[zoom], Vm_2[zoom], ls='--', c='#bd0026', label=r'Simulated $V_{m}$')
#ax2_1.set_ylabel('Voltage (mV)', fontsize=14)

ax2_2.plot(times_2[zoom], data_2[zoom], label=r'Measured $I_{out}$')
ax2_2.plot(times_2[zoom], Iout_2[zoom], ls='--', label='Simulated $I_{out}$')
ax2_2.set_ylim([-1200, 1200])
#ax2_2.set_ylabel('Current (pA)', fontsize=14)

ax2_1.text(-0.15, 0.95, '(B)', transform=ax2_1.transAxes, size=12, weight='bold')
ax2_1.set_xlim((t_lower, t_upper))
ax2_2.set_xlim((t_lower, t_upper))

# 3
ax3_1.plot(times_3[zoom], data_cc_3[zoom], c='#feb24c', label=r'Measured $V_{m}$')
ax3_1.plot(times_3[zoom], Vc_3[zoom], ls='-', c='#7f7f7f', label=r'Input $V_{cmd}$')
ax3_1.plot(times_3[zoom], Vm_3[zoom], ls='--', c='#bd0026', label=r'Simulated $V_{m}$')
ax3_1.set_ylabel('Voltage\n(mV)', fontsize=12)

ax3_2.plot(times_3[zoom], data_3[zoom], label=r'Measured $I_{out}$')
ax3_2.plot(times_3[zoom], Iout_3[zoom], ls='--', label='Simulated $I_{out}$')
ax3_2.set_ylim([-400, 200])
ax3_2.set_ylabel('Current\n(pA)', fontsize=12)
ax3_2.set_xlabel('Time (ms)', fontsize=12)

ax3_1.text(-0.15, 0.95, '(C)', transform=ax3_1.transAxes, size=12, weight='bold')
ax3_1.set_xlim((t_lower, t_upper))
ax3_2.set_xlim((t_lower, t_upper))

# 4
ax4_1.plot(times_4[zoom], data_cc_4[zoom], c='#feb24c', label=r'Measured $V_{m}$')
ax4_1.plot(times_4[zoom], Vc_4[zoom], ls='-', c='#7f7f7f', label=r'Input $V_{cmd}$')
ax4_1.plot(times_4[zoom], Vm_4[zoom], ls='--', c='#bd0026', label=r'Simulated $V_{m}$')
#ax4_1.set_ylabel('Voltage (mV)', fontsize=14)

ax4_2.plot(times_4[zoom], data_4[zoom], label=r'Measured $I_{out}$')
ax4_2.plot(times_4[zoom], Iout_4[zoom], ls='--', label='Simulated $I_{out}$')
ax4_2.set_ylim([-1200, 1200])
#ax4_2.set_ylabel('Current (pA)', fontsize=14)
ax4_2.set_xlabel('Time (ms)', fontsize=12)

ax4_1.text(-0.15, 0.95, '(D)', transform=ax4_1.transAxes, size=12, weight='bold')
ax4_1.set_xlim((t_lower, t_upper))
ax4_2.set_xlim((t_lower, t_upper))

# Done
plt.savefig('%s/simulate-all-%s.pdf' % (savedir, which_sim),
        format='pdf', bbox_inches='tight')
plt.savefig('%s/simulate-all-%s' % (savedir, which_sim), dpi=300,
        bbox_inches='tight')
plt.close()
