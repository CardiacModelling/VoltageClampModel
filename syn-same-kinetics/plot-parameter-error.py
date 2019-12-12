#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib/')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pints
import pickle
import seaborn as sns
from scipy.interpolate import interp1d

import model as m
import syn
from parameters import full2vc, full2vc_typical_values
import parametertransform
from priors import IKrLogPrior, VoltageClampLogPrior
from protocols import leak_staircase as protocol_def

loaddir = 'out/syn-full2vclinleak-full2vclinleak'
loaddir2 = 'out/syn-full2vclinleak-full2vclinleak-true'

parameter_names = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
        r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', r'$C_m$', r'$R_s$',
        r'$C_{prs}$', r'$V^{\dagger}_{off}$', r'$g_{Leak}$']

# Model
model = m.Model('../mmt-model-files/full2-voltage-clamp-ikr-linleak.mmt',
                protocol_def=protocol_def,
                temperature=273.15 + 25,  # K
                transform=parametertransform.donothing,
                useFilterCap=False)  # ignore capacitive spike
# Set which parameters to be inferred
model.set_parameters(full2vc + ['voltageclamp.gLeak'])

fakedatanoise = 20.0

# Time
times = np.arange(0, 15.2e3, 0.5)  # dt=0.5ms

fig = plt.figure(figsize=(10, 3))
grid = plt.GridSpec(2, 2, hspace=0, wspace=0)
axes = np.empty(3, dtype=object)

axes[0] = fig.add_subplot(grid[:1, :1])
axes[1] = fig.add_subplot(grid[1:, :1])
axes[2] = fig.add_subplot(grid[:, 1:])
axes[0].set_xlim((times[0], times[-1]))
axes[0].set_ylabel('Voltage (mV)')
axes[0].set_xticks([])
axes[1].set_xlim((times[0], times[-1]))
axes[1].set_xlim([0, 15e3])
axes[1].set_ylim([-600, 1750])
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Current (pA)')

err = []
trueparams_all = []
for cell in range(10):
    parameters = np.loadtxt('%s/solution-%s.txt' % (loaddir, cell))

    trueparams = np.loadtxt('%s/solution-%s.txt' % (loaddir2, cell))

    with open('%s/fixparam-%s.pkl' % (loaddir2, cell), 'rb') as f:
        fixparams = pickle.load(f)

    model.set_fix_parameters(fixparams)  #NOTE: assume we get these right!

    # generate from model + add noise
    data = model.simulate(trueparams, times)
    data += np.random.normal(0, fakedatanoise, size=data.shape)

    # Voltage
    voltage = model.voltage(times, trueparams)

    # Simulate
    sim = model.simulate(parameters, times)

    axes[0].plot(times, voltage, lw=0.5, c='#7f7f7f',
            label='__nl__' if cell else r'$V_m$')

    axes[1].plot(times, data, c='C0', alpha=0.5, lw=0.5,
            label='__nl__' if cell else 'Data')
    axes[1].plot(times, sim, c='C1', alpha=0.75, lw=1,
            label='__nl__' if cell else 'Simulation')

    #err.append((parameters - trueparams) / trueparams * 100)
    #err.append((2 * np.arctan(trueparams / parameters) - np.pi / 2.)
    #           * 100. * 2. / np.pi)
    trueparams_all.append(trueparams)
    err.append((parameters - trueparams))

# Protocol
prt = np.loadtxt('../protocol-time-series/protocol-staircaseramp.csv',
        delimiter=',', skiprows=1)
prt_v_func = interp1d(prt[:, 0] * 1e3, prt[:, 1], kind='linear')
axes[0].plot(times, prt_v_func(times), c='C0', ls='--', lw=0.5,
        label=r'$V_{cmd}$')

# Error
trueparams_all = np.asarray(trueparams_all)
# Use SNR of the true values to determine whether use mean or SD to normalise.
snr = (np.mean(trueparams_all, axis=0) / np.std(trueparams_all, axis=0)) < 1
norm = np.mean(trueparams_all, axis=0)
norm[snr] = np.std(trueparams_all, axis=0)[snr]
err = np.asarray(err) / norm * 100

x = range(len(parameters))
axes[2].set_ylim([-15, 15])
axes[2].axhline(0)
#axes[2].bar(x, np.mean(err, axis=0), color='C0')
#axes[2].errorbar(x, np.mean(err, axis=0), yerr=np.std(err, axis=0), ls='',
#        color='#7f7f7f')
sp_x = np.repeat(x, len(err))
sp_y = err.flatten('F')
sns.swarmplot(sp_x, sp_y, size=3, ax=axes[2])
axes[2].set_xticks(x)
axes[2].set_xticklabels(parameter_names)
yticks = [-15, -10, -5, 0, 5, 10, 15]
axes[2].set_yticks(yticks)
axes[2].set_yticklabels(['%.1f' % i + '%' for i in yticks])
axes[2].set_ylabel('Percentage error')

# Finishing
axes[0].legend()
axes[1].legend(loc=2)
grid.tight_layout(fig, pad=1.0, rect=(0.0, 0.0, 1, 1))
grid.update(wspace=0.21, hspace=0.0)

plt.savefig('figs/parameters-error-full2vclinleak-full2vclinleak.pdf',
        format='pdf')
plt.savefig('figs/parameters-error-full2vclinleak-full2vclinleak', dpi=200)
print('Done')
