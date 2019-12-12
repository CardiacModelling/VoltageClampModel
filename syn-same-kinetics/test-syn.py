#!/usr/bin/env python3
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import model as m
import parameters
from protocols import leak_staircase as stair_prt

import syn

savedir = './figs'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Setup model
temperature = 23.0 + 273.15  # K

stair_model = m.Model('../mmt-model-files/simplified-voltage-clamp-ikr.mmt',
                    protocol_def=stair_prt,
                    temperature=temperature,  # K
                    transform=None,
                    useFilterCap=False)
t_stair = np.arange(0, 15.5e3, 0.5)  # ms

# Ideal case
stair_model_ideal = m.Model('../mmt-model-files/ideal-ikr.mmt',
                    protocol_def=stair_prt,
                    temperature=temperature,  # K
                    transform=None,
                    useFilterCap=False)
typical_p = parameters.idealvc_typical_values
v_command = stair_model_ideal.voltage(t_stair)
i_stair_ideal = stair_model_ideal.simulate(typical_p, t_stair)

# Plot
fig = plt.figure(figsize=(6, 8))
grid = plt.GridSpec(5, 1, hspace=0.0, wspace=0.2)
axes = []
axes.append(fig.add_subplot(grid[0:1, :]))
axes.append(fig.add_subplot(grid[1:2, :]))
axes.append(fig.add_subplot(grid[2:3, :]))
axes.append(fig.add_subplot(grid[3:4, :]))
axes.append(fig.add_subplot(grid[4:5, :]))

all_p = []

for i in range(100):
    p, _ = syn.generate_simvc(stair_model, t_stair, icell=i,
            return_parameters=True)
    voltage = syn.generate_simvc(stair_model, t_stair, icell=i,
            return_voltage=True)
    current = syn.generate_simvc(stair_model, t_stair, icell=i, add_linleak=True,
            fakedatanoise=0)
    noisycurrent = syn.generate_simvc(stair_model, t_stair, icell=i,
            add_linleak=True)

    all_p.append(p)

    # Normalise the same way as in paper 1
    from scipy.optimize import minimize
    res_s = minimize(lambda x: np.sum(np.abs(current / x
                                             - i_stair_ideal)), x0=1.0)
    norm_s = res_s.x[0]
    res_d = minimize(lambda x: np.sum(np.abs(noisycurrent / x
                                             - i_stair_ideal)), x0=norm_s)
    norm_d = res_d.x[0]
    if norm_d > 1e2 or not np.isfinite(norm_d):
        # Maybe smoothing making fitting harder?
        norm_d = norm_s
    if norm_s > 1e2 or not np.isfinite(norm_s):
        # Simulation went wrong?!
        raise RuntimeError('Simulation for %s %s %s seems' % \
                (file_name, cell, prt) + ' problematic')

    axes[0].plot(t_stair, voltage, lw=0.5)
    axes[1].plot(t_stair, current, lw=0.5)
    axes[2].plot(t_stair, noisycurrent, lw=0.5)
    axes[3].plot(t_stair, current / norm_s, lw=0.5)
    axes[4].plot(t_stair, noisycurrent / norm_d, lw=0.5)

# Plot ideal case
axes[0].plot(t_stair, v_command, ls='--', c='k', label=r'$V_{cmd}$')
axes[0].legend(loc=4)
axes[1].plot(t_stair, i_stair_ideal, ls='--', c='k', label='Ideal')
axes[2].plot(t_stair, i_stair_ideal, ls='--', c='k', label='Ideal')
axes[3].plot(t_stair, i_stair_ideal / 1.0, ls='--', c='k',
        label='Ideal')
axes[4].plot(t_stair, i_stair_ideal / 1.0, ls='--', c='k',
        label='Ideal')
axes[4].legend(loc=4)

axes[0].set_ylabel('Voltage\n[mV]')
axes[0].set_xticks([])
axes[1].set_ylabel('Current\n[pA]')
axes[1].set_xticks([])
axes[2].set_ylabel('Current\n[pA]')
axes[2].set_xticks([])
axes[3].set_ylabel('Normalised\nCurrent')
axes[3].set_xticks([])
axes[4].set_ylabel('Normalised\nCurrent')
axes[4].set_xlabel('Time [ms]')
plt.savefig('%s/test-syn-data.png' % savedir, dpi=300, bbox_inches='tight')
plt.close()

# Plot parameters
fig, axes = plt.subplots(3, 5, figsize=(16, 7))

all_p = np.asarray(all_p)

for i in range(3):
    for j in range(5):
        n = i * 5 + j
        if n < all_p.shape[1]:
            axes[i, j].hist(all_p[:, n])
            axes[i, j].set_xlabel(r'$p_{%s}$' % n, fontsize=14)
            axes[i, j].ticklabel_format(style='sci', scilimits=(-2,4),
                    axis='both')
        else:
            axes[i, j].axis('off')

axes[1, 0].set_ylabel('Frequency', fontsize=16)
plt.subplots_adjust(wspace=0.225, hspace=0.4)
plt.savefig('%s/test-syn-parameters.png' % savedir, bbox_inches='tight')
plt.close()

