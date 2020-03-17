#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib/')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model as m
from protocols import leak_staircase as protocol_def

model = m.Model('../mmt-model-files/ideal-ikr.mmt',
                protocol_def=protocol_def,
                temperature=273.15 + 25,  # K
                transform=None,
                useFilterCap=False)

times = np.arange(0, 15.5e3, 0.5)  # ms

HBM_mean = np.array([
    7.65e-2,
    9.05e+1,
    2.84e-2,
    4.74e+1,
    1.03e+2,
    2.13e+1,
    8.01e+0,
    2.96e+1
]) * 1e-3  # V, s -> mV, ms
fit = np.loadtxt('out/herg25oc1-scheme3-simvclinleak/' +
        'herg25oc1-solution-717354021.txt')  # mV, ms

HBM_mean = np.append(1, HBM_mean)
fit = np.append(1, fit)  # Append g_Kr = 1

voltage = model.voltage(times)
sim1 = model.simulate(HBM_mean, times)
sim2 = model.simulate(fit, times)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 3.5),
        gridspec_kw={'height_ratios': [1, 2]})
axes[0].plot(times, voltage, c='#7f7f7f')
axes[1].plot(times, sim1, label='Hypothesis 1', color='#d95f02')
axes[1].plot(times, sim2, label='Hypothesis 2', color='#1b9e77')
axes[0].set_ylabel('Voltage (mV)')
axes[1].set_ylabel(r'Current ($g_{Kr}=1$)')
axes[1].set_xlabel('Time (ms)')

# Zooms
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

set_xlim_ins = [[9750, 13000],]
set_ylim_ins = [[5, 20],]
inset_setup = [(3, 0.75, 'lower center'),]
mark_setup = [(2, 4),]

for i_zoom, (w, h, loc) in enumerate(inset_setup):
    axins = inset_axes(axes[1], width=w, height=h, loc=loc,
            axes_kwargs={"facecolor" : "#f0f0f0"})
    axins.plot(times, sim1, alpha=1, color='#d95f02')
    axins.plot(times, sim2, alpha=1, color='#1b9e77')
    axins.set_xlim(set_xlim_ins[i_zoom])
    axins.set_ylim(set_ylim_ins[i_zoom])
    #axins.yaxis.get_major_locator().set_params(nbins=3)
    #axins.xaxis.get_major_locator().set_params(nbins=3)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    pp, p1, p2 = mark_inset(axes[1], axins, loc1=mark_setup[i_zoom][0],
            loc2=mark_setup[i_zoom][1], fc="none", lw=0.75, ec='k')
    pp.set_fill(True); pp.set_facecolor("#f0f0f0")

axes[1].legend()
plt.tight_layout()
plt.savefig('figs/hypotheses-comparison-staircase.pdf', format='pdf',
        bbox_inches='tight')
plt.savefig('figs/hypotheses-comparison-staircase.png', dpi=200,
        bbox_inches='tight')

