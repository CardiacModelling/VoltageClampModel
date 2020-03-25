#!/usr/bin/env python2
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import pickle

import sys
sys.path.append('./hERGRapidCharacterisation-lib')
import plot_hbm_func as plot_func

savedir = 'figs'
n_param = 2
param_name = [r'$V_{off}$', r'$g_{Leak}$']

if not os.path.isdir(savedir):
    os.makedirs(savedir)

param_syn = []
try:
    filename = sys.argv[1]
except IndexError:
    print('Usage: python %s [str:filename]' % os.path.basename(__file__))
    sys.exit()
path_to_syn = './out/%s' % filename
files_syn = glob.glob(path_to_syn + '/*542811797.txt')
for file_syn in files_syn:
    p = np.loadtxt(file_syn)
    param_syn.append(p[-1 * n_param:])
param_syn = np.array(param_syn)

# Plot the params!
fig_size = (3 * n_param, 3 * n_param)
# fig_size = (12, 12)
fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)

for i in range(n_param):
    for j in range(n_param):
        if i == j:
            # Diagonal: no plot
            axes[i, j].hist(param_syn[:, i], bins=15)

        elif i < j:
            # Top-right: no plot
            axes[i, j].axis('off')

        else:
            # Lower-left: plot scatters
            px_s = param_syn[:, j]
            py_s = param_syn[:, i]
            axes[i, j].scatter(px_s, py_s, c='C0',
                    label=filename)

            #xmin = np.min(px_s)
            #xmax = max(np.max(px_e), np.max(px_s))
            #ymin = min(np.min(py_e), np.min(py_s))
            #ymax = max(np.max(py_e), np.max(py_s))

            #axes[i, j].set_xlim(xmin, xmax)
            #axes[i, j].set_ylim(ymin, ymax)
            
        # Set tick labels
        if i < n_param - 1 and i >= j:
            # Only show x tick labels for the last row
            axes[i, j].set_xticklabels([])
        if j > 0 and i >= j:
            # Only show y tick labels for the first column
            axes[i, j].set_yticklabels([])

    # Set axis labels and ticks
    if i >= 0:
        axes[i, 0].set_ylabel(param_name[i], fontsize=14)
        axes[i, 0].tick_params('y', labelsize=12)
    if i <= n_param - 1:
        axes[-1, i].set_xlabel(param_name[i], fontsize=14)
        axes[-1, i].tick_params('x', labelsize=12, rotation=30)


#axes[1, 0].legend(fontsize=32, loc="lower left", bbox_to_anchor=(1.15, 1.15),
#                  bbox_transform=axes[1, 0].transAxes)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# plt.savefig('%s/%s-0.png' % (savedir, filename), bbox_inch='tight', dpi=300)

plt.savefig('%s/%s-artefacts.png' % (savedir, filename), bbox_inch='tight', dpi=100)

plt.close()
