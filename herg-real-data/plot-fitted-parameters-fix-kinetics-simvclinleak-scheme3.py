#!/usr/bin/env python2
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys

savedir = 'figs'
n_model_param = 3
param_name = [r'$\ln(g_{Kr})$', r'$V^\dagger_{off}$', r'$g^\dagger_{leak}$']

if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Load syn param from voltage-artefact
path_to_file = './out/herg25oc1-scheme3-simvclinleak/' \
        + 'herg25oc1-solution_i-717354021.txt'
param_exp = np.loadtxt(path_to_file)

n_param = param_exp.shape[1]

# Take log for conductances
param_exp[:, 0] = np.log(param_exp[:, 0])
# param_exp[:, 2] = np.log(param_exp[:, 2])

# Plot the params!
fig_size = (12, 12)
fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)

for i in range(n_param):
    for j in range(n_param):
        if i == j:
            '''
            # Diagonal: no plot
            # axes[i, j].axis('off')
            axes[i, j].set_xticklabels([])
            axes[i, j].set_yticklabels([])
            axes[i, j].tick_params(axis='both', which='both', bottom=False,
                                   top=False, left=False, right=False, 
                                   labelleft=False, labelbottom=False)
            '''
            axes[i, j].hist(param_exp[:, i], bins=15)

        elif i < j:
            # Top-right: no plot
            axes[i, j].axis('off')

        else:
            # Lower-left: plot scatters
            px_e = param_exp[:, j]
            py_e = param_exp[:, i]
            axes[i, j].scatter(px_e, py_e, c='C0', alpha=0.5,
                    label='fitting results')

            #axes[i, j].set_xlim(xmin, xmax)
            #axes[i, j].set_ylim(ymin, ymax)

        # Set tick labels
        if i < n_param - 1:
            # Only show x tick labels for the last row
            axes[i, j].set_xticklabels([])
        if j > 0:
            # Only show y tick labels for the first column
            axes[i, j].set_yticklabels([])

    # Set axis labels and ticks
    if True:
        if i == 0:
            axes[i, 0].set_ylabel('Frequncey', fontsize=32)
        else:
            axes[i, 0].set_ylabel(param_name[i], fontsize=32)
        axes[i, 0].tick_params('y', labelsize=26)
    if True:
        axes[-1, i].set_xlabel(param_name[i], fontsize=32)
        axes[-1, i].tick_params('x', labelsize=26, rotation=30)

#axes[1, 0].legend(fontsize=32, loc="lower left", bbox_to_anchor=(1.15, 1.15),
#                  bbox_transform=axes[1, 0].transAxes)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('%s/herg25oc1-simvclinleak-scheme3-fitted-parameters.pdf' \
        % (savedir), bbox_inch='tight', format='pdf')
plt.savefig('%s/herg25oc1-simvclinleak-scheme3-fitted-parameters' % (savedir),
        bbox_inch='tight', dpi=300)
plt.close()
