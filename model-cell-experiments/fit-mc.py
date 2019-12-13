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

import model as m; m.vhold = 0
from protocols import leak_staircase as protocol_def

"""
Run fit for single model cell experiment data
"""

savedir = './out/'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Load data
import util
staircase_idx = [0, 1, 0]
f = 'data/20191002_mc_nocomp.dat'
whole_data, times = util.load(f, staircase_idx, vccc=True)
times = times * 1e3  # s -> ms
data = (whole_data[0] + whole_data[3]) * 1e12  # A -> pA
print('Injecting (pA): ', np.mean(whole_data[3]) * 1e12)

# Down sample...
times = times[::5]
data = data[::5]

print('Fitting to ', f)
saveas = 'mcnocomp'

# Control fitting seed --> OR DONT
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)

# Set parameter transformation
def transform_to_model_param(v):
    i = [0, 1, 2, 3]# + [5, 6]
    o = np.copy(v)
    o[i] = np.exp(o[i])
    return o

def transform_from_model_param(v):
    i = [0, 1, 2, 3]# + [5, 6]
    o = np.copy(v)
    o[i] = np.log(o[i])
    return o

# Model
model = m.Model('../mmt-model-files/full2-voltage-clamp-mc.mmt',
                protocol_def=protocol_def,
                temperature=273.15 + 23.0,  # K
                transform=transform_to_model_param,
                readout='voltageclamp.Iout',
                useFilterCap=False)
parameters = [
    'mc.g',
    'voltageclamp.cprs',
    'membrane.cm',
    'voltageclamp.rseries',
    'voltageclamp.voffset_eff',
#'voltageclamp.tau_clamp',
#'voltageclamp.tau_sum',
]
model.set_parameters(parameters)
parameter_to_fix = [
    'voltageclamp.cprs_est',
    'voltageclamp.cm_est',
    'voltageclamp.rseries_est',
]
parameter_to_fix_values = [
    0.,  # pF; Cprs*
    0.,  # pF; Cm*
    0.,  # GOhm; Rs*
]
fix_p = {}
for i, j in zip(parameter_to_fix, parameter_to_fix_values):
    fix_p[i] = j
model.set_fix_parameters(fix_p)


#
# Fit
#

# Create Pints stuffs
problem = pints.SingleOutputProblem(model, times, data)
error = pints.RootMeanSquaredError(problem)
lower = [
    0.1,  # pA/mV = 1/GOhm; g_membrane
    0.1,  # pF; Cprs
    1.0,  # pF; Cm
    1e-3,  # GOhm; Rs
    -20.,  # mV; Voffset+
#0.1e-3,  # s; tau_clamp
#1e-3,  # s; tau_sum
]
upper = [
    100.,  # pA/mV = 1/GOhm; g_membrane
    10.,  # pF; Cprs
    100.,  # pF; Cm
    50e-3,  # GOhm; Rs
    20.,  # mV; Voffset+
#10e-3,  # s; tau_clamp
#100e-3,  # s; tau_sum
]
boundaries = pints.RectangularBoundaries(transform_from_model_param(lower),
        transform_from_model_param(upper))

# Check error is working fine
idealparams = [
    1. / 0.5,  # pA/mV = 1/GOhm; g_membrane
    4.7,  # pF; Cprs
    22.0,  # pF; Cm
    30e-3,  # GOhm; Rs
    0.0,  # mV; Voffset+
#0.8e-3,  # s; tau_clamp
#40e-3,  # s; tau_sum
]# + [2.5]  # pA; noise
priorparams = [
    1.,  # pA/mV = 1/GOhm; g_membrane
    5.,  # pF; Cprs
    10.,  # pF; Cm
    10e-3,  # GOhm; Rs
    0.,  # mV; Voffset+
#1e-3,  # s; tau_clamp
#10e-3,  # s; tau_sum
]# + [2.5]  # pA; noise
transform_priorparams = transform_from_model_param(priorparams)
print('Score at prior parameters: ',
        error(transform_priorparams))
print(error(transform_from_model_param(idealparams)))
p = [
 1.04988672817989661e-05,
 3.62247251960526144e-03,
 1.17891302405355911e-04,
 8.90479353755912195e-02,
 2.04243756821879145e+01,
#1e-3,  # s; tau_clamp
#10e-3,  # s; tau_sum
]
print(error(transform_from_model_param(p)))
for _ in range(10):
    assert(error(transform_priorparams) ==\
            error(transform_priorparams))

# Run
try:
    N = int(sys.argv[1])
except IndexError:
    N = 3

params, errors = [], []

for i in range(N):

    for _ in range(100):
        try:
            if i == 0:
                x0 = transform_priorparams
            else:
                # Randomly pick a starting point
                x0 = boundaries.sample()[0]
            error(x0)
        except ValueError:
            continue
        break
    print('Starting point: ', x0)

    # Create optimiser
    print('Starting error: ', error(x0))
    opt = pints.Optimisation(error, x0.T, boundaries=boundaries,
            method=pints.NelderMead)
    opt.set_max_iterations(None)
    opt.set_max_unchanged_iterations(iterations=100, threshold=1e-5)
    opt.set_parallel(False)

    # Run optimisation
    try:
        with np.errstate(all='ignore'):
            # Tell numpy not to issue warnings
            p, s = opt.run()
            p = transform_to_model_param(p)
            params.append(p)
            errors.append(s)
            print('Found solution:          Ideal parameters:' )
            for k, x in enumerate(p):
                print(pints.strfloat(x) + '    ' + \
                        pints.strfloat(idealparams[k]))
    except ValueError:
        import traceback
        traceback.print_exc()

#
# Done
#

# Order from best to worst
order = np.argsort(errors)  # (use [::-1] for LL)
errors = np.asarray(errors)[order]
params = np.asarray(params)[order]

# Show results
bestn = min(3, N)
print('Best %d errors:' % bestn)
for i in range(bestn):
    print(errors[i])
print('Mean & std of error:')
print(np.mean(errors))
print(np.std(errors))
print('Worst error:')
print(errors[-1])

# Extract best 3
obtained_error0 = errors[0]
obtained_parameters0 = params[0]
obtained_error1 = errors[1]
obtained_parameters1 = params[1]
obtained_error2 = errors[2]
obtained_parameters2 = params[2]

# Show results
print('Found solution:          Ideal parameters:' )
# Store output
with open('%s/%s-solution-%s-1.txt' % (savedir, saveas, fit_seed), 'w') as f:
    for k, x in enumerate(obtained_parameters0):
        print(pints.strfloat(x) + '    ' + pints.strfloat(idealparams[k]))
        f.write(pints.strfloat(x) + '\n')
print('Found solution:          Ideal parameters:' )
# Store output
with open('%s/%s-solution-%s-2.txt' % (savedir, saveas, fit_seed), 'w') as f:
    for k, x in enumerate(obtained_parameters1):
        print(pints.strfloat(x) + '    ' + pints.strfloat(idealparams[k]))
        f.write(pints.strfloat(x) + '\n')
print('Found solution:          Ideal parameters:' )
# Store output
with open('%s/%s-solution-%s-3.txt' % (savedir, saveas, fit_seed), 'w') as f:
    for k, x in enumerate(obtained_parameters2):
        print(pints.strfloat(x) + '    ' + pints.strfloat(idealparams[k]))
        f.write(pints.strfloat(x) + '\n')

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
sol0 = problem.evaluate(transform_from_model_param(obtained_parameters0))
sol1 = problem.evaluate(transform_from_model_param(obtained_parameters1))
sol2 = problem.evaluate(transform_from_model_param(obtained_parameters2))
vol = model.voltage(times)
axes[0].plot(times, vol, c='#7f7f7f')
axes[0].set_ylabel('Voltage (mV)')
axes[1].plot(times, data, alpha=0.5, label='data')
axes[1].plot(times, sol0, label='found solution 1')
axes[1].plot(times, sol1, label='found solution 2')
axes[1].plot(times, sol2, label='found solution 3')
axes[1].legend()
axes[1].set_ylabel('Current (pA)')
axes[1].set_xlabel('Time (ms)')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s-solution-%s.png' % (savedir, saveas, fit_seed),
        bbox_inches='tight')
plt.close()
