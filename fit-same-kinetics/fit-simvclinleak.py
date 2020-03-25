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

import model as m
from parameters import ikr as ikr_parameters
from parameters import get_qc
from parameters import simvc_fix, simvc_fix_typical_values
from parameters import idealvc_typical_values
import parametertransform
from priors import IKrLogPrior, SimplifiedVoltageClampLogPrior
from protocols import leak_staircase as protocol_def

"""
Run fit for single experiment data with simvc voltage clamp set-up with linear
leak.
"""

try:
    file_name = sys.argv[1]
    cell = sys.argv[2]
except:
    print('Usage: python %s [str:file_name]' % os.path.basename(__file__)
          + ' [str:cell_id] --optional [N_repeats]')
    sys.exit()

file_dir = '../data'
file_list = [
        'herg25oc1',
        'herg27oc1',
        'herg27oc2',
        'herg30oc1',
        'herg30oc2',
        'herg33oc1',
        'herg33oc2',
        'herg37oc3',
        ]
temperatures = [25.0, 27.0, 27.0, 30.0, 30.0, 33.0, 33.0, 37.0]
useFilterCap = False

if file_name not in file_list:
    raise ValueError('Input `file_name` must be in `file_list`')
temperature = temperatures[file_list.index(file_name)]

savedir = './out/' + file_name + '-simvclinleak'
if not os.path.isdir(savedir):
    os.makedirs(savedir)


data_file_name = file_name + '-staircaseramp-' + cell + '.csv'
time_file_name = file_name + '-staircaseramp-times.csv'
print('Fitting to ', data_file_name)
print('Temperature: ', temperature)
saveas = data_file_name[:-4]

# Control fitting seed --> OR DONT
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)


# Set parameter transformation
transform_to_ikr = parametertransform.log_transform_to_ikr
transform_from_ikr = parametertransform.log_transform_from_ikr
transform_to_vc = parametertransform.donothing
transform_from_vc = parametertransform.donothing
transform_to_leak = parametertransform.donothing
transform_from_leak = parametertransform.donothing

n_ikr_parameters = IKrLogPrior(None, None).n_parameters()
n_vc_parameters = 1
transform_to_ikr_vc = parametertransform.ComposeTransformation(
        transform_to_ikr, transform_to_vc, n_ikr_parameters)
transform_from_ikr_vc = parametertransform.ComposeTransformation(
        transform_from_ikr, transform_from_vc, n_ikr_parameters)
transform_to_model_param = parametertransform.ComposeTransformation(
        transform_to_ikr_vc, transform_to_leak,
        n_ikr_parameters + n_vc_parameters)
transform_from_model_param = parametertransform.ComposeTransformation(
        transform_from_ikr_vc, transform_from_leak,
        n_ikr_parameters + n_vc_parameters)

# Model
model = m.Model('../mmt-model-files/simplified-voltage-clamp-ikr-linleak.mmt',
                protocol_def=protocol_def,
                temperature=273.15 + temperature,  # K
                transform=transform_to_model_param,
                useFilterCap=useFilterCap)  # ignore capacitive spike
# Set which parameters to be inferred
simvclinleak = ikr_parameters + [
    'voltageclamp.voffset_eff',
    'voltageclamp.gLeak'
    ]
model.set_parameters(simvclinleak)

def get_fix_param(var, val):
    """
        var: variable name.
        val: variable value to fix.
    """
    out = {}
    for i, j in zip(var, val):
        out[i] = j
    return out

rseal, cm, rseries = get_qc('../qc', file_name, cell)
print('Est. Rseal, Cm, Rseries:', rseal, cm, rseries, '(GOhm, pF, GOhm)')
alpha = 0.8  # rseries %compensation
simvc_fix_values = [cm, rseries * alpha, rseries]
fix_p = get_fix_param(simvc_fix + ['voltageclamp.rseries'],
        simvc_fix_values)

model.set_fix_parameters(fix_p)


# Load data
data = np.loadtxt(file_dir + '/' + data_file_name,
                  delimiter=',', skiprows=1) # headers
times = np.loadtxt(file_dir + '/' + time_file_name,
                   delimiter=',', skiprows=1) # headers
times = times * 1e3  # s -> ms
noise_sigma = np.std(data[:500])
print('Estimated noise level: ', noise_sigma)

if useFilterCap:
    # Apply capacitance filter to data
    data = data * model.cap_filter(times)

#
# Fit
#

# Create Pints stuffs
problem = pints.SingleOutputProblem(model, times, data)
loglikelihood = pints.GaussianKnownSigmaLogLikelihood(problem, noise_sigma)
ikr_prior = IKrLogPrior(transform_to_ikr, transform_from_ikr)
vc_prior = pints.UniformLogPrior([-15], [15])
ileak_prior = pints.UniformLogPrior([transform_from_leak(-1e3)],
        [transform_from_leak(1e3)])
logprior = pints.ComposedLogPrior(ikr_prior, vc_prior, ileak_prior)
logposterior = pints.LogPosterior(loglikelihood, logprior)

# Check logposterior is working fine
priorparams = np.append(idealvc_typical_values, [0, -1e-2])
transform_priorparams = transform_from_model_param(priorparams)
print('Score at prior parameters: ',
        logposterior(transform_priorparams))
for _ in range(10):
    assert(logposterior(transform_priorparams) ==\
            logposterior(transform_priorparams))


# Run
try:
    N = int(sys.argv[3])
except IndexError:
    N = 3

params, logposteriors = [], []

for i in range(N):

    for _ in range(100):
        try:
            if i == 0:  # OK for real data
                x0 = transform_priorparams
            else:
                # Randomly pick a starting point
                x0 = logprior.sample()[0]
            logposterior(x0)
        except ValueError:
            continue
        break
    print('Starting point: ', x0)

    # Create optimiser
    print('Starting logposterior: ', logposterior(x0))
    opt = pints.Optimisation(logposterior, x0.T, method=pints.CMAES)
    opt.set_max_iterations(None)
    opt.set_parallel(False)

    # Run optimisation
    try:
        with np.errstate(all='ignore'):
            # Tell numpy not to issue warnings
            p, s = opt.run()
            p = transform_to_model_param(p)
            params.append(p)
            logposteriors.append(s)
            print('Found solution:          Prior parameters:' )
            for k, x in enumerate(p):
                print(pints.strfloat(x) + '    ' + \
                        pints.strfloat(priorparams[k]))
    except ValueError:
        import traceback
        traceback.print_exc()

#
# Done
#

# Order from best to worst
order = np.argsort(logposteriors)[::-1]  # (use [::-1] for LL)
logposteriors = np.asarray(logposteriors)[order]
params = np.asarray(params)[order]

# Show results
bestn = min(3, N)
print('Best %d logposteriors:' % bestn)
for i in range(bestn):
    print(logposteriors[i])
print('Mean & std of logposterior:')
print(np.mean(logposteriors))
print(np.std(logposteriors))
print('Worst logposterior:')
print(logposteriors[-1])

# Extract best 3
obtained_logposterior0 = logposteriors[0]
obtained_parameters0 = params[0]
obtained_logposterior1 = logposteriors[1]
obtained_parameters1 = params[1]
obtained_logposterior2 = logposteriors[2]
obtained_parameters2 = params[2]

# Show results
print('Found solution:          Prior parameters:' )
# Store output
with open('%s/%s-solution-%s.txt' % (savedir, saveas, fit_seed), 'w') as f:
    for k, x in enumerate(obtained_parameters0):
        print(pints.strfloat(x) + '    ' + pints.strfloat(priorparams[k]))
        f.write(pints.strfloat(x) + '\n')
print('Found solution:          Prior parameters:' )
# Store output
with open('%s/%s-solution-%s-2.txt' % (savedir, saveas, fit_seed), 'w') as f:
    for k, x in enumerate(obtained_parameters1):
        print(pints.strfloat(x) + '    ' + pints.strfloat(priorparams[k]))
        f.write(pints.strfloat(x) + '\n')
print('Found solution:          Prior parameters:' )
# Store output
with open('%s/%s-solution-%s-3.txt' % (savedir, saveas, fit_seed), 'w') as f:
    for k, x in enumerate(obtained_parameters2):
        print(pints.strfloat(x) + '    ' + pints.strfloat(priorparams[k]))
        f.write(pints.strfloat(x) + '\n')

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
sol0 = problem.evaluate(transform_from_model_param(obtained_parameters0))
sol1 = problem.evaluate(transform_from_model_param(obtained_parameters1))
sol2 = problem.evaluate(transform_from_model_param(obtained_parameters2))
vol = model.voltage(times) * 1e3
axes[0].plot(times, vol, c='#7f7f7f')
axes[0].set_ylabel('Voltage [mV]')
axes[1].plot(times, data, alpha=0.5, label='data')
axes[1].plot(times, sol0, label='found solution')
axes[1].plot(times, sol1, label='found solution')
axes[1].plot(times, sol2, label='found solution')
axes[1].legend()
axes[1].set_ylabel('Current [pA]')
axes[1].set_xlabel('Time [ms]')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s-solution-%s.png' % (savedir, saveas, fit_seed), bbox_inches='tight')
plt.close()
