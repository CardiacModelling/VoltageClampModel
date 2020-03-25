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

import model as m
import syn
from parameters import simvc, simvc_typical_values
import parametertransform
from priors import IKrLogPrior, SimplifiedVoltageClampLogPrior
from protocols import leak_staircase as protocol_def

"""
Run fit for single experiment-synthetic data study assuming correct voltage 
clamp (simvc) set-up with linear leak fitting to synthetic data from simvc
model with linear leak residual.
"""

try:
    cell = sys.argv[1]
    int(cell)
except:
    print('Usage: python %s [int:cell_id]' % os.path.basename(__file__)
        + ' --optional [int:N_repeats]')
    sys.exit()

savedir = 'out/syn-simvclinleak-simvclinleak'
if not os.path.isdir(savedir):
    os.makedirs(savedir)
savedir2 = 'out/syn-simvclinleak-simvclinleak-true'
if not os.path.isdir(savedir2):
    os.makedirs(savedir2)

temperature = 25.0
useFilterCap = False

# Set parameter transformation
transform_to_ikr = parametertransform.log_transform_to_ikr
transform_from_ikr = parametertransform.log_transform_from_ikr
transform_to_vc = parametertransform.log_transform_to_vc
transform_from_vc = parametertransform.log_transform_from_vc
transform_to_leak = parametertransform.log_transform_to_linleak
transform_from_leak = parametertransform.log_transform_from_linleak

n_ikr_parameters = IKrLogPrior(None, None).n_parameters()
n_vc_parameters = SimplifiedVoltageClampLogPrior(None, None).n_parameters()
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
model.set_parameters(simvc + ['voltageclamp.gLeak'])

times = np.arange(0, 15.5e3, 0.5)  # ms


#
# Generate syn. data
#
model_simvc = m.Model('../mmt-model-files/simplified-voltage-clamp-ikr.mmt',
                protocol_def=protocol_def,
                temperature=273.15 + temperature,  # K
                transform=parametertransform.donothing,
                useFilterCap=useFilterCap)  # ignore capacitive spike

fakedatanoise = 20.0

trueparams, fixparams = syn.generate_simvc(model_simvc, times,
        icell=int(cell), return_parameters=True)

linleakparams = syn.generate_simvc(model_simvc, times, icell=int(cell),
        add_linleak=True, return_leak_param=True)

with open('%s/solution-%s.txt' % (savedir2, cell), 'w') as f:
    for x in trueparams:
        f.write(pints.strfloat(x) + '\n')

with open('%s/fixparam-%s.pkl' % (savedir2, cell), 'wb') as f:
    pickle.dump(fixparams, f, protocol=2)  # for Python 2 compatibility

with open('%s/linleakparam-%s.txt' % (savedir2, cell), 'w') as f:
    for x in linleakparams:
        f.write(pints.strfloat(x) + '\n')

# lump leak parameters with model parameters
trueparams = np.append(trueparams, linleakparams)

data = syn.generate_simvc(model_simvc, times, icell=int(cell),
        add_linleak=True, transform=parametertransform.donothing,
        fakedatanoise=fakedatanoise)

if useFilterCap:
    # Apply capacitance filter to data
    data = data * model.cap_filter(times)

# Estimate noise from start of data
noise_sigma = np.std(data[:200])


#
# Fit
#
# Set cell dependent parameters which not to be inferred
model.set_fix_parameters(fixparams)  #NOTE: assume we get these right!

# Create Pints stuffs
problem = pints.SingleOutputProblem(model, times, data)
loglikelihood = pints.GaussianKnownSigmaLogLikelihood(problem, noise_sigma)
ikr_prior = IKrLogPrior(transform_to_ikr, transform_from_ikr)
vc_prior = SimplifiedVoltageClampLogPrior(transform_to_vc, transform_from_vc)
ileak_prior = pints.UniformLogPrior([transform_from_leak(1e-3)],
        [transform_from_leak(1e3)])
logprior = pints.ComposedLogPrior(ikr_prior, vc_prior, ileak_prior)
logposterior = pints.LogPosterior(loglikelihood, logprior)

# Check logposterior is working fine
priorparams = np.append(simvc_typical_values, 1e-2)
transform_priorparams = transform_from_model_param(priorparams)
print('Score at prior parameters: ',
        logposterior(transform_priorparams))
for _ in range(10):
    assert(logposterior(transform_priorparams) ==\
            logposterior(transform_priorparams))

print('Score at true parameters: ',
        logposterior(transform_from_model_param(trueparams)))

# Run
try:
    N = int(sys.argv[2])
except IndexError:
    N = 3

params, logposteriors = [], []

for i in range(N):

    for _ in range(100):
        try:
            if False:  #NOTE: not this for syn data study
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
            print('Found solution:          True parameters:' )
            for k, x in enumerate(p):
                print(pints.strfloat(x) + '    ' + \
                        pints.strfloat(trueparams[k]))
    except ValueError:
        import traceback
        traceback.print_exc()

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
print('Found solution:          True parameters:' )
# Store output
with open('%s/solution-%s.txt' % (savedir, cell), 'w') as f:
    for k, x in enumerate(obtained_parameters0):
        print(pints.strfloat(x) + '    ' + pints.strfloat(trueparams[k]))
        f.write(pints.strfloat(x) + '\n')
print('Found solution:          True parameters:' )
# Store output
with open('%s/solution-%s-2.txt' % (savedir, cell), 'w') as f:
    for k, x in enumerate(obtained_parameters1):
        print(pints.strfloat(x) + '    ' + pints.strfloat(trueparams[k]))
        f.write(pints.strfloat(x) + '\n')
print('Found solution:          True parameters:' )
# Store output
with open('%s/solution-%s-3.txt' % (savedir, cell), 'w') as f:
    for k, x in enumerate(obtained_parameters2):
        print(pints.strfloat(x) + '    ' + pints.strfloat(trueparams[k]))
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
axes[1].set_xlabel('Time [s]')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/solution-%s.png' % (savedir, cell), bbox_inches='tight')
plt.close()
