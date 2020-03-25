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
from parameters import simvc, get_qc
from parameters import simvc_fix, simvc_fix_typical_values
from parameters import simvc_typical_values
import parametertransform
from priors import IKrLogPrior, VoltageOffsetWithConductanceLogPrior
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

updateELeakCorrection = True
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

savedir = './out/' + file_name + '-fixkinetics-simvclinleak-loosebound-fixELeak'
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
transform_to_vc = parametertransform.log_transform_to_vc
transform_from_vc = parametertransform.log_transform_from_vc
transform_to_leak = parametertransform.donothing
transform_from_leak = parametertransform.donothing

n_gvc_parameters = VoltageOffsetWithConductanceLogPrior(None,
        None).n_parameters()
transform_to_gvc = parametertransform.ComposeTransformation(
        np.exp, transform_to_vc, 1)
transform_from_gvc = parametertransform.ComposeTransformation(
        np.log, transform_from_vc, 1)
transform_to_model_param = parametertransform.ComposeTransformation(
        transform_to_gvc, transform_to_leak, n_gvc_parameters)
transform_from_model_param = parametertransform.ComposeTransformation(
        transform_from_gvc, transform_from_leak, n_gvc_parameters)

# Model
model = m.Model('../mmt-model-files/simplified-voltage-clamp-ikr-linleak.mmt',
                protocol_def=protocol_def,
                temperature=273.15 + temperature,  # K
                transform=transform_to_model_param,
                useFilterCap=useFilterCap)  # ignore capacitive spike
model_ideal = m.Model('../mmt-model-files/ideal-ikr.mmt',
                protocol_def=protocol_def,
                temperature=273.15 + temperature,  # K
                transform=parametertransform.donothing,
                useFilterCap=useFilterCap)  # ignore capacitive spike
# Set which parameters to be inferred
model.set_parameters([
    'ikr.g',
    #'voltageclamp.rseries',
    'voltageclamp.voffset_eff',
    'voltageclamp.gLeak'])

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
ikr_param = [
        'ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4',
        'ikr.p5', 'ikr.p6', 'ikr.p7', 'ikr.p8',  
        ]
p_ikr = np.array([  # paper 1 supplement HBM mean parameters
        9.48e-2, 8.69e+1, 2.98e-2, 4.69e+1,
        1.04e+2, 2.19e+1, 8.05e+0, 2.99e+1]) * 1e-3  # V, s -> mV, ms
simvc_fix_values = [cm, rseries * alpha, rseries]
extra_fix = ['voltageclamp.rseries']
if updateELeakCorrection:
    leakbeforeparam = np.loadtxt('../qc/' + file_name + '-staircaseramp-leak_before.txt')
    leakafterparam = np.loadtxt('../qc/' + file_name + '-staircaseramp-leak_after.txt')
    cell_id_file = '../qc/%s-staircaseramp-cell_id.txt' % file_name
    cell_ids = []
    with open(cell_id_file, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                cell_ids.append(l.split()[0])
    cell_idx = cell_ids.index(cell)
    ga, Ea = leakbeforeparam[cell_idx]
    gb, Eb = leakafterparam[cell_idx]
    ELeakCorrection = - (ga * Ea - gb * Eb) / (gb - ga)
    print('E_Leak correction: ', ELeakCorrection, ' (mV)')
    extra_fix += ['voltageclamp.ELeak']
    simvc_fix_values += [ELeakCorrection]
fix_p = get_fix_param(ikr_param + simvc_fix + extra_fix,
        np.append(p_ikr, simvc_fix_values))
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
vc_prior = VoltageOffsetWithConductanceLogPrior(transform_to_gvc,
        transform_from_gvc)
ileak_prior = pints.UniformLogPrior([transform_from_leak(-1e2)],
        [transform_from_leak(1e2)])
logprior = pints.ComposedLogPrior(vc_prior, ileak_prior)
logposterior = pints.LogPosterior(loglikelihood, logprior)

# Check logposterior is working fine
priorparams = np.array([3.23e+4 * 1e-3, -1.5, -1e-2])
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
ideal = model_ideal.simulate(np.append(obtained_parameters0[0], p_ikr), times)
vol = model.voltage(times)
axes[0].plot(times, vol, c='#7f7f7f')
axes[0].set_ylabel('Voltage [mV]')
axes[1].plot(times, data, alpha=0.5, label='data')
axes[1].plot(times, ideal, label='same kinetics')
axes[1].plot(times, sol0, label='found solution')
axes[1].plot(times, sol1, label='found solution')
axes[1].plot(times, sol2, label='found solution')
axes[1].legend()
axes[1].set_ylabel('Current [pA]')
axes[1].set_xlabel('Time [ms]')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s-solution-%s.png' % (savedir, saveas, fit_seed), bbox_inches='tight')
plt.close()
