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
import pints.io
import pints.plot

# Model etc.
import model as m
import parameters
from parameters import simvc_fix, get_qc
from parameters import simvc_typical_values
import parametertransform
from priors import IKrWithoutConductanceLogPrior
from priors import VoltageOffsetWithConductanceLogPrior
from protocols import leak_staircase as protocol_def

"""
Run fit for mutliple cell experiment data with same kinetics:

Optimise a likelihood
    L(\theta) = \prod_i L_{kinetics, i}(\theta | \phi^*_i)
where
    i, measurement index
    \theta, the parameters of the kinetics
    L_{kinetics, i}, the likelihood of the kinetic parameters for measurement i
and
    \phi^*_i = \argmax_{\phi_i} L_{individual, i}(\phi_i | \theta)
where
    \phi_i, the parameters for individual measurement, such as g_{Kr} and
        voltage artefact parameters
    L_{individual, i}, the likelihood of \phi_i given \theta.

This returns us \theta* (= \argmax_\theta L(\theta)) and {\phi^*_i}.
"""

try:
    file_name = sys.argv[1]
    run_id = int(sys.argv[2])
except:
    print('Usage: python %s [str:file_name]' % os.path.basename(__file__)
            + '[int:run_id]')
    sys.exit()

common_param = parameters.ikr[1:]
independent_param = ['ikr.g',
    #'voltageclamp.rseries',
    'voltageclamp.voffset_eff',
    'voltageclamp.gLeak']
n_common_param = len(common_param)
n_independent_param = len(independent_param)


def get_param(common, independent):
    model_p = np.append(independent[0], common)
    vclamp_p = independent[1:]
    return np.append(model_p, vclamp_p)


def get_fix_param(var, val):
    """
        var: variable name.
        val: variable value to fix.
    """
    out = {}
    for i, j in zip(var, val):
        out[i] = j
    return out


def update_logposterior_fix_param(l, p):
    """
        l: log-posterior.
        p: dict, fix params.

        NOTE: Only works for specific implementation.
    """
    l._log_likelihood._problem._model.set_fix_parameters(p)


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

savedir = './out/' + file_name + '-scheme3-simvclinleak'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

print('Temperature: ', temperature)

# Control fitting seed --> OR DONT
np.random.seed(run_id)
fit_seed = np.random.randint(0, 2**30)
# fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)


# Get cells
selectedfile = '../manualselection/manualv2selected-%s.txt' % (file_name)
selectedwell = []
with open(selectedfile, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            selectedwell.append(l.split()[0])
selectedwell = selectedwell[:]
n_cells = len(selectedwell)
print(file_name + ' selected ' + str(n_cells) + ' cells')


# Set parameter transformation
transform_to_ikr = parametertransform.log_transform_to_ikr
transform_from_ikr = parametertransform.log_transform_from_ikr
transform_to_vc = parametertransform.log_transform_to_vc
transform_from_vc = parametertransform.log_transform_from_vc
transform_to_leak = parametertransform.donothing
transform_from_leak = parametertransform.donothing

transform_to_common_param = transform_to_ikr
transform_from_common_param = transform_from_ikr

n_gvc_parameters = VoltageOffsetWithConductanceLogPrior(None,
        None).n_parameters()
transform_to_gvc = parametertransform.ComposeTransformation(
        np.exp, transform_to_vc, 1)
transform_from_gvc = parametertransform.ComposeTransformation(
        np.log, transform_from_vc, 1)
transform_to_independent_param = parametertransform.ComposeTransformation(
        transform_to_gvc, transform_to_leak, n_gvc_parameters)
transform_from_independent_param = parametertransform.ComposeTransformation(
        transform_from_gvc, transform_from_leak, n_gvc_parameters)


# For individual bit in MwG
# Model
model_common = m.Model('../mmt-model-files/simplified-voltage-clamp-ikr-linleak.mmt',
                protocol_def=protocol_def,
                temperature=273.15 + temperature,  # K
                transform=transform_to_common_param,
                useFilterCap=useFilterCap)  # ignore capacitive spike
model_common.set_parameters(common_param)

model_independent = m.Model('../mmt-model-files/simplified-voltage-clamp-ikr-linleak.mmt',
                protocol_def=protocol_def,
                temperature=273.15 + temperature,  # K
                transform=transform_to_independent_param,
                useFilterCap=useFilterCap)  # ignore capacitive spike
model_independent.set_parameters(independent_param)


# Load all data
log_posteriors_common = []
log_posteriors_independent = []
data_all = []
times_all = []
noise_sigma_all = []
x0_common_all = []
x0_independent_all = []
transform_x0_common_all = []
transform_x0_independent_all = []
for cell in selectedwell:
    # Load data
    data_file_name = file_name + '-staircaseramp-' + cell + '.csv'
    time_file_name = file_name + '-staircaseramp-times.csv'
    print('Fitting to ', data_file_name)

    data = np.loadtxt(file_dir + '/' + data_file_name,
                      delimiter=',', skiprows=1) # headers
    times = np.loadtxt(file_dir + '/' + time_file_name,
                       delimiter=',', skiprows=1) # headers
    times = times * 1e3  # s -> ms
    noise_sigma = np.std(data[:500])
    print('Estimated noise level: ', noise_sigma)

    if useFilterCap:
        # Apply capacitance filter to data
        data = data * model_common.cap_filter(times)

    data_all.append(data)
    times_all.append(times)
    noise_sigma_all.append(noise_sigma)

    # Get independent CMA-ES results as x0
    cma_file = './fit-results/%s/%s-staircaseramp-%s-solution-542811797.txt' \
            % (file_name, file_name, cell)
    x0i_param = np.loadtxt(cma_file)
    # Change unit... (note, conductance unit different to Beattie's ones)
    x0i_param = x0i_param * 1e-3  # in mV, ms
    x0_common = x0i_param[1:n_common_param + 1]  # remove conductance
    # Get 'independent parameters'
    cma_ip_file = './out/%s-fixkinetics-simvclinleak/%s-staircaseramp-%s-solution-542811797.txt' \
            % (file_name, file_name, cell)
    x0_independent = np.loadtxt(cma_ip_file)
    # x0_independent = np.array([-1.5, 1e-2])
    tx0_common = transform_from_common_param(x0_common)
    # tx0_independent = transform_from_vc(x0i_param[n_common_param + 1:])
    tx0_independent = transform_from_independent_param(x0_independent)

    x0_common_all.append(x0_common)
    x0_independent_all.append(x0_independent)
    transform_x0_common_all.append(tx0_common)
    transform_x0_independent_all.append(tx0_independent)

    # Get fix parameters' value
    rseal, cm, rseries = get_qc('../qc', file_name, cell)
    print('Est. Rseal, Cm, Rseries:', rseal, cm, rseries, '(GOhm, pF, GOhm)')
    alpha = 0.8  # rseries %compensation
    simvc_fix_values = [cm, rseries * alpha, rseries]
    fix_p = get_fix_param(simvc_fix + ['voltageclamp.rseries'],
            simvc_fix_values)

    #
    # For individual bit in MwG
    #
    model_common.set_fix_parameters(fix_p)
    model_independent.set_fix_parameters(fix_p)

    problem_common = pints.SingleOutputProblem(model_common, times, data)
    loglikelihood_common = pints.GaussianKnownSigmaLogLikelihood(
            problem_common, noise_sigma)
    logprior_common = IKrWithoutConductanceLogPrior(transform_to_common_param,
            transform_from_common_param)
    log_posterior_common = pints.LogPosterior(loglikelihood_common,
            logprior_common)
    log_posteriors_common.append(log_posterior_common)

    problem_independent = pints.SingleOutputProblem(model_independent, times,
            data)
    loglikelihood_independent = pints.GaussianKnownSigmaLogLikelihood(
            problem_independent, noise_sigma)
    vc_prior = VoltageOffsetWithConductanceLogPrior(
            transform_to_gvc, transform_from_gvc)
    ileak_prior = pints.UniformLogPrior([-1e3], [1e3])
    prior_independent = pints.ComposedLogPrior(vc_prior, ileak_prior)
    log_posterior_independent = pints.LogPosterior(loglikelihood_independent,
            prior_independent)
    log_posteriors_independent.append(log_posterior_independent)

    print('Score at default parameters: ',
            log_posterior_common(transform_x0_common_all[-1]))

    fix_p = get_fix_param(independent_param, x0_independent_all[-1])
    update_logposterior_fix_param(log_posterior_common, fix_p)
    print('Score at updated parameters: ',
            log_posterior_common(transform_x0_common_all[-1]))

    print('Score at default parameters: ',
            log_posterior_independent(transform_x0_independent_all[-1]))

    fix_p = get_fix_param(common_param, x0_common_all[-1])
    update_logposterior_fix_param(log_posterior_independent, fix_p)
    print('Score at updated parameters: ',
            log_posterior_independent(transform_x0_independent_all[-1]))

    for _ in range(5):
        assert(log_posterior_common(transform_x0_common_all[-1]) ==\
                log_posterior_common(transform_x0_common_all[-1]))
        assert(log_posterior_independent(transform_x0_independent_all[-1]) ==\
                log_posterior_independent(transform_x0_independent_all[-1]))

data_all = np.asarray(data_all).T
times_all = np.asarray(times_all).T
times = times_all[:, 0]

transform_x0_common_all_mean = np.mean(transform_x0_common_all, axis=0)
x0_common_all_mean = np.copy(transform_to_common_param(transform_x0_common_all_mean))
total_log_posteriors_common = pints.SumOfIndependentLogPDFs(
        log_posteriors_common)

# quick sanity checks
assert(n_cells == len(log_posteriors_independent))
assert(n_cells == len(log_posteriors_common))


#
# Define the big loglikelihood
#
tip0 = np.copy(transform_x0_independent_all)
tcp0 = np.copy(transform_x0_common_all_mean)
n_opt_iterations_independent = 200

final_posterior = m.ParallelMultiLevelLogLikelihood(
            total_common_logposterior=total_log_posteriors_common,
            list_of_independent_logposteriors=log_posteriors_independent,
            common_param=common_param,
            independent_param=independent_param,
            transform_to_common_param=transform_to_common_param,
            transform_to_independent_param=transform_to_independent_param,
            transformed_independent_parameters0=tip0,
            n_opt=n_opt_iterations_independent,
            restart_fit=True,
            onlyNelderMead=True,
            n_workers=None)

print('Test total_log_posteriors_common:', total_log_posteriors_common(tcp0))
# This is important to run once...
# To set what's the prior lp within final_posterior object
print('Test final_posterior: ', final_posterior(tcp0))


#
# Run optimisation
# 
N = 1

logposteriors = []
params_c = []
params_i = []
n_iter = 1000
epsilon = 0.5  # when to switch from CMA-ES to Nelder-Mead

for _ in range(N):

    x0 = tcp0

    if epsilon > 0:
        # Create optimiser part 1
        opt = pints.OptimisationController(
                final_posterior, x0,
                method=pints.CMAES)
        opt.set_max_iterations(int(epsilon * n_iter))
        opt.set_parallel(False)

        # Run optimisation
        try:
            with np.errstate(all='ignore'):
                # Tell numpy not to issue warnings
                p, s = opt.run()
        except ValueError:
            import traceback
            traceback.print_exc()

        del(opt)
        x0 = np.copy(p)

    if epsilon < 1:
        # Create optimiser part 2
        opt = pints.OptimisationController(
                final_posterior, x0,
                method=pints.NelderMead)
        #opt.set_max_iterations(int((1 - epsilon) * n_iter))
        opt.set_parallel(False)

        # Run optimisation
        try:
            with np.errstate(all='ignore'):
                # Tell numpy not to issue warnings
                p, s = opt.run()
        except ValueError:
            import traceback
            traceback.print_exc()

        del(opt)

    # Found parameters
    found_tcp = np.copy(p)
    found_cp = transform_to_common_param(p)
    _ = final_posterior(found_tcp)
    found_ip = final_posterior.get_current_independent_param()

    logposteriors.append(s)
    params_c.append(found_cp)
    params_i.append(found_ip)


#
# Done
#
obtained_logposterior0 = logposteriors[0]
obtained_parameters0 = params_c[0]
obtained_parameters_i0 = params_i[0]

# Show results
saveas = file_name
print('Found solution:          Prior parameters:' )
# Store output
with open('%s/%s-solution-%s.txt' % (savedir, saveas, fit_seed), 'w') as f:
    for k, x in enumerate(obtained_parameters0):
        print(pints.strfloat(x) + '    ' + pints.strfloat(x0_common_all_mean[k]))
        f.write(pints.strfloat(x) + '\n')
np.savetxt('%s/%s-solution_i-%s.txt' % (savedir, saveas, fit_seed), obtained_parameters_i0)

with open('%s/%s-cells-%s.txt' % (savedir, saveas, fit_seed), 'w') as f:
    for c in selectedwell:
        f.write(c + '\n')

# Plot results
model_ideal = m.Model('../mmt-model-files/ideal-ikr.mmt',
                protocol_def=protocol_def,
                temperature=273.15 + temperature,  # K
                transform=parametertransform.donothing,
                useFilterCap=useFilterCap)  # ignore capacitive spike
voltage = model_ideal.voltage(times)
output0 = final_posterior.problem_evaluate(
        transform_from_common_param(obtained_parameters0),
        obtained_parameters_i0)
if not os.path.isdir(savedir + '/plot-solution-%s' % fit_seed):
    os.makedirs(savedir + '/plot-solution-%s' % fit_seed)
for i in range(n_cells):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axes[0].plot(times, voltage, c='#7f7f7f')
    axes[0].set_ylabel('Voltage (mV)')
    axes[1].plot(times, data_all[:, i], alpha=0.5, label='data')
    ideal = model_ideal.simulate(
            np.append(obtained_parameters_i0[i][0], obtained_parameters0),
            times)
    axes[1].plot(times, ideal, label='same kinetics')
    axes[1].plot(times, output0[:, i], label='found solution 1')
    axes[1].legend()
    axes[1].set_ylabel('Current (pA)')
    axes[1].set_xlabel('Time (ms)')
    plt.subplots_adjust(hspace=0)
    plt.savefig(savedir + '/plot-solution-%s/%s' % (fit_seed, selectedwell[i]))
    plt.close()

print('Done.')

## eof
