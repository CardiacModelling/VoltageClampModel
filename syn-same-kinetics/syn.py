#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('../lib')
import numpy as np

import model as m
import parameters

def generate_simvc(model, times, icell=None, set_parameters=None,
        return_parameters=False, return_voltage=False,
        return_leak_param=False, add_leak=False, add_linleak=False,
        transform=None, fakedatanoise=20.0):
    """
    Generate synthetic data

    Input
    =====
    model: Pints forward model
    times: time points for synthetic data
    icell: integer ID for the synthetic cell
    set_parameters: a fixed set of parameters with
                    [simulate_parameters, fixed_parameters]
    return_parameters: if True, return parameters
    return_voltage: if True, return voltage trace
    return_leak_param: if True, return leak parameter
    add_leak: if True, add (non-linear) leak residual to current
    add_linleak: if True, add (linear) leak residual to current
    transform: parameter transformation function ('from' transform)
    fakedatanoise: noise level in pA
    """

    if icell is None and set_parameters is None:
        raise ValueError('Either `icell` or `set_parameters` must not be None')
    elif icell is not None and set_parameters is not None:
        raise ValueError('Only one `icell` or `set_parameters` can be set')

    if transform is None:
        from parametertransform import donothing
        transform = donothing

    if set_parameters is None:
        # Set mean parameters
        p_voffset_mean = 0  # mV
        p_rseries_mean = 7.5e-3  # GOhm
        p_cm_mean = 12.5  # pF
        p_ikr = np.array([  # paper 1 supplement HBM mean parameters
            3.23e+4,
            9.48e-2, 8.69e+1, 2.98e-2, 4.69e+1,
            1.04e+2, 2.19e+1, 8.05e+0, 2.99e+1]) * 1e-3  # V, s -> mV, ms
        p_ikr_g_mean = p_ikr[0]
        p_ikr_kinetics = p_ikr[1:]
        alpha = 0.8  # 80% compensation

        # Set std of the parameters
        std_voffset = 1.5  # mV, see paper 1
        std_rseries = 2.5e-3  # GOhm; LogNormal
        std_cm = 5.0  # pF; LogNormal
        std_g = p_ikr_g_mean  # big variability for conductance; LogNormal
        std_est_error = 0.1  # 10% error for rseries estimation?

        # Fix seed
        np.random.seed(icell)
        fit_seed = np.random.randint(0, 2**30)
        print('Using seed: ', fit_seed)
        np.random.seed(fit_seed)

        # Generate parameter sample
        voffset = np.random.normal(p_voffset_mean, std_voffset)
        rseries_logmean = np.log(p_rseries_mean) \
                - 0.5 * np.log((std_rseries / p_rseries_mean) ** 2 + 1.)
        rseries_scale = np.sqrt(np.log((std_rseries / p_rseries_mean) ** 2 \
                + 1.))
        rseries = np.random.lognormal(rseries_logmean, rseries_scale)
        cm_logmean = np.log(p_cm_mean) \
                - 0.5 * np.log((std_cm / p_cm_mean) ** 2 + 1.)
        cm_scale = np.sqrt(np.log((std_cm / p_cm_mean) ** 2 + 1.))
        cm = np.random.lognormal(cm_logmean, cm_scale)
        est_rseries = rseries * (1.0 + np.random.normal(0, std_est_error))

        p_ikr_g_mean = p_ikr_g_mean * 2./3.
        std_g = std_g * 2./3.
        g_logmean = np.log(p_ikr_g_mean) \
                - 0.5  * np.log((std_g / p_ikr_g_mean) ** 2 + 1.)
        g_scale = np.sqrt(np.log((std_g / p_ikr_g_mean) ** 2 + 1.))
        ikr_g = p_ikr_g_mean / 2. + np.random.lognormal(g_logmean, g_scale)

        # Lump parameters together
        p_ikr = np.append(ikr_g, p_ikr_kinetics) 
        p = np.append(p_ikr, [
                rseries,  # GOhm
                voffset,  # mV
                ])
        simvc_param_to_fix = np.array([
                cm,  # pF
                alpha * est_rseries,  # GOhm
                ])

        fix_p = {}
        for i, j in zip(parameters.simvc_fix, simvc_param_to_fix):
            fix_p[i] = j
    else:
        p, fix_p = set_parameters

    # Return parameters
    if return_parameters:
        return p, fix_p

    # Set model parameters
    model.set_parameters(parameters.simvc)
    model.set_fix_parameters(fix_p)

    # Return voltage
    if return_voltage:
        return model.voltage(times, parameters=p)

    # Simulate
    i = model.simulate(transform(p), times)
    i += np.random.normal(0, fakedatanoise, size=i.shape)

    # Add leak residual
    if add_leak:
        voltage =  model.voltage(times, parameters=p)

        i_s_mean = 6.0
        i_s_std = 3.5
        i_s_logmean = np.log(i_s_mean) \
                - 0.5 * np.log((i_s_std / i_s_mean) ** 2 + 1.)
        i_s_scale = np.sqrt(np.log((i_s_std / i_s_mean) ** 2 + 1.))
        i_s = np.random.lognormal(i_s_logmean, i_s_scale)

        n_shift = 1.0
        n_logmean = np.log(0.35)
        n_scale = 0.55
        n = n_shift + np.random.lognormal(n_logmean, n_scale)
        kbTq = 0.02586e3  # 300 K

        if return_leak_param:
            return [i_s, n]

        leak = i_s * (np.exp(voltage / (n * kbTq)) - 1.)
        i += leak

    # Add linear leak residual
    if add_linleak:
        voltage =  model.voltage(times, parameters=p)

        i_s_mean = 0.25
        i_s_std = 0.1
        i_s_logmean = np.log(i_s_mean) \
                - 0.5 * np.log((i_s_std / i_s_mean) ** 2 + 1.)
        i_s_scale = np.sqrt(np.log((i_s_std / i_s_mean) ** 2 + 1.))
        i_s = np.random.lognormal(i_s_logmean, i_s_scale)

        if return_leak_param:
            return [i_s]

        leak = i_s * (voltage + 80.0)
        i += leak

    return i


def generate_full2vc(model, times, icell=None, set_parameters=None,
        return_parameters=False, return_voltage=False,
        add_linleak=False, transform=None, fakedatanoise=20.0):
    """
    Generate synthetic data

    Input
    =====
    model: Pints forward model
    times: time points for synthetic data
    icell: integer ID for the synthetic cell
    set_parameters: a fixed set of parameters with
                    [simulate_parameters, fixed_parameters]
    return_parameters: if True, return parameters
    return_voltage: if True, return voltage trace
    add_linleak: if True, add (linear) leak residual to current
    transform: parameter transformation function ('from' transform)
    fakedatanoise: noise level in pA
    """

    if icell is None and set_parameters is None:
        raise ValueError('Either `icell` or `set_parameters` must not be None')
    elif icell is not None and set_parameters is not None:
        raise ValueError('Only one `icell` or `set_parameters` can be set')

    if transform is None:
        from parametertransform import donothing
        transform = donothing

    if set_parameters is None:
        # Set mean parameters
        p_voffset_mean = 0  # mV
        p_rseries_mean = 12.5e-3  # GOhm
        p_cprs_mean = 4.  # pF
        p_cm_mean = 15.  # pF
        p_ikr = np.array([  # paper 1 supplement HBM mean parameters
            3.23e+4,
            9.48e-2, 8.69e+1, 2.98e-2, 4.69e+1,
            1.04e+2, 2.19e+1, 8.05e+0, 2.99e+1]) * 1e-3  # V, s -> mV, ms
        p_ikr_g_mean = p_ikr[0]
        p_ikr_kinetics = p_ikr[1:]
        alpha = 0.8  # 80% compensation

        # Set std of the parameters
        std_voffset = 1.5  # mV, see paper 1
        std_rseries = 2e-3  # GOhm; LogNormal
        std_cprs = 1.0  # pF; LogNormal
        std_cm = 2.5  # pF; LogNormal
        std_g = p_ikr_g_mean  # big variability for conductance; LogNormal
        std_est_error = 0.1  # 10% error for rseries estimation?

        # Fix seed
        np.random.seed(icell)
        fit_seed = np.random.randint(0, 2**30)
        print('Using seed: ', fit_seed)
        np.random.seed(fit_seed)

        # Generate parameter sample
        voffset = np.random.normal(p_voffset_mean, std_voffset)
        rseries_logmean = np.log(p_rseries_mean) \
                - 0.5 * np.log((std_rseries / p_rseries_mean) ** 2 + 1.)
        rseries_scale = np.sqrt(np.log((std_rseries / p_rseries_mean) ** 2 \
                + 1.))
        rseries = np.random.lognormal(rseries_logmean, rseries_scale)
        cprs_logmean = np.log(p_cprs_mean) \
                - 0.5 * np.log((std_cprs / p_cprs_mean) ** 2 + 1.)
        cprs_scale = np.sqrt(np.log((std_cprs / p_cprs_mean) ** 2 + 1.))
        cprs = np.random.lognormal(cprs_logmean, cprs_scale)
        cm_logmean = np.log(p_cm_mean) \
                - 0.5 * np.log((std_cm / p_cm_mean) ** 2 + 1.)
        cm_scale = np.sqrt(np.log((std_cm / p_cm_mean) ** 2 + 1.))
        cm = np.random.lognormal(cm_logmean, cm_scale)
        est_rseries = rseries * (1.0 + np.random.normal(0, std_est_error))
        est_cm = cm * (1.0 + np.random.normal(0, std_est_error))
        est_cprs = cprs * (1.0 + np.random.normal(0, std_est_error))

        p_ikr_g_mean = p_ikr_g_mean * 2./3.
        std_g = std_g * 2./3.
        g_logmean = np.log(p_ikr_g_mean) \
                - 0.5  * np.log((std_g / p_ikr_g_mean) ** 2 + 1.)
        g_scale = np.sqrt(np.log((std_g / p_ikr_g_mean) ** 2 + 1.))
        ikr_g = p_ikr_g_mean / 2. + np.random.lognormal(g_logmean, g_scale)

        # Lump parameters together
        p_ikr = np.append(ikr_g, p_ikr_kinetics) 
        p = np.append(p_ikr, [
                cm,  # pF
                rseries,  # GOhm
                cprs,  # pF
                voffset,  # mV
                ])
        full2vc_param_to_fix = np.array([
                est_cm,  # pF
                alpha * est_rseries,  # GOhm
                est_cprs,  # pF
                ])

        fix_p = {}
        for i, j in zip(parameters.full2vc_fix, full2vc_param_to_fix):
            fix_p[i] = j
    else:
        p, fix_p = set_parameters

    # Add linear leak residual
    if add_linleak:
        i_s_mean = 0.25
        i_s_std = 0.1
        i_s_logmean = np.log(i_s_mean) \
                - 0.5 * np.log((i_s_std / i_s_mean) ** 2 + 1.)
        i_s_scale = np.sqrt(np.log((i_s_std / i_s_mean) ** 2 + 1.))
        i_s = np.random.lognormal(i_s_logmean, i_s_scale)
        p = np.append(p, i_s)
    else:
        p = np.append(p, 0)

    # Return parameters
    if return_parameters:
        return p, fix_p

    # Set model parameters
    model.set_parameters(parameters.full2vc + ['voltageclamp.gLeak'])
    model.set_fix_parameters(fix_p)

    # Return voltage
    if return_voltage:
        return model.voltage(times, parameters=p)

    # Simulate
    i = model.simulate(transform(p), times)
    i += np.random.normal(0, fakedatanoise, size=i.shape)

    return i

