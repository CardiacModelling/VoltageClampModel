#/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
if not '--show' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import protocols
import model as m
from parameters import simvc, get_qc
from parameters import simvc_fix, simvc_fix_typical_values
from parameters import simvc_typical_values
from releakcorrect import I_releak, score_leak, protocol_leak_check

from scipy.optimize import fmin
# Set seed
np.random.seed(101)

def get_fix_param(var, val):
    """
        var: variable name.
        val: variable value to fix.
    """
    out = {}
    for i, j in zip(var, val):
        out[i] = j
    return out


#
# Protocols
#
protocol_funcs = {
    'staircaseramp': protocols.leak_staircase,
    'pharma': protocols.pharma,  # during drug application
    'apab': 'protocol-apab.csv',
    'apabv3': 'protocol-apabv3.csv',
    'ap05hz': 'protocol-ap05hz.csv',
    'ap1hz': 'protocol-ap1hz.csv',
    'ap2hz': 'protocol-ap2hz.csv',
    'sactiv': protocols.sactiv,
    'sinactiv': protocols.sinactiv,
}
protocol_dir = '../protocol-time-series'
protocol_list = [
        'staircaseramp',
        'pharma',
        'apab',
        'apabv3',
        'ap05hz',
        'ap1hz',
        'ap2hz',
        'sactiv',
        'sinactiv',
        ]
protocol_list = ['staircaseramp']
prt_names = ['Staircase', 'pharma', 'EAD', 'DAD', 'AP05Hz', 'AP1Hz', 'AP2Hz', 'actIV', 'inactIV']

# IV protocol special treatment
protocol_iv = [
    'sactiv',
    'sinactiv',
]
protocol_iv_times = {
    'sactiv': protocols.sactiv_times,
    'sinactiv': protocols.sinactiv_times,
}
protocol_iv_convert = {
    'sactiv': protocols.sactiv_convert,
    'sinactiv': protocols.sinactiv_convert,
}
protocol_iv_args = {
    'sactiv': protocols.sactiv_iv_arg,
    'sinactiv': protocols.sinactiv_iv_arg,
}
protocol_iv_v = {
    'sactiv': protocols.sactiv_v,
    'sinactiv': protocols.sinactiv_v,
}

data_dir = '../data-autoLC'
data_dir_staircase = '../data'
file_dir = './out'
file_name = 'herg25oc1'
temperature = 25.0 + 273.15  # in K
fit_seed = '209652396'
withfcap = False

savepath = 'figs/membrane-voltage-scheme3'
if not os.path.isdir(savepath):
    os.makedirs(savepath)

# Leak param
leakbeforeparam = np.loadtxt('../qc/' + file_name + '-staircaseramp-leak_before.txt')
leakafterparam = np.loadtxt('../qc/' + file_name + '-staircaseramp-leak_after.txt')
cell_id_file = '../qc/%s-staircaseramp-cell_id.txt' % file_name
cell_ids = []
with open(cell_id_file, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            cell_ids.append(l.split()[0])

# Get selected cells
files_dir = os.path.realpath(os.path.join(file_dir, file_name))
searchwfcap = '-fcap' if withfcap else ''
selectedfile = '../manualselection/manualv2selected-%s.txt' % (file_name)
selectedwell = []
with open(selectedfile, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            selectedwell.append(l.split()[0])

for prt in protocol_list:
    # Model
    protocol_def = protocol_funcs[prt]
    if type(protocol_def) is str:
        protocol_def = '%s/%s' % (protocol_dir, protocol_def)

    ideal_model = m.Model('../mmt-model-files/ideal-ikr.mmt',
                        protocol_def=protocol_def,
                        temperature=temperature,  # K
                        transform=None,
                        useFilterCap=False)  # ignore capacitive spike

    model = m.Model(
            '../mmt-model-files/simplified-voltage-clamp-ikr-linleak.mmt',
            protocol_def=protocol_def,
            temperature=temperature,  # K
            transform=None,
            useFilterCap=False)  # ignore capacitive spike
    # Set which parameters to be inferred
    model.set_parameters([
        'ikr.g',
        #'voltageclamp.rseries',
        'voltageclamp.voffset_eff',
        'voltageclamp.gLeak'])

    # Time points
    times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, file_name,
            prt), delimiter=',', skiprows=1)
    times = times * 1e3  # s -> ms

    # Voltage protocol
    if prt not in protocol_iv:
        times_sim = np.copy(times)
        voltage_c = ideal_model.voltage(times_sim)
    else:
        times_sim = protocol_iv_times[prt](times[1] - times[0])
        voltage_c = ideal_model.voltage(times_sim)
        voltage_c, t = protocol_iv_convert[prt](voltage_c, times_sim)
        assert(np.mean(np.abs(t - times)) < 1e-6)

    # Parameters
    fn = '%s-scheme3-simvclinleak/%s-cells-%s.txt' % \
         (files_dir, file_name, fit_seed)
    scheme3_cell_list = []
    with open(fn, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                scheme3_cell_list.append(l.split()[0])

    param_file = '%s-scheme3-simvclinleak/%s-solution_i-%s.txt' % \
            (files_dir, file_name, fit_seed)
    obtained_parameters_all = np.loadtxt(param_file)

    ikr_param = [
            'ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4',
            'ikr.p5', 'ikr.p6', 'ikr.p7', 'ikr.p8',  
            ]
    p_ikr = np.loadtxt('%s-scheme3-simvclinleak/%s-solution-%s.txt' % \
            (files_dir, file_name, fit_seed))

    # Calculate ranking
    rmsd = []

    for i_cell, cell in enumerate(selectedwell):
        # Data
        if prt == 'staircaseramp':
            data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir_staircase,
                    file_name, prt, cell), delimiter=',', skiprows=1)
        elif prt not in protocol_iv:
            data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, file_name,
                    prt, cell), delimiter=',', skiprows=1)
            # Re-leak correct the leak corrected data...
            g_releak = fmin(score_leak, [0.0], args=(data, voltage, times,
                                protocol_leak_check[prt]), disp=False)
            data = I_releak(g_releak[0], data, voltage)
        else:
            data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, file_name,
                    prt, cell), delimiter=',', skiprows=1)
            # Re-leak correct the leak corrected data...
            for i in range(data.shape[1]):
                g_releak = fmin(score_leak, [0.0], args=(data[:, i],
                                    voltage[:, i], times,
                                    protocol_leak_check[prt]), disp=False)
                data[:, i] = I_releak(g_releak[0], data[:, i], voltage[:, i])
        assert(len(data) == len(times))

        # For fix kinetics model
        rseal, cm, rseries = get_qc('../qc', file_name, cell)
        #print('Est. Rseal, Cm, Rseries:', rseal, cm, rseries, '(GOhm, pF, GOhm)')
        alpha = 0.8  # rseries %compensation
        simvc_fix_values = [cm, rseries * alpha, rseries]
        extra_fix = ['voltageclamp.rseries']
        updateELeakCorrection = False
        cell_idx = cell_ids.index(cell)
        ga, Ea = leakbeforeparam[cell_idx]
        gb, Eb = leakafterparam[cell_idx]
        if updateELeakCorrection:
            ELeakCorrection = - (ga * Ea - gb * Eb) / (gb - ga)
            #print('E_Leak correction: ', ELeakCorrection, ' (mV)')
            if np.abs(ELeakCorrection) > 200: print('==' * 30, ga, Ea, gb, Eb)
            extra_fix += ['voltageclamp.ELeak']
            simvc_fix_values += [ELeakCorrection]

        scheme3_cell_idx = scheme3_cell_list.index(cell)
        obtained_parameters = obtained_parameters_all[scheme3_cell_idx]

        # Simulation
        fix_p = get_fix_param(ikr_param + simvc_fix + extra_fix + ['voltageclamp.gLeak_est', 'voltageclamp.ELeak_est'],
                np.append(p_ikr, simvc_fix_values + [0, -80]))
        model.set_fix_parameters(fix_p)
        simulation = model.simulate(obtained_parameters, times_sim)
        if prt != 'staircaseramp' and prt not in protocol_iv:
            # Re-leak correct the leak corrected simulation... TODO?
            g_releak_simulation = fmin(score_leak, [0.1], args=(simulation, voltage, times,
                                protocol_leak_check[prt]), disp=False)
            simulation = I_releak(g_releak_simulation[0], simulation, voltage)
        if prt in protocol_iv:
            simulation, t = protocol_iv_convert[prt](simulation, times_sim)
            assert(np.mean(np.abs(t - times)) < 1e-6)
            simulation, t = protocol_iv_convert[prt](
                    simulation, times_sim)
            assert(np.mean(np.abs(t - times)) < 1e-6)
            # Re-leak correct the leak corrected simulation... TODO?
            for i in range(simulation.shape[1]):
                g_releak_simulation = fmin(score_leak, [0.1], args=(simulation[:, i],
                                    voltage[:, i], times,
                                    protocol_leak_check[prt]), disp=False)
                simulation[:, i] = I_releak(g_releak_simulation[0], simulation[:, i], voltage[:, i])

        if prt not in protocol_iv:
            times_sim = np.copy(times)
            voltage = model.voltage(times, parameters=obtained_parameters)
        else:
            times_sim = protocol_iv_times[prt](times[1] - times[0])
            voltage = model.voltage(times_sim, parameters=obtained_parameters)
            voltage, t = protocol_iv_convert[prt](voltage, times_sim)
            assert(np.mean(np.abs(t - times)) < 1e-6)

        # Use ga TODO only for staircase for now
        fix_p = get_fix_param(ikr_param + simvc_fix + extra_fix + ['voltageclamp.gLeak_est', 'voltageclamp.ELeak_est'],
                np.append(p_ikr, simvc_fix_values + [ga, Ea]))
        model.set_fix_parameters(fix_p)
        simulation_2 = model.simulate(obtained_parameters, times_sim)
        #preE4031_leak_param = np.copy(obtained_parameters)
        #preE4031_leak_param[2] = ga  # replace with g_leak^* for pre-E4031
        #simulation_2 = model.simulate(preE4031_leak_param, times_sim)
        #simulation_2 -= ga * (voltage_c - (-80))  # take out leak from trace
        #simulation_2 += obtained_parameters[2] * (voltage_c - (-80)) # replace with g_leak^\dagger
        voltage_2 = model.voltage(times_sim, parameters=obtained_parameters)

        rmsd.append(np.sqrt(np.mean((voltage_2 - voltage) ** 2)))

        #
        # Plot
        #
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        if prt not in protocol_iv:
            # protocol
            axes[0].plot(times * 1e-3, voltage_c, c='#7f7f7f', label=r'$V_c$')
            # membrane voltage
            axes[0].plot(times * 1e-3, voltage, c='C0', label=r'$V_m$')
            # Just offset?
            axes[0].plot(times * 1e-3, voltage_c + obtained_parameters[1], '--', c='C1', label=r'$V_c + V_{off}$')
            # g_leak^* pre-E4031
            axes[0].plot(times * 1e-3, voltage_2, ':', c='C2', label=r'$V_m$ with $g_{leak}^*$ from pre-E4031')
        else:
            for i in range(voltage.shape[1]):
                # protocol
                axes[0].plot(times * 1e-3, voltage_c[:, i], c='#7f7f7f')
                # membrane voltage
                axes[0].plot(times * 1e-3, voltage[:, i], c='#C0')

        if prt not in protocol_iv:
            # recording
            axes[1].plot(times * 1e-3, data, lw=1, alpha=0.5,
                    c='#9ecae1', label='data')
            # simulation
            axes[1].plot(times * 1e-3, simulation, lw=2,
                    c='#d95f02', label='model; $g_{Kr}=%.1f, V^\dagger_{off}=%.2f, g^\dagger_{Leak}=%.2f$' % (obtained_parameters[0], obtained_parameters[1], obtained_parameters[2]))
            # g_leak^* pre-E4031
            axes[1].plot(times * 1e-3, simulation_2, ':', lw=2, c='C2', label='Simulated with $g_{leak}^*=%s$ from pre-E4031' % ga)
        else:
            iv_v = protocol_iv_v[prt]()  # mV
            # recording
            iv_i = protocols.get_corrected_iv(data, times,
                                              *protocol_iv_args[prt]())
            axes[1].plot(iv_v, iv_i / np.max(iv_i), lw=2,
                    alpha=0.25, c='#9ecae1', label='data')
            # simulation
            iv_i = protocols.get_corrected_iv(simulation, times,
                                              *protocol_iv_args[prt]())
            axes[1].plot(iv_v, iv_i / np.max(iv_i), lw=2,
                    alpha=1, c='#d95f02', label='model prediction')
            if prt == 'sactiv':
                axes[1].set_ylim([-0.05, 1.05])
            elif prt == 'sinactiv':
                axes[1].set_ylim([-5, 1.05])

        axes[0].legend()
        axes[1].legend()
        plt.savefig('%s/membrane-voltage-scheme3-%s' % (savepath, cell), dpi=200)
        plt.close()

    cell_sorted = [x for _, x in sorted(zip(rmsd, selectedwell))]
    ids = np.argsort(rmsd)
    rmsd_sorted = np.asarray(rmsd)[ids]
    np.savetxt('%s/membrane-voltage-scheme3-rmsd.txt' % (savepath), rmsd_sorted)
    with open('%s/membrane-voltage-scheme3-cell.txt' % (savepath), 'w') as f:
        for c in cell_sorted:
            f.write(c + '\n')
