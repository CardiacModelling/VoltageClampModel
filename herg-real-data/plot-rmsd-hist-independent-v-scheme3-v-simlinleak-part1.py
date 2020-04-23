#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import string

import protocols
import model as m
from parameters import ikr as ikr_parameters
from parameters import simvc, get_qc
from parameters import simvc_fix, simvc_fix_typical_values
from parameters import simvc_typical_values
from releakcorrect import I_releak, score_leak, protocol_leak_check

from scipy.optimize import fmin
# Set seed
np.random.seed(101)

savedir = './figs'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

#refcell = 'D19'

def get_fix_param(var, val):
    """
        var: variable name.
        val: variable value to fix.
    """
    out = {}
    for i, j in zip(var, val):
        out[i] = j
    return out

def rmsd_compute(t1, t2):
    # Normalised RMSD value between trace 1 ``t1`` and trace 2 ``t2``
    #
    # Note, usually normalise to data, so 
    # - ``t2`` data (or anything as reference)
    # - ``t1`` simulation (or anything for comparison)
    return np.sqrt(np.mean((t1 - t2) ** 2)) / np.sqrt(np.mean(t2 ** 2))


#
# Protocol info
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
    # 'sactiv',
    # 'sinactiv',
    'pharma',
    'apab',
    'apabv3',
    # 'ap05hz',
    'ap1hz',
    'ap2hz',
]
validation_idx = [
    None,
    # 1,
    # 2,
    3,
    4,
    5,
    # 6,
    7,
    8,
]

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

data_dir_staircase = '../data'
data_dir = '../data-autoLC'
file_dir = '../../hERGRapidCharacterisation/room-temperature-only/out'
file_dir2 = './out'
file_list = [
        'herg25oc1',
        ]
temperatures = np.array([25.0])
temperatures += 273.15  # in K
fit_seed = '542811797'
fit_seed2 = '717354021'

file_name = file_list[0]
temperature = temperatures[0]

# Load RMSD matrix
rmsd_matrix_file = '../../hERGRapidCharacterisation/room-temperature-only/figs/rmsd-hist-%s-autoLC-releak/rmsd-matrix.txt' \
                   % file_name
rmsd_cells_file = '../../hERGRapidCharacterisation/room-temperature-only/figs/rmsd-hist-%s-autoLC-releak/rmsd-matrix-cells.txt' \
                  % file_name

rmsd_matrix = np.loadtxt(rmsd_matrix_file)

with open(rmsd_matrix_file, 'r') as f:
    rmsd_prt = f.readline().strip('\n').strip('#').split()

rmsd_cells = []
with open(rmsd_cells_file, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            rmsd_cells.append(l.strip('\n').split('-')[1])


rmsd_matrix_file3 = './out/rmsd-hist-herg25oc1-fit-simvclinleak/rmsd-matrix.txt'
rmsd_cells_file3 = './out/rmsd-hist-herg25oc1-fit-simvclinleak/rmsd-matrix-cells.txt'

rmsd_matrix3 = np.loadtxt(rmsd_matrix_file3)

with open(rmsd_matrix_file3, 'r') as f:
    rmsd_prt3 = f.readline().strip('\n').strip('#').split()

rmsd_cells3 = []
with open(rmsd_cells_file3, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            rmsd_cells3.append(l.strip('\n').split('-')[1])


rmsd_matrix_file2 = './out/rmsd-hist-%s-fixkinetics-simvclinleak-scheme3/rmsd-matrix.txt' \
                   % file_name
rmsd_cells_file2 = './out/rmsd-hist-%s-fixkinetics-simvclinleak-scheme3/rmsd-matrix-cells.txt' \
                  % file_name

rmsd_matrix2 = np.loadtxt(rmsd_matrix_file2)

with open(rmsd_matrix_file2, 'r') as f:
    rmsd_prt2 = f.readline().strip('\n').strip('#').split()

rmsd_cells2 = []
with open(rmsd_cells_file2, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            rmsd_cells2.append(l.strip('\n').split('-')[1])


rankedlabels = [r'$*$',
                u'\u2021',
                r'#',
                u'\u2666']


#
# Do a very very tailored version........ :(
#
fig = plt.figure(figsize=(16, 15))
bigxgap = 12
n_xgrid = 84
bigygap = 5
n_ygrid = 31
grid = plt.GridSpec(2 * n_ygrid + 1 * bigygap, 3 * n_xgrid + 2 * bigxgap,
                    hspace=0.0, wspace=0.0)
axes = np.empty([10, int(len(protocol_list) / 2)], dtype=object)
# long list here:
for i in range(int(len(protocol_list) / 2)):
    i_grid = i * (n_xgrid + bigxgap)
    f_grid = (i + 1) * n_xgrid + i * bigxgap

    # First 'row'
    axes[0, i] = fig.add_subplot(grid[0:3, i_grid:f_grid])
    axes[0, i].set_xticklabels([])
    axes[1, i] = fig.add_subplot(grid[3:9, i_grid:f_grid])
    axes[1, i].set_xticklabels([])
    axes[2, i] = fig.add_subplot(grid[9:15, i_grid:f_grid])
    axes[2, i].set_xticklabels([])
    axes[3, i] = fig.add_subplot(grid[15:21, i_grid:f_grid])
    # Histogram
    axes[4, i] = fig.add_subplot(grid[24:31, i_grid:f_grid])

    # Second 'row'
    n_shift = n_ygrid + bigygap
    axes[5, i] = fig.add_subplot(grid[n_shift+0:n_shift+3, i_grid:f_grid])
    axes[5, i].set_xticklabels([])
    axes[6, i] = fig.add_subplot(grid[n_shift+3:n_shift+9, i_grid:f_grid])
    axes[6, i].set_xticklabels([])
    axes[7, i] = fig.add_subplot(grid[n_shift+9:n_shift+15, i_grid:f_grid])
    axes[7, i].set_xticklabels([])
    axes[8, i] = fig.add_subplot(grid[n_shift+15:n_shift+21, i_grid:f_grid])
    # Histogram
    axes[9, i] = fig.add_subplot(grid[n_shift+24:n_shift+31, i_grid:f_grid])

    # Set x-labels
    axes[3, i].set_xlabel('Time [s]', fontsize=14)
    axes[4, i].set_xlabel('RRMSE', fontsize=14)
    axes[8, i].set_xlabel('Time [s]', fontsize=14)
    axes[9, i].set_xlabel('RRMSE', fontsize=14)

# Set labels
axes[0, 0].set_ylabel('Voltage\n[mV]', fontsize=14)
axes[1, 0].set_ylabel(r'Best ($*$)', fontsize=14, color='#d95f02')
axes[2, 0].set_ylabel(u'Median (\u2021)', fontsize=14, color='#d95f02')
axes[3, 0].set_ylabel(r'90%ile (#)', fontsize=14, color='#d95f02')
axes[4, 0].set_ylabel('Frequency\n[N=%s]' % len(rmsd_cells), fontsize=14)
axes[5, 0].set_ylabel('Voltage\n[mV]', fontsize=14)
axes[6, 0].set_ylabel(r'Best ($*$)', fontsize=14, color='#d95f02')
axes[7, 0].set_ylabel(u'Median (\u2021)', fontsize=14, color='#d95f02')
axes[8, 0].set_ylabel(r'90%ile (#)', fontsize=14, color='#d95f02')
axes[9, 0].set_ylabel('Frequency\n[N=%s]' % len(rmsd_cells), fontsize=14)

axes[2, 0].text(-0.25, 0.5, 'Current [pA]', rotation=90, fontsize=18,
        transform=axes[2, 0].transAxes, ha='center', va='center')
axes[7, 0].text(-0.25, 0.5, 'Current [pA]', rotation=90, fontsize=18,
        transform=axes[7, 0].transAxes, ha='center', va='center')


#
# Model
#
prt2model = {}
prt2model2 = {}
prt2fixkineticsmodel = {}
for prt in protocol_list:

    protocol_def = protocol_funcs[prt]
    if type(protocol_def) is str:
        protocol_def = '%s/%s' % (protocol_dir, protocol_def)

    prt2model[prt] = m.Model('../mmt-model-files/ideal-ikr.mmt',
                        protocol_def=protocol_def,
                        temperature=temperature,  # K
                        transform=None,
                        useFilterCap=False)  # ignore capacitive spike

    prt2model2[prt] = m.Model(
            '../mmt-model-files/simplified-voltage-clamp-ikr-linleak.mmt',
            protocol_def=protocol_def,
            temperature=temperature,  # K
            transform=None,
            useFilterCap=False)  # ignore capacitive spike

    prt2fixkineticsmodel[prt] = m.Model(
                '../mmt-model-files/simplified-voltage-clamp-ikr-linleak.mmt',
                protocol_def=protocol_def,
                temperature=temperature,  # K
                transform=None,
                useFilterCap=False)  # ignore capacitive spike


#
# Plot
#
for i_prt, prt in enumerate(protocol_list):

    # Calculate axis index
    ai, aj = 5 * int(i_prt / 3), i_prt % 3

    # Title
    if prt == 'staircaseramp':
        axes[ai, aj].set_title('Calibration', fontsize=16)
    else:
        axes[ai, aj].set_title('Validation %s' % validation_idx[i_prt],
                fontsize=16)

    # Add label!
    axes[ai, aj].text(-0.1, 1.4, string.ascii_uppercase[i_prt],
                      transform=axes[ai, aj].transAxes, size=20,
                      weight='bold')

    # Time point
    times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, file_name,
        prt), delimiter=',', skiprows=1) * 1e3  # s -> ms

    # Protocol
    model = prt2model[prt]
    modelfixkinetics = prt2fixkineticsmodel[prt]
    # Set which parameters to be inferred
    modelfixkinetics.set_parameters([
        'ikr.g',
        #'voltageclamp.rseries',
        'voltageclamp.voffset_eff',
        'voltageclamp.gLeak'])

    model2 = prt2model2[prt]
    simvclinleak = ikr_parameters + [
        'voltageclamp.voffset_eff',
        'voltageclamp.gLeak'
        ]
    model2.set_parameters(simvclinleak)


    if prt not in protocol_iv:
        times_sim = np.copy(times)
        voltage = model.voltage(times_sim)
    else:
        times_sim = protocol_iv_times[prt](times[1] - times[0])
        voltage = model.voltage(times_sim)
        voltage, t = protocol_iv_convert[prt](voltage, times_sim)
        assert(np.mean(np.abs(t - times)) < 1e-8)
    axes[ai, aj].set_ylim((np.min(voltage) - 10, np.max(voltage) + 15))

    # Plot protocol
    if prt not in protocol_iv:
        axes[ai, aj].plot(times, voltage, c='#696969')
    else:
        # protocol
        for i in range(voltage.shape[1]):
            axes[ai, aj].plot(times, voltage[:, i], c='#696969')

    # Calculate ranking
    rmsd = rmsd_matrix[:, rmsd_prt.index(prt)]
    best_cell = np.argmin(rmsd)
    median_cell = np.argsort(rmsd)[len(rmsd)//2]
    p90_cell = np.argsort(rmsd)[int(len(rmsd)*0.9)]
    rankedcells = [rmsd_cells[best_cell],
                   rmsd_cells[median_cell],
                   rmsd_cells[p90_cell]]
    rankedvalues = [rmsd[best_cell],
                    rmsd[median_cell],
                    rmsd[p90_cell]]
                    #rmsd[rmsd_cells.index(refcell)]]

    rmsd2 = rmsd_matrix2[:, rmsd_prt2.index(prt)]
    #best_cell2 = np.argmin(rmsd2)
    #median_cell2 = np.argsort(rmsd2)[len(rmsd2)//2]
    #p90_cell2 = np.argsort(rmsd2)[int(len(rmsd2)*0.9)]
    #NOTE Compare with the 'red' cells; not its own ranking!!
    best_cell2 = rmsd_cells2.index(rmsd_cells[best_cell])
    median_cell2 = rmsd_cells2.index(rmsd_cells[median_cell])
    p90_cell2 = rmsd_cells2.index(rmsd_cells[p90_cell])
    rankedcells2 = [rmsd_cells2[best_cell2],
                   rmsd_cells2[median_cell2],
                   rmsd_cells2[p90_cell2]]
    rankedvalues2 = [rmsd2[best_cell2],
                    rmsd2[median_cell2],
                    rmsd2[p90_cell2]]
                    #rmsd2[rmsd_cells2.index(refcell)]]

    rmsd3 = rmsd_matrix3[:, rmsd_prt3.index(prt)]
    #best_cell3 = np.argmin(rmsd3)
    #median_cell3 = np.argsort(rmsd3)[len(rmsd3)//2]
    #p90_cell3 = np.argsort(rmsd3)[int(len(rmsd3)*0.9)]
    #NOTE Compare with the 'red' cells; not its own ranking!!
    best_cell3 = rmsd_cells3.index(rmsd_cells[best_cell])
    median_cell3 = rmsd_cells3.index(rmsd_cells[median_cell])
    p90_cell3 = rmsd_cells3.index(rmsd_cells[p90_cell])
    rankedcells3 = [rmsd_cells3[best_cell3],
                   rmsd_cells3[median_cell3],
                   rmsd_cells3[p90_cell3]]
    rankedvalues3 = [rmsd3[best_cell3],
                    rmsd3[median_cell3],
                    rmsd3[p90_cell3]]
                    #rmsd3[rmsd_cells3.index(refcell)]]

    # Parameters
    fn = '%s/%s-scheme3-simvclinleak/%s-cells-%s.txt' % \
         (file_dir2, file_name, file_name, fit_seed2)
    scheme3_cell_list = []
    with open(fn, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                scheme3_cell_list.append(l.split()[0])

    param_file = '%s/%s-scheme3-simvclinleak/%s-solution_i-%s.txt' % \
            (file_dir2, file_name, file_name, fit_seed2)
    obtained_parameters_all = np.loadtxt(param_file)

    ikr_param = [
            'ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4',
            'ikr.p5', 'ikr.p6', 'ikr.p7', 'ikr.p8',  
            ]
    p_ikr = np.loadtxt('%s/%s-scheme3-simvclinleak/%s-solution-%s.txt' % \
            (file_dir2, file_name, file_name, fit_seed2))

    p_ikr_hbm = np.array([  # paper 1 supplement HBM mean parameters
            9.48e-2, 8.69e+1, 2.98e-2, 4.69e+1,
            1.04e+2, 2.19e+1, 8.05e+0, 2.99e+1]) * 1e-3  # V, s -> mV, ms


    for i_cell, cell in enumerate(rankedcells):
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

        # Fitted parameters
        param_file = '%s/%s/%s-staircaseramp-%s-solution-%s.txt' % \
                (file_dir, file_name, file_name, cell, fit_seed)
        obtained_parameters = np.loadtxt(param_file) * 1e-3  # V, s -> mV, ms

        # For fix kinetics model
        rseal, cm, rseries = get_qc('../qc', file_name, cell)
        #print('Est. Rseal, Cm, Rseries:', rseal, cm, rseries, '(GOhm, pF, GOhm)')
        alpha = 0.8  # rseries %compensation
        ikr_param = [
                'ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4',
                'ikr.p5', 'ikr.p6', 'ikr.p7', 'ikr.p8',  
                ]
        p_ikr = np.array([  # paper 1 supplement HBM mean parameters
                9.48e-2, 8.69e+1, 2.98e-2, 4.69e+1,
                1.04e+2, 2.19e+1, 8.05e+0, 2.99e+1]) * 1e-3  # V, s -> mV, ms
        simvc_fix_values = [cm, rseries * alpha, rseries]
        fix_p = get_fix_param(ikr_param + simvc_fix + ['voltageclamp.rseries'],
                np.append(p_ikr, simvc_fix_values))
        modelfixkinetics.set_fix_parameters(fix_p)

        fix_p2 = get_fix_param(simvc_fix + ['voltageclamp.rseries'],
                simvc_fix_values)
        model2.set_fix_parameters(fix_p2)

        param_file = 'out/herg25oc1-simvclinleak/%s-staircaseramp-%s-solution-542811797.txt' % \
                (file_name, cell)
        obtained_parameters2 = np.loadtxt(param_file)

        scheme3_cell_idx = scheme3_cell_list.index(cell)
        obtained_parameters3 = obtained_parameters_all[scheme3_cell_idx]

        # Simulation
        simulation = model.simulate(obtained_parameters, times_sim)
        simulationfixkinetics = model2.simulate(obtained_parameters2, times_sim)
        simulationfixkinetics_s3 = modelfixkinetics.simulate(obtained_parameters3, times_sim)
        if prt != 'staircaseramp' and prt not in protocol_iv:
            # Re-leak correct the leak corrected simulationfixkinetics... TODO?
            g_releak_simulationfixkinetics = fmin(score_leak, [0.1], args=(simulationfixkinetics, voltage, times,
                                protocol_leak_check[prt]), disp=False)
            simulationfixkinetics = I_releak(g_releak_simulationfixkinetics[0], simulationfixkinetics, voltage)
            g_releak_simulationfixkinetics_s3 = fmin(score_leak, [0.1], args=(simulationfixkinetics_s3, voltage, times,
                                protocol_leak_check[prt]), disp=False)
            simulationfixkinetics_s3 = I_releak(g_releak_simulationfixkinetics_s3[0], simulationfixkinetics_s3, voltage)
        if prt in protocol_iv:
            simulation, t = protocol_iv_convert[prt](simulation, times_sim)
            assert(np.mean(np.abs(t - times)) < 1e-6)
            simulationfixkinetics, t = protocol_iv_convert[prt](
                    simulationfixkinetics, times_sim)
            assert(np.mean(np.abs(t - times)) < 1e-6)
            simulationfixkinetics_s3, t = protocol_iv_convert[prt](
                    simulationfixkinetics_s3, times_sim)
            assert(np.mean(np.abs(t - times)) < 1e-6)
            # Re-leak correct the leak corrected simulationfixkinetics... TODO?
            for i in range(simulationfixkinetics.shape[1]):
                g_releak_simulationfixkinetics = fmin(score_leak, [0.1], args=(simulationfixkinetics[:, i],
                                    voltage[:, i], times,
                                    protocol_leak_check[prt]), disp=False)
                simulationfixkinetics[:, i] = I_releak(g_releak_simulationfixkinetics[0], simulationfixkinetics[:, i], voltage[:, i])
            for i in range(simulationfixkinetics_s3.shape[1]):
                g_releak_simulationfixkinetics_s3 = fmin(score_leak, [0.1], args=(simulationfixkinetics_s3[:, i],
                                    voltage[:, i], times,
                                    protocol_leak_check[prt]), disp=False)
                simulationfixkinetics_s3[:, i] = I_releak(g_releak_simulationfixkinetics_s3[0], simulationfixkinetics_s3[:, i], voltage[:, i])

        # Work out ylim
        maximum = np.percentile(simulation, 100)
        minimum = np.percentile(simulation, 0.0)
        amplitude = maximum - minimum
        if prt in ['apabv3', 'ap05hz']:
            maximum += 0.6 * amplitude
            minimum -= 0.6 * amplitude
        elif prt in ['apab', 'ap1hz']:
            maximum += 0.3 * amplitude
            minimum -= 0.3 * amplitude
        else:
            maximum += 0.15 * amplitude
            minimum -= 0.15 * amplitude
        axes[ai + i_cell + 1, aj].set_ylim([minimum, maximum])
        
        # Plot
        if prt not in protocol_iv:
            # recording
            axes[ai + i_cell + 1, aj].plot(times, data, lw=1, alpha=0.8,
                    c='#1f77b4', label='data')
            # simulation
            if prt == 'staircaseramp':
                axes[ai + i_cell + 1, aj].plot(times, simulation, lw=2,
                        c='#d95f02', label='Hypothesis 1')
                axes[ai + i_cell + 1, aj].plot(times, simulationfixkinetics_s3, lw=2,
                        c='#1b9e77', label='Hypothesis 2')
                axes[ai + i_cell + 1, aj].plot(times, simulationfixkinetics, lw=2,
                        c='#9467bd', label='Hypothesis 3')
            else:
                axes[ai + i_cell + 1, aj].plot(times, simulation, lw=2,
                        c='#d95f02', label='H1 prediction')
                axes[ai + i_cell + 1, aj].plot(times, simulationfixkinetics_s3, lw=2,
                        c='#1b9e77', label='H2 prediction')
                axes[ai + i_cell + 1, aj].plot(times, simulationfixkinetics, lw=2,
                        c='#9467bd', label='H3 prediction')
        else:
            iv_v = protocol_iv_v[prt]() * 1000  # V -> mV
            # recording
            iv_i = protocols.get_corrected_iv(data, times,
                                              *protocol_iv_args[prt]())
            axes[ai + i_cell + 1, aj].plot(iv_v, iv_i / np.max(iv_i), lw=2,
                    alpha=1, c='#1f77b4', label='data')
            # simulation
            iv_i = protocols.get_corrected_iv(simulation, times,
                                              *protocol_iv_args[prt]())
            axes[ai + i_cell + 1, aj].plot(iv_v, iv_i / np.max(iv_i), lw=2,
                    alpha=1, c='#d95f02', label='model prediction')
            # simulationfixkinetics scheme3
            iv_i_fixkinetics_s3 = protocols.get_corrected_iv(simulationfixkinetics_s3, times,
                                              *protocol_iv_args[prt]())
            axes[ai + i_cell + 1, aj].plot(iv_v, iv_i_fixkinetics_s3 / np.max(iv_i_fixkinetics_s3), lw=2,
                    alpha=1, c='#1b9e77', label='model (fix kinetics) prediction new')
            # simulationfixkinetics
            iv_i_fixkinetics = protocols.get_corrected_iv(simulationfixkinetics, times,
                                              *protocol_iv_args[prt]())
            axes[ai + i_cell + 1, aj].plot(iv_v, iv_i_fixkinetics / np.max(iv_i_fixkinetics), lw=2,
                    alpha=1, c='#9467bd', label='model (fix kinetics) prediction HBM')
            if prt == 'sactiv':
                axes[ai + i_cell + 1, aj].set_ylim([-0.05, 1.05])
            elif prt == 'sinactiv':
                axes[ai + i_cell + 1, aj].set_ylim([-5, 1.05])
        
        if False:
            print(prt, i_cell, cell)
            print('red', rmsd_compute(simulation, data))
            print('green', rmsd_compute(simulationfixkinetics, data))
    
    # Plot rmsd histogram
    rmse_min = min(np.min(rmsd), np.min(rmsd2))                                 
    rmse_max = max(np.max(rmsd), np.max(rmsd2))                                 
    rmse_range = rmse_max - rmse_min                                            
    bins = np.linspace(rmse_min - 0.1 * rmse_range, rmse_max + 0.1 * rmse_range, 20)
    n, b, _ = axes[ai + 4, aj].hist(rmsd, bins=bins, color='#d95f02', alpha=0.25)
    n2, b2, _ = axes[ai + 4, aj].hist(rmsd2, bins=bins, color='#1b9e77', alpha=0.25)
    n3, b3, _ = axes[ai + 4, aj].hist(rmsd3, bins=bins, color='#8856a7', alpha=0.5)

    # Add labels
    rankedidx = []
    for i, v in enumerate(rankedvalues):
        idx = np.where(b <= v)[0][-1]
        if idx in rankedidx:
            print('Ref. marker might clash with other markers...')
        axes[ai + 4, aj].text((b[idx] + b[idx + 1]) / 2., n[idx] + 3,
                rankedlabels[i], fontsize=16, color='#d95f02',
                ha='center', va='center')
        if n[idx] == np.max(n):
            axes[ai + 4, aj].set_ylim([0, n[idx] + 6])
        rankedidx.append(idx)

    rankedidx2 = []
    for i, v in enumerate(rankedvalues2):
        idx = np.where(b2 <= v)[0][-1]
        if idx in rankedidx2:
            print('Ref. marker might clash with other markers...')
        axes[ai + 4, aj].text((b2[idx] + b2[idx + 1]) / 2., n2[idx] + 3,
                rankedlabels[i], fontsize=16, color='#1b9e77',
                ha='center', va='center')
        if n2[idx] == np.max(n2):
            axes[ai + 4, aj].set_ylim([0, n2[idx] + 6])
        rankedidx2.append(idx)

    rankedidx3 = []
    for i, v in enumerate(rankedvalues3):
        idx = np.where(b3 <= v)[0][-1]
        if idx in rankedidx3:
            print('Ref. marker might clash with other markers...')
        axes[ai + 4, aj].text((b3[idx] + b3[idx + 1]) / 2., n3[idx] + 3,
                rankedlabels[i], fontsize=16, color='#8856a7',
                ha='center', va='center')
        if n3[idx] == np.max(n3):
            axes[ai + 4, aj].set_ylim([0, n3[idx] + 6])
        rankedidx3.append(idx)


#
# Final adjustment and save
#
#axes[1, 0].legend()
#axes[1, 1].legend()
import matplotlib.patches as mpatches
data_patch = mpatches.Patch(color='#9ecae1', label='Data')
h1_patch = mpatches.Patch(color='#d95f02', label='Hypothesis 1: independent kinetics models')
h2_patch = mpatches.Patch(color='#1b9e77', label='Hypothesis 2: identical kinetics models')
h3_patch = mpatches.Patch(color='#8856a7', label='Hypothesis 3: cell-specific conductance, kinetics and artefacts')
axes[0, 0].legend(handles=[data_patch, h3_patch, h1_patch, h2_patch], loc='upper left', bbox_to_anchor=(-.025, 3.75), fontsize=14, ncol=2, columnspacing=5.5)
#grid.tight_layout(fig, pad=0.6, rect=(0.02, 0.0, 1, 0.99))
#grid.update(wspace=0.2, hspace=0.0)
plt.savefig('%s/rmsd-hist-independent-v-scheme3-v-simlinleak-part1.png' % (savedir), bbox_inch='tight',
        pad_inches=0, dpi=300)

print('Done')
