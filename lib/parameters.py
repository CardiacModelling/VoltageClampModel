#!/usr/bin/env python3
# List of parameters to be inferred
import numpy as np
from priors import prior_parameters

# Just IKr model
ikr = [
    'ikr.g',
    'ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4',
    'ikr.p5', 'ikr.p6', 'ikr.p7', 'ikr.p8',  
    ]

p = np.asarray(prior_parameters['25.0'])
p[1:] = p[1:] * 1e-3  # in mV, ms

ikr_names = [r'$g_{Kr}$',
             r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
             r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$',]

# Ideal voltage clamp model
idealvc = list(ikr)

idealvc_typical_values = np.copy(p)

idealvc_names = list(ikr_names)

# Realistic voltage clamp model
fullvc = ikr + [
    'membrane.cm',
    'voltageclamp.rseries',
    'voltageclamp.cprs_eff',
    'voltageclamp.voffset_eff'
    ]

fullvc_typical_values = np.append(idealvc_typical_values, [
    15.0,  # pF
    8.0e-3,  # GOhm
    4.0 * 0.5,  # pF
    -5.0,  # mV
    ])

fullvc_best_values = np.append(idealvc_typical_values, [
    10.0,  # pF
    5.0e-3,  # GOhm
    4.0 * 0.1,  # pF
    5.0,  # mV
    ])

fullvc_names = idealvc_names + [
        r'$C_m$',
        r'$R_{series}$',
        r'$C^{\dagger}_{prs}$',
        r'$V^{\dagger}_{off}$'
        ]

fullvc_fix = ['voltageclamp.cm_est', 'voltageclamp.rseries_est']

fullvc_fix_typical_values = [
    0.9 * 10.0,  # pF
    0.9 * 5.0e-3,  # GOhm
#12.0,  # pF
#14.0e-3,  # GOhm
    ]


# Realistic voltage clamp model (full)
full2vc = ikr + [
    'membrane.cm',
    'voltageclamp.rseries',
    'voltageclamp.cprs',
    'voltageclamp.voffset_eff'
    ]

full2vc_typical_values = np.append(idealvc_typical_values, [
    15.0,  # pF
    8.0e-3,  # GOhm
    4.0,  # pF
    -5.0,  # mV
    ])

full2vc_best_values = np.append(idealvc_typical_values, [
    10.0,  # pF
    5.0e-3,  # GOhm
    4.0,  # pF
    5.0,  # mV
    ])

full2vc_names = idealvc_names + [
        r'$C_m$',
        r'$R_{series}$',
        r'$C_{prs}$',
        r'$V^{\dagger}_{off}$'
        ]

full2vc_fix = ['voltageclamp.cm_est',
        'voltageclamp.rseries_est',
        'voltageclamp.cprs_est',]

full2vc_fix_typical_values = [
    0.9 * 10.0,  # pF
    0.9 * 5.0e-3,  # GOhm
    0.9 * 4,  # pF
    ]


# Simplified voltage clamp model
simvc = ikr + [
    'voltageclamp.rseries',
    'voltageclamp.voffset_eff'
    ]

simvc_typical_values = np.append(idealvc_typical_values, [
    8.0e-3,  # GOhm
    -5.0,  # mV
    ])

simvc_best_values = np.append(idealvc_typical_values, [
    5.0e-3,  # GOhm
    5.0,  # mV
    ])

simvc_names = idealvc_names + [
        r'$R_{series}$',
        r'$V^{\dagger}_{off}$'
        ]

simvc_fix = ['voltageclamp.cm_est', 'voltageclamp.rseries_est']

simvc_fix_typical_values = [
    10.0,  # pF
    0.9 * 5.0e-3,  # GOhm
    ]


def get_qc(path, filename, cell):
    # path: path to files
    # filename: e.g. herg25oc1
    # cell: e.g. A01
    # 
    # return: rseal, cm, rseries (in GOhm, pF, GOhm)
    path2cellid = path + '/' + filename + '-staircaseramp-cell_id.txt'
    path2cellrseal = path + '/' + filename + '-staircaseramp-Rseal_before.txt'
    path2cellcm = path + '/' + filename + '-staircaseramp-Cm_before.txt'
    path2cellrseries = path + '/' + filename + \
            '-staircaseramp-Rseries_before.txt'
    with open(path2cellid, 'r') as f:
        cellid = f.read().splitlines()
    idx = cellid.index(cell)
    rseal = np.loadtxt(path2cellrseal)[idx] * 1e-9  # GOhm
    cm = np.loadtxt(path2cellcm)[idx] * 1e12  # pF
    rseries = np.loadtxt(path2cellrseries)[idx] * 1e-9  # GOhm
    return rseal, cm, rseries


''' # Leak model 3
leak = ['inlleak.A1',
        'inlleak.B1',
        'inlleak.gd',
        'inlleak.Ed',
        'inlleak.gt',
        'inlleak.tauta',
        'inlleak.tautb',
        'inlleak.Et']

leak_typical_values = [
    2e-2,  # for no cell recording
    2e-7,
    2e-3,  # TODO
    2e-8,  # TODO
    0.5e-1,  # pA/mV/mV
    200.,  # ms
    2.,  # ms/mV
    5.0,  # mV
    ]

''' # Leak model 4
leak = ['inlleak.A1a',
        'inlleak.A1b',
        'inlleak.B1a',
        'inlleak.B1b',
        'inlleak.gt',
        'inlleak.tauta',
        'inlleak.tautb',
        'inlleak.Et']

leak_typical_values = [
    2e-2,  # for no cell recording
    2e-7,
    2e-3,
    2e-8,
    0.5e-1,  # pA/mV/mV
    200.,  # ms
    2.,  # ms/mV
    5.0,  # mV
    ]
#'''

# Realistic voltage clamp model with leak model
fullvc_wLeak = fullvc + leak

