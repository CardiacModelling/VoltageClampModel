#!/usr/bin/env python3
import numpy as np

#
# model cell v1
#
mc_name = [
    r'$R_\text{m}$ (\si{\mega\ohm})',
    r'$C_\text{prs}$ (\si{\pico\farad})',
    r'$C_\text{m}$ (\si{\pico\farad})',
    r'$R_\text{s}$ (\si{\mega\ohm})',
    r'$V_\text{off}$ (\si{\milli\volt})',
]
mc_true = [
    500,  # GOhm; R_membrane
    4.7,  # pF; Cprs
    22.0,  # pF; Cm
    30.0,  # GOhm; Rs
    0.0,  # mV; Voffset+
]
mc_machine = [
    498,  # GOhm; R_membrane
    7.8,  # pF; Cprs
    32.85,  # pF; Cm
    32.6,  # GOhm; Rs
    0.2,  # mV; Voffset+
]
mc_fit = np.loadtxt('out/mcnocomp-solution-542811797-1.txt')
mc_fit[0] = 1e3 / mc_fit[0]  # g -> R; G -> M
mc_fit[3] = mc_fit[3] * 1e3  # G -> M

mc_tex = ""

mc_tex += '\\toprule\n'
for i in mc_name:
    mc_tex += ' & ' + i
mc_tex += ' \\\\\n'
mc_tex += '\\midrule\n'
mc_tex += 'True'
for i in mc_true:
    mc_tex += ' & ' + '%.2f' % i
mc_tex += ' \\\\\n'
mc_tex += 'Machine'
for i in mc_machine:
    mc_tex += ' & ' + '%.2f' % i
mc_tex += ' \\\\\n'
mc_tex += 'Fitted'
for i in mc_fit:
    mc_tex += ' & ' + '%.2f' % i
mc_tex += ' \\\\\n'
mc_tex += '\\bottomrule\n'
print(mc_tex)


#
# model cell v3
#
mc3_name = [
    r'$R_\text{k}$ (\si{\mega\ohm})',
    r'$C_\text{k}$ (\si{\pico\farad})',
    r'$R_\text{m}$ (\si{\mega\ohm})',
    r'$C_\text{prs}$ (\si{\pico\farad})',
    r'$C_\text{m}$ (\si{\pico\farad})',
    r'$R_\text{s}$ (\si{\mega\ohm})',
    r'$V_\text{off}$ (\si{\milli\volt})',
]
mc3_true = [
    100,  # GOhm; R_kinetics
    1000.,  # GOhm; C_kinetics
    500,  # GOhm; R_membrane
    4.7,  # pF; Cprs
    22.0,  # pF; Cm
    30.0,  # GOhm; Rs
    0.0,  # mV; Voffset+
]
mc3_machine = [
    np.NaN,  # GOhm; R_kinetics
    np.NaN,  # GOhm; C_kinetics
    91.3,  # GOhm; R_membrane
    8.8,  # pF; Cprs
    41.19,  # pF; Cm
    33.6,  # GOhm; Rs
    -1.2,  # mV; Voffset+
]
mc3_fit = np.loadtxt('out/mc3nocomp-solution-542811797-1.txt')
mc3_fit[0] = 1e3 / mc3_fit[0]  # g -> R; G -> M
mc3_fit[2] = 1e3 / mc3_fit[2]  # g -> R; G -> M
mc3_fit[5] = mc3_fit[5] * 1e3  # G -> M

mc3_tex = ""

mc3_tex += '\\toprule\n'
for i in mc3_name:
    mc3_tex += ' & ' + i
mc3_tex += ' \\\\\n'
mc3_tex += '\\midrule\n'
mc3_tex += 'True'
for i in mc3_true:
    mc3_tex += ' & ' + '%.2f' % i
mc3_tex += ' \\\\\n'
mc3_tex += 'Machine'
for i in mc3_machine:
    mc3_tex += ' & ' + '%.2f' % i
mc3_tex += ' \\\\\n'
mc3_tex += 'Fitted'
for i in mc3_fit:
    mc3_tex += ' & ' + '%.2f' % i
mc3_tex += ' \\\\\n'
mc3_tex += '\\bottomrule\n'
print(mc3_tex)
