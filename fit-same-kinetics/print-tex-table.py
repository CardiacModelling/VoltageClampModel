#!/usr/bin/env python3
import numpy as np

param_name = [
    r'$p_1$ $(s^{-1})$',
    r'$p_2$ $(V^{-1})$',
    r'$p_3$ $(s^{-1})$',
    r'$p_4$ $(V^{-1})$',
    r'$p_5$ $(s^{-1})$',
    r'$p_6$ $(V^{-1})$',
    r'$p_7$ $(s^{-1})$',
    r'$p_8$ $(V^{-1})$',
]
HBM_mean = np.array([
    7.65e-2,
    9.05e+1,
    2.84e-2,
    4.74e+1,
    1.03e+2,
    2.13e+1,
    8.01e+0,
    2.96e+1
]) * 1e-3  # V, s -> mV, ms
fit = np.loadtxt('out/herg25oc1-scheme3-simvclinleak/' +
        'herg25oc1-solution-209652396.txt')  # mV, ms

tex = ""

tex += '\\toprule\n'
for i in param_name:
    tex += ' & ' + i
tex += ' \\\\\n'
tex += '\\midrule\n'
tex += 'HBM mean (Lei et al.\\ \\cite{lei2019a})'
for i in HBM_mean:
    tex += ' & ' \
            + np.format_float_scientific(i, precision=2, exp_digits=1)
tex += ' \\\\\n'
tex += 'Hypothesis 2'
for i in fit:
    tex += ' & ' \
            + np.format_float_scientific(i, precision=2, exp_digits=1)
tex += ' \\\\\n'
tex += '\\bottomrule\n'
print(tex)

