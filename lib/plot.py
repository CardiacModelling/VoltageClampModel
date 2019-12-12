#
# Quick diagnostic plots.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import warnings
import numpy as np
import pints

def function(f, x, lower=None, upper=None, evaluations=20):
    """
    Creates 1d plots of a :class:`LogPDF` or a :class:`ErrorMeasure` around a
    point `x` (i.e. a 1-dimensional plot in each direction).

    Arguments:

    ``f``
        A :class:`pints.LogPDF` or :class:`pints.ErrorMeasure` to plot.
    ``x``
        A point in the function's input space.
    ``lower``
        (Optional) Lower bounds for each parameter, used to specify the lower
        bounds of the plot.
    ``upper``
        (Optional) Upper bounds for each parameter, used to specify the upper
        bounds of the plot.
    ``evaluations``
        (Optional) The number of evaluations to use in each plot.

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt

    # Check function get dimension
    if not (isinstance(f, pints.LogPDF) or isinstance(f, pints.ErrorMeasure)):
        raise ValueError(
            'Given function must be pints.LogPDF or pints.ErrorMeasure.')
    dimension = f.n_parameters()

    # Check point
    x = pints.vector(x)
    if len(x) != dimension:
        raise ValueError(
            'Given point `x` must have same dimension as function.')

    # Check boundaries
    if lower is None:
        # Guess boundaries based on point x
        lower = x * 0.95
        lower[lower == 0] = -1
    else:
        lower = pints.vector(lower)
        if len(lower) != dimension:
            raise ValueError(
                'Lower bounds must have same dimension as function.')
    if upper is None:
        # Guess boundaries based on point x
        upper = x * 1.05
        upper[upper == 0] = 1
    else:
        upper = pints.vector(upper)
        if len(upper) != dimension:
            raise ValueError(
                'Upper bounds must have same dimension as function.')

    # Check number of evaluations
    evaluations = int(evaluations)
    if evaluations < 1:
        raise ValueError('Number of evaluations must be greater than zero.')

    # Create points to plot
    xs = np.tile(x, (dimension * evaluations, 1))
    for j in range(dimension):
        i1 = j * evaluations
        i2 = i1 + evaluations
        xs[i1:i2, j] = np.linspace(lower[j], upper[j], evaluations)

    import os
    if not os.path.isdir('likelihood1d'):
        os.makedirs('likelihood1d')

    # Evaluate points
    fs = pints.evaluate(f, xs, parallel=False)

    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(9, 4))
    for j, p in enumerate(x):
        i1 = j * evaluations
        i2 = i1 + evaluations
        a1 = j % 4
        a2 = j // 4
        axes[a1, a2].plot(xs[i1:i2, j], fs[i1:i2], c='C2', label='Function')
        np.savetxt('likelihood1d/p%s.txt' % (j + 1),
                np.asarray([xs[i1:i2, j], fs[i1:i2]]).T)
        axes[a1, a2].axvline(p, c='C0', label='Value')
        axes[a1, a2].set_xlabel(r'$p_{%s}$' % str(1 + j))
        if j == 0:
            axes[a1, a2].legend()

    plt.tight_layout()
    return fig, axes

