#!/usr/bin/env python3
"""
Shared functions for the artefact notebooks.
"""
import myokit
import numpy as np


def _fit_exponential(t, I, Iss, i1, i2, cutoff, invert, plot=False):
    """
    Fits a single exponential to ``I - Iss`` on the segment ``i1:i2``.

    The exponential is assumed to be decreasing. For an increasing exponential,
    set ``invert=True``.

    If the signal on ``i1:i2`` dips below ``cutoff``, the upper bound ``i2``
    will be reduced.

    Returns ``tau, I0``.
    """

    # Find points for exponential
    ilog = I[i1:i2] - Iss
    if invert:
        ilog = -ilog

    # If any zeroes are found, this is almost certainly due to noise.
    if np.any(ilog < 0):
        # Find where the signal dips below a set signal-to-noise ratio
        i = np.where(ilog < cutoff)[0][0]
        # Cut off values after that point, and update i2 accordingly
        ilog = ilog[:i]
        i2 = i1 + i
    tlog = t[i1:i2]
    ilog = np.log(ilog)

    # Calculate means, residuals, coefficients
    mx = np.mean(tlog)
    my = np.mean(ilog)
    rx = tlog - mx
    ry = ilog - my
    b = np.sum(rx * ry) / np.sum(rx ** 2)
    a = my - b * mx

    # Get tau and I0 estimates
    tau = -1 / b
    if invert:
        I0 = -np.exp(a) + Iss
    else:
        I0 = np.exp(a) + Iss

    if plot:
        print(f'Tau* = {tau:.3f} ms')
        print(f'I0* = {I0:.3f} pA')

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(tlog, ilog, lw=3, label='Data')
        ax.plot(tlog, a + b * tlog, '--', label='Least squares fit')
        ax.legend()
        plt.show()

    return tau, I0


def _integrate_current(t, I, Iss, i0, i3, cutoff, dt, invert):
    """
    Integrates the I[i0:i3] and returns the result.

    If the initial points are below cutoff, ``i0`` will be increased.
    """
    # Get segment containing transient
    iup = I[i0:i3] - Iss
    if invert:
        iup = -iup

    # Increase i0 if necessary
    i = np.where(iup > cutoff)[0][0]

    iup = iup[i:]
    i0 += i

    # Integrate
    if dt is None:
        return np.trapezoid(iup, t[i0:i3])
    return np.trapezoid(iup, dx=dt)


def estimate_cell_parameters(
        t, I, T, dV, dt=None, f1=0.1, f2=0.8, f3=0.8, f4=1):
    """

    Arguments:

    ``t``
        A time vector, starting at 0 and going up to and/or including time 2T.
    ``I``
        The corresponding current vector.
    ``T``
        The duration of both steps (each has duration T).
    ``dV``
        The difference ``V1 - V2``, where ``V1`` is the command voltage during
        the first, and ``V2`` during the second step.
    ``dt``
        Sampling interval, or ``None`` to assume irregular sampling.
    ``f1=0.1``
        The start of the segment where an exponential is fit, as a fraction of
        ``T``.
    ``f2=0.8``
        The end of the segment where an exponential is fit, as a fraction of
        ``T``. If the given signal is noisy, a shorter interval may be used.
    ``f3=0.8``
        The start of the segment used to estimate the steady-state current, as
        a fraction of ``T``.
    ``f4=1.0``
        The end of the segment used to estimate the steady-state current, as
        a fraction of ``T``.

    Returns:

    ``Rs``
        The estimated series (or access) resistance.
    ``Rm``
        The estimated membrane resistance.
    ``Cm``
        The estimated membrane capacitance.
    ``points``
        A tuple ``(tau, I01, a1, b1, Iss1, c1, d1, I02, a2, b2, Iss2, c2, d2)``
        where ``tau`` is the estimated time constant, ``Iss`` are current
        steady-state values, ``I0`` are initial values for the fitted
        transients, and where the remaining numbers give array indices suitable
        for drawing fitted transients and steady states.

    """
    # Get indices
    f = np.array((f1, f2, f3, f4, 1, 1 + f1, 1 + f2, 1 + f3, 1 + f4)) * T
    if dt is None:
        i = np.searchsorted(t, f)
    else:
        i = np.rint(f / dt).astype(int)
    i1, i2, i3, i4, iT, i5, i6, i7, i8 = i

    # Estimate I1 and I2
    I1 = np.mean(I[i3:i4])
    I2 = np.mean(I[i7:i8])
    dI = I1 - I2

    # Estimate the noise
    cutoff = np.std(I[i3:i4]) + np.std(I[i7:i8])

    # Estimate tau and I0
    tau1, I01 = _fit_exponential(t, I, I1, i1, i2, cutoff, False)
    tau2, I02 = _fit_exponential(t - T, I, I2, i5, i6, cutoff, True)
    tau = 0.5 * (tau1 + tau2)

    # Estimate charge
    Q11 = _integrate_current(t, I, I1, 0, i3, cutoff, dt, False)
    Q12 = _integrate_current(t, I, I2, iT, i7, cutoff, dt, True)
    Qm = 0.5 * (Q11 + Q12) + tau * dI

    # Estimate rest
    Rs = tau * dV / Qm
    Rm = dV / dI - Rs
    Cm = Qm * (Rm + Rs) / (Rm * dV)

    # Gather points for drawing
    points = (tau, I01, i1, i2, I1, i3, i4, I02, i5, i6, I2, i7, i8)

    return Rs, Rm, Cm, points


def _test_one_shot():
    """ Generates data and shows the results of a one-shot test. """

    import matplotlib.pyplot as plt

    m = myokit.parse_model('''
        [[model]]
        amp.Vm = -70

        [engine]
        time = 0 [ms] in [ms] bind time
        pace = 0 bind pace

        [amp]
        Rs = 11.7e-3 [GOhm] in [GOhm]
        Cm = 31.89 [pF] in [pF]
        Rm = 0.5003 [GOhm] in [GOhm]
        Vc = 1 [mV] * engine.pace
            in [mV]
        dot(Vm) = (Rm * I_obs - Vm) / (Rm * Cm)
            in [mV]
        I_obs = (Vc - Vm) / Rs
            in [pA]
        ''')
    m.check_units(myokit.UNIT_STRICT)

    T = 10
    V1 = -60
    V2 = -70
    dV = V1 - V2

    p = myokit.Protocol()
    p.schedule(start=0, level=V1, duration=T, period=2*T)
    p.schedule(start=T, level=V2, duration=T, period=2*T)

    if True:
        N = 2000
        dt = (2 * T) / N
        print(f'Using dt={dt} for a total of {N} samples')
    else:
        dt=None
        print('Using adaptive time steps')

    s = myokit.Simulation(m, p)
    s.set_tolerance(1e-12, 1e-12)
    s.pre(2 * T)
    s.reset()
    d = s.run(2 * T, log_interval=dt).npview()
    t, I = d.time(), d['amp.I_obs']

    #I += np.random.normal(0, 5, size=t.shape)

    Rs, Rm, Cm, points = estimate_cell_parameters(t, I, T, dV, dt)
    print(f'Estimated Rs {1e3 * Rs:>5.1f} MOhm')
    print(f'Estimated Rm {1e3 * Rm:>5.1f} MOhm')
    print(f'Estimated Cm {Cm:>5.2f} pF')

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot()
    ax.plot(t, I, label='$Iobs$')
    kw = dict(color='tab:orange', lw=2)
    tau, I01, a1, b1, I1, c1, d1, I02, a2, b2, I2, c2, d2 = points
    ax.plot((t[c1], t[d1 - 1]), (I1, I1), **kw)
    ax.plot((t[c2], t[d2 - 1]), (I2, I2), **kw)
    te = t[a1:(b1 + a1) // 3]
    ax.plot(te, I1 - (I1 - I01) * np.exp(-te / tau), **kw)
    ax.plot(T + te, I2 - (I2 - I02) * np.exp(-te / tau), **kw)
    ax.legend(loc='lower right')
    plt.show()


def bode(magnitude, argument, axes=None, lo=1e-2, hi=1e5, **kwargs):
    """
    Creates a bode plot for the given ``magnitude`` and ``argument`` functions.

    Returns a tuple ``(ax0, ax1)``.
    """

    lo, hi = np.log10(lo), np.log10(hi)
    w = np.logspace(lo, hi, 1001, base=np.e)

    if axes is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(9, 6))
        fig.subplots_adjust(hspace=0.2)
        ax0 = fig.add_subplot(2, 1, 1)
        ax0.set_xscale('log')
        ax0.set_yscale('log')
        ax0.set_ylabel('Gain')
        ax0.grid()

        ax1 = fig.add_subplot(2, 1, 2)
        ax1.set_xscale('log')
        ax1.set_xlabel('Angular frequency')
        ax1.set_ylabel('Phase shift (degrees)')
        ax1.set_ylim(-195, 195)
        ax1.set_yticks([-180, -90, 0, 90, 180])

        ax1.grid()
    else:
        ax0, ax1 = axes

    label=None
    if kwargs:
        label=','.join(f'{k}={v}' for k, v in kwargs.items())

    ax0.plot(w, magnitude(w, **kwargs), label=None)
    ax1.plot(w, argument(w, **kwargs) * 180 / np.pi, label=label)
    if label is not None:
        ax1.legend()

    return ax0, ax1


def simplification_plot1(model, euler_dt=None):
    """
    Runs a short simulation and shows the fastest changing states.
    """
    import matplotlib.pyplot as plt

    sim = myokit.Simulation(model, myokit.pacing.constant(-140))
    sim.pre(2)
    protocol = myokit.Protocol()
    protocol.add_step(level=-140, duration=0.1)
    protocol.add_step(level=+60, duration=2)
    sim.set_protocol(protocol)

    dt = 1e-6
    d = sim.run(0.1 - dt)
    d = sim.run(1.9 + dt, log=d).npview()
    t = d.time()

    fig = plt.figure(figsize=(12, 7))
    fig.subplots_adjust(0.06, 0.1, 0.99, 0.975, wspace=0.28, hspace=0.3)
    grid = fig.add_gridspec(3, 3)

    sg0 = grid[0:2, 0].subgridspec(3, 1, hspace=0.07)
    ax0a = fig.add_subplot(sg0[0])
    ax0b = fig.add_subplot(sg0[1:])
    for ax in (ax0a, ax0b):
        ax.set_ylabel('V (mV)')
        ax.plot(t, d['amp.Vc'], label='Vc')
        if 'amp.Vs' in d:
            ax.plot(t, d['amp.Vs'], label='Vs')
        ax.plot(t, d['amp.Vm'], label='Vm')
    ax0a.set_ylim(55, 65)
    ax0a.set_xticklabels([])
    ax0b.set_xlabel('Time (ms)')
    ax0b.legend()

    sg1 = grid[0:2, 1].subgridspec(3, 1, hspace=0.05)
    ax1a = fig.add_subplot(sg1[0])
    ax1b = fig.add_subplot(sg1[1:])
    for ax in (ax1a, ax1b):
        ax.set_ylabel('I (pA)')
        ax.plot(t, d['amp.I'], label='I')
        ax.plot(t, d['amp.I_obs'], label='I_obs')
    ax1a.set_ylim(-1700, 1000)
    ax1a.set_xticklabels([])
    ax1b.set_xlabel('Time (ms)')
    ax1b.legend()

    dxdts = np.array([np.abs(d[x.lhs()]) for x in model.states()])
    names = np.array([x.qname() for x in model.states()])
    dmaxs = np.max(dxdts, axis=1)
    i = list(np.argsort(dmaxs)[::-1])
    names, dxdts, dmaxs = names[i], dxdts[i], dmaxs[i]
    ticks = np.arange(len(dxdts))

    ax2 = fig.add_subplot(grid[0:2, 2])
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('|dX/dt| (?/ms)')
    for dxdt in dxdts:
        ax2.plot(t, dxdt)
    ax2.set_xlim(0.09, 0.5)

    ax3 = fig.add_subplot(grid[-1, :])
    ax3.set_ylabel('|dx/dt|')
    ax3.set_yscale('log')
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(names, rotation=45 if len(names) > 6 else 0, ha='right')
    for x, dmax in zip(ticks, dmaxs):
        ax3.bar(x, dmax)

    # Forward Euler
    if euler_dt is not None:
        x = np.array(sim.default_state())
        ts = np.arange(0, 2, euler_dt)
        ps = protocol.value_at_times(ts)
        xs = np.zeros((len(ts), len(x)))
        for i, (ti, pi) in enumerate(zip(ts, ps)):
            xs[i] = x
            dx = np.array(sim.evaluate_derivatives(x, {'pace': pi}))
            x += euler_dt * dx
        ylim = ax0b.get_ylim()
        ax0b.plot(ts, xs[:, 0], 'k--', label='Forward Euler Vm')
        ax0b.set_ylim(*ylim)

    return fig, (ax0a, ax0b, ax1a, ax1b, ax2, ax3)


def simplification_plot2(m0, m1):
    """
    Runs a short simulation and compares a simplified model (``m1``) with an
    original model (``m0``).
    """
    import matplotlib.pyplot as plt

    protocol = myokit.pacing.constant(-140)
    s0 = myokit.Simulation(m0, protocol)
    s1 = myokit.Simulation(m1, protocol)
    s0.pre(2)
    s1.pre(2)
    protocol = myokit.Protocol()
    protocol.add_step(level=-140, duration=0.1)
    protocol.add_step(level=+60, duration=2)
    s0.set_protocol(protocol)
    s1.set_protocol(protocol)

    dt = 1e-6
    d0 = s0.run(0.1 - dt)
    d0 = s0.run(1.9 + dt, log=d0).npview()
    t0 = d0.time()
    d1 = s1.run(0.1 - dt)
    d1 = s1.run(1.9 + dt, log=d1).npview()
    t1 = d1.time()

    fig = plt.figure(figsize=(12, 4))
    fig.subplots_adjust(0.06, 0.1, 0.99, 0.975, wspace=0.32, hspace=0.15)
    grid = fig.add_gridspec(4, 3)
    ax0a = fig.add_subplot(grid[0, 0])
    ax0b = fig.add_subplot(grid[1:, 0])
    vaxes = (ax0a, ax0b)
    if 'amp.Vrc' in d0 and 'amp.Vrc' in d1:
        ax1a = fig.add_subplot(grid[:2, 1])
        ax1b = fig.add_subplot(grid[2:, 1])
        ax2a = fig.add_subplot(grid[:2, 2])
        ax2b = fig.add_subplot(grid[2:, 2])
        iaxes = (ax1a, ax1b, ax2a, ax2b)
    else:
        ax1a = fig.add_subplot(grid[:, 1])
        ax2a = fig.add_subplot(grid[:, 2])
        iaxes = (ax1a, ax2a)

    # Voltages
    for ax in vaxes:
        ax.set_xlim(0, 0.8)
        ax.set_ylabel('V (mV)')
        ax.plot(t0, d0['amp.Vc'], label='Vc')
        if 'amp.Vs' in d0:
            ax.plot(t0, d0['amp.Vs'], label='Vs')
        if 'amp.Vs' in d1:
            ax.plot(t1, d1['amp.Vs'], 'k--')
        ax.plot(t0, d0['amp.Vm'], label='Vm')
        ax.plot(t1, d1['amp.Vm'], 'k--', label='Reduced')
    ax0a.set_xticklabels([])
    ax0a.set_ylim(57, 63)
    ax0b.set_xlabel('Time (ms)')
    ax0b.legend()

    # Currents
    for ax in iaxes:
        ax.set_ylabel('I (pA)')
    ax1a.set_xlim(0, 1)
    ax2a.set_ylim(540, 640)
    for ax in (ax1a, ax2a):
        ax.plot(t0, d0['amp.I'], label='I')
        ax.plot(t1, d1['amp.I'], 'k--')
        ax.plot(t0, d0['amp.I_obs'], label='I_obs')
        ax.plot(t1, d1['amp.I_obs'], 'k--')
        ax.legend()
    if 'amp.Vrc' in d0 and 'amp.Vrc' in d1:
        ax1b.set_xlim(0, 1)
        ax2b.set_ylim(540, 640)
        for ax in (ax1a, ax2a):
            ax.set_xticklabels([])
        for ax in (ax1b, ax2b):
            ax.set_xlabel('Time (ms)')
            ax.plot(t0, d0['amp.Vrc'] / m0.get('amp.Rf').eval(),
                    label='Vrc / Rf')
            ax.plot(t1, d1['amp.Vrc'] / m1.get('amp.Rf').eval(), 'k--')
            ax.legend()

    return fig, vaxes + iaxes


if __name__ == '__main__':
    _test_one_shot()
