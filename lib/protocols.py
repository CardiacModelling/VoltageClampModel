#!/usr/bin/env python
import numpy as np

#
# Protocol info
#
capmaskdt = 2e-3  # 3 ms (minimum 2 ms)
vhold = -80e-3 # + 0.13549805e-3

# TODO only protocols marked with 'in mV, ms' are converted/ready to use

def sine_wave(model, return_capmask=False):
    # sine wave protocol on JPhysio paper
    # in mV, ms
    steps = [
        (-80, 250),
        (-120, 50),
        (-80, 200),
        (40, 1000),
        (-120, 500),
        (-80, 1000),
        (-30, 3500),
        (-120, 500),
        (-80, 1000),
    ]
    # Set sine-wave
    model.get('voltageclamp.Vc').set_rhs(
                'if(engine.time >= 3.0001e3 and engine.time < 6.5001e3,'
                + ' - 30'
                + ' + 54 * sin(0.007 * (engine.time - 2.5001e3))'
                + ' + 26 * sin(0.037 * (engine.time - 2.5001e3))'
                + ' + 10 * sin(0.190 * (engine.time - 2.5001e3))'
                + ', engine.pace)')
    return model, steps


def sine_wave_1(model, return_capmask=False):
    # Known as original sine wave protocol
    # prototype of the sine_wave() protocol
    steps = [
        (-80e-3 + 0.13549805e-3, 100),  # just for however long it is...
    ]
    # Set original sine wave
    # the diff in 0.0001 for the start time match what is shown in the
    # protocol files
    model.get('voltageclamp.Vc').set_rhs(
                'if(engine.time >= 0.1001 and engine.time < 5.0000,'
                + ' + 0.13549805e-3'
                + ' + 57e-3 * sin(0.194778745e3 * (engine.time - 0.1001))'
                + ' + 28e-3 * sin(0.502654825e3 * (engine.time - 0.1001))'
                + ' + 18e-3 * sin(0.703716754e3 * (engine.time - 0.1001))'
                + ', engine.pace)')
    return model, steps


def sine_wave_2(model, return_capmask=False):
    # Known as Maz Wang Diff (MWD) protocol
    # prototype of the sine_wave() protocol
    steps = [
        (-80e-3 + 0.13549805e-3, 100),  # just for however long it is...
    ]
    # Set original sine wave
    # the diff in 0.0813 for the start time match what is shown in the
    # protocol files
    model.get('voltageclamp.Vc').set_rhs(
                'if(engine.time >= 0.1813 and engine.time < 5.0813,'
                + ' + 0.13549805e-3'
                + ' - 25.3e-3 * sin(0.000439823e3 * (engine.time - 0.0814))'
                + ' + 99.7e-3 * sin(0.018849556e3 * (engine.time - 0.0814))'
                + ' - 4.2e-3 * sin(1.005309649e3 * (engine.time - 0.0814))'
                + ', engine.pace)')
    return model, steps

def sine_wave_3(model, return_capmask=False):
    # Known as equal proportion protocol
    # prototype of the sine_wave() protocol
    steps = [
        (-80e-3 + 0.13549805e-3, 100),  # just for however long it is...
    ]
    # Set original sine wave
    # the diff in 0.0001 for the start time match what is shown in the
    # protocol files
    model.get('voltageclamp.Vc').set_rhs(
                'if(engine.time >= 0.1001 and engine.time < 5.0001,'
                + ' + 0.13549805e-3'
                + ' - 51e-3 * sin(0.000502655e3 * (engine.time - 0.0001))'
                + ' + 71e-3 * sin(0.025132741e3 * (engine.time - 0.0001))'
                + ' + 17e-3 * sin(0.031415927e3 * (engine.time - 0.0001))'
                + ', engine.pace)')
    return model, steps


def leak_staircase(model, return_capmask=False):
    # in ms, mV
    # My 'test6_v3'/staircase-ramp protocol
    # model: myokit model
    # return_capmask: if True, return an extra function that takes time series
    #                 as argument and return a mask to filter off capacitance
    #                 effect.

    tpre  = 0.2e3           # Time before step to variable V
    tstep = 0.5e3           # Time at variable V
    tpost = 0.1e3           # Time after step to variable V
    vhold = -80
    vmin = -60
    vmax = 40
    vres = 20        # Difference in V between steps
    v = np.arange(vmin, vmax + vres, vres)

    steps = []
    # Leak estimate
    steps += [(vhold, 0.25e3)]
    steps += [(-120, 0.05e3)]
    steps += [(-30, 400)]  # ramp step
    # Staircase
    steps += [(vhold, 0.2e3)]
    steps += [(40, 1.0e3)]
    steps += [(-120, 0.5e3)]
    steps += [(vhold, 1.0e3)]
    for vstep in v[1::]:
        steps += [(vstep, tstep)]
        steps += [(vstep-vres, tstep)]
    for vstep in v[::-1][:-1]:
        steps += [(vstep, tstep)]
        steps += [(vstep-2*vres, tstep)]
    steps += [(vhold, 1.0e3 - tstep)]  # extend a bit the ending...
    # EK estimate
    steps += [(40, tstep)]
    steps += [(-70, 10)]  # Michael's suggestion
    steps += [(-120, tstep - 10)]  # second ramp step
    steps += [(vhold, 100e3)]
    # Set ramp bit
    model.get('voltageclamp.Vc').set_rhs(
                'piecewise('
                +
                'engine.time >= 0.300e3 and engine.time < 0.700001e3,'
                + '-150 + 0.1 * engine.time'
                +
                ', engine.time >= 14.410e3 and engine.time < 14.510001e3,'
                + ' + 5.694e3 - 0.4 * engine.time'
                +
                ', engine.pace)')

    if return_capmask:

        def capmask(times, capmaskdt=capmaskdt):
            fcap = np.ones(times.shape)
            currentt = 0
            for v, dur in steps:
                idxi = np.where(times > currentt)[0][0] - 1  # inclusive
                idxf = np.where(times > currentt + capmaskdt)[0][0]
                fcap[idxi:idxf] = 0
                currentt += dur
            return fcap

        return model, steps, capmask
    else:
        return model, steps


def pharma(model, return_capmask=False):
    # in ms, mV
    # Roche hERG screening protocol
    # model: myokit model
    # return_capmask: if True, return an extra function that takes time series
    #                 as argument and return a mask to filter off capacitance
    #                 effect.
    tpre  = 100           # Time before step to variable V
    tpre2  = 50           # Time before step to variable V
    tstep = 500           # Time at variable V
    tstep2 = 500           # Time at variable V
    tpost = 200           # Time after step to variable V
    vhold = -80
    vstep = 20
    vstep2 = -40

    steps = [(vhold, tpre)]
    steps += [(vstep2, tpre2)]
    steps += [(vstep, tstep)]
    steps += [(vstep2, tstep2)]
    steps += [(vhold, tpost)]

    if return_capmask:
        return model, steps, None
    else:
        return model, steps


def sactiv(model, return_capmask=False):
    # in mV, ms
    # Short activation IV
    # time
    tsweepinterval = 5e3   # Time that is out side recording
    tpre  = 100          # Start bit
    tstep = 1e3            # Time at variable V
    tpost = 500          # Time after step to variable V
    tpost2 = 100         # End bit
    # voltage
    vhold = -80
    vmin = -50
    vres = 15         # Difference in V between steps
    n_sweeps = 7
    vpost = -40
    v = vmin + np.arange(n_sweeps) * vres
    steps = []
    for vstep in v:
        steps += [(vhold, tsweepinterval)]
        steps += [(vhold, tpre)]
        steps += [(vstep, tstep)]
        steps += [(vpost, tpost)]
        steps += [(vhold, tpost2)]

    if return_capmask:
        return model, steps, None
    else:
        return model, steps

def sactiv_times(dt):
    # in mV, ms
    # Return simulation time series with given dt
    # time
    tsweepinterval = 5e3   # Time that is out side recording
    tpre  = 100          # Start bit
    tstep = 1e3            # Time at variable V
    tpost = 500          # Time after step to variable V
    tpost2 = 100         # End bit

    n_sweeps = 7

    t_sweep = tpre + tstep + tpost + tpost2

    return np.arange(0, (tsweepinterval + t_sweep) * n_sweeps, dt)


def sactiv_convert(x, t):
    # in mV, ms
    # trim simulation output ``x`` to data format:
    # [step_1, step_2, ...].T
    # ``t`` in unit of second and must match ``x``
    assert(len(x) == len(t))
    t = np.asarray(t)
    x = np.asarray(x)
    # time
    tsweepinterval = 5e3   # Time that is out side recording
    tpre  = 100          # Start bit
    tstep = 1e3            # Time at variable V
    tpost = 500          # Time after step to variable V
    tpost2 = 100         # End bit

    n_sweeps = 7

    t_sweep = tpre + tstep + tpost + tpost2

    n_discard = np.abs(t - tsweepinterval).argmin()
    n_total = np.abs(t - (tsweepinterval + t_sweep)).argmin()

    x_out =  np.zeros((n_total - n_discard, n_sweeps))
    for i in range(n_sweeps):
        x_out[:, i] = x[i * n_total + n_discard : (i + 1) * n_total]
    t_out = t[n_discard:n_total]
    t_out = t_out - t_out[0]  # shift it to 0
    return x_out, t_out


def sinactiv(model, return_capmask=False):
    # in mV, ms
    # Short steady-stete inactivation IV
    # time
    tsweepinterval = 5e3   # Time that is out side recording
    tpre  = 100          # Start bit
    tpre2 = 500          # Time before step to variable V
    tstep = 500          # Time at variable V
    tpost = 100          # End bit
    # voltage
    vhold = -80
    vmin = -140
    vres = 20         # Difference in V between steps
    n_sweeps = 10
    vpre = 20
    v = vmin + np.arange(n_sweeps) * vres
    steps = []
    for vstep in v:
        steps += [(vhold, tsweepinterval)]
        steps += [(vhold, tpre)]
        steps += [(vpre, tpre2)]
        steps += [(vstep, tstep)]
        steps += [(vhold, tpost)]

    if return_capmask:
        return model, steps, None
    else:
        return model, steps


def sinactiv_times(dt):
    # in mV, ms
    # Return simulation time series with given dt
    # time
    tsweepinterval = 5e3   # Time that is out side recording
    tpre  = 100          # Start bit
    tpre2 = 500          # Time before step to variable V
    tstep = 500          # Time at variable V
    tpost = 100          # End bit

    n_sweeps = 10

    t_sweep = tpre + tpre2 + tstep + tpost

    return np.arange(0, (tsweepinterval + t_sweep) * n_sweeps, dt)


def sinactiv_convert(x, t):
    # in mV, ms
    # trim simulation output ``x`` to data format:
    # [step_1, step_2, ...].T
    # ``t`` in unit of second and must match ``x``
    assert(len(x) == len(t))
    # time
    tsweepinterval = 5e3   # Time that is out side recording
    tpre  = 100          # Start bit
    tpre2 = 500          # Time before step to variable V
    tstep = 500          # Time at variable V
    tpost = 100          # End bit

    n_sweeps = 10

    t_sweep = tpre + tpre2 + tstep + tpost

    n_discard = np.abs(t - tsweepinterval).argmin()
    n_total = np.abs(t - (tsweepinterval + t_sweep)).argmin()

    x_out =  np.zeros((n_total - n_discard, n_sweeps))
    for i in range(n_sweeps):
        x_out[:, i] = x[i * n_total + n_discard : (i + 1) * n_total]
    t_out = t[n_discard:n_total]
    t_out = t_out - t_out[0]  # shift it to 0
    return x_out, t_out


def sactiv_v():
    # in mV, ms
    # Return voltage step
    # voltage
    vmin = -50
    vres = 15         # Difference in V between steps
    n_sweeps = 7
    v = vmin + np.arange(n_sweeps) * vres
    return v

def sinactiv_v():
    # in mV, ms
    # Return voltage step
    # voltage
    vmin = -140
    vres = 20         # Difference in V between steps
    n_sweeps = 10
    v = vmin + np.arange(n_sweeps) * vres
    return v


def sactiv_iv_arg(nosweepinterval=True):
    # in mV, ms
    # return arg for get_iv()
    # time
    tsweepinterval = 5e3   # Time that is out side recording
    tpre  = 100          # Start bit
    tstep = 1e3            # Time at variable V
    tpost = 500          # Time after step to variable V
    tpost2 = 100         # End bit

    n_sweeps = 7

    t_start = tpre + tstep
    t_end = tpre + tstep + tpost
    if not nosweepinterval:
        t_start += tsweepinterval
        t_end += tsweepinterval

    # t_start, t_end, [(t_trim, t_fit_until)_1, (...)_2]
    return t_start, t_end, [(60, 400), (60, 400)]


def sinactiv_iv_arg(nosweepinterval=True):
    # in mV, ms
    # return arg for get_iv()
    # time
    tsweepinterval = 5e3   # Time that is out side recording
    tpre  = 100          # Start bit
    tpre2 = 500          # Time before step to variable V
    tstep = 500          # Time at variable V
    tpost = 100          # End bit

    n_sweeps = 10

    t_start = tpre + tpre2
    t_end = tpre + tpre2 + tstep
    if not nosweepinterval:
        t_start += tsweepinterval
        t_end += tsweepinterval

    # t_start, t_end, [(t_trim, t_fit_until)_1, (...)_2]
    return t_start, t_end, [(50, 400), (20, 100)]


def get_iv(folded_current, times, t_start, t_end):
    # in mV, ms
    times = np.asarray(times)
    n_samples, n_steps = folded_current.shape
    time_window = np.where(np.logical_and(times > t_start, times <= t_end))[0]
    time_window_90 = time_window[int(len(time_window) * 0.05):
            int(len(time_window) * 0.25)]
    I = []
    for i in range(n_steps):
        if (folded_current[:, i][time_window_90] <= 0).all():
            peak_I = np.min(folded_current[:, i][time_window])
        else:
            peak_I = np.max(folded_current[:, i][time_window])
        I.append(peak_I)
    return I


def get_corrected_iv(folded_current, times,
                     t_start, t_end, fit_windows,
                     debug=False, dt=2e-1):
    # in mV, ms
    # use 2-parameters exponential fit to the tail
    # folded_current: current in (n_samples, n_steps) format
    # times: time points corresponding to n_samples in `folded_current`
    # t_start: time at voltage step of interest begin
    # t_end: time at voltage step of interest finish
    # fit_windows: [(t_trim, t_fit_until)_1, (...)_2],
    #              1: if current >= 0,
    #              2: if current < 0,
    #              t_trim: time trimming _from `t_start`_ not used for fitting
    #              t_fit_until: time counted _from `t_start`_ used for fitting
    # dt: should match time step in `times`
    import scipy
    def exp_func(t, a, b):
        # do a "proper exponential" decay fit
        # i.e. shift the t to t' where t' has zero at the start of the 
        # voltage step
        return - a * np.exp( -b * (t - x[0]))
    times = np.asarray(times)
    assert(np.abs((times[1] - times[0]) - dt) < 1e-6)
    n_samples, n_steps = folded_current.shape
    time_window = np.where(np.logical_and(times > t_start, times <= t_end))[0]
    I = np.zeros(n_steps)
    fitting = []
    if debug:
        import matplotlib.pyplot as plt
        fig = plt.figure()
    for i in range(n_steps):
        if np.mean(folded_current[:, i][
                    time_window[0] + int(fit_windows[0][0] // dt)
                    :time_window[0] + int(fit_windows[0][1] // dt)]) >= 0:
            i_trim = int(fit_windows[0][0] // dt)
            i_fit_until = int(fit_windows[0][1] // dt)
        else:
            i_trim = int(fit_windows[1][0] // dt)
            i_fit_until = int(fit_windows[1][1] // dt)
        # trim off the first i_trim in case it is still shooting down...
        x = times[time_window[0] + i_trim:time_window[0] + i_fit_until]
        if i == 0:
            fitting.append(x)
        y = folded_current[:, i][time_window[0] + i_trim:
                                 time_window[0] + i_fit_until]
        try:
            # give it a bound for fitting: 
            # 1. "decay => all positive"  or  maybe some not 'decay'?
            #    => a bit negative...
            # 2. all current < 500 A/F...
            # 3. delay tend to be slow! (in unit of second though!)
            x = x * 1e-3
            popt, pcov = scipy.optimize.curve_fit(exp_func, x, y)#,
            #                                       bounds=(-10., [500., 10.]))
            fitted = exp_func(times[time_window[0]:
                                    time_window[0] + i_fit_until]*1e-3, *popt)
            I[i] = np.max(fitted[0])
            fitting.append(fitted)
        except:
            # give up, just print out a warning and use old method
            print('WARNING: CANNOT FIT TO voltage step %d'%(i))
            raise Exception('Maybe not here!')
        if debug:
            plt.plot(times[time_window[0] - 500:time_window[-1] + 500],
                     folded_current[:, i][time_window[0] - 500:
                                          time_window[-1] + 500], c='#d62728')
            plt.plot(times[time_window[0]:time_window[0] + i_fit_until],
                     fitted, '--', c='#1f77b4')
            plt.plot(times[time_window][0], fitted[0], 'kx')
    if debug:
        plt.axvline(x=times[time_window[0] + i_trim])
        plt.axvline(x=times[time_window[0] + i_fit_until])
        plt.savefig('figs/test-iv-fig.png')
        plt.close()
    return I


#
# Vandenberg et al. 2006 protocols
#

def Vandenberg2006_isochronal_tail_current(model, return_capmask=False,
                                           return_times=False,
                                           return_voltage=False):
    # A standard isochronal tail current protocol to measure voltage
    # dependence of activation
    # model: myokit model
    # return_capmask: if True, return an extra function that takes time series
    #                 as argument and return a mask to filter off capacitance
    #                 effect.
    # return_times: if True, _only_ return times, total time, time of interest
    # return_voltage: if True, _only_ return voltage steps
    tpre2 = 5            # Time before and after step to variable V
    tstep = 30           # Time at variable V
    tpost = 0.5          # Time shortly after step to variable V
    # activation
    vhold = -80e-3
    vmin = -80e-3
    vmax = 40e-3
    vres = 10e-3        # Difference in V between steps
    vpost = -100e-3
    v = np.arange(vmin, vmax + vres, vres)
    steps = []
    for vstep in v:
        steps += [(vhold, tpre2)]
        steps += [(vpost, tpost)]
        steps += [(vhold, tpost)]
        steps += [(vstep, tstep)]
        steps += [(vpost, tpost)]
        steps += [(vhold, tpre2)]

    if return_times:
        ttotal = tpre2 + tpost + tpost + tstep + tpost + tpre2
        tmeasure = tpre2 + tpost + tpost + tstep
        # Default time
        DT = 1.0e-04
        TTOTAL = np.sum([a[1] for a in steps])
        times = np.arange(0, TTOTAL - DT, DT)
        return times, ttotal, tmeasure

    if return_voltage:
        return v

    if return_capmask:
        return model, steps, None
    else:
        return model, steps


def Vandenberg2006_double_pulse(model, return_capmask=False,
                                return_times=False,
                                return_voltage=False):
    # A standard isochronal tail current protocol to measure voltage
    # dependence of activation
    # model: myokit model
    # return_capmask: if True, return an extra function that takes time series
    #                 as argument and return a mask to filter off capacitance
    #                 effect.
    # return_times: if True, _only_ return times, total time, time of interest
    # return_voltage: if True, _only_ return voltage steps
    tpre2 = 5.0            # Time before and after step to variable V
    tstep = 0.5           # Time at variable V
    tpost = 1.0          # Time shortly after step to variable V
    # inactivation
    vhold = -80e-3
    vmin = -120e-3  # -150e-3
    vmax = 50e-3
    vres = 10e-3        # Difference in V between steps
    vpost = +40e-3
    v = np.arange(vmin, vmax + vres, vres)
    steps = []
    for vstep in v:
        steps += [(vhold, tpre2)]
        steps += [(vpost, tpost)]
        steps += [(vstep, tstep)]
        steps += [(vhold, tpre2)]

    if return_times:
        ttotal = tpre2 + tpost + tstep + tpre2
        tmeasure = tpre2 + tpost
        # Default time
        DT = 1.0e-04
        TTOTAL = np.sum([a[1] for a in steps])
        times = np.arange(0, TTOTAL - DT, DT)
        return times, ttotal, tmeasure

    if return_voltage:
        return v

    if return_capmask:
        return model, steps, None
    else:
        return model, steps


def Vandenberg2006_envelope_of_tails(model, return_capmask=False,
                                     thold=1, vhold=40e-3,
                                     return_times=False):
    # An envelope of tails protocol to measure activation rate
    # model: myokit model
    # return_capmask: if True, return an extra function that takes time series
    #                 as argument and return a mask to filter off capacitance
    #                 effect.
    # thold, vhold: holding time and holding voltage before stepping to -50 mV,
    #               defualt are 1 s and +40 mV.
    #               To change this, use for example:
    #
    #               def envelope_of_tails_new(model, return_capmask=False):
    #                   thold = 0.5  # e.g. 500 ms
    #                   vhold = 0  # e.g. +0 mV
    #                   return envelope_of_tails(model, return_capmask,
    #                                            thold, vhold)
    #
    # return_times: if True, _only_ return times, total time, time of interest
    tpre  = 0.1           # Time before stepping to vhold
    tpost = 0.5           # Time holding at -50 mV
    tend = 0.5            # Time after stepping from -50 mV
    vpre = -80e-3
    vpost = -50e-3
    vend = vpre

    steps = [(vpre, tpre)]
    steps += [(vhold, thold)]
    steps += [(vpost, tpost)]
    steps += [(vend, tend)]

    if return_times:
        ttotal = tpre + thold + tpost + tend
        tmeasure = tpre + thold
        # Default time
        DT = 1.0e-04
        TTOTAL = np.sum([a[1] for a in steps])
        times = np.arange(0, TTOTAL - DT, DT)
        return times, ttotal, tmeasure

    if return_capmask:
        return model, steps, None
    else:
        return model, steps


def Vandenberg2006_triple_pulse(model, return_capmask=False,
                                return_times=False):
    # A triple pulse protocol to measure deactivation, inactivation and
    # recovery from inactivation rates
    # model: myokit model
    # return_capmask: if True, return an extra function that takes time series
    #                 as argument and return a mask to filter off capacitance
    #                 effect.
    # return_times: if True, _only_ return times, total time, time of interest
    tpre  = 0.1           # Time before protocol
    thold  = 1.0          # Time at first +40 mV
    tdown = 10e-3         # Time at repolarization to -80 mV
    thold2 = 0.2          # Time at second +40 mV
    tpost = 0.5           # Time after stepping down to final -120 mV
    vpre = -80e-3
    vhold = +40e-3
    vdown = -80e-3
    vhold2 = +40e-3
    vpost = -120e-3

    steps = [(vpre, tpre)]
    steps += [(vhold, thold)]
    steps += [(vdown, tdown)]
    steps += [(vhold2, thold2)]
    steps += [(vpost, tpost)]

    if return_times:
        ttotal = tpre + thold + tdown + thold2 + tpost
        tmeasure = [tpre + thold + tdown,
                    tpre + thold + tdown + thold2]
        # Default time
        DT = 1.0e-04
        TTOTAL = np.sum([a[1] for a in steps])
        times = np.arange(0, TTOTAL - DT, DT)
        return times, ttotal, tmeasure

    if return_capmask:
        return model, steps, None
    else:
        return model, steps


#
# Zhou et al. 1998 protocols
#

def Zhou1998_activation_deactivation(model,
                                     return_capmask=False,
                                     return_times=False):
    # An envelope of tails protocol to measure activation and deactivation
    # rates
    # model: myokit model
    # return_capmask: if True, return an extra function that takes time series
    #                 as argument and return a mask to filter off capacitance
    #                 effect.
    # return_times: if True, _only_ return times, total time, time of interest
    tpre  = 0.1           # Time before stepping to vhold
    thold = 5             # Time during holding step
    tpost = 5             # Time holding at -50 mV
    tend = 0.5            # Time after stepping from -50 mV
    vpre = -80e-3
    vhold = 0
    vpost = -50e-3
    vend = vpre

    steps = [(vpre, tpre)]
    steps += [(vhold, thold)]
    steps += [(vpost, tpost)]
    steps += [(vend, tend)]

    if return_times:
        ttotal = tpre + thold + tpost + tend
        tmeasure = [tpre,
                    tpre + thold]
        # Default time
        DT = 1.0e-04
        TTOTAL = np.sum([a[1] for a in steps])
        times = np.arange(0, TTOTAL - DT, DT)
        return times, ttotal, tmeasure

    if return_capmask:
        return model, steps, None
    else:
        return model, steps


def Zhou1998_inactivation(model, return_capmask=False, return_times=False):
    # A triple pulse protocol to measure inactivation rate
    # model: myokit model
    # return_capmask: if True, return an extra function that takes time series
    #                 as argument and return a mask to filter off capacitance
    #                 effect.
    # return_times: if True, _only_ return times, total time, time of interest
    tpre  = 0.1           # Time before protocol
    thold  = 0.2          # Time at first +60 mV
    tdown = 2e-3          # Time at repolarization to -100 mV  # TODO 10ms?
    thold2 = 0.3          # Time at 0 mV (measure)
    tpost = 0.1           # Finish
    vpre = -80e-3
    vhold = +60e-3
    vdown = -100e-3
    vhold2 = 0
    vpost = -80e-3

    steps = [(vpre, tpre)]
    steps += [(vhold, thold)]
    steps += [(vdown, tdown)]
    steps += [(vhold2, thold2)]
    steps += [(vpost, tpost)]

    if return_times:
        ttotal = tpre + thold + tdown + thold2 + tpost
        tmeasure = tpre + thold + tdown
        # Default time
        DT = 1.0e-04
        TTOTAL = np.sum([a[1] for a in steps])
        times = np.arange(0, TTOTAL - DT, DT)
        return times, ttotal, tmeasure

    if return_capmask:
        return model, steps, None
    else:
        return model, steps


def Zhou1998_recovery(model, return_capmask=False, return_times=False):
    # A double pulse protocol to measure recovery from inactivation rate
    # model: myokit model
    # return_capmask: if True, return an extra function that takes time series
    #                 as argument and return a mask to filter off capacitance
    #                 effect.
    # return_times: if True, _only_ return times, total time, time of interest
    tpre  = 0.1           # Time before protocol
    thold  = 0.2          # Time at first +60 mV
    tdown = 0.3           # Time at repolarization to -50 mV (measure)
    tpost = 0.1           # Finish
    vpre = -80e-3
    vhold = +60e-3
    vdown = -50e-3
    vpost = -80e-3

    steps = [(vpre, tpre)]
    steps += [(vhold, thold)]
    steps += [(vdown, tdown)]
    steps += [(vpost, tpost)]

    if return_times:
        ttotal = tpre + thold + tdown + tpost
        tmeasure = tpre + thold
        # Default time
        DT = 1.0e-04
        TTOTAL = np.sum([a[1] for a in steps])
        times = np.arange(0, TTOTAL - DT, DT)
        return times, ttotal, tmeasure

    if return_capmask:
        return model, steps, None
    else:
        return model, steps

