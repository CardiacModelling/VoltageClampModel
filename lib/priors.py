#!/usr/bin/env python
import sys
sys.path.append('../lib/')
import numpy as np
import pints

param_names = ['ikr.g',
               'ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4',
               'ikr.p5', 'ikr.p6', 'ikr.p7', 'ikr.p8']

prior_parameters = {
        '25.0': [  # herg25oc1A01
         3.20021270937667614e+01,  # conductance [pA/mV]
         1.14638102637862893e-01,
         9.62556044732309886e+01,
         2.41591031546071096e-02,
         4.73551745851820414e+01,
         1.04325557236535232e+02,
         2.81709051196671112e+01,
         7.75475091376362169e+00,
         3.12745030498797831e+01,],
        '37.0': [  # herg37oc3O10
         5.20892708628531982e+01,  # conductance [pA/mV]
         3.66770880038180325e+00,
         7.03658277896161337e+01,
         3.44869494468236087e-02,
         6.63979406057673600e+01,
         4.52039468100492343e+02,
         2.51257322547301776e+01,
         5.33347088237617299e+01,
         2.82415900626202863e+01,]
    }

defaultparams = np.asarray(prior_parameters['37.0'])
bound = 100  # 1 + 1e-1
lower = defaultparams * bound ** -1
upper = defaultparams * bound


#
# Set up voltage clamp model prior
#
class VoltageClampLogPrior(pints.LogPrior):
    """
    Unnormalised prior for fully compensated voltage clamp model.
    """
    def __init__(self, transform, inv_transform):
        super(VoltageClampLogPrior, self).__init__()

        self.lower = np.array([
            0.1,     # Cm [pF]
            0.1e-3,  # Rseries [GOhm]
            4e-2,    # Cprs [pF]
            -15.0,   # Voffset [mV]
            ])

        self.upper = np.array([
            50.0,    # Cm [pF]
            50e-3,   # Rseries [GOhm]
            30.0,     # Cprs [pF]
            15.0,    # Voffset [mV]
            ])

        self.minf = -float('inf')

        self.transform = transform
        self.inv_transform = inv_transform

    def n_parameters(self):
        return 4

    def __call__(self, parameters):
        debug = False
        parameters = self.transform(parameters)

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug: print('Lower')
            return self.minf
        if np.any(parameters > self.upper):
            if debug: print('Upper')
            return self.minf

        # Return
        return 0

    def sample(self, n=1):
        out = np.zeros((n, self.n_parameters()))

        for i in range(n):
            p = np.zeros(self.n_parameters())

            p[:-1] = np.exp(np.random.uniform(
                np.log(self.lower[:-1]), np.log(self.upper[:-1])))
            # contain -ve
            p[-1:] = np.random.uniform(self.lower[-1:], self.upper[-1:])

            out[i, :] = self.inv_transform(p)

        # Return
        return out


class SimplifiedVoltageClampLogPrior(VoltageClampLogPrior):
    """
    Unnormalised prior for simplified compensated voltage clamp model.
    """
    def __init__(self, transform, inv_transform):
        super(SimplifiedVoltageClampLogPrior, self).__init__(transform,
                inv_transform)

        self.lower = np.array([
            0.1e-3,  # Rseries [GOhm]
            -15.0,   # Voffset [mV]
            ])

        self.upper = np.array([
            50e-3,   # Rseries [GOhm]
            15.0,    # Voffset [mV]
            ])

    def n_parameters(self):
        return 2


class SimplifiedVoltageClamp2LogPrior(VoltageClampLogPrior):
    """
    Unnormalised prior for simplified compensated voltage clamp model.
    """
    def __init__(self, transform, inv_transform):
        super(SimplifiedVoltageClamp2LogPrior, self).__init__(transform,
                inv_transform)

        self.lower = np.array([
            0.1e-3,  # Rseries [GOhm]
            -15.0,   # Voffset [mV]
            0.1e-3,  # Rseries_est [GOhm]
            ])

        self.upper = np.array([
            50e-3,   # Rseries [GOhm]
            15.0,    # Voffset [mV]
            50e-3,   # Rseries_est [GOhm]
            ])

    def n_parameters(self):
        return 3

    def sample(self, n=1):
        out = np.zeros((n, self.n_parameters()))

        for i in range(n):
            p = np.zeros(self.n_parameters())

            p[:1] = np.exp(np.random.uniform(
                np.log(self.lower[:1]), np.log(self.upper[:1])))
            p[2:] = np.exp(np.random.uniform(
                np.log(self.lower[2:]), np.log(self.upper[2:])))
            # contain -ve
            p[1] = np.random.uniform(self.lower[1], self.upper[1])

            out[i, :] = self.inv_transform(p)

        # Return
        return out


class SimplifiedInformativeVoltageClampLogPrior(pints.LogPrior):
    """
    Unnormalised prior for fully compensated voltage clamp model.
    """
    def __init__(self, transform, inv_transform, prior_rseal=10e-3):
        super(SimplifiedInformativeVoltageClampLogPrior, self).__init__()

        self.transform = transform
        self.inv_transform = inv_transform

        self._rseal_mean = m = prior_rseal  # [GOhm]
        self._rseal_std = s = 1.0e-3  # [GOhm]
        self._rseal_logmean = np.log(m) - 0.5 * np.log((s / m) ** 2 + 1.)
        self._rseal_scale = np.sqrt(np.log((s / m) ** 2 + 1.))

        self._voffset_mean = 0.0  # [mV]
        self._voffset_std = 5.0  # [mV]

        self._rseal_prior = pints.LogNormalLogPrior(self._rseal_logmean,
                self._rseal_scale)
        self._voffset_prior = pints.GaussianLogPrior(self._voffset_mean,
                self._voffset_std)

    def n_parameters(self):
        return 2

    def sample(self, n=1):
        output = np.zeros((n, self.n_parameters()))
        output[:, 0] = self._rseal_prior.sample(n)
        output[:, 1] = self._voffset_prior.sample(n)
        for i in range(n):
            output[i, :] = self.inv_transform(output[i, :])
        return output

    def __call__(self, parameters):
        parameters = self.transform(parameters)
        output = 0
        output += self._rseal_prior([parameters[0]])
        output += self._voffset_prior([parameters[1]])
        return output


#
# Set up non-linear time-dependent leak model prior
#
class LeakLogPrior(pints.LogPrior):
    """
    Unnormalised prior for non-linear time-dependent leak current model.
    """
    def __init__(self, transform, inv_transform):
        super(LeakLogPrior, self).__init__()

        self.lower = np.array([
            1e-12,  # A1a []
            1e-12,  # A1b [/ms]
            1e-12,  # B1a []
            1e-12,  # B1b [/ms]
            1e-5,  # gt [pA/mV/mV]
            25.0,    # tauta [ms]
            1e-2,    # tautb [ms/mV]
            -80,    # Et [mV]
            ])

        self.upper = np.array([
            1.0,  # A1a []
            1.0,  # A1b [/ms]
            1.0,  # B1a []
            1.0,  # B1b [/ms]
            1.0,  # gt [pA/mV/mV]
            1e3,  # tauta [ms]
            10.,  # tautb [ms/mV]
            80,   # Et [mV]
            ])

        self.minf = -float('inf')

        self.transform = transform
        self.inv_transform = inv_transform

    def n_parameters(self):
        return 8

    def __call__(self, parameters):
        debug = False
        parameters = self.transform(parameters)

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug: print('Lower')
            return self.minf
        if np.any(parameters > self.upper):
            if debug: print('Upper')
            return self.minf

        # Return
        return 0

    def sample(self, n=1):
        out = np.zeros((n, self.n_parameters()))

        for i in range(n):
            p = np.zeros(self.n_parameters())

            p[:-1] = np.exp(np.random.uniform(
                np.log(self.lower[:-1]), np.log(self.upper[:-1])))
            # contain -ve
            p[-1:] = np.random.uniform(self.lower[-1:], self.upper[-1:])

            out[i, :] = self.inv_transform(p)

        # Return
        return out


#
# Set up IKr prior
#
class IKrLogPrior(pints.LogPrior):
    """
    Unnormalised prior with constraint on the rate constants.

    # Partially adapted from 
    https://github.com/pints-team/ikr/blob/master/beattie-2017/beattie.py

    # Added parameter transformation everywhere
    """
    def __init__(self, transform, inv_transform):
        super(IKrLogPrior, self).__init__()

        # Give it a big bound...
        self.lower_conductance = 1e2 * 1e-3  # pA/mV
        self.upper_conductance = 5e5 * 1e-3  # pA/mV

        # change unit...
        self.lower_alpha = 1e-7              # Kylie: 1e-7
        self.upper_alpha = 1e3               # Kylie: 1e3
        self.lower_beta  = 1e-7              # Kylie: 1e-7
        self.upper_beta  = 0.4               # Kylie: 0.4

        self.lower = np.array([
            self.lower_conductance,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
        ])
        self.upper = np.array([
            self.upper_conductance,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
        ])

        self.minf = -float('inf')

        self.rmin = 1.67e-5# * 1e3
        self.rmax = 1000# * 1e3

        self.vmin = -120# * 1e-3
        self.vmax =  60# * 1e-3

        self.transform = transform
        self.inv_transform = inv_transform

    def n_parameters(self):
        return 8 + 1

    def __call__(self, parameters):

        debug = False
        parameters = self.transform(parameters)

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug: print('Lower')
            return self.minf
        if np.any(parameters > self.upper):
            if debug: print('Upper')
            return self.minf

        # Check rate constant boundaries
        g, p1, p2, p3, p4, p5, p6, p7, p8 = parameters[:]

        # Check forward rates
        r = p1 * np.exp(p2 * self.vmax)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r1')
            return self.minf
        r = p5 * np.exp(p6 * self.vmax)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r2')
            return self.minf

        # Check backward rates
        r = p3 * np.exp(-p4 * self.vmin)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r3')
            return self.minf
        r = p7 * np.exp(-p8 * self.vmin)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r4')
            return self.minf

        return 0

    def _sample_partial(self, v):
        for i in range(100):
            a = np.exp(np.random.uniform(
                np.log(self.lower_alpha), np.log(self.upper_alpha)))
            b = np.random.uniform(self.lower_beta, self.upper_beta)
            r = a * np.exp(b * v)
            if r >= self.rmin and r <= self.rmax:
                return a, b
        raise ValueError('Too many iterations')

    def sample(self, n=1):
        out = np.zeros((n, 9))
        
        for i in range(n):
            p = np.zeros(9)

            # Sample forward rates
            p[1:3] = self._sample_partial(self.vmax)
            p[5:7] = self._sample_partial(self.vmax)

            # Sample backward rates
            p[3:5] = self._sample_partial(-self.vmin)
            p[7:9] = self._sample_partial(-self.vmin)

            # Sample conductance
            p[0] = np.random.uniform(
                self.lower_conductance, self.upper_conductance)

            out[i, :] = self.inv_transform(p)

        # Return
        return out



#
# Set up voltage clamp model with conductance value prior
#
class VoltageClampWithConductanceLogPrior(pints.LogPrior):
    """
    Unnormalised prior for fully compensated voltage clamp model wtih
    conductnace. Mainly for MultiCellModel(), where we assume same kinetics
    across cells but with different conductance values.
    """
    def __init__(self, transform, inv_transform):
        super(VoltageClampWithConductanceLogPrior, self).__init__()

        # Give it a big bound...
        self.lower_conductance = 1e2 * 1e-3  # pA/mV
        self.upper_conductance = 5e5 * 1e-3  # pA/mV

        self.lower = np.array([
            self.lower_conductance,
            0.1,     # Cm [pF]
            0.01e-3,  # Rseries [GOhm]
            4e-2,    # Cprs [pF]
            -15.0,   # Voffset [mV]
            ])

        self.upper = np.array([
            self.upper_conductance,
            50.0,    # Cm [pF]
            50e-3,   # Rseries [GOhm]
            30.0,     # Cprs [pF]
            15.0,    # Voffset [mV]
            ])

        self.minf = -float('inf')

        self.transform = transform
        self.inv_transform = inv_transform

    def n_parameters(self):
        return 1 + 4

    def __call__(self, parameters):
        debug = False
        parameters = self.transform(parameters)

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug: print('Lower')
            return self.minf
        if np.any(parameters > self.upper):
            if debug: print('Upper')
            return self.minf

        # Return
        return 0

    def sample(self, n=1):
        out = np.zeros((n, self.n_parameters()))

        for i in range(n):
            p = np.zeros(self.n_parameters())

            p[:-1] = np.exp(np.random.uniform(
                np.log(self.lower[:-1]), np.log(self.upper[:-1])))
            # contain -ve
            p[-1:] = np.random.uniform(self.lower[-1:], self.upper[-1:])

            out[i, :] = self.inv_transform(p)

        # Return
        return out


class SimplifiedVoltageClampWithConductanceLogPrior(
        VoltageClampWithConductanceLogPrior):
    """
    Unnormalised prior for simplified compensated voltage clamp model.
    """
    def __init__(self, transform, inv_transform):
        super(SimplifiedVoltageClampWithConductanceLogPrior, self).__init__(
                transform, inv_transform)

        self.lower = np.array([
            self.lower_conductance,
            0.1e-3,  # Rseries [GOhm]
            -15.0,   # Voffset [mV]
            ])

        self.upper = np.array([
            self.upper_conductance,
            50e-3,   # Rseries [GOhm]
            15.0,    # Voffset [mV]
            ])

    def n_parameters(self):
        return 1 + 2


class VoltageOffsetWithConductanceLogPrior(
        VoltageClampWithConductanceLogPrior):
    """
    Unnormalised prior for voltage offset model with conductance value.
    """
    def __init__(self, transform, inv_transform):
        super(VoltageOffsetWithConductanceLogPrior, self).__init__(
                transform, inv_transform)

        self.lower = np.array([
            self.lower_conductance,
            -15.0,   # Voffset [mV]
            ])

        self.upper = np.array([
            self.upper_conductance,
            15.0,    # Voffset [mV]
            ])

    def n_parameters(self):
        return 1 + 1


class SimplifiedInformativeVoltageClampWithConductanceLogPrior(pints.LogPrior):
    """
    Unnormalised prior for fully compensated voltage clamp model.
    """
    def __init__(self, transform, inv_transform, prior_rseal=10e-3):
        super(SimplifiedInformativeVoltageClampWithConductanceLogPrior,
                self).__init__()
        raise NotImplementedError


#
# Set up IKr without conductance value prior
#
class IKrWithoutConductanceLogPrior(pints.LogPrior):
    """
    Unnormalised prior with constraint on the rate constants without
    conductnace value. Mainly for MultiCellModel() where we assume same
    kinetics across cells but with different conductance values.

    # Partially adapted from 
    https://github.com/pints-team/ikr/blob/master/beattie-2017/beattie.py

    # Added parameter transformation everywhere
    """
    def __init__(self, transform, inv_transform):
        super(IKrWithoutConductanceLogPrior, self).__init__()

        # change unit...
        self.lower_alpha = 1e-7              # Kylie: 1e-7
        self.upper_alpha = 1e3               # Kylie: 1e3
        self.lower_beta  = 1e-7              # Kylie: 1e-7
        self.upper_beta  = 0.4               # Kylie: 0.4

        self.lower = np.array([
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
        ])
        self.upper = np.array([
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
        ])

        self.minf = -float('inf')

        self.rmin = 1.67e-5# * 1e3
        self.rmax = 1000# * 1e3

        self.vmin = -120# * 1e-3
        self.vmax =  60# * 1e-3

        self.transform = transform
        self.inv_transform = inv_transform

    def n_parameters(self):
        return 8

    def __call__(self, parameters):

        debug = False
        parameters = self.transform(parameters)

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug: print('Lower')
            return self.minf
        if np.any(parameters > self.upper):
            if debug: print('Upper')
            return self.minf

        # Check rate constant boundaries
        p1, p2, p3, p4, p5, p6, p7, p8 = parameters[:]

        # Check forward rates
        r = p1 * np.exp(p2 * self.vmax)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r1')
            return self.minf
        r = p5 * np.exp(p6 * self.vmax)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r2')
            return self.minf

        # Check backward rates
        r = p3 * np.exp(-p4 * self.vmin)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r3')
            return self.minf
        r = p7 * np.exp(-p8 * self.vmin)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r4')
            return self.minf

        return 0

    def _sample_partial(self, v):
        for i in range(100):
            a = np.exp(np.random.uniform(
                np.log(self.lower_alpha), np.log(self.upper_alpha)))
            b = np.random.uniform(self.lower_beta, self.upper_beta)
            r = a * np.exp(b * v)
            if r >= self.rmin and r <= self.rmax:
                return a, b
        raise ValueError('Too many iterations')

    def sample(self, n=1):
        out = np.zeros((n, 8))
        
        for i in range(n):
            p = np.zeros(8)

            # Sample forward rates
            p[0:2] = self._sample_partial(self.vmax)
            p[4:6] = self._sample_partial(self.vmax)

            # Sample backward rates
            p[2:4] = self._sample_partial(-self.vmin)
            p[6:8] = self._sample_partial(-self.vmin)

            out[i, :] = self.inv_transform(p)

        # Return
        return out


#
# Multiple priori
#
class MultiPriori(pints.LogPrior):
    """
    Combine multiple priors
    """
    def __init__(self, priors):
        self._priors = priors
        self._n_parameters = self._priors[0].n_parameters()
        for p in self._priors:
            assert(self._n_parameters == p.n_parameters())

    def n_parameters(self):
        return self._n_parameters

    def __call__(self, x):
        t = 0
        for p in self._priors:
            t += p(x)
        return t


