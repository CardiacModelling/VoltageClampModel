#!/usr/bin/env python
import numpy as np


class ComposeTransformation(object):
    """Combine two transformations as one."""

    def __init__(self, transform1, transform2, splitat):
        """
            transform1: first transformation.
            transform2: second transformation.
            splitat: input parameter before ``splitat`` uses ``transform1``,
                     and second half uses ``transform2``.
        """
        self.transform1 = transform1
        self.transform2 = transform2
        self.splitat = splitat

    def __call__(self, param):
        # Apply compose transformation
        p1 = param[:self.splitat]
        p2 = param[self.splitat:]
        tp1 = self.transform1(p1)
        tp2 = self.transform2(p2)
        return np.append(tp1, tp2)


def log_transform_from_model_param(param):
    # Apply natural log transformation to selected parameters
    out = np.copy(param)
    out = np.append(
            log_transform_from_ikr(out[:9]),
            log_transform_from_vc(out[9:])
            )
    return out


def log_transform_to_model_param(param):
    # Inverse of log_transform_from_model_param()
    # Apply natural exp transformation to all parameters
    out = np.copy(param)
    out = np.append(
            log_transform_to_ikr(out[:9]),
            log_transform_to_vc(out[9:])
            )
    return out


def log_transform_from_vc(param):
    # Apply natural log transformation to selected parameters
    out = np.copy(param)
    out[:-1] = np.log(out[:-1])
    return out


def log_transform_to_vc(param):
    # Inverse of log_transform_from_model_param()
    # Apply natural exp transformation to selected parameters
    out = np.copy(param)
    out[:-1] = np.exp(out[:-1])
    return out


def logit_transform_from_vc(param):
    # Apply logit transformation to selected parameters
    # Note: logit transformation does not have the properties of log-transform,
    #       i.e. if the bound span several order of magnitude, it does not re-
    #       scale it to log-like scale.
    #TODO: maybe turn it into a class?
    lower = np.array([
        0.1e-3,  # Rseries [GOhm]
        -15.0,   # Voffset [mV]
        ])

    upper = np.array([
        50e-3,   # Rseries [GOhm]
        15.0,    # Voffset [mV]
        ])
    out = np.copy(param)
    out = np.log((out - lower) / (upper - out))
    return out


def logit_transform_to_vc(param):
    # Inverse of logit_transform_from_model_param()
    # Apply logitic transformation to selected parameters
    #TODO: maybe turn it into a class?
    lower = np.array([
        0.1e-3,  # Rseries [GOhm]
        -15.0,   # Voffset [mV]
        ])

    upper = np.array([
        50e-3,   # Rseries [GOhm]
        15.0,    # Voffset [mV]
        ])
    out = np.copy(param)
    out = (lower + upper * np.exp(out)) / (1. + np.exp(out))
    return out


def log_transform_from_linleak(param):
    # Apply natural log transformation to selected parameters
    out = np.copy(param)
    out = np.log(out)
    return out


def log_transform_to_linleak(param):
    # Inverse of log_transform_from_linleak()
    # Apply natural exp transformation to selected parameters
    out = np.copy(param)
    out = np.exp(out)
    return out


def log_transform_from_leak(param):
    # Apply natural log transformation to selected parameters
    out = np.copy(param)
    out[:-1] = np.log(out[:-1])
    return out


def log_transform_to_leak(param):
    # Inverse of log_transform_from_model_param()
    # Apply natural exp transformation to selected parameters
    out = np.copy(param)
    out[:-1] = np.exp(out[:-1])
    return out


def log_transform_from_ikr(param):
    # Apply natural log transformation to all parameters
    out = np.copy(param)
    out = np.log(out)
    return out


def log_transform_to_ikr(param):
    # Inverse of log_transform_from_model_param()
    # Apply natural exp transformation to all parameters
    out = np.copy(param)
    out = np.exp(out)
    return out


def donothing(param):
    out = np.copy(param)
    return out
