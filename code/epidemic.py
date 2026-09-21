"""Logistic disease progress, and what a late detection costs.

Van der Plank's logistic model is the standard description of a polycyclic foliar
epidemic: severity x grows as dx/dt = r x (1 - x). A leaf cleared in error is not
looked at again until the next inspection, so the cost of a miss is the severity
the epidemic has reached by the time it is found.
"""
import numpy as np
from scipy.special import expit


def severity(t, x0=1e-3, r=0.15):
    """Fraction of tissue diseased t days after onset.

    Written through the logit rather than as x0 e^{rt} / (1 - x0 + x0 e^{rt}).
    The textbook form overflows for large r t, which happens whenever an optimiser
    wanders into a long interval, and returns nan where it should return 1.0.
    """
    return expit(np.log(x0 / (1.0 - x0)) + r * np.asarray(t, float))


def time_to_severity(x, x0=1e-3, r=0.15):
    """Days from onset for severity to reach x: the inverse of severity()."""
    return np.log(x * (1 - x0) / (x0 * (1 - x))) / r


def detection_delay(interval_days, p_detect):
    """Expected days from onset to detection.

    Inspections happen every `interval_days` and each one detects an existing
    infection with probability p_detect, so the number of inspections needed is
    geometric. Onset falls uniformly within an interval, which contributes half an
    interval on average, so the delay is half an interval when p_detect = 1. A
    detection probability of zero never detects.
    """
    if p_detect <= 0:
        return np.inf
    return interval_days * (1.0 / p_detect - 0.5)


def miss_cost(delay_days, x0=1e-3, r=0.15, cap=1.0):
    """Crop loss from detecting late: the severity reached at detection, capped at one.

    This is the loss the objective uses. It is one defensible choice among several,
    and stationarity.LOSSES replaces it with others of different shape. The cost is
    per infection, not per day, so it is not divided by the interval.
    """
    return float(np.minimum(severity(delay_days, x0, r), cap))
