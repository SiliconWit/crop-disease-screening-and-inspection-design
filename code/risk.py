"""The composed objective, and the optimum over threshold and interval.

The screening threshold fixes how often the node offloads and how often it wrongly
clears a diseased leaf. The inspection interval fixes how long a wrongly cleared
leaf waits before it is looked at again. The two meet in one expected daily cost.
Nothing here assumes how the optimum in one variable depends on the other;
stationarity.py and run_stationarity.py examine that.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import screening as S
import uplink as U
import epidemic as E

# Cost weights. They convert three incommensurable things into one number, so they
# are among the most consequential assumptions in the study; run_value.py scales
# each of them up and down.
C_LOSS = 1.0          # per unit of crop severity at detection
C_ENERGY = 2.0e-4     # per joule per day, a proxy for battery and servicing
C_DATA = 3.0e-6       # per byte per day, the airtime bill

# Leaves imaged per inspection visit. A real node sweeps a section of the plot.
CAPTURES_PER_VISIT = 20

# The base case. Every value is an assumption.
BASE = dict(prior=0.15, r=0.15, goodput=20000.0, payload=25000,
            p_success=0.85, cloud_recall=0.92)
CONVENTIONAL_INTERVAL = 7.0      # weekly scouting, the usual field practice
TARGET_MISS = 0.02               # edge miss rate of the reported operating point

# Search region of the optimisers. An optimum on one of these bounds is a corner
# of the search, not a stationary point, and is reported as such.
THETA_BOUNDS = (0.01, 0.95)
INTERVAL_BOUNDS = (0.5, 60.0)


def on_bound(theta, interval_days, tol=1e-6):
    """True if a (threshold, interval) pair lies on a bound of the search region."""
    return (interval_days <= INTERVAL_BOUNDS[0] + tol
            or interval_days >= INTERVAL_BOUNDS[1] - tol
            or theta <= THETA_BOUNDS[0] + tol or theta >= THETA_BOUNDS[1] - tol)


def risk(theta, interval_days, prior=0.15, r=0.15, goodput=20000.0,
         payload=25000, p_success=0.85, cloud_recall=0.92, verbose=False):
    """Expected cost per day of a (threshold, interval) pair.

    A diseased leaf is caught only if the edge sends it AND the cloud calls it.
    Crop loss is per infection event, not per day: infections arrive at a rate
    that does not depend on how often the node looks, so that rate is a constant
    folded into C_LOSS, and dividing the loss by the interval would make a long
    interval look cheap for no physical reason. The sleep budget is not part of
    the objective, because neither design variable moves it.

    With verbose=True the pieces are returned as well, so that the share of each
    term can be reported at any optimum.
    """
    op = S.operating_point(theta, prior)
    p_detect = op["p_detect_given_diseased"] * cloud_recall
    delay = E.detection_delay(interval_days, p_detect)
    loss = E.miss_cost(delay, r=r)
    duty = U.duty_cycle_cost(interval_days, op["offload_rate"], payload,
                             goodput, p_success, CAPTURES_PER_VISIT)
    total = (C_LOSS * loss
             + C_ENERGY * duty["controllable_J_per_day"]
             + C_DATA * duty["bytes_per_day"])
    if verbose:
        return dict(risk=total, loss=loss, delay_days=delay,
                    offload_rate=op["offload_rate"], miss_rate=op["miss_rate"],
                    energy_J_per_day=duty["controllable_J_per_day"],
                    idle_J_per_day=duty["idle_J_per_day"],
                    bytes_per_day=duty["bytes_per_day"], p_detect=p_detect)
    return total


def optimise(prior=0.15, r=0.15, goodput=20000.0, payload=25000,
             p_success=0.85, cloud_recall=0.92):
    """Joint minimisation over threshold and interval.

    The surface is not convex, so a single local search from one start can land
    in the wrong place. A coarse grid is searched first, then Nelder-Mead refines
    from the best grid point, with both variables clipped to the search region.
    The grid minimum is returned too, so that a refinement that made things worse
    would be visible.
    """
    lo_t, hi_t = THETA_BOUNDS
    lo_i, hi_i = INTERVAL_BOUNDS
    ths = np.linspace(0.05, 0.60, 34)
    ivs = np.linspace(1.0, 45.0, 45)
    Z = np.array([[risk(t, i, prior, r, goodput, payload, p_success, cloud_recall)
                   for i in ivs] for t in ths])
    a, b = np.unravel_index(np.argmin(Z), Z.shape)
    x0 = [ths[a], ivs[b]]
    res = minimize(lambda x: risk(np.clip(x[0], lo_t, hi_t),
                                  float(np.clip(x[1], lo_i, hi_i)), prior, r, goodput,
                                  payload, p_success, cloud_recall),
                   x0, method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-10, maxiter=800))
    th, iv = float(np.clip(res.x[0], lo_t, hi_t)), float(np.clip(res.x[1], lo_i, hi_i))
    out = risk(th, iv, prior, r, goodput, payload, p_success, cloud_recall, verbose=True)
    out.update(theta=th, interval_days=iv, grid_min=float(Z.min()),
               on_bound=on_bound(th, iv))
    return out


def unconditional_baseline(interval_days, prior=0.15, r=0.15, goodput=20000.0,
                           payload=25000, p_success=0.85, cloud_recall=0.92):
    """Offload every capture: the node with no edge screening at all.

    Every capture is sent, so detection depends on the cloud alone. This is the
    comparison a transmitted-volume saving is measured against, evaluated at the
    same interval as the design it is compared with.
    """
    delay = E.detection_delay(interval_days, cloud_recall)
    loss = E.miss_cost(delay, r=r)
    duty = U.duty_cycle_cost(interval_days, 1.0, payload, goodput, p_success,
                             CAPTURES_PER_VISIT)
    return dict(risk=(C_LOSS * loss
                      + C_ENERGY * duty["controllable_J_per_day"]
                      + C_DATA * duty["bytes_per_day"]),
                loss=loss, delay_days=delay, offload_rate=1.0,
                energy_J_per_day=duty["controllable_J_per_day"],
                bytes_per_day=duty["bytes_per_day"])


def opt_interval_given(theta, **kw):
    """Best interval with the threshold held fixed; returns (interval, risk).

    A bounded scalar search over INTERVAL_BOUNDS. A result on a bound is a corner,
    not a stationary point.
    """
    lo, hi = INTERVAL_BOUNDS
    f = lambda i: risk(theta, float(np.clip(i, lo, hi)), **kw)
    res = minimize_scalar(f, bounds=(lo, hi), method="bounded",
                          options=dict(xatol=1e-4))
    return float(res.x), float(res.fun)


def opt_theta_given(interval, **kw):
    """Best threshold with the interval held fixed; returns (threshold, risk)."""
    lo, hi = THETA_BOUNDS
    f = lambda t: risk(float(np.clip(t, lo, hi)), interval, **kw)
    res = minimize_scalar(f, bounds=(lo, hi), method="bounded",
                          options=dict(xatol=1e-5))
    return float(res.x), float(res.fun)
