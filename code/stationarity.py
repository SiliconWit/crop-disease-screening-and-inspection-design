"""Where the optimal threshold comes from.

Write the two stationarity conditions of the objective, one in the threshold and
one in the interval, and eliminate the interval between them. What survives the
elimination is

    Q'(t) = Q(t) P'(t) / (P(t) - rho P(t)^2 / 2),

with P the probability that a diseased leaf is sent, Q the offload rate over all
captures and rho the cloud recall. It contains no deployment parameter other than
the densities (through P and Q) and rho. This module evaluates the condition,
finds its roots, and provides a tight optimiser and a set of alternative loss
functions with which to test it.

The condition is necessary at an interior optimum, not sufficient: it can have
more than one root, and a root need not be an optimum.
"""
import numpy as np
from scipy.optimize import brentq, minimize
import screening as S
import uplink as U
import epidemic as E
import risk as R

PRIOR = 0.15


def condition(t, cloud_recall=0.92, prior=PRIOR):
    """The stationarity condition in the threshold, as Q'/Q - P'/(P - rho P^2/2).

    Dividing through by Q > 0 leaves the sign, and hence every root, unchanged, and
    keeps the value well scaled in the tails. The derivatives are the densities
    themselves (P' = -f_d, Q' = -(prior f_d + (1 - prior) f_h)), so no difference
    quotient is needed, and P and Q are taken from survival functions rather than
    as one minus a distribution function. Either shortcut loses all precision
    once the tail probabilities fall below about 1e-10, and then produces
    spurious roots at high thresholds.
    """
    rho = cloud_recall
    P = S.sf_diseased(t)
    Q = prior * P + (1 - prior) * S.sf_healthy(t)
    f1, f0 = S.pdf_diseased(t), S.pdf_healthy(t)
    dP = -f1
    dQ = -(prior * f1 + (1 - prior) * f0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(dQ / Q - dP / (P - 0.5 * rho * P * P))


def analytic_theta(cloud_recall=0.92, prior=PRIOR, r=0.15, goodput=20000.0,
                   payload=25000, p_success=0.85):
    """The threshold that satisfies the condition, bracketed on [0.05, 0.9].

    Takes every deployment parameter so that the signature does not presuppose
    which ones survive the elimination; only the densities, the prior and the
    cloud recall are used. Raises ValueError if the bracket does not contain a
    sign change; use condition_roots where more than one root may exist.
    """
    return float(brentq(condition, 0.05, 0.9, args=(cloud_recall, prior), xtol=1e-12))


def condition_roots(cloud_recall=0.92, prior=PRIOR, r=0.15, goodput=20000.0,
                    payload=25000, p_success=0.85, lo=0.02, hi=0.95, n=400):
    """Every root of the condition on [lo, hi], in increasing order.

    A single bracketing call finds at most one root, or fails outright when the
    end points share a sign, so a grid is scanned for sign changes and each is
    refined. Which root an optimum sits on, if any, is for the caller to decide.
    """
    ts = np.linspace(lo, hi, n)
    v = np.array([condition(t, cloud_recall, prior) for t in ts])
    return [float(brentq(condition, ts[i], ts[i + 1], args=(cloud_recall, prior),
                         xtol=1e-12))
            for i in range(n - 1) if np.isfinite(v[i]) and np.isfinite(v[i + 1])
            and np.sign(v[i]) != np.sign(v[i + 1])]


def general_optimum(loss_fn, per_visit_J=0.0, prior=PRIOR, r=0.15,
                    goodput=20000.0, payload=25000, p_success=0.85,
                    cloud_recall=0.92):
    """Joint optimum under an arbitrary loss, with an optional cost per visit.

    `loss_fn(D, r)` maps a detection delay to a loss. `per_visit_J` is energy
    charged once per inspection visit whatever is offloaded (waking, focusing,
    capturing), which is a cost of the form A/T rather than proportional to the
    offload rate. Returns (threshold, interval, objective).

    The differences this optimiser has to resolve are far below the grid spacing
    in risk.optimise, so its tolerances are much tighter, and it keeps the best of
    sixteen starting points: a single start can stop on a flat stretch of the
    surface and report a threshold that is merely where the search gave up.
    """
    def J(x):
        th = float(np.clip(x[0], 0.02, 0.94))
        T = float(np.clip(x[1], 0.5, 90.0))
        op = S.operating_point(th, prior)
        pdet = op["p_detect_given_diseased"] * cloud_recall
        D = E.detection_delay(T, pdet)
        duty = U.duty_cycle_cost(T, op["offload_rate"], payload, goodput,
                                 p_success, R.CAPTURES_PER_VISIT)
        return (R.C_LOSS * loss_fn(D, r)
                + R.C_ENERGY * (duty["controllable_J_per_day"] + per_visit_J / T)
                + R.C_DATA * duty["bytes_per_day"])
    best = None
    for t0 in (0.12, 0.25, 0.40, 0.60):
        for T0 in (3.0, 8.0, 20.0, 40.0):
            res = minimize(J, [t0, T0], method="Nelder-Mead",
                           options=dict(xatol=1e-8, fatol=1e-14, maxiter=4000))
            if best is None or res.fun < best.fun:
                best = res
    return (float(np.clip(best.x[0], 0.02, 0.94)),
            float(np.clip(best.x[1], 0.5, 90.0)), float(best.fun))


# Four increasing losses of quite different shape. If a claim about the optimal
# threshold were really a property of the logistic model, one of these would break
# it. The square root is not differentiable at zero delay, so its argument is kept
# non-negative.
LOSSES = {
    "logistic (van der Plank)": lambda D, r: E.severity(D, r=r),
    "linear in delay": lambda D, r: r * D,
    "exponential, no saturation": lambda D, r: 1e-3 * np.exp(r * D),
    "square root": lambda D, r: np.sqrt(max(r * D, 0.0)),
}
