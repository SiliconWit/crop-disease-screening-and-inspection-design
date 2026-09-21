"""Edge screening as a likelihood ratio test, and what the threshold fixes.

The node computes a scalar index g in [0,1] from an HSV partition of each leaf
image: the fraction of leaf pixels that are chlorotic or necrotic. It offloads the
image over 2G only when g exceeds a threshold. Everything the threshold controls,
the offload rate, the edge miss rate and the false offload rate, follows in closed
form from the two class-conditional densities of g, provided a threshold on g is
the right test at all.
"""
import numpy as np
from scipy.stats import beta

# Class-conditional densities of the edge index. These are assumed, not fitted to
# labelled images. Beta is the natural two-parameter family on [0,1].
HEALTHY = dict(a=2.0, b=18.0)     # mean 0.10: a healthy leaf still has some senescence
DISEASED = dict(a=5.0, b=9.0)     # mean 0.36, with real overlap into the healthy range


def pdf_healthy(g):
    """Density of the index for healthy leaves."""
    return beta.pdf(g, HEALTHY["a"], HEALTHY["b"])


def pdf_diseased(g):
    """Density of the index for diseased leaves."""
    return beta.pdf(g, DISEASED["a"], DISEASED["b"])


def cdf_healthy(g):
    """Probability a healthy leaf's index is at or below g."""
    return beta.cdf(g, HEALTHY["a"], HEALTHY["b"])


def cdf_diseased(g):
    """Probability a diseased leaf's index is at or below g."""
    return beta.cdf(g, DISEASED["a"], DISEASED["b"])


def sf_healthy(g):
    """Probability a healthy leaf's index exceeds g, accurate far into the tail."""
    return beta.sf(g, HEALTHY["a"], HEALTHY["b"])


def sf_diseased(g):
    """Probability a diseased leaf's index exceeds g, accurate far into the tail."""
    return beta.sf(g, DISEASED["a"], DISEASED["b"])


def likelihood_ratio(g):
    """f_diseased(g) / f_healthy(g), guarded against a zero denominator.

    HEALTHY and DISEASED are read at call time, not at import, so that
    densities.swapped can replace them and every function here follows.
    """
    return pdf_diseased(g) / np.maximum(pdf_healthy(g), 1e-300)


def is_lr_monotone(n=2000):
    """True if the likelihood ratio never decreases across (0, 1).

    A threshold test on g is a likelihood ratio test only if the ratio is monotone
    in g. For two Beta densities that holds when a_1 >= a_0 and b_1 <= b_0; it is
    checked numerically here rather than assumed, because if it fails no single
    threshold is the optimal test. The check works on the log of the ratio and
    stays away from the end points, where both densities can vanish together.
    """
    g = np.linspace(1e-4, 1 - 1e-4, n)
    lr = np.log(likelihood_ratio(g))
    return bool(np.all(np.diff(lr) > -1e-9))


def operating_point(theta, prior):
    """What a screening threshold buys, in closed form.

    Returns the offload probability over all captures, the edge miss rate (a
    diseased leaf cleared at the edge and never sent, as a fraction of ALL
    captures, so it is weighted by the prior), the false offload rate, and the
    probability that a diseased leaf is sent, which is what the detection model
    needs.
    """
    p_send_d = 1.0 - cdf_diseased(theta)     # diseased leaf is offloaded
    p_send_h = 1.0 - cdf_healthy(theta)      # healthy leaf is offloaded
    q = prior * p_send_d + (1 - prior) * p_send_h
    return dict(offload_rate=q,
                miss_rate=prior * (1 - p_send_d),
                false_offload=(1 - prior) * p_send_h,
                p_detect_given_diseased=p_send_d)


def threshold_for_miss_rate(target_miss, prior, lo=1e-6, hi=1 - 1e-6):
    """The threshold at which the edge miss rate equals `target_miss`.

    The miss rate rises with the threshold, so a bracketing root finder works. If
    the target is outside what any threshold can give, the end of the bracket is
    returned rather than raising.
    """
    from scipy.optimize import brentq
    f = lambda t: operating_point(t, prior)["miss_rate"] - target_miss
    if f(lo) > 0:
        return lo
    if f(hi) < 0:
        return hi
    return brentq(f, lo, hi)
