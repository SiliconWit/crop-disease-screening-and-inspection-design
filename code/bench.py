"""Turn bench measurements into the quantities the model assumes.

Labelled leaf images become fitted densities, transfer logs become goodput and a
per-attempt success probability, a scope capture becomes a mean transmit current,
and a labelled evaluation set becomes a cloud recall. Each function takes raw
measurements and returns a summary. The model inputs used in this repository are
assumptions (see data/MANIFEST.md).
"""
import numpy as np


def fit_beta(values, eps=1e-4):
    """Maximum-likelihood Beta(a, b) on [0, 1]; returns (a, b, n_clipped).

    The location is fixed at 0 and the scale at 1, or the fit would be free to move
    the support and the parameters would stop meaning what screening.py assumes. An
    index of exactly 0 (a spotless leaf) or 1 has zero likelihood under most Beta
    densities, so values are clipped to [eps, 1 - eps] first and the number clipped
    is returned: if it is a large share of the sample, a Beta density is the wrong
    family.
    """
    from scipy.stats import beta
    v = np.asarray(values, float)
    clipped = int(np.sum((v < eps) | (v > 1 - eps)))
    v = np.clip(v, eps, 1 - eps)
    a, b, _, _ = beta.fit(v, floc=0, fscale=1)
    return float(a), float(b), clipped


def goodput_summary(bytes_sent, seconds):
    """Goodput per transfer in bits per second, summarised by quantiles.

    Wall-clock seconds for the whole transfer, retries included, divided into the
    payload bits delivered once. The modem's reported bitrate is not goodput. The
    median and the 10th and 90th percentiles are returned: a congested cell has a
    distribution, and the mean hides the hours when it is worst.
    """
    b = np.asarray(bytes_sent, float)
    s = np.asarray(seconds, float)
    g = 8.0 * b / s
    return dict(n=int(g.size), median_bps=float(np.median(g)),
                p10_bps=float(np.percentile(g, 10)), p90_bps=float(np.percentile(g, 90)))


def first_try_success(attempts):
    """Fraction of transfers that completed on the first attempt.

    Under the geometric model this estimates the per-attempt success probability.
    Transfers that never completed are failures, not missing data, and count as such.
    """
    a = np.asarray(attempts, float)
    return float(np.mean(a == 1))


def mean_current(t_s, current_A):
    """Mean current over a capture, by trapezoidal integration over its duration.

    A scope trace is not evenly sampled after decimation, so the current is
    integrated against the time stamps rather than the samples averaged. The 2 A datasheet peak lasts well
    under a millisecond; a mean close to it means the capture is too short or the
    shunt reading is being taken at the peak.
    """
    t = np.asarray(t_s, float)
    i = np.asarray(current_A, float)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(i, t) / (t[-1] - t[0]))


def detection_recall(true_labels, predicted_labels, healthy="healthy"):
    """Share of diseased images the cloud classifier does not call healthy.

    The model needs the probability the cloud acts on a diseased leaf that was sent.
    A black rot leaf called ring spot is still acted on, so this is detection recall,
    not per-class recall and not macro F1. A different definition changes the
    threshold, so it would have to be stated.
    """
    t = np.asarray(true_labels)
    p = np.asarray(predicted_labels)
    sick = t != healthy
    return float(np.mean(p[sick] != healthy)) if sick.any() else float("nan")
