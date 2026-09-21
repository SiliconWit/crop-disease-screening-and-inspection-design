"""How much rests on the two assumed class-conditional densities.

Every operating point in the study is computed from two Beta densities that were
chosen, not fitted. This module swaps other pairs in, re-runs the design, and
summarises each pair's overlap in one number so that the results can be plotted
against something a reader can interpret.
"""
import contextlib
import numpy as np
import screening as S
import risk as R


def overlap(h, d):
    """Bhattacharyya coefficient between two Beta densities, each given as dict(a=, b=).

    The integral of sqrt(f_h f_d) over [0, 1]: one for identical densities, near zero
    for densities that barely touch. Integrated numerically on a fine grid that stays
    clear of the end points, where a Beta density with a parameter below one is
    infinite.
    """
    from scipy.stats import beta
    trap = getattr(np, "trapezoid", None) or np.trapz
    g = np.linspace(1e-4, 1 - 1e-4, 4000)
    return float(trap(np.sqrt(beta.pdf(g, **h) * beta.pdf(g, **d)), g))


@contextlib.contextmanager
def swapped(h, d):
    """Temporarily replace the densities in screening.py, and always put them back.

    Everything in screening.py reads HEALTHY and DISEASED at call time, so updating
    those dicts in place is enough. They are restored in a finally block, so that
    an exception part-way through a sweep cannot leave the next computation on the
    wrong pair.
    """
    h0, d0 = dict(S.HEALTHY), dict(S.DISEASED)
    S.HEALTHY.update(h); S.DISEASED.update(d)
    try:
        yield
    finally:
        S.HEALTHY.update(h0); S.DISEASED.update(d0)


def with_densities(h, d):
    """The joint optimum and the 2 percent operating point under another pair.

    Records whether the optimum sits on a bound of the search. Where the densities
    overlap heavily, screening can become nearly useless and the optimiser can run
    to the edge of the region it is allowed to search; such a point is a corner,
    and its threshold is set by the bound, not by the densities.
    """
    BASE = R.BASE
    with swapped(h, d):
        o = R.optimise(**BASE)
        base = R.unconditional_baseline(o["interval_days"], **BASE)
        th2 = S.threshold_for_miss_rate(R.TARGET_MISS, BASE["prior"])
        i2, _ = R.opt_interval_given(th2, **BASE)
        op2 = R.risk(th2, i2, **BASE, verbose=True)
        b2 = R.unconditional_baseline(i2, **BASE)
        return dict(bhattacharyya=overlap(dict(a=h["a"], b=h["b"]),
                                          dict(a=d["a"], b=d["b"])),
                    healthy=h, diseased=d,
                    theta=o["theta"], interval=o["interval_days"],
                    on_bound=o["on_bound"],
                    miss=o["miss_rate"], offload=o["offload_rate"],
                    delay=o["delay_days"], risk=o["risk"],
                    volume_cut=1 - o["bytes_per_day"] / base["bytes_per_day"],
                    at_target_theta=th2, at_target_interval=i2,
                    at_target_offload=op2["offload_rate"],
                    at_target_delay=op2["delay_days"],
                    at_target_volume_cut=1 - op2["bytes_per_day"] / b2["bytes_per_day"],
                    at_target_extra_delay=op2["delay_days"] - b2["delay_days"])
