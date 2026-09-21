"""Screening: the closed forms must agree with their own definitions."""
import numpy as np
import pytest

import screening as S


@pytest.fixture
def pair():
    """Put the module's densities back after a test swaps them."""
    h0, d0 = dict(S.HEALTHY), dict(S.DISEASED)
    yield
    S.HEALTHY.update(h0); S.DISEASED.update(d0)


def test_offload_rate_is_the_mixture_of_the_two_tails():
    for theta in (0.05, 0.2, 0.45):
        op = S.operating_point(theta, 0.3)
        ph, pd = 1 - S.cdf_healthy(theta), 1 - S.cdf_diseased(theta)
        assert np.isclose(op["offload_rate"], 0.3 * pd + 0.7 * ph)
        assert np.isclose(op["p_detect_given_diseased"], pd)
        assert np.isclose(op["miss_rate"] + 0.3 * pd, 0.3), (
            "the miss rate is a share of all captures, so it is weighted by the prior")


def test_raising_the_threshold_never_sends_more():
    q = [S.operating_point(t, 0.15)["offload_rate"] for t in np.linspace(0.01, 0.99, 60)]
    assert np.all(np.diff(q) <= 1e-12)


def test_threshold_for_miss_rate_inverts_the_miss_rate():
    for target in (0.01, 0.03, 0.08):
        t = S.threshold_for_miss_rate(target, 0.15)
        assert abs(S.operating_point(t, 0.15)["miss_rate"] - target) < 1e-8


def test_monotonicity_check_can_say_no(pair):
    S.HEALTHY.update(a=2.0, b=10.0); S.DISEASED.update(a=4.0, b=8.0)
    assert S.is_lr_monotone() is True
    S.HEALTHY.update(a=5.0, b=5.0); S.DISEASED.update(a=1.5, b=1.5)
    assert S.is_lr_monotone() is False, (
        "a diseased density wider than the healthy one on both sides is not monotone")
