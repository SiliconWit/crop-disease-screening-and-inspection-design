"""The objective: the two ways a composed cost goes wrong without saying so."""
import numpy as np

import risk as R
import uplink as U

BASE = dict(prior=0.15, r=0.15, goodput=20000.0, payload=25000, p_success=0.85,
            cloud_recall=0.92)


def test_crop_loss_is_not_spread_over_the_interval():
    w = (R.C_ENERGY, R.C_DATA)
    try:
        R.C_ENERGY, R.C_DATA = 0.0, 0.0
        costs = [R.risk(0.3, T, **BASE) for T in (2.0, 5.0, 10.0, 20.0, 40.0, 100.0, 200.0)]
    finally:
        R.C_ENERGY, R.C_DATA = w
    assert np.all(np.diff(costs) >= -1e-12), (
        "with loss as the only term, looking less often cannot be cheaper")


def test_sleep_current_does_not_enter_the_objective():
    a = R.risk(0.3, 10.0, **BASE)
    i0 = U.I_IDLE
    try:
        U.I_IDLE = i0 * 40
        b = R.risk(0.3, 10.0, **BASE)
    finally:
        U.I_IDLE = i0
    assert np.isclose(a, b, rtol=0, atol=1e-15)


def test_optimise_is_not_beaten_by_a_grid():
    o = R.optimise(**BASE)
    grid = min(R.risk(t, T, **BASE) for t in np.linspace(0.06, 0.58, 27)
               for T in np.linspace(1.5, 44.0, 30))
    assert o["risk"] <= grid + 1e-12
    assert 0.01 <= o["theta"] <= 0.95 and 0.5 <= o["interval_days"] <= 60.0
    b = R.unconditional_baseline(o["interval_days"], **BASE)
    assert b["offload_rate"] == 1.0 and b["bytes_per_day"] > o["bytes_per_day"]
