"""Uplink: retransmissions, units, and the sleep budget kept apart."""
import numpy as np

import uplink as U


def test_retransmissions_are_geometric():
    c = U.offload_cost(10000, 16000.0, p_success=0.5, attach=False)
    assert np.isclose(c["attempts"], 2.0)
    assert np.isclose(c["bytes_on_air"], 20000)
    assert np.isclose(c["time_s"], 2 * 8 * 10000 / 16000.0), "bits, not bytes"


def test_a_slower_link_costs_time_not_bytes():
    fast = U.offload_cost(25000, 40000.0)
    slow = U.offload_cost(25000, 10000.0)
    assert np.isclose(fast["bytes_on_air"], slow["bytes_on_air"])
    assert slow["time_s"] > fast["time_s"] and slow["energy_J"] > fast["energy_J"]


def test_sleep_is_kept_apart_from_what_the_design_controls():
    a = U.duty_cycle_cost(10.0, 0.2, 25000, 20000.0, 0.85, 20)
    i0 = U.I_IDLE
    try:
        U.I_IDLE = i0 * 40
        b = U.duty_cycle_cost(10.0, 0.2, 25000, 20000.0, 0.85, 20)
    finally:
        U.I_IDLE = i0
    assert np.isclose(a["controllable_J_per_day"], b["controllable_J_per_day"])
    assert np.isclose(b["idle_J_per_day"], U.V_SUPPLY * i0 * 40 * 86400.0)
    assert np.isclose(a["energy_J_per_day"], a["controllable_J_per_day"] + a["idle_J_per_day"])
