"""Stationarity: the condition's root, and two optimisers that must agree."""
import numpy as np

import risk as R
import stationarity as ST


def _slope_in_theta(theta, kw, h=1e-4):
    """dJ/dtheta with the interval re-optimised for this threshold."""
    T, _ = R.opt_interval_given(theta, **kw)
    return (R.risk(theta + h, T, **kw) - R.risk(theta - h, T, **kw)) / (2 * h)


def test_analytic_theta_is_stationary_where_it_is_computed():
    for change in ({}, {"r": 0.30}):
        kw = dict(R.BASE, **change)
        t = ST.analytic_theta(**kw)
        near = abs(_slope_in_theta(t, kw))
        away = abs(_slope_in_theta(t + 0.02, kw))
        assert near < 0.02 * away, (
            "with the interval at its best, the objective should be flat in the threshold "
            "at the threshold the condition gives")


def test_general_optimum_matches_the_grid_optimiser_on_the_base_problem():
    th, T, J = ST.general_optimum(ST.LOSSES["logistic (van der Plank)"])
    o = R.optimise(**R.BASE)
    assert abs(th - o["theta"]) < 1e-3 and abs(T - o["interval_days"]) < 0.1
    assert J <= o["risk"] + 1e-12, "the tighter optimiser should never do worse"
