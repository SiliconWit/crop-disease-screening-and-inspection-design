"""Epidemic: no overflow, a true inverse, and a delay that behaves at its limits."""
import numpy as np

import epidemic as E


def test_severity_does_not_overflow():
    x = E.severity(np.array([0.0, 50.0, 1e4]), x0=1e-3, r=0.45)
    assert np.all(np.isfinite(x)), "the textbook form returns nan for large r t"
    assert np.isclose(x[0], 1e-3) and np.isclose(x[-1], 1.0)


def test_time_to_severity_inverts_severity():
    for r in (0.1, 0.3):
        t = E.time_to_severity(0.05, r=r)
        assert np.isclose(E.severity(t, r=r), 0.05)


def test_detection_delay_at_its_limits():
    assert np.isclose(E.detection_delay(10.0, 1.0), 5.0), "half an interval when every look finds it"
    assert np.isclose(E.detection_delay(20.0, 0.5), 2 * E.detection_delay(10.0, 0.5))
    assert np.isinf(E.detection_delay(10.0, 0.0))
    assert E.miss_cost(1e5, r=0.45) <= 1.0
