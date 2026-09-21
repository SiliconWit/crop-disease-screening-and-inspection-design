"""Bench summaries: each must recover a known answer from synthetic measurements."""
import numpy as np

import bench as B


def test_beta_fit_recovers_known_parameters():
    rng = np.random.default_rng(3)
    v = rng.beta(3.0, 11.0, 20000)
    v[:5] = 0.0
    a, b, clipped = B.fit_beta(v)
    assert abs(a / 3.0 - 1) < 0.06 and abs(b / 11.0 - 1) < 0.06
    assert clipped == 5


def test_mean_current_integrates_over_time_not_samples():
    t = np.concatenate([np.linspace(0, 0.001, 200), np.linspace(0.0011, 1.0, 20)])
    i = np.where(t <= 0.001, 2.0, 0.3)
    m = B.mean_current(t, i)
    assert 0.29 < m < 0.32, "averaging the samples would weight the short peak by its sample count"


def test_goodput_and_first_try_success():
    s = B.goodput_summary([25000, 25000, 25000], [10.0, 20.0, 40.0])
    assert np.isclose(s["median_bps"], 10000.0) and s["n"] == 3
    assert np.isclose(B.first_try_success([1, 2, 1, 1, 3]), 0.6)
    assert np.isclose(B.detection_recall(["healthy", "black rot", "ring spot", "black rot"],
                                         ["black rot", "ring spot", "healthy", "black rot"]),
                      2 / 3)
