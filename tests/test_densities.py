"""Densities: an overlap that means what it says, and a swap that always undoes itself."""
import numpy as np
import pytest

import screening as S
import densities as D


def test_overlap_is_one_for_identical_densities_and_symmetric():
    a, b = dict(a=2.0, b=18.0), dict(a=5.0, b=9.0)
    assert np.isclose(D.overlap(a, a), 1.0, atol=1e-3)
    assert np.isclose(D.overlap(a, b), D.overlap(b, a))
    assert 0.0 < D.overlap(a, b) < 1.0


def test_swapped_puts_the_originals_back_even_after_an_error():
    h0, d0 = dict(S.HEALTHY), dict(S.DISEASED)
    seen = []
    with pytest.raises(KeyError):
        with D.swapped(dict(a=3.0, b=3.0), dict(a=4.0, b=2.0)):
            seen.append((S.HEALTHY["a"], S.DISEASED["b"]))
            raise KeyError("part-way through a sweep")
    assert seen == [(3.0, 2.0)], "inside the block the swapped pair must be in force"
    assert S.HEALTHY == h0 and S.DISEASED == d0
