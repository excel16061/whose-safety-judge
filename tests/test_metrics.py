import math

import numpy as np
import pytest

from wsj import metrics as M


def test_binary_entropy_known_values():
    assert M.binary_entropy([0.5])[0] == pytest.approx(1.0)
    assert M.binary_entropy([0.0, 1.0]).tolist() == [0.0, 0.0]
    assert math.isnan(M.binary_entropy([np.nan])[0])


def test_jsd_identity_extremes_and_symmetry():
    assert M.bernoulli_jsd(0.3, 0.3) == pytest.approx(0.0, abs=1e-12)
    assert M.bernoulli_jsd(0.0, 1.0) == pytest.approx(1.0)
    assert M.bernoulli_jsd(0.2, 0.7) == pytest.approx(M.bernoulli_jsd(0.7, 0.2))


def test_jsd_hand_computed():
    # M = Bern(0.25); JSD = H(0.25) - (H(0.5) + H(0)) / 2
    h025 = -(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75))
    assert M.bernoulli_jsd(0.5, 0.0) == pytest.approx(h025 - 0.5)


def test_jsd_matches_scipy_definition():
    from scipy.spatial.distance import jensenshannon
    for p, q in [(0.1, 0.9), (0.33, 0.5), (0.99, 0.01)]:
        ref = jensenshannon([p, 1 - p], [q, 1 - q], base=2) ** 2
        assert M.bernoulli_jsd(p, q) == pytest.approx(ref, abs=1e-9)


def test_brier_and_skill():
    h = np.array([0.0, 0.5, 1.0])
    assert M.brier([0.0, 0.5, 1.0], h) == 0.0
    assert M.brier([1.0, 0.5, 0.0], h) == pytest.approx(2 / 3)
    ref = np.full(3, 0.5)
    assert M.brier_skill(h, h, ref) == pytest.approx(1.0)
    assert M.brier_skill(ref, h, ref) == pytest.approx(0.0)


def test_ece():
    assert M.ece([0.05, 0.05, 0.95, 0.95], [0.0, 0.1, 1.0, 0.9]) == pytest.approx(0.0)
    assert M.ece([0.2, 0.2], [0.6, 0.6]) == pytest.approx(0.4)
    # p == 1.0 lands in the top bin instead of overflowing
    assert M.ece([1.0], [1.0]) == pytest.approx(0.0)


def test_rank_metrics_undefined_for_constant_predictor():
    assert math.isnan(M.spearman([0.3, 0.3, 0.3], [0.1, 0.5, 0.9]))
    assert math.isnan(M.auroc([0.3, 0.3], [0, 1]))
    assert M.auroc([0.1, 0.9], [0, 1]) == 1.0


def test_kappa():
    assert M.kappa([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert M.kappa([0.1, 0.9, 0.2, 0.8], [1, 0, 1, 0]) == pytest.approx(-1.0)
