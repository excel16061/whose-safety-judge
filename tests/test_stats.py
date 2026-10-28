import numpy as np
import pytest

from wsj import stats as S


def test_holm_known_example():
    assert S.holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    out = S.holm([0.2, np.nan])
    assert out[0] == pytest.approx(0.2) and np.isnan(out[1])


def test_bootstrap_ci_is_deterministic_and_covers_mean():
    x = np.random.default_rng(0).normal(5, 1, 200)
    idx = S.bootstrap_indices(len(x), 500, seed=1)
    point, (lo, hi) = S.bootstrap_ci(np.mean, [x], idx)
    assert lo < point < hi and lo < 5 < hi
    assert S.bootstrap_ci(np.mean, [x], S.bootstrap_indices(len(x), 500, seed=1))[1] == (lo, hi)


def test_permutation_test_detects_difference_and_not_noise():
    rng = np.random.default_rng(0)
    ec = np.full(60, 0.1)
    # same judge error distribution in both groups -> large p
    _, p_same = S.permutation_test_bss_diff(rng.uniform(0, 0.1, 60), ec, rng.uniform(0, 0.1, 60), ec, 2000, 1)
    # much worse judge in the top group -> small p, negative difference
    diff, p_diff = S.permutation_test_bss_diff(np.full(60, 0.3), ec, np.full(60, 0.01), ec, 2000, 1)
    assert p_same > 0.05
    assert diff < 0 and p_diff < 0.01


def test_same_size_null_shapes_and_edge_cases():
    labels = [np.array([1.0, 1.0, 0.0, 0.0]), np.array([1.0, np.nan, 0.0])]
    h_sub, h_rest = S.same_size_null(labels, np.array([2, 3]), n_draws=200, seed=0)
    assert h_sub.shape == (200, 2)
    # item 0: subsets of 2 from {1,1,0,0} average to 0.5 over draws
    assert np.nanmean(h_sub[:, 0]) == pytest.approx(0.5, abs=0.08)
    # item 1: the whole item is the subset, so "rest" is empty -> NaN; subset is always 1/2
    assert np.all(np.isnan(h_rest[:, 1])) and np.allclose(h_sub[:, 1], 0.5)


def test_null_test_p_value():
    null = np.random.default_rng(0).normal(0.2, 0.01, 999)
    assert S.null_test(0.2, null)["p"] > 0.5
    far = S.null_test(0.5, null)
    assert far["p"] == pytest.approx(1 / 1000) and far["excess"] == pytest.approx(0.3, abs=0.01)


def test_rater_permutation_null_uses_whole_raters():
    # rater 0 always says Yes, rater 1 always No; rater 2 rated only item 0
    num = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    den = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    h_sub, h_rest = S.rater_permutation_null(num, den, k=1, n_draws=300, seed=0)
    assert h_sub.shape == (300, 2)
    # a single whole rater is drawn, so each draw is that rater's own labels
    picked_strict = h_sub[:, 1] == 1.0
    assert np.all(h_sub[picked_strict, 0] == 1.0)
    # drawing rater 2 leaves item 1 undefined for the subset and uses raters 0+1 for the rest
    only_r2 = np.isnan(h_sub[:, 1])
    assert only_r2.any() and np.allclose(h_rest[only_r2, 1], 0.5)
    with pytest.raises(ValueError):
        S.rater_permutation_null(num, den, k=3, n_draws=10, seed=0)


def test_holm_padding_keeps_preregistered_family_size():
    import pandas as pd
    from wsj.analysis import apply_holm_across
    t = pd.DataFrame({"predictor": ["j", "j"], "p_perm": [0.01, 0.04]})
    padded = apply_holm_across([t], ["j"], family_size=6)["p_holm"].tolist()
    assert padded == pytest.approx([0.06, 0.2])  # 6*0.01, then max(0.06, 5*0.04)
    assert apply_holm_across([t], ["j"])["p_holm"].tolist() == pytest.approx([0.02, 0.04])
