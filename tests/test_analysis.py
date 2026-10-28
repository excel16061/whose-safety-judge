import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES
from wsj import analysis as A
from wsj.data import items_table, load_raw

FAST = A.Settings(n_boot=200, n_perm=500, n_null=200, seed=1)


def _setup(dataset="350", cfg=A.CONFIGS[0]):
    raw = load_raw(FIXTURES / f"diverse_safety_adversarial_dialog_{dataset}.csv")
    other = "990" if dataset == "350" else "350"
    other_t = A.prepare(load_raw(FIXTURES / f"diverse_safety_adversarial_dialog_{other}.csv"), other, cfg)[1]
    return raw, items_table(raw), other_t


def test_perfect_judge_scores_perfectly():
    raw, items, other_t = _setup()
    _, targets = A.prepare(raw, "350", A.CONFIGS[0])
    perfect = dict(zip(targets["item_id"], targets["h"]))
    res = A.analyse_dataset(raw, items, "350", A.CONFIGS[0], other_t, {"perfect": perfect}, FAST)
    o = res["overall"].set_index(["predictor", "metric"])["value"]
    assert o[("perfect", "jsd")] == pytest.approx(0.0, abs=1e-12)
    assert o[("perfect", "bss")] == pytest.approx(1.0)
    assert len(res["targets"]) == 7  # the all-Unsure item is excluded
    assert set(res["preds"]) == {"constant", "majority", "keyword", "perfect"}


def test_judge_missing_items_is_skipped_and_reported():
    raw, items, other_t = _setup()
    res = A.analyse_dataset(raw, items, "350", A.CONFIGS[0], other_t, {"partial": {1: 0.5}}, FAST)
    assert res["missing_judges"] == ["partial"]


def test_outputs_have_expected_structure():
    raw, items, other_t = _setup("990")
    _, targets = A.prepare(raw, "990", A.CONFIGS[0])
    judge = {i: 0.5 for i in targets["item_id"]}
    res = A.analyse_dataset(raw, items, "990", A.CONFIGS[0], other_t, {"j": judge}, FAST)
    assert set(res["tertiles"]["tertile"]) == {0, 1, 2}
    assert set(res["tests"]["predictor"]) == {"majority", "keyword", "j"}
    g = res["groups"]
    assert (g["n_raters"] >= 10).all()
    assert {"humans:group_vs_rest", "j"} <= set(g["predictor"])
    # constant baseline uses the other dataset's mean
    assert res["constant_value"] == pytest.approx(other_t["h"].mean())


def test_tertiles_are_balanced_and_rank_based():
    t = A.tertiles(np.array([0.9, 0.1, 0.5, 0.5, 0.2, 0.8]))
    assert np.bincount(t).tolist() == [2, 2, 2]
    assert t[1] == 0 and t[0] == 2


def _synthetic_ratings(n_items=60, seed=0, deviant=True):
    """40 raters; 'A' group (10 raters) says the opposite of everyone else if deviant."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.05, 0.95, n_items)
    rows = []
    for r in range(40):
        grp = "A" if r < 10 else "B"
        for i in range(n_items):
            yes = rng.random() < ((1 - q[i]) if (deviant and grp == "A") else q[i])
            rows.append({"item_id": i, "rater_id": str(r), "label": "Yes" if yes else "No",
                         "gender": grp, "race": "x", "age": "x", "education": "x"})
    return pd.DataFrame(rows), q


def test_group_null_flags_a_deviant_group_but_not_a_random_one():
    for deviant, expect_sig in [(True, True), (False, False)]:
        ratings, q = _synthetic_ratings(deviant=deviant)
        from wsj.data import human_targets
        targets = human_targets(ratings)
        preds = {"judge": targets["h"].to_numpy()}  # judge = overall human view
        g = A.group_tables(ratings, targets, preds, "350", A.CONFIGS[0], FAST)
        a = g[(g["group"] == "A") & (g["predictor"] == "judge")].iloc[0]
        if expect_sig:
            assert a["excess"] > 0 and a["p"] < 0.01
        else:
            assert a["p"] > 0.01


def test_disagreement_test_detects_worse_high_entropy_items():
    n = 90
    h = np.concatenate([np.full(30, 0.02), np.full(30, 0.25), np.full(30, 0.5)])
    targets = pd.DataFrame({"item_id": range(n), "h": h, "m": (h >= 0.5).astype(int)})
    from wsj.metrics import binary_entropy
    targets["d"] = binary_entropy(h)
    judge = np.where(h == 0.5, 0.95, h)  # exact on easy items, far off on contested ones
    preds = {"constant": np.full(n, 0.3), "judge": judge}
    _, tests = A.disagreement_tables(targets, preds, FAST)
    row = tests[tests["predictor"] == "judge"].iloc[0]
    assert row["bss_top_minus_bottom"] < 0 and row["p_perm"] < 0.01


def test_group_null_is_not_fooled_by_stable_rater_strictness():
    """Amendment 1: random groups of raters with stable strictness must not look significant."""
    rng = np.random.default_rng(3)
    n_items, n_raters = 60, 40
    q = rng.uniform(0.1, 0.9, n_items)
    offset = rng.normal(0, 1.5, n_raters)
    logit = np.log(q / (1 - q))
    rows = []
    for r in range(n_raters):
        for i in range(n_items):
            yes = rng.random() < 1 / (1 + np.exp(-(logit[i] + offset[r])))
            rows.append({"item_id": i, "rater_id": str(r), "label": "Yes" if yes else "No",
                         "gender": "A" if r % 4 == 0 else "B", "race": "x", "age": "x", "education": "x"})
    ratings = pd.DataFrame(rows)
    from wsj.data import human_targets
    targets = human_targets(ratings)
    g = A.group_tables(ratings, targets, {"judge": targets["h"].to_numpy()}, "350", A.CONFIGS[0], FAST)
    a = g[(g["group"] == "A") & (g["predictor"] == "judge")].iloc[0]
    assert a["p"] > 0.05
