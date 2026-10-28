"""Why the RQ2 null was amended (methodology Amendment 1).

Synthetic check: false-positive rate of two nulls when groups are RANDOM (no real
demographic meaning) but each rater has a stable strictness. Run: python scripts/null_simulation.py
"""
import numpy as np

from wsj.metrics import bernoulli_jsd
from wsj.stats import null_test, rater_permutation_null, same_size_null

rng = np.random.default_rng(0)
n_items, n_raters, k, n_sims, n_null = 150, 100, 12, 200, 300
fp_item, fp_rater = 0, 0
for s in range(n_sims):
    q = rng.uniform(0.05, 0.95, n_items)
    logit = np.log(q / (1 - q))
    offset = rng.normal(0, 1.0, n_raters)            # stable per-rater strictness
    Y = (rng.random((n_raters, n_items)) < 1 / (1 + np.exp(-(logit[None] + offset[:, None])))).astype(float)
    judge = Y.mean(0)                                 # judge = overall human view
    grp = rng.choice(n_raters, k, replace=False)      # a group with NO real demographic meaning
    obs = bernoulli_jsd(judge, Y[grp].mean(0)).mean()
    # (a) per-item resampling (current pre-registered null)
    h_sub, _ = same_size_null([Y[:, i] for i in range(n_items)], np.full(n_items, k), n_null, s)
    fp_item += null_test(obs, bernoulli_jsd(judge, h_sub).mean(1))["p"] < 0.05
    # (b) whole-rater permutation (the amended null, as used in wsj.analysis)
    h_rand, _ = rater_permutation_null(Y, np.ones_like(Y), k, n_null, s)
    fp_rater += null_test(obs, bernoulli_jsd(judge, h_rand).mean(1))["p"] < 0.05
print(f"false-positive rate at alpha=0.05 over {n_sims} random groups:")
print(f"  per-item resampling null (superseded): {fp_item / n_sims:.2f}")
print(f"  whole-rater permutation (amended)   : {fp_rater / n_sims:.2f}")
