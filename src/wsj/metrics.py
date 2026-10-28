"""Metric definitions (docs/methodology.md, section 6). All inputs are 1-D arrays."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def _xlog2(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, x * np.log2(np.where(x > 0, x, 1.0)), 0.0)


def binary_entropy(p) -> np.ndarray:
    """H(p) in bits; 0 at p in {0, 1}; NaN stays NaN."""
    p = np.asarray(p, dtype=float)
    out = -(_xlog2(p) + _xlog2(1 - p))
    return np.where(np.isnan(p), np.nan, out)


def bernoulli_jsd(p, q) -> np.ndarray:
    """Element-wise Jensen-Shannon divergence between Bernoulli(p) and Bernoulli(q), base 2, in [0, 1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    # JSD = H(M) - (H(P) + H(Q)) / 2
    out = binary_entropy(m) - 0.5 * (binary_entropy(p) + binary_entropy(q))
    return np.clip(out, 0.0, 1.0)


def mean_jsd(p, h) -> float:
    return float(np.mean(bernoulli_jsd(p, h)))


def brier(p, h) -> float:
    p, h = np.asarray(p, float), np.asarray(h, float)
    return float(np.mean((p - h) ** 2))


def brier_skill(p, h, reference) -> float:
    """1 - Brier(p) / Brier(reference). Positive = better than the reference."""
    ref = brier(reference, h)
    return float(1.0 - brier(p, h) / ref) if ref > 0 else float("nan")


def ece(p, h, n_bins: int = 10) -> float:
    """Expected calibration error against a soft target, equal-width bins on p."""
    p, h = np.asarray(p, float), np.asarray(h, float)
    # bin index in 0..n_bins-1; p == 1 goes into the last bin
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(p[sel].mean() - h[sel].mean())
    return float(total)


def _is_constant(p) -> bool:
    return float(np.ptp(np.asarray(p, float))) == 0.0


def spearman(p, h) -> float:
    if _is_constant(p) or _is_constant(h):
        return float("nan")
    return float(spearmanr(p, h).statistic)


def auroc(p, m) -> float:
    m = np.asarray(m)
    if _is_constant(p) or len(np.unique(m)) < 2:
        return float("nan")
    return float(roc_auc_score(m, p))


def kappa(p, m) -> float:
    pred = (np.asarray(p, float) >= 0.5).astype(int)
    m = np.asarray(m, int)
    if len(np.unique(pred)) < 2 and len(np.unique(m)) < 2 and pred[0] == m[0]:
        return 1.0
    return float(cohen_kappa_score(pred, m))


def all_metrics(p, h, m, reference) -> dict:
    return {
        "jsd": mean_jsd(p, h),
        "brier": brier(p, h),
        "bss": brier_skill(p, h, reference),
        "ece": ece(p, h),
        "spearman": spearman(p, h),
        "auroc": auroc(p, m),
        "kappa": kappa(p, m),
    }
