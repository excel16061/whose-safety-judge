"""Analyses for RQ1-RQ3 (docs/methodology.md, sections 6-8)."""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import SEED
from .baselines import constant_baseline, keyword_rule, majority_baseline
from .data import (GROUP_COLUMNS, MIN_GROUP_RATERS, analysable_groups, human_targets,
                   label_counts, ratings_table, soft_label)
from .metrics import all_metrics, bernoulli_jsd
from .stats import (bootstrap_indices, bss_from_errors, holm, mean_jsd_rows, null_test,
                    percentile_ci, permutation_test_bss_diff, rater_permutation_null)

METRICS = ["jsd", "brier", "bss", "ece", "spearman", "auroc", "kappa"]


@dataclass(frozen=True)
class Config:
    """One analysis setting. 'primary' is the pre-registered main analysis."""
    name: str
    filter_raters: bool = True
    unsure: str = "drop"


CONFIGS = [
    Config("primary"),
    Config("sens_unsure_half", unsure="half"),
    Config("sens_all_raters", filter_raters=False),
]


@dataclass
class Settings:
    n_boot: int = 2000
    n_perm: int = 10000
    n_null: int = 1000
    seed: int = SEED


def prepare(raw: pd.DataFrame, dataset: str, cfg: Config):
    ratings = ratings_table(raw, dataset, filter_raters=cfg.filter_raters)
    targets = human_targets(ratings, unsure=cfg.unsure)
    targets = targets[targets["h"].notna()].sort_values("item_id").reset_index(drop=True)
    return ratings, targets


def baseline_inputs(other_targets: pd.DataFrame) -> tuple[float, float]:
    return float(other_targets["h"].mean()), float(other_targets["m"].mean())


def build_predictions(targets, items, judge_scores: dict[str, dict[int, float]],
                      other_mean_h: float, other_majority_rate: float) -> dict[str, np.ndarray]:
    """Aligned prediction arrays. Judges missing any item are skipped (and reported by the caller)."""
    n = len(targets)
    resp = items.set_index("item_id").loc[targets["item_id"], "response"]
    preds = {
        "constant": constant_baseline(n, other_mean_h),
        "majority": majority_baseline(n, other_majority_rate),
        "keyword": keyword_rule(resp.tolist()),
    }
    for name, scores in judge_scores.items():
        if all(i in scores for i in targets["item_id"]):
            preds[name] = np.array([scores[i] for i in targets["item_id"]])
    return preds


def overall_table(targets, preds, settings: Settings) -> pd.DataFrame:
    h, m = targets["h"].to_numpy(), targets["m"].to_numpy()
    ref = preds["constant"]
    idx = bootstrap_indices(len(h), settings.n_boot, settings.seed)
    rows = []
    for name, p in preds.items():
        point = all_metrics(p, h, m, ref)
        boots = {k: [] for k in METRICS}
        for r in idx:
            b = all_metrics(p[r], h[r], m[r], ref[r])
            for k in METRICS:
                boots[k].append(b[k])
        for k in METRICS:
            lo, hi = percentile_ci(boots[k])
            rows.append({"predictor": name, "metric": k, "value": point[k], "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def tertiles(d: np.ndarray) -> np.ndarray:
    """0 = lowest disagreement, 2 = highest. Rank-based; ties broken by position (item order)."""
    rank = pd.Series(d).rank(method="first").to_numpy() - 1
    return np.floor(rank * 3 / len(d)).astype(int)


def disagreement_tables(targets, preds, settings: Settings):
    h, d = targets["h"].to_numpy(), targets["d"].to_numpy()
    t = tertiles(d)
    ref = preds["constant"]
    err_c = (ref - h) ** 2
    rng_seed = settings.seed
    per_tertile, tests = [], []
    for name, p in preds.items():
        err_j = (p - h) ** 2
        for k in range(3):
            sel = np.where(t == k)[0]
            idx = bootstrap_indices(len(sel), settings.n_boot, rng_seed + k)
            bss = bss_from_errors(err_j[sel], err_c[sel])
            boots = [bss_from_errors(err_j[sel][r], err_c[sel][r]) for r in idx]
            lo, hi = percentile_ci(boots)
            per_tertile.append({
                "predictor": name, "tertile": k, "n_items": len(sel),
                "entropy_min": float(d[sel].min()), "entropy_max": float(d[sel].max()),
                "mean_h": float(h[sel].mean()),
                "bss": bss, "bss_lo": lo, "bss_hi": hi,
                "brier": float(err_j[sel].mean()),
                "jsd": float(bernoulli_jsd(p[sel], h[sel]).mean()),
            })
        if name == "constant":
            continue
        top, bot = np.where(t == 2)[0], np.where(t == 0)[0]
        diff, pval = permutation_test_bss_diff(err_j[top], err_c[top], err_j[bot], err_c[bot],
                                               settings.n_perm, settings.seed)
        # CI for the difference: independent bootstrap within each tertile
        it = bootstrap_indices(len(top), settings.n_boot, settings.seed + 10)
        ib = bootstrap_indices(len(bot), settings.n_boot, settings.seed + 11)
        diffs = [bss_from_errors(err_j[top][a], err_c[top][a]) - bss_from_errors(err_j[bot][b], err_c[bot][b])
                 for a, b in zip(it, ib)]
        lo, hi = percentile_ci(diffs)
        rho = spearmanr(d, err_j - err_c)
        tests.append({"predictor": name, "bss_top_minus_bottom": diff, "ci_lo": lo, "ci_hi": hi,
                      "p_perm": pval, "spearman_entropy_vs_excess_error": float(rho.statistic),
                      "spearman_p": float(rho.pvalue)})
    return pd.DataFrame(per_tertile), pd.DataFrame(tests)


def _rater_matrices(ratings, item_ids, unsure: str):
    """(n_raters, n_items) numerator/denominator of P(unsafe), matching data.soft_label."""
    lab = ratings["label"]
    if unsure == "drop":
        num, den = (lab == "Yes").astype(float), lab.isin(["Yes", "No"]).astype(float)
    else:
        num = (lab == "Yes").astype(float) + 0.5 * (lab == "Unsure")
        den = lab.isin(["Yes", "No", "Unsure"]).astype(float)
    df = pd.DataFrame({"r": ratings["rater_id"], "i": ratings["item_id"], "num": num, "den": den})
    wide = df.pivot_table(index="r", columns="i", values=["num", "den"], aggfunc="sum", fill_value=0.0)
    num_m = wide["num"].reindex(columns=item_ids, fill_value=0.0)
    den_m = wide["den"].reindex(columns=item_ids, fill_value=0.0)
    return num_m.index.to_numpy(), num_m.to_numpy(), den_m.to_numpy()


def group_tables(ratings, targets, preds, dataset: str, cfg: Config, settings: Settings) -> pd.DataFrame:
    """RQ2: per group, divergence of each predictor from the group vs random groups of as many whole raters."""
    item_ids = targets["item_id"].to_numpy()
    rater_ids, num, den = _rater_matrices(ratings, item_ids, cfg.unsure)
    h_all = targets["h"].to_numpy()
    rows = []
    gseed = settings.seed
    n_total = ratings["rater_id"].nunique()
    for col, val, n_raters in analysable_groups(ratings, dataset):
        if n_raters < MIN_GROUP_RATERS or n_raters == n_total:
            continue  # too small, or the "group" is everyone
        gseed += 1
        in_g = ratings[col] == val
        gc = label_counts(ratings[in_g]).reindex(item_ids, fill_value=0)
        rc = label_counts(ratings[~in_g]).reindex(item_ids, fill_value=0)
        h_g = soft_label(gc["Y"], gc["N"], gc["U"], cfg.unsure)
        h_r = soft_label(rc["Y"], rc["N"], rc["U"], cfg.unsure)
        sizes = (gc["Y"] + gc["N"] + gc["U"]).to_numpy().astype(int)
        ok = ~np.isnan(h_g)
        h_sub, h_rest = rater_permutation_null(num[:, ok], den[:, ok], n_raters, settings.n_null, gseed)
        base = {"dataset": f"DICES-{dataset}", "group_column": col, "group": str(val),
                "n_raters": n_raters, "n_items": int(ok.sum()),
                "mean_raters_per_item": float(sizes[ok].mean()),
                "group_mean_h": float(np.nanmean(h_g)), "all_mean_h": float(h_all[ok].mean())}
        # human vs human context
        hh = float(np.nanmean(bernoulli_jsd(h_g[ok], h_r[ok])))
        hh_null = np.nanmean(bernoulli_jsd(h_sub, h_rest), axis=1)
        rows.append({**base, "predictor": "humans:group_vs_rest", **null_test(hh, hh_null),
                     "mean_signed_gap": float(np.nanmean(h_r[ok] - h_g[ok]))})
        for name, p in preds.items():
            obs = float(np.mean(bernoulli_jsd(p[ok], h_g[ok])))
            nul = mean_jsd_rows(p[ok], h_sub)
            rows.append({**base, "predictor": name, **null_test(obs, nul),
                         "mean_signed_gap": float(np.mean(p[ok] - h_g[ok]))})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = np.nan
        for name, g in df.groupby("predictor"):
            df.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
    return df


def analyse_dataset(raw, items, dataset, cfg, other_targets, judge_scores, settings: Settings, item_ids=None):
    """All RQ1-RQ3 tables for one dataset; item_ids restricts the analysis to those conversations."""
    ratings, targets = prepare(raw, dataset, cfg)
    if item_ids is not None:
        targets = targets[targets["item_id"].isin(set(item_ids))].reset_index(drop=True)
    mean_h, maj_rate = baseline_inputs(other_targets)
    preds = build_predictions(targets, items, judge_scores, mean_h, maj_rate)
    missing = [j for j in judge_scores if j not in preds]
    overall = overall_table(targets, preds, settings)
    per_t, tests = disagreement_tables(targets, preds, settings)
    groups = group_tables(ratings, targets, preds, dataset, cfg, settings)
    for df in (overall, per_t, tests):
        df.insert(0, "dataset", f"DICES-{dataset}")
    return {"targets": targets, "preds": preds, "overall": overall, "tertiles": per_t,
            "tests": tests, "groups": groups, "missing_judges": missing,
            "constant_value": mean_h, "majority_rate_other": maj_rate}


def apply_holm_across(test_tables: list[pd.DataFrame], judge_names: list[str],
                      family_size: int | None = None) -> pd.DataFrame:
    """Holm over the pre-registered family: judges (not baselines) x datasets.

    family_size: number of tests in the pre-registered family. Tests not run yet count as
    p = 1, so the correction is never weaker than planned when some judges are missing.
    """
    df = pd.concat(test_tables, ignore_index=True)
    fam = df["predictor"].isin(judge_names)
    p = df.loc[fam, "p_perm"].to_numpy()
    pad = max(0, (family_size or 0) - len(p))
    df["p_holm"] = np.nan
    df.loc[fam, "p_holm"] = holm(np.concatenate([p, np.ones(pad)]))[:len(p)]
    return df


def to_json(obj) -> str:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    return json.dumps(obj, indent=2, default=default)
