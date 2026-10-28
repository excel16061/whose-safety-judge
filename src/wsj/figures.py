"""Figures (PNG) built from the CSVs written by `wsj evaluate`."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SHORT = {
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "SmolLM2-1.7B",
}
ORDER = ["constant", "majority", "keyword", *SHORT.values()]
COLORS = {"constant": "#9e9e9e", "majority": "#bdbdbd", "keyword": "#8d6e63",
          "Qwen2.5-0.5B": "#1f77b4", "Qwen2.5-1.5B": "#2a9d8f", "SmolLM2-1.7B": "#d62728"}


_CAPTION = {"text": ""}


def set_caption(text: str):
    """Footer added to every figure saved afterwards (which judges and how many conversations)."""
    _CAPTION["text"] = text


def short(name: str) -> str:
    return SHORT.get(name, name)


def _save(fig, path: Path):
    fig.tight_layout()
    if _CAPTION["text"]:
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.08)
        fig.text(0.5, 0.01, _CAPTION["text"], ha="center", va="bottom", fontsize=7, color="#555555")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def human_distribution(items: pd.DataFrame, out: Path):
    """Histogram of per-item human P(unsafe) for each dataset."""
    ds = sorted(items["dataset"].unique())
    fig, axes = plt.subplots(1, len(ds), figsize=(5 * len(ds), 3.4), sharey=False)
    for ax, d in zip(np.atleast_1d(axes), ds):
        h = items.loc[items["dataset"] == d, "h"]
        ax.hist(h, bins=np.linspace(0, 1, 21), color="#2a9d8f", edgecolor="white")
        ax.set_title(f"{d}: share of raters saying unsafe (n={len(h)})")
        ax.set_xlabel("human P(unsafe) per conversation, Unsure excluded")
        ax.set_ylabel("conversations")
    _save(fig, out)


def overall(overall_df: pd.DataFrame, out: Path):
    """BSS and JSD with 95% bootstrap CIs, per predictor and dataset."""
    df = overall_df.copy()
    df["predictor"] = df["predictor"].map(short)
    ds = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, label in zip(axes, ["bss", "jsd"],
                                 ["Brier skill score vs constant (higher is better)",
                                  "mean Jensen-Shannon divergence (lower is better)"]):
        preds = [p for p in ORDER if p in set(df["predictor"])]
        for j, d in enumerate(ds):
            sub = df[(df["dataset"] == d) & (df["metric"] == metric)].set_index("predictor").reindex(preds)
            y = np.arange(len(preds)) + (j - 0.5 * (len(ds) - 1)) * 0.25
            v = sub["value"].to_numpy()
            lo, hi = sub["ci_lo"].to_numpy(), sub["ci_hi"].to_numpy()
            ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="o", capsize=3, label=d)
        ax.set_yticks(np.arange(len(preds)), preds)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel(label)
        ax.invert_yaxis()
    axes[0].legend(fontsize=8)
    _save(fig, out)


def reliability(items: pd.DataFrame, judges: list[str], out: Path):
    """Judge P(unsafe) against human P(unsafe): scatter plus 10-bin means."""
    ds = sorted(items["dataset"].unique())
    fig, axes = plt.subplots(len(ds), len(judges), figsize=(3.6 * len(judges), 3.4 * len(ds)),
                             squeeze=False)
    bins = np.linspace(0, 1, 11)
    for r, d in enumerate(ds):
        sub = items[items["dataset"] == d]
        for c, j in enumerate(judges):
            ax = axes[r, c]
            if j not in sub or sub[j].isna().any():
                ax.set_visible(False)
                continue
            p, h = sub[j].to_numpy(), sub["h"].to_numpy()
            ax.scatter(p, h, s=6, alpha=0.35, color=COLORS.get(short(j), "C0"))
            idx = np.minimum((p * 10).astype(int), 9)
            xs = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
            ys = [h[idx == b].mean() for b in range(10) if (idx == b).any()]
            ax.plot(xs, ys, "k-o", ms=3, lw=1)
            ax.plot([0, 1], [0, 1], "k--", lw=0.6)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{short(j)} · {d}", fontsize=9)
            ax.set_xlabel("judge P(unsafe)")
            ax.set_ylabel("human P(unsafe)")
    _ = bins
    _save(fig, out)


def disagreement(tertile_df: pd.DataFrame, out: Path):
    """BSS per disagreement tertile with 95% CIs."""
    df = tertile_df[tertile_df["predictor"] != "constant"].copy()
    df["predictor"] = df["predictor"].map(short)
    ds = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(ds), figsize=(5.5 * len(ds), 3.8), squeeze=False)
    for ax, d in zip(axes[0], ds):
        sub = df[df["dataset"] == d]
        preds = [p for p in ORDER if p in set(sub["predictor"])]
        for k, pname in enumerate(preds):
            s = sub[sub["predictor"] == pname].sort_values("tertile")
            x = s["tertile"].to_numpy() + (k - 0.5 * (len(preds) - 1)) * 0.1
            v = s["bss"].to_numpy()
            ax.errorbar(x, v, yerr=[v - s["bss_lo"], s["bss_hi"] - v], fmt="-o", capsize=2,
                        label=pname, color=COLORS.get(pname))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks([0, 1, 2], ["low", "middle", "high"])
        ax.set_xlabel("human disagreement tertile (entropy)")
        ax.set_ylabel("Brier skill score vs constant")
        ax.set_title(d)
        lo = max(-3.0, float(sub["bss_lo"].min()) - 0.1)
        ax.set_ylim(lo, max(0.6, float(sub["bss_hi"].max()) + 0.1))
    axes[0, 0].legend(fontsize=7)
    _save(fig, out)


def groups(group_df: pd.DataFrame, judges: list[str], out: Path):
    """Excess divergence over the same-size random-rater null, per demographic group."""
    keep = [*judges, "constant", "humans:group_vs_rest"]
    df = group_df[group_df["predictor"].isin(keep)].copy()
    df["label"] = df["group_column"] + ": " + df["group"]
    ds = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(ds), figsize=(6.5 * len(ds), 6), squeeze=False)
    for ax, d in zip(axes[0], ds):
        sub = df[df["dataset"] == d]
        labels = list(dict.fromkeys(sub["label"]))
        for k, pname in enumerate(keep):
            s = sub[sub["predictor"] == pname].set_index("label").reindex(labels)
            y = np.arange(len(labels)) + (k - 0.5 * (len(keep) - 1)) * 0.15
            name = "humans (group vs rest)" if pname.startswith("humans") else short(pname)
            ax.errorbar(s["excess"], y,
                        xerr=[s["null_mean"] - s["null_lo"], s["null_hi"] - s["null_mean"]],
                        fmt="o", ms=4, capsize=2, label=name, color=COLORS.get(short(pname), "k"))
            sig = s["p_holm"] < 0.05
            ax.scatter(s.loc[sig, "excess"], y[sig.to_numpy()], marker="*", s=80, color="gold",
                       edgecolor="k", zorder=5)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("observed JSD minus null mean (bars: null 95% range around 0)")
        ax.set_title(d)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    _save(fig, out)


def prompt_sensitivity(items_by_prompt: dict[str, pd.DataFrame], model: str, out: Path):
    """Same model, three prompts: P(unsafe) under P1 and P2 against P0."""
    p0 = items_by_prompt.get("P0")
    others = [k for k in ("P1", "P2") if k in items_by_prompt]
    if p0 is None or not others:
        return False
    fig, axes = plt.subplots(1, len(others), figsize=(4.2 * len(others), 4), squeeze=False)
    for ax, k in zip(axes[0], others):
        a = p0[["dataset", "item_id", model]].merge(
            items_by_prompt[k][["dataset", "item_id", model]], on=["dataset", "item_id"], suffixes=("_P0", f"_{k}"))
        for d, s in a.groupby("dataset"):
            ax.scatter(s[f"{model}_P0"], s[f"{model}_{k}"], s=6, alpha=0.4, label=d)
        ax.plot([0, 1], [0, 1], "k--", lw=0.6)
        ax.set_xlabel("P(unsafe), prompt P0")
        ax.set_ylabel(f"P(unsafe), prompt {k}")
        ax.set_title(short(model))
        ax.legend(fontsize=7)
    _save(fig, out)
    return True
