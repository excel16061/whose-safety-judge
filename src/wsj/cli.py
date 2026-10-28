"""Command-line entry point: `wsj download|clean|judge|evaluate|figures|all`."""

import argparse
import gc
import json
import sys
import time

import numpy as np
import pandas as pd

from . import SEED
from .analysis import CONFIGS, Config, Settings, analyse_dataset, apply_holm_across, prepare, to_json
from .data import CSV_NAME, cleaning_report, items_table, load_raw
from .download import download_all, manual_instructions
from .judges import MODELS, SENSITIVITY_MODEL, HFBackend, ScoreCache, get_prompt, load_scores, score_items
from .paths import default_paths

DATASETS = ("350", "990")
CACHE_FILE = "judge_scores.jsonl"
SUBSET_PER_DATASET = 150


def _load(paths, dataset):
    path = paths.raw / CSV_NAME[dataset]
    if not path.exists():
        sys.exit(f"{path} not found. Run `wsj download` first.\n\n{manual_instructions(paths.raw)}")
    return load_raw(path)


def cmd_download(args, paths):
    problems = download_all(paths.raw)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        print(manual_instructions(paths.raw), file=sys.stderr)
        return 1
    print(f"All files present and verified in {paths.raw}")
    return 0


def cmd_clean(args, paths):
    for d in args.datasets:
        rep = cleaning_report(_load(paths, d), d, paths.raw / f"README_{d}.md")
        (paths.results / f"cleaning_report_{d}.json").write_text(to_json(rep))
        print(f"DICES-{d}: {rep['items']} items, raters {rep['raters_raw']} -> {rep['raters_after_filter']}, "
              f"labels (filtered) {rep['label_counts_filtered']}, "
              f"all-Unsure items excluded {rep['items_excluded_all_unsure']}")
    return 0


def judge_plan(models, prompts):
    """(model, prompt) pairs. Default: every model with P0, plus P1/P2 for the sensitivity model."""
    if models or prompts:
        return [(m, p) for m in (models or list(MODELS)) for p in (prompts or ["P0"])]
    return [(m, "P0") for m in MODELS] + [(SENSITIVITY_MODEL, "P1"), (SENSITIVITY_MODEL, "P2")]


def subset_ids(items, dataset: str, per_dataset: int | None = None, seed: int = SEED) -> list[int]:
    """Fixed-seed random subset of a dataset's conversations (README, Deviations from plan, item 2)."""
    per_dataset = SUBSET_PER_DATASET if per_dataset is None else per_dataset
    ids = np.sort(items["item_id"].to_numpy())
    rng = np.random.default_rng([seed, int(dataset)])
    return sorted(int(i) for i in rng.choice(ids, size=min(per_dataset, len(ids)), replace=False))


def cmd_judge(args, paths):
    cache = ScoreCache(paths.cache / CACHE_FILE)
    backend = None
    for model, prompt_id in judge_plan(args.models, args.prompts):
        if backend is None or backend.name != model:
            backend = None  # free the previous model before loading the next one
            gc.collect()
            print(f"loading {model} @ {MODELS[model][:10]}", flush=True)
            backend = HFBackend(model, MODELS[model], threads=args.threads)
        for d in args.datasets:
            items = items_table(_load(paths, d))
            if args.limit:
                items = items.head(args.limit)
            if args.subset:
                items = items[items["item_id"].isin(subset_ids(items, d))]
            t0 = time.time()

            def progress(k, n):
                if k % 50 == 0 or k == n:
                    print(f"  {model} {prompt_id} DICES-{d}: {k}/{n} ({time.time() - t0:.0f}s)", flush=True)

            _, s = score_items(backend, get_prompt(prompt_id), items, d, cache,
                               run_log=paths.results / "run_log.jsonl", progress=progress)
            per = s.seconds / max(s.n_scored, 1)
            print(f"done {model} {prompt_id} DICES-{d}: {s.n_scored} scored, {s.n_cached} cached, "
                  f"{s.seconds:.1f}s ({per:.2f}s per new item)", flush=True)
    return 0


def _item_frame(res, dataset):
    t = res["targets"][["item_id", "Y", "N", "U", "h", "m", "d"]].copy()
    for name, p in res["preds"].items():
        t[name] = p
    t.insert(0, "dataset", f"DICES-{dataset}")
    return t


def evaluation_scopes(scores, targets, items, models):
    """Item sets to analyse, as (run-dir suffix, scope label, {dataset: item_ids or None}, judges).

    Judges scored on every conversation are analysed on all of them. Judges scored only on
    the fixed-seed subset are analysed on the part of it they have scored, together with every
    judge that also covers those conversations, so judges are compared on the same items.
    """
    def covers(m, wanted):
        return all(set(wanted[d]) <= set(scores[d].get(m, {})) for d in targets)

    full = {d: targets[d]["item_id"].tolist() for d in targets}
    with_scores = [m for m in models if any(scores[d].get(m) for d in targets)]
    full_judges = [m for m in with_scores if covers(m, full)]
    scopes = [("", "all conversations", {d: None for d in targets}, full_judges)]
    partial = [m for m in with_scores if m not in full_judges]
    if partial:
        # the subset, narrowed to conversations every partial judge has scored (an interrupted
        # run is analysed on what it finished; resuming it widens this back to the full subset)
        sub = {d: [i for i in subset_ids(items[d], d)
                   if i in set(full[d]) and all(i in scores[d].get(m, {}) for m in partial)]
               for d in targets}
        label = f"fixed-seed subset ({SUBSET_PER_DATASET} per dataset, seed {SEED}), scored items"
        scopes.append(("__subset", label, sub, [m for m in with_scores if covers(m, sub)]))
    return scopes


def run_evaluation(paths, settings: Settings, configs=CONFIGS, prompts=("P0", "P1", "P2"), datasets=DATASETS,
                   scopes=("all", "subset"), only_models=None):
    raws = {d: _load(paths, d) for d in datasets}
    items = {d: items_table(raws[d]) for d in datasets}
    summary_path = paths.results / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"runs": {}}
    summary["settings"] = settings.__dict__
    cache_path = paths.cache / CACHE_FILE
    for prompt_id in prompts:
        planned = list(MODELS) if prompt_id == "P0" else [SENSITIVITY_MODEL]
        # judges to analyse; leaving out pilot-only judges keeps them from shrinking the subset
        models = [m for m in planned if not only_models or m in only_models]
        cfgs = configs if prompt_id == "P0" else [c for c in configs if c.name == "primary"]
        for cfg in cfgs:
            scores = {d: {m: load_scores(cache_path, m, prompt_id, d) for m in models} for d in datasets}
            scores = {d: {m: s for m, s in v.items() if s} for d, v in scores.items()}
            if not any(scores.values()):
                print(f"skip {cfg.name}__{prompt_id}: no judge scores cached")
                continue
            targets = {d: prepare(raws[d], d, cfg)[1] for d in datasets}
            for suffix, label, item_ids, judges in evaluation_scopes(scores, targets, items, models):
                if not judges or ("subset" if suffix else "all") not in scopes:
                    continue
                run = f"{cfg.name}__{prompt_id}{suffix}"
                out = paths.results / run
                out.mkdir(parents=True, exist_ok=True)
                results = {}
                for d in datasets:
                    other = [o for o in datasets if o != d]
                    other_t = targets[other[0]] if other else targets[d]
                    js = {m: scores[d][m] for m in judges}
                    results[d] = analyse_dataset(raws[d], items[d], d, cfg, other_t, js, settings,
                                                 item_ids=item_ids[d])
                    print(f"{run} DICES-{d}: done on {len(results[d]['targets'])} conversations "
                          f"(missing judges: {results[d]['missing_judges']})", flush=True)
                # Holm over the pre-registered family size even when some judges were not run yet
                tests = apply_holm_across([results[d]["tests"] for d in datasets], judges,
                                          family_size=len(planned) * len(datasets))
                tables = {
                    "overall.csv": [results[d]["overall"] for d in datasets],
                    "disagreement_tertiles.csv": [results[d]["tertiles"] for d in datasets],
                    "disagreement_tests.csv": [tests],
                    "groups.csv": [results[d]["groups"] for d in datasets],
                    "item_predictions.csv": [_item_frame(results[d], d) for d in datasets],
                }
                n_conv = {f"DICES-{d}": len(results[d]["targets"]) for d in datasets}
                for name, parts in tables.items():
                    df = pd.concat(parts, ignore_index=True)
                    if len(df):
                        df.insert(1, "scope", label)
                        df.insert(2, "n_conversations", df["dataset"].map(n_conv))
                    df.to_csv(out / name, index=False)
                summary["runs"][run] = {
                    "config": cfg.__dict__, "prompt": prompt_id, "judges": judges, "scope": label,
                    "n_conversations": n_conv,
                    "constant_value": {d: results[d]["constant_value"] for d in datasets},
                    "missing_judges": {d: results[d]["missing_judges"] for d in datasets},
                }
    summary_path.write_text(to_json(summary))
    return summary


def cmd_evaluate(args, paths):
    settings = Settings(n_boot=args.n_boot, n_perm=args.n_perm, n_null=args.n_null, seed=args.seed)
    run_evaluation(paths, settings, datasets=tuple(args.datasets), scopes=tuple(args.scopes),
                   only_models=args.models)
    return 0


def cmd_figures(args, paths):
    from . import figures as F
    if not (paths.results / "primary__P0").exists():
        sys.exit("results/primary__P0 not found; run `wsj evaluate` first")
    for suffix in ("", "__subset"):
        prim = paths.results / f"primary__P0{suffix}"
        if not prim.exists():
            continue
        items = pd.read_csv(prim / "item_predictions.csv")
        judges = [m for m in MODELS if m in items.columns]
        n = items.groupby("dataset").size()
        F.set_caption(f"{items['scope'].iloc[0]}; judges: {', '.join(F.short(j) for j in judges)}; "
                      + ", ".join(f"{d}: {k} conversations" for d, k in n.items()))
        tag = suffix.replace("__", "_")
        F.human_distribution(items, paths.figures / f"fig1_human_distribution{tag}.png")
        F.overall(pd.read_csv(prim / "overall.csv"), paths.figures / f"fig2_overall{tag}.png")
        F.reliability(items, judges, paths.figures / f"fig3_reliability{tag}.png")
        F.disagreement(pd.read_csv(prim / "disagreement_tertiles.csv"), paths.figures / f"fig4_disagreement{tag}.png")
        F.groups(pd.read_csv(prim / "groups.csv"), judges, paths.figures / f"fig5_groups{tag}.png")
    F.set_caption("")
    by_prompt = {p: pd.read_csv(paths.results / f"primary__{p}" / "item_predictions.csv")
                 for p in ("P0", "P1", "P2") if (paths.results / f"primary__{p}").exists()}
    F.prompt_sensitivity(by_prompt, SENSITIVITY_MODEL, paths.figures / "fig6_prompt_sensitivity.png")
    print(f"figures written to {paths.figures}")
    return 0


def cmd_all(args, paths):
    for step in (cmd_download, cmd_clean, cmd_judge, cmd_evaluate, cmd_figures):
        rc = step(args, paths)
        if rc:
            return rc
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="wsj", description="Whose safety does an LLM judge reflect? (DICES)")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("download", "clean", "judge", "evaluate", "figures", "all"):
        p = sub.add_parser(name)
        p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
        if name in ("judge", "all"):
            p.add_argument("--models", nargs="+", choices=list(MODELS))
            p.add_argument("--prompts", nargs="+", choices=["P0", "P1", "P2"])
            p.add_argument("--limit", type=int, help="score only the first N items (pilot)")
            p.add_argument("--subset", action="store_true",
                           help=f"score only the fixed-seed subset ({SUBSET_PER_DATASET} conversations per dataset)")
            p.add_argument("--threads", type=int, default=None)
        if name in ("evaluate", "all"):
            p.add_argument("--n-boot", type=int, default=2000)
            p.add_argument("--n-perm", type=int, default=10000)
            p.add_argument("--n-null", type=int, default=1000)
            p.add_argument("--seed", type=int, default=SEED)
            p.add_argument("--scopes", nargs="+", default=["all", "subset"], choices=["all", "subset"],
                           help="analyse judges scored on all conversations, on the fixed-seed subset, or both")
        if name == "evaluate":
            p.add_argument("--models", nargs="+", choices=list(MODELS),
                           help="analyse only these judges (default: every judge with cached scores)")
    return ap


COMMANDS = {"download": cmd_download, "clean": cmd_clean, "judge": cmd_judge,
            "evaluate": cmd_evaluate, "figures": cmd_figures, "all": cmd_all}


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = default_paths().ensure()
    return COMMANDS[args.command](args, paths) or 0


if __name__ == "__main__":
    sys.exit(main())
