# Design

How the code is organised and why. The analysis decisions themselves are in
`methodology.md`; this document covers the software.

## Pipeline

```
wsj download  ->  data/raw/            (pinned DICES CSVs + READMEs, SHA-256 checked)
wsj clean     ->  results/cleaning_report_{350,990}.json
wsj judge     ->  cache/judge_scores.jsonl   (one line per model x prompt x dataset x conversation)
                  results/run_log.jsonl      (settings, versions, timings of every judge run)
wsj evaluate  ->  results/<config>__<prompt>/*.csv, results/summary.json
                  (results/<config>__<prompt>__subset/ only while a judge covers just the subset)
wsj figures   ->  figures/*.png
wsj all       ->  all of the above in order
```

Each step reads only files written by earlier steps, so any step can be
re-run on its own.

## Modules (`src/wsj/`)

| Module | Responsibility |
|---|---|
| `download.py` | Source URLs pinned to one DICES commit, expected SHA-256s, manual-download instructions if fetching fails |
| `data.py` | Loading, the authors' removed-rater lists, soft labels `h`, majority `m`, disagreement `d`, demographic groups, cleaning report |
| `prompts.py` | The frozen prompt strings P0/P1/P2 |
| `judges.py` | The `Backend` protocol, the transformers backend, logit scoring, the score cache and the run log |
| `baselines.py` | Constant, majority and keyword-rule baselines |
| `metrics.py` | JSD, Brier, BSS, ECE, Spearman, AUROC, κ, entropy |
| `stats.py` | Paired bootstrap, permutation test, whole-rater null for groups, Holm |
| `analysis.py` | RQ1–RQ3 tables for one dataset and one configuration |
| `figures.py` | PNGs from the CSVs only (figures never recompute statistics) |
| `cli.py` | The `wsj` command, run orchestration, the fixed-seed subset, scopes |
| `paths.py` | Directory layout (root = current directory, or `WSJ_ROOT`) |

## Key decisions

**Probabilities from logits, no generation.** A judge's probability of
"unsafe" is the softmax over just two next-token logits, those for "Yes" and
"No", after the chat-formatted prompt. This avoids parsing free text and
sampling noise, it is deterministic, and the project never generates any text.
The share of probability mass on {Yes, No} is logged to show how well each
model follows the answer format.

**A cache keyed on everything that changes the score.** The cache key covers
the model, its pinned revision, the prompt id, the dataset, the item id, and
the fully rendered prompt text. An earlier version of the key left out the
dataset and item. That made identical texts in the two datasets collide, and a
regression test now covers it. Results are appended after every conversation,
so an interrupted run resumes where it stopped and a finished run is never
repeated.

**Backends behind a protocol.** Tests use a fake backend with the same
interface as the transformers one, so the whole pipeline is tested in seconds
without downloading a model. One slow test runs a real model and is skipped
if the model is not cached.

**Shared bootstrap indices.** All predictors are evaluated on the same
bootstrap resamples, so a difference between two predictors gets a paired
confidence interval.

**The whole-rater null for groups.** The demographic test draws random groups
of whole raters (matrix multiplication over a rater × item table). It does not
redraw raters for each item. Methodology Amendment 1 and
`scripts/null_simulation.py` explain why.

**Scopes for partial runs.** A judge scored on every conversation is analysed
on all of them (`results/<config>__<prompt>/`). A judge scored only on the fixed-seed
subset (`--subset`: 150 conversations per dataset, seed 20261027) is analysed
on the part of the subset it has scored. Every fully-scored judge and the
baselines are analysed on the same conversations alongside it
(`results/<config>__<prompt>__subset/`), so comparisons within a folder are
always on identical items. Every CSV carries `scope` and `n_conversations`
columns, and every figure has a footer naming the judges and the conversation
counts. This was used for the preliminary results. In the final results every
judge covers all conversations, so no subset folders are produced; the
earlier ones are kept in `results/archive_partial_runs_2026-10-28/`.

**Holm with the planned family size.** The RQ3 family is the judges planned
for that prompt × the datasets (6 tests for P0, 2 each for P1 and P2, which
run on one judge). If a planned test has not run, it counts as p = 1, so the
correction is never weaker than pre-registered
(`analysis.apply_holm_across`). In the final results every planned test has
run, so no padding is applied. Group tests are corrected within each judge
and dataset.

**Fixed seeds everywhere.** The seed is 20261027 for bootstraps, permutations,
nulls and the subset. torch runs single-process in float32 with deterministic
algorithms.

## Testing

`tests/` runs on two small synthetic fixture CSVs with the real DICES column
layout (`tests/fixtures/make_fixtures.py`). Real DICES data is never needed for
the tests. The tests cover:

- loading and cleaning, including the removed-rater lists and odd labels;
- the metrics against hand-computed values and against scipy;
- the statistical procedures, including that a random group with stable rater
  strictness is *not* flagged and a genuinely deviant group *is*;
- caching and resuming;
- subset selection and the analysis of interrupted subset runs;
- the CLI end to end, through to the figures.

Run them with `.venv/bin/python -m pytest -q`.
