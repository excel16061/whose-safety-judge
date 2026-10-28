# Index

**Open first:** `README.md` covers the question, the final results with
their caveats, and how to run everything. Then read `docs/methodology.md`
for exact definitions.

| Path | What it is |
|---|---|
| `README.md` | Overview, research questions, final results with caveats, changes from the preliminary results, step-by-step run instructions with timings, reproducing from the cache, tests, deviations from plan |
| `INDEX.md` | This file |
| `docs/problem_analysis.md` | The problem, the research questions, and why these measurements |
| `docs/methodology.md` | Pre-registered methodology and evaluation: exact metric and test definitions, plus Amendment 1 (the whole-rater null) |
| `docs/design.md` | Software design: pipeline, modules, key decisions, testing |
| `docs/limitations.md` | Limitations (small weak judges; prompt dependence; unexplained 1.5B behaviour; small groups; how the fixed guess shapes the skill scores) |
| `docs/related_work.md` | The five papers read, how this project differs, and how each citation was checked |
| `docs/errors_and_fixes.md` | Every problem hit during the project: what happened, the cause, the fix, and the effect on results |
| `docs/project_log.md` | Dated timeline of the project, from the pre-registration to the final results |
| `src/wsj/` | The Python package (`wsj` command): data, judges, metrics, statistics, analysis, figures |
| `scripts/download_data.py` | Downloads DICES at a pinned commit and checks SHA-256s (same as `wsj download`) |
| `scripts/null_simulation.py` | Simulation behind Amendment 1 (false-positive rates of the two group nulls) |
| `scripts/recheck_judge_scores.py` | Read-only sanity check: re-scores a few cached Qwen2.5-1.5B conversations and compares with the cache |
| `scripts/run_remaining.sh` | Finishes every pre-registered run on 8 threads, archives the partial-run outputs, then re-runs the analysis, figures and tests |
| `tests/` | pytest suite on small synthetic fixtures (`tests/fixtures/`) |
| `cache/judge_scores.jsonl` | Every judge probability used in the results, one line per model × prompt × dataset × conversation (6,700 lines, 3 MB). Lets you rebuild the tables and figures without re-running the judges |
| `results/` | Final analysis tables (`<config>__<prompt>/`), `summary.json`, cleaning reports, `run_log.jsonl` (settings, versions and timing of every judge run), and `logs/` from each run |
| `results/archive_partial_runs_2026-10-28/` | Outputs of the earlier partial runs, kept for reference and superseded by the final results: the old `*__P0` and `*__subset` folders and `summary.json` |
| `figures/` | Final PNG figures; `figures/archive_partial_runs_2026-10-28/` holds the figures from the partial runs |
| `requirements.txt`, `pyproject.toml` | Exact dependency pins and the package definition |
| `data/raw/` | Downloaded DICES CSVs and READMEs. Not in the repository, because Google distributes them: run `wsj download` |
| `private_notes/` | Personal notes (defence questions, learning list, email draft). Kept on the author's computer only and git-ignored, so not in the repository |

Folders in `results/`: `primary` uses the main settings, `sens_unsure_half`
counts Unsure as half, and `sens_all_raters` keeps all raters. `__P0`,
`__P1` and `__P2` name the prompt. All final folders cover all 1,340
conversations; `__subset` folders exist only in the archive.
