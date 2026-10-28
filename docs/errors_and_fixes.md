# Errors and fixes

Every problem hit during the project: what happened, the cause, how it was
fixed, and the effect on the results. The facts come from `results/logs/`,
`results/run_log.jsonl`, the git history, the README's "Deviations from
plan" and the run history kept while the project was done. Times are PST. A dated timeline is in `project_log.md`.

The short version: **none of these problems changed a reported number**,
except the README and docs claims that were wrong and were corrected. The two
that most affect how the results should be *read* are SmolLM2's near-constant
scores (§12) and the role of the fixed guess in the disagreement test (§13).

## Setup and design

### 1. scikit-learn was not installed (2026-10-27, ~22:49)

- **What happened.** The first environment check found torch and
  transformers installed system-wide, but importing scikit-learn failed with
  `ModuleNotFoundError: No module named 'sklearn'`.
- **Cause.** The system Python did not have it, and the metrics need it
  (`roc_auc_score`, `cohen_kappa_score`).
- **Fix.** The project got its own virtual environment (`.venv`) with every
  dependency pinned in `requirements.txt`, including `scikit-learn==1.9.1`.
  All runs used that environment (Python 3.12.3, torch 2.14.1, transformers
  5.21.0, as recorded in `results/run_log.jsonl`).
- **Effect on results.** None. It was found before any code ran.

### 2. The pre-registered group test flagged random groups ~20% of the time (2026-10-27, before any results)

- **What happened.** The RQ2 test as first written compared each
  demographic group with random raters drawn separately for every
  conversation. A synthetic simulation (`scripts/null_simulation.py`) built
  random 12-rater groups with no demographic meaning. The planned test still
  called 21% of them significant at α = 0.05 (19% in the committed re-run),
  instead of about 5%.
- **Cause.** DICES raters differ in how strict they are, and each rater's
  strictness is stable across conversations. A real group is a fixed set of
  raters, so its strictness carries across every conversation. Redrawing
  raters per conversation averages strictness away, so the null was too
  narrow and any fixed set of raters looked unusual.
- **Fix.** Methodology Amendment 1: random groups are drawn as *whole
  raters*, and the same raters are used on every conversation. The
  simulation gives 3% (5% in the committed re-run) false positives with this
  null. The re-run was repeated on 2026-10-29 and again printed 0.19 and
  0.05. The old test is not used in any reported result.
- **Effect on results.** The change was made before any judge output was
  compared with any human rating, so no result was chosen by it. Without it,
  roughly one random group in five would have looked "significant".

### 3. A git commit was blocked (2026-10-27, ~23:31)

- **What happened.** After Amendment 1 was written and tested, the commit was
  blocked by the tool permission settings, which classed `git commit` as a
  destructive git action. The changes stayed in the working tree.
- **Fix.** After approval, the work was committed at 23:38 in two commits
  ("Free each judge model before loading the next" and "Amend RQ2 null to
  whole-rater permutation").
- **Effect on results.** None. It delayed the commit by a few minutes.

### 4. The README had a wrong Colab/GPU note (2026-10-27, ~23:51)

- **What happened.** A README draft said the remaining runs could be done on
  a free Colab GPU by installing the requirements and running the same
  commands.
- **Cause.** The code always runs on CPU: it never moves the model to a GPU,
  and the logged device is fixed to `cpu`.
- **Fix.** Corrected within a minute, before it was committed. The README
  now says GPU support would need a small change in `HFBackend` (move the
  model and inputs to `cuda`, and log the device), and that GPU results can
  differ slightly from CPU ones.
- **Effect on results.** None. Every score was computed on CPU.

## Running the judges

### 5. The first full run was stopped after 1.5 minutes, and compute was cut to 30 minutes (2026-10-27, 23:38–23:41)

- **What happened.** A run of all three judges with P0 started at about
  23:38. It was stopped at about 23:39 on request, while Qwen2.5-0.5B was
  scoring DICES-350 (`results/logs/judge_P0_main.log` ends at 50/350). Only
  about 30 minutes of compute were then available.
- **Fix.** The cache check after the stop found complete lines only (73 new
  scores on top of 120 pilot scores). The scope was reduced for that
  session: Qwen2.5-0.5B on all 1,340 conversations, and Qwen2.5-1.5B on a
  fixed-seed subset of 150 conversations per dataset (seed 20261027).
  SmolLM2 and the prompt checks were postponed. This is README deviation 2.
- **Effect on results.** Temporary. The subset analyses were reported as
  preliminary and are superseded. Every pre-registered run was completed
  later (§9).

### 6. The PC shut down during the run (2026-10-27, ~23:59)

- **What happened.** The PC lost power while Qwen2.5-1.5B was loading its
  weights for the subset run. Qwen2.5-0.5B had finished all 1,340
  conversations at 23:59:17 (`results/logs/pipeline.log`). Its analysis had
  just started and was lost (`results/logs/evaluate_all.log` was empty).
- **Cache check.** After the restart, `cache/judge_scores.jsonl` (1,420
  lines) and `results/run_log.jsonl` (12 lines) both ended in a newline and
  every line parsed as JSON. There were no duplicate (model, prompt,
  dataset, conversation) keys and no non-finite probabilities. No partial
  last entry had to be dropped.
- **Fix.** The run was resumed on 2026-10-28, 00:02–00:15
  (`results/logs/pipeline_resume.log`). Re-running Qwen2.5-0.5B scored
  nothing new (all cached). Qwen2.5-1.5B was given a 550-second budget and
  stopped at it (exit code 124). By then it had scored 150 of 150 DICES-350
  subset conversations and 30 of 150 DICES-990 ones.
- **Effect on results.** None on the final results, because every score was
  cached per conversation and the cache was intact. The preliminary
  DICES-990 subset results rested on only 30 conversations.

### 7. Threads switched from 12 to 8 (2026-10-28)

- **What happened.** The resume, and every later run, used 8 CPU threads
  instead of 12, to keep the machine cooler.
- **Cause of concern.** The number of threads can change floating-point
  results in the last digits. The Qwen2.5-0.5B scores and the pilot rows
  were computed with 12 threads, everything else with 8. Each row's thread
  count is in `results/run_log.jsonl`.
- **Check.** `scripts/recheck_judge_scores.py` re-scored three Qwen2.5-1.5B
  conversations from scratch at 8 and at 12 threads. P(unsafe) changed by at
  most 1e-5.
- **Effect on results.** Negligible: far below anything that changes a
  reported number.

### 8. Pilot-only judges shrank the subset analysis (2026-10-28)

- **What happened.** SmolLM2 and Qwen2.5-1.5B had scores for the first 20
  conversations per dataset from the timing pilot (`--limit 20`). The
  analysis treated them as partial runs, so the subset analysis was cut down
  to the few conversations they shared with it.
- **Fix.** `wsj evaluate` gained a `--models` option so that pilot-only
  judges could be left out. The RQ3 Holm correction kept the pre-registered
  family size, counting tests not yet run as p = 1.
- **Effect on results.** Only the preliminary subset analysis was affected.
  In the final analysis every judge covers all conversations, so the option
  is not needed.

### 9. The final runs were interrupted once (2026-10-28, 01:51)

- **What happened.** `scripts/run_remaining.sh` (started 00:56) was stopped
  on request at 01:51, before a planned shutdown, during the Qwen2.5-1.5B
  P1/P2/P0 run. The runner was killed and the judge was stopped with SIGINT,
  which only interrupts between cache writes.
- **Cache check before resuming.** 2,759 lines, all valid JSON, ending in a
  newline, no duplicate keys. A copy of the cache was kept before resuming.
- **Fix.** Resumed at 21:32 with the same script. It finished on 2026-10-29
  at 01:01, with every step succeeding on its first attempt and 6,700 cached
  scores (5 judge-prompt pairs × 1,340 conversations). The script repeats
  its first steps on every start, which cost about 6 minutes of redundant
  analysis.
- **Effect on results.** None.

### 10. The archive step missed the P1 subset folder (2026-10-29)

- **What happened.** After the final run, `results/primary__P1__subset/` was
  still next to the final results.
- **Cause.** When the resumed script repeated its subset-analysis step
  (21:32–21:38 on 2026-10-28), Qwen2.5-1.5B already had partial P1 scores,
  so the step also wrote a P1 subset folder (created 21:38:16). The
  script's archive step only moved `*__P0` and `*__P0__subset` folders.
- **Fix.** The folder was moved (not deleted) into
  `results/archive_partial_runs_2026-10-28/`, and the script now archives
  every `*__subset` folder. The new archive line was tested on dummy
  folders.
- **Effect on results.** None. The folder was never part of the final
  results.

## Interpreting the results

### 11. Qwen2.5-1.5B was at chance level on DICES-350 (2026-10-28 onwards)

- **What happened.** On the preliminary subset, Qwen2.5-1.5B ranked
  DICES-350 conversations at chance (AUROC 0.47) while the smaller 0.5B
  model did well on the same conversations.
- **Check for a bug (2026-10-28).** Token IDs ("Yes" = 9454 and "No" = 2753,
  single tokens), the rendered prompt and chat template (byte-identical for
  the two Qwen models), the answer format (≥ 99.99% of the next-token
  probability on Yes/No) and reproducibility (re-scored conversations
  matched the cache to within 1e-5) were all checked. No bug and no
  difference from the 0.5B run was found.
- **Final runs.** The result holds on all 350 conversations and with every
  prompt (AUROC 0.47, 0.51 and 0.33 for P0, P1 and P2), while the same model
  ranks DICES-990 moderately well (0.67–0.73).
- **Status.** Unexplained. The README and `limitations.md` say so, and the
  1.5B DICES-350 results, including its large positive disagreement Δ, are
  not interpreted.

### 12. SmolLM2-1.7B gives nearly the same score to every conversation (2026-10-29)

- **What happened.** SmolLM2 beats the fixed guess on both datasets (BSS
  +0.10 and +0.15), but its P(unsafe) has a mean of 0.28–0.29 and an SD of
  0.04–0.05, and it is never above 0.5.
- **Cause.** Not a bug: SmolLM2 puts at least 98.9% of its next-token
  probability on Yes/No on every conversation, so it follows the format. Its
  near-constant value is simply closer to each dataset's real average than
  the fixed guess, which borrows the *other* dataset's average and is off
  by about 0.1 in each.
- **Fix.** The README does not present its positive BSS as skill. It reports
  Spearman ρ and AUROC (0.54–0.58) next to it, and adds a clearly labelled
  post-hoc comparison against each dataset's own average, which no judge
  clearly beats.
- **Effect on results.** It changes how SmolLM2's numbers are read, not the
  numbers.

### 13. "Worse on disagreement" was first overstated (2026-10-29)

- **What happened.** The first summary of the final results said the judges
  do worse where humans disagree on DICES-990.
- **Cause.** The pre-registered measure is skill *relative to the fixed
  guess*. On DICES-990 the fixed guess (0.36) is close to the
  most-disagreed third's average human answer (0.45), so its Brier score
  there is tiny (0.021, compared with 0.089 in the least-disagreed third).
  In raw Brier score, every judge's error is *smaller* in the most-disagreed
  third, on both datasets.
- **Fix.** The README, `limitations.md` and the other notes now say the
  judges "fall further behind the fixed guess" there, not that they make
  bigger errors, and give both readings.
- **Effect on results.** None on the numbers. It changes the conclusion's
  wording.

## Documentation

### 14. README and docs claims that were stale or wrong

Each was found by checking the text against the CSVs, logs or code, and was
corrected:

| When found | Claim | Problem | Correction |
|---|---|---|---|
| 2026-10-28 | Preliminary README: the subset run used 12 threads, and the subset was finished | It used 8 threads and stopped at its time limit | Fixed in the README |
| 2026-10-29 | "Status: preliminary", "Runs not done yet", and limitations and notes saying runs were missing | Out of date after the final runs | Replaced with the final results |
| 2026-10-29 | SmolLM2's DICES-350 disagreement Δ "significant in 2 of 3 analyses", shown next to p = 0.75 | The p = 0.75 belongs to the analysis where it is *not* significant, so the sentence read as a contradiction | Now gives each analysis: +0.19 (Holm p = 0.0006), +0.12 (Holm p = 0.03), −0.02 (Holm p = 1.0, not significant) |
| 2026-10-29 | Groups have "11–25 raters" (README) | 11 comes from the all-raters analysis; in the main analysis the smallest groups have 13–26 | Corrected |
| 2026-10-29 | Groups have "10–84 raters each" (`limitations.md`) | The 25 tested groups have 13–141 raters | Corrected |
| 2026-10-29 | SmolLM2 download "~3.4 GB" | The downloaded model is 3.2 GB (Qwen2.5-1.5B 2.9 GB, Qwen2.5-0.5B about 1 GB) | Corrected |
| 2026-10-29 | `design.md`: Holm padding "when some planned tests have not run yet" | Every planned test has run | Rewritten to describe the final families (6 tests for P0, 2 each for P1 and P2) |
| 2026-10-29 | `INDEX.md`: the script's archive step "does not cover" P1 subset folders | Out of date after the fix in §10 | Updated |

### 15. Preparing the repository (2026-10-29)

- **Personal paths.** Five log files in `results/logs/` contained the local
  home folder path. They were replaced with `/path/to/whose-safety-judge`.
- **A scan that skipped files.** The first scan for personal paths returned
  nothing. The shell's `grep` was a wrapper that skips git-ignored files,
  and `results/` and `cache/` were git-ignored at the time. The scan was
  repeated with the system `grep`, which found the five log files.
- **Personal notes.** Notes written for the author (defence questions,
  learning list, email draft) were moved to `private_notes/`, which is
  git-ignored, and removed from the git history.
- **Effect on results.** None.
