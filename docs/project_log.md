# Project log

A dated timeline of the project. Times are IST (UTC+5:30). Sources: the git
history, `results/logs/` (including `pipeline.log`, `pipeline_resume.log`
and `pipeline_final.log`), `results/run_log.jsonl`, and the run history kept
while the project was done. Each problem is described in more detail in
`errors_and_fixes.md` (§ numbers below).

## 2026-10-27: design, code and first runs

| Time | Event |
|---|---|
| ~22:49 | Environment check: scikit-learn missing from the system Python. A pinned virtual environment is created (§1). |
| 23:00 | Pre-registered methodology committed (`docs/methodology.md`): research questions, metrics, tests, seeds, judges and prompts, before any results. |
| 23:01 | Judge prompts P0, P1, P2 and the keyword baseline frozen and committed. |
| 23:08 | The `wsj` package, CLI and tests committed. |
| before 23:31 | 20-conversation timing pilot per model and dataset. A synthetic simulation shows that the planned group test calls about 20% of random groups significant (§2). Amendment 1 is written. |
| ~23:31 | The commit is blocked by the tool permission settings (§3). |
| 23:38 | Committed: freeing each judge model before loading the next, and Amendment 1. The full three-judge run starts. |
| ~23:39 | The run is stopped on request after about 1.5 minutes. Only about 30 minutes of compute are available, so the scope is reduced: Qwen2.5-0.5B on everything, Qwen2.5-1.5B on a fixed-seed subset (§5). |
| 23:41 | Qwen2.5-0.5B starts on all 1,340 conversations (12 threads). |
| 23:46 | Fixed-seed subset runs and per-scope evaluation committed. |
| 23:50 | Docs, index and notes committed. |
| ~23:51 | A wrong Colab/GPU note in the README draft is corrected before it is committed (§4). |
| 23:59 | Qwen2.5-0.5B finishes (1,085 s). The PC loses power while Qwen2.5-1.5B is loading (§6). |

## 2026-10-28: resume, preliminary results, final runs

| Time | Event |
|---|---|
| 00:02 | Resume after the cache integrity check (1,420 lines, all valid). 8 threads from now on (§6, §7). |
| 00:02–00:15 | Qwen2.5-1.5B subset run stops at its 550 s limit (150/150 DICES-350, 30/150 DICES-990 conversations). Analysis and figures for the 0.5B run and the subset. `wsj evaluate --models` added (§8). |
| ~00:15 | Preliminary results written up, with README corrections about the thread count and the incomplete subset (§14). |
| ~00:39 | Sanity check of the 1.5B chance-level result: no bug found (§11). |
| 00:56 | `scripts/run_remaining.sh` starts the remaining pre-registered runs. |
| 01:03 | Step 1 (finish the 1.5B subset) done. |
| 01:08 | Step 2 (subset analysis and figures) done. Step 3 (Qwen2.5-1.5B, P1, P2, P0 on everything) starts. |
| 01:51 | Stopped on request before a shutdown (§9). |
| 21:32 | Resumed after a cache check (2,759 lines, valid, no duplicates). Steps 1 and 2 repeat; the repeated subset analysis writes `primary__P1__subset` (§10). |
| 21:38 | Step 3 starts again from the cache. |
| 23:36 | Step 3 done (7,102 s). Step 4, SmolLM2-1.7B with P0, starts. |

## 2026-10-29: final results and documentation

| Time | Event |
|---|---|
| 00:55 | SmolLM2 done (4,722 s). Partial-run outputs archived. |
| 01:00 | Final analysis done (308 s): 5 analyses, all on 350 + 990 conversations, no judge missing. |
| 01:01 | Figures done and all 46 tests pass. 6,700 cached scores. |
| after 01:01 | `primary__P1__subset` moved into the archive and the script's archive step fixed (§10). Final results written into the README, limitations and docs. SmolLM2's near-constant scores (§12) and the raw-Brier reading of the disagreement test (§13) documented. |
| later | Every number in the README and docs checked against `results/`. Wrong or stale claims corrected (§14). The tables were rebuilt from the cache in a fresh copy of the project to check that they reproduce. Personal paths cleaned and personal notes moved out of the repository (§15). |

## Final run times (from the logs)

| Step | Time |
|---|---|
| Qwen2.5-0.5B, P0, 1,340 conversations (12 threads) | ~18 min (0.84–0.89 s each) |
| Qwen2.5-1.5B, each of P0, P1, P2 (8 threads) | ~55–60 min (2.3–2.7 s each) |
| SmolLM2-1.7B, P0 (8 threads) | ~80 min (3.5–3.7 s each) |
| `wsj evaluate`, all analyses | 5–6 min |
| `wsj figures` | ~5 s |
| Tests | ~30 s |
