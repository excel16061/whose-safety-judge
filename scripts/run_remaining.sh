#!/bin/bash
# Finish every pre-registered run on 8 CPU threads (cores 0-7), then re-run the analysis.
# Each judge step resumes from cache/judge_scores.jsonl and is retried if it stops.
# Usage (repo root):  nohup scripts/run_remaining.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
L=results/logs; P=$L/pipeline_final.log
Q05=Qwen/Qwen2.5-0.5B-Instruct; Q15=Qwen/Qwen2.5-1.5B-Instruct; SMOL=HuggingFaceTB/SmolLM2-1.7B-Instruct
log() { echo "[$(date '+%F %T')] $*" | tee -a "$P"; }

step() {  # step <name> <command...>: run on cores 0-7, retry up to 5 times
    local name=$1; shift
    for attempt in 1 2 3 4 5; do
        log "start $name (attempt $attempt)"
        local s=$(date +%s)
        taskset -c 0-7 "$@" >> "$L/$name.log" 2>&1
        local rc=$?
        log "end $name exit $rc ($(( $(date +%s) - s ))s)"
        [ $rc -eq 0 ] && return 0
        sleep 30
    done
    log "GAVE UP on $name"; exit 1
}

step final_1_judge_q15_P0_subset   .venv/bin/wsj judge --models $Q15 --prompts P0 --subset --threads 8
step final_2_evaluate_subset       .venv/bin/wsj evaluate --scopes all subset --models $Q05 $Q15
step final_2_figures               .venv/bin/wsj figures
step final_3_judge_q15_P1_P2_P0    .venv/bin/wsj judge --models $Q15 --prompts P1 P2 P0 --threads 8
step final_4_judge_smol_P0         .venv/bin/wsj judge --models $SMOL --prompts P0 --threads 8

# Every judge now covers all conversations, so the subset scope is no longer produced.
# Move the earlier partial-run outputs aside instead of leaving stale files next to the final ones.
A=results/archive_partial_runs_2026-10-28; mkdir -p "$A" figures/archive_partial_runs_2026-10-28
for d in results/*__P0 results/*__subset; do [ -d "$d" ] && mv "$d" "$A/"; done  # every prompt's subset folders
[ -f results/summary.json ] && mv results/summary.json "$A/"
mv figures/*.png figures/archive_partial_runs_2026-10-28/ 2>/dev/null
log "archived partial-run outputs to $A and figures/archive_partial_runs_2026-10-28"

step final_5_evaluate              .venv/bin/wsj evaluate
step final_5_figures               .venv/bin/wsj figures
step final_6_pytest                .venv/bin/python -m pytest -q
log "ALL DONE"
