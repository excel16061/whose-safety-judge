# Limitations

> **Scope of every conclusion.** All pre-registered runs are done: three
> small open judges (Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct,
> SmolLM2-1.7B-Instruct) scored all 1,340 DICES conversations with the
> primary prompt P0, and Qwen2.5-1.5B also ran the prompt checks P1 and P2.
> Every claim in the README is limited to those judges, those prompts and
> those conversations. None of them is a claim about "small LLM judges", let
> alone LLM judges in general.

## 1. The judges are small, weak models

The judges have 0.5B–1.7B parameters and run on CPU. Models this size are much
weaker readers of nuanced, adversarial conversations than the API models
studied by Movva et al. (GPT-4 and others), or than the judges used in
practice. A poor fit to human ratings here says something about these
particular models, not about LLM judges in general. The results bear this
out: none of the three tracks the human ratings well. The probability of
"unsafe" is also read from just two next-token logits. That is one reasonable
way to get a probability, not the only one. On every cached conversation,
the Qwen models put at least 99.97% of the next-token probability on
"Yes"/"No" (all prompts), and SmolLM2 at least 98.9% (median 99.6%), so
following the answer format is not the problem. That mass is logged
(`results/run_log.jsonl`, cache) but not used to exclude items.

## 2. Results depend on the prompt

LLM judgements can shift a lot with prompt wording. One primary prompt (P0)
was frozen before any results, and two checks were run on Qwen2.5-1.5B only:
a minimal prompt (P1) and a reversed-polarity prompt (P2).

- The 1.5B model's at-or-below-chance ranking on DICES-350 appeared with all
  three prompts, so that pattern is not an artefact of P0's wording.
- Its calibration did depend strongly on the wording: BSS was about −0.7 to
  −1.1 under P0 and P1, but about −3.4 under P2, where its scores spread out
  and became confidently wrong.
- The prompt checks were run on one judge only. There is no evidence here
  about how the 0.5B or SmolLM2 results would move under other prompts.
- P0's definition of "unsafe" paraphrases the DICES rater categories, and a
  different paraphrase could change the results.

## 3. Some judge behaviour is unexplained

Qwen2.5-1.5B ranks DICES-990 conversations moderately well (AUROC 0.67–0.73)
but DICES-350 conversations at or below chance with every prompt (AUROC
0.33–0.51), and significantly backwards under P2. A sanity check ruled out a
scoring bug for P0 (token IDs, chat template, answer format,
reproducibility), and the same code gives the 0.5B model its best result on
DICES-350. What causes the difference has not been investigated. Until it
is, the 1.5B results on DICES-350, including its large positive
disagreement Δ, should not be interpreted.

## 4. Demographic groups are small

After the rater filter, the 25 tested demographic groups have 13–141 raters
each, and several have only 13–26. For example, in DICES-350 the Multiracial
group has 13 raters and each race group has 21–25; in DICES-990 the Latinx
group has 16. Groups under 10 raters (such as Nonbinary, with 1 rater in
DICES-990) are listed but not tested. With groups this small:

- each group's per-conversation share of "unsafe" is noisy;
- the tests have limited power, so "no significant difference" does not mean
  "no difference";
- Movva et al. reached the same conclusion for DICES-350 with GPT-4 and
  correlation-based tests: the data is too small to resolve demographic
  differences.

The one group result that passes correction in some runs (judges fit US
raters on DICES-990 worse than random groups) is borderline: it passes Holm
in three runs, fails in the other two P0-family runs and in both robustness
checks, and is small in size. It should be treated as a hint for follow-up,
not a finding.

## 5. Groups are coarse, self-reported, and confounded

Gender, race, age, education and locale are self-reported categories that the
dataset itself defines (for example, DICES-990 has a large "Other" race
category). They overlap. In DICES-990, every rater recorded as White, Black or
LatinX is in the US locale, and the India-locale raters are all recorded as
"Asian/Asian subcontinent" (41) or "Other" (43). The analysis looks at one
attribute at a time, so it cannot say which attribute "explains" a gap. For
example, the US-locale and Latinx results on DICES-990 may reflect the same
raters. Raters also differ a lot *within* groups. The rater strictness that
motivated Amendment 1 is one example.

## 6. One chatbot, one time, one rater pool

All conversations come from adversarial sessions with a single chatbot
(LaMDA), collected in 2021–2022, and were rated by crowdworkers from the
dataset's vendor pool. The "human distribution" is the distribution of *that
pool*. It is not "society" or any population of users. Results may not carry
over to other chatbots, other kinds of conversation, or other rater pools.

## 7. Choices in building the human target

- **Unsure ratings** are left out of the primary target, and counted as half
  in a sensitivity analysis. Other treatments are possible.
- **Rater filter.** The primary analysis uses the DICES authors' removed-rater
  lists, with all raters kept as a sensitivity check.
- **Items rated "nonsensical/unfamiliar"** in DICES's first question are kept.
- The judge rates one binary question, "is the final response unsafe?". Human
  raters answered several safety questions, and only `Q_overall` is used.

## 8. The fixed guess borrows the other dataset, and shapes every skill score

So that the baselines are not fitted on the data they are scored on, the
fixed guess (constant baseline) for DICES-350 is the mean human `P(unsafe)`
of DICES-990, and the other way round. The two datasets do differ: 0.36 on
DICES-350 and 0.26 on DICES-990. So the fixed guess is too low on DICES-350
and too high on DICES-990. The Brier skill score is measured against it,
which has consequences:

- **A judge can beat the fixed guess without telling conversations apart.**
  SmolLM2-1.7B gives almost the same P(unsafe) to every conversation (mean
  0.28–0.29, SD 0.04–0.05, never above 0.5), and its positive BSS on both
  datasets comes mainly from that value being closer to each dataset's
  average. Its ranking is weak or at chance. Spearman ρ and AUROC, which do
  not use the fixed guess, are reported alongside BSS for this reason.
- **Post-hoc only:** against the same dataset's own average, a stronger bar
  that uses the test data's base rate, no judge does clearly better (BSS
  between +0.04 and −1.52). This comparison was not pre-registered and has
  no CIs.
- **The disagreement test (RQ3) inherits it.** The fixed guess is much more
  accurate in some thirds than others: on DICES-990 its Brier score is 0.021
  in the most-disagreed third and 0.089 in the least-disagreed third. In raw
  Brier score, every judge makes *smaller* errors on the most-disagreed
  third, on both datasets. The significant negative Δ on DICES-990 means
  "falls further behind the fixed guess", not "makes bigger errors".

## 9. Statistical caveats

- Bootstrap confidence intervals resample conversations. They treat the
  raters as fixed and do not include uncertainty from which raters happened
  to be recruited.
- Many quantities are reported. Holm corrections apply only within the
  families named in the methodology (groups within a judge and dataset; the
  RQ3 tests across judges and datasets within a run). There is no correction
  across prompts, robustness analyses or research questions, so a single
  Holm p just under 0.05 among dozens of group tests is weak evidence.
- The RQ3 tertile split (rank-based, ties broken by item order) is one way to
  define "high disagreement". Entropy near its maximum and moderate entropy
  can still contain quite different conversations.

## 10. What was changed after planning

- The demographic null was changed before any results (Amendment 1).
- The runs were done in stages for compute reasons, with a time-limited
  subset run first; the final runs cover the full planned scope (README,
  Deviations from plan).
- No analysis decision was changed after seeing results. The in-sample
  baseline comparison and the raw-Brier reading of RQ3 were added after the
  final results as extra context; they do not replace the pre-registered
  numbers.
