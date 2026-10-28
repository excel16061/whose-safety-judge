# Related work

Only papers that were downloaded and read for this project are cited. The
table at the end says how each one was checked.

## The dataset: DICES

Aroyo et al. (2023), *DICES Dataset: Diversity in Conversational AI Evaluation
for Safety* (arXiv:2306.11247), released the conversations and ratings used
here. Crowdworkers held adversarial conversations with the LaMDA chatbot, and a
separate, demographically diverse pool of raters judged the safety of the
chatbot's final response. The paper presents DICES-350 and DICES-990. DICES-990
was rated by raters from two locales (India and the US), and DICES-350 also has
expert ("gold") safety labels. DICES treats disagreement between raters as
information to study, not noise to average away. This project takes the same
view: the target is the full share of raters who say "unsafe", not a single
majority label.

## Disagreement between rater groups: GRASP

Prabhakaran, Homan et al. (2024), *GRASP: A Disagreement Analysis Framework to
Assess Group Associations in Perspectives* (arXiv:2311.05074), give metrics for
whether rater subgroups systematically disagree, together with significance
tests based on permutation. Their permutation shuffles the demographic labels
across raters while keeping each rater's ratings together. That respects the
fact that one rater's ratings are not independent of each other. This project's
demographic test (RQ2) uses the same idea of drawing whole raters (methodology,
Amendment 1). It compares a *judge* with each group rather than groups with
each other.

## The closest prior work: annotation alignment

Movva, Koh and Pierson (EMNLP 2024), *Annotation alignment: Comparing LLM and
human annotations of conversational safety* (arXiv:2406.06369), ask three
questions on DICES-350:

- **RQ1:** how well LLM ratings correlate with the average of the 112 raters.
  GPT-4 reaches Pearson r = 0.59, higher than the median single rater's
  correlation with the average (r = 0.51).
- **RQ2:** whether that correlation differs between race and gender groups.
  None of the group correlations falls outside a null built from 5,000
  permutations of rater demographics. They conclude that DICES-350 is too
  small to resolve demographic differences, and that there is substantial
  idiosyncratic variation within groups.
- **RQ3:** whether GPT-4 can predict when one group rates a conversation as
  more unsafe than another. It cannot.

Their models were large API and open-weight models: GPT-3.5, GPT-4, GPT-4o,
Gemini 1.5 Pro and Llama 3.1 405B Instruct. Their ratings were 1–5 Likert
scores, binarised at ≥ 3 when compared with individual raters. They removed
11 annotators who disagreed with the majority more than 80% of the time.
**They also used DICES-990, but only as held-out data for engineering their
prompt.** It was not an evaluation set.

How this project differs, without claiming to be first at anything:

1. **Target.** Each judge's probability is compared with the whole per-item
   human distribution (the share saying unsafe), using Jensen–Shannon
   divergence, Brier score and calibration error. Movva et al. use correlation
   with the average rating. Correlation can be high even when every predicted
   probability is badly off in level or spread. A Spearman correlation is still
   reported so the two can be compared.
2. **Data.** DICES-990 is used as a second *evaluation* set, analysed
   separately from DICES-350. It has more conversations, and a locale group.
3. **Judges.** These are small open models (0.5B–1.7B parameters) run on CPU.
   Their probabilities come from next-token logits, with no sampling. This is
   cheaper and fully reproducible, but the models are much weaker than those
   Movva et al. tested.
4. **Disagreement.** RQ3 here asks whether the judge loses skill on the
   conversations humans disagree on most. Movva et al.'s RQ3 is a different
   question: they ask whether an LLM can predict differences *between* groups.

## Why disagreement matters: Artificial Hivemind

Jiang et al. (NeurIPS 2025, Datasets and Benchmarks track), *Artificial Hivemind: The Open-Ended Homogeneity
of Language Models (and Beyond)* (arXiv:2510.22954), introduce INFINITY-CHAT,
a set of open-ended queries with dense human ratings. They report that "LMs,
reward models, and LM judges are less well calibrated to human ratings on model
generations that elicit differing idiosyncratic annotator preferences, despite
maintaining comparable overall quality." This project's RQ3 asks the analogous
question for safety judgements: do judges lose skill relative to a baseline on
the conversations where human safety raters disagree most? The settings differ:
here it is binary safety ratings of chatbot replies, not quality ratings of
creative generations. A result in one setting does not carry over to the other.

## Pluralistic alignment: what "matching the distribution" means

Sorensen et al. (ICML 2024), *Position: A Roadmap to Pluralistic Alignment*
(arXiv:2402.05070), distinguish three kinds of pluralism. One is *distributional*
pluralism: models "well-calibrated to a given population in distribution". They
advocate "more distributionally pluralistic evaluations with respect to clearly
specified groups of people". RQ1 and RQ2 here are a small, concrete example of
that kind of evaluation. The population is the DICES rater pool, and the
specified groups are its self-reported demographic groups.

## How each paper was checked

All PDFs were downloaded from arXiv on 2026-10-27 and converted to text with
`pdftotext`. Every claim above was found in that text by searching for the
specific passage, not taken from memory or from an abstract alone.

| Paper | arXiv | What was checked in the text |
|---|---|---|
| Aroyo et al., DICES | 2306.11247 | title and authors; LaMDA conversations; DICES-350/990; India and US locales for DICES-990; gold labels on DICES-350 |
| Prabhakaran, Homan et al., GRASP | 2311.05074 | title and authors; permutation tests that shuffle rater demographics (pseudocode in the paper) |
| Sorensen et al., Pluralistic alignment | 2402.05070 | venue line (ICML 2024, PMLR 235); title and authors (includes Liwei Jiang); the three kinds of pluralism; the quoted sentences |
| Movva, Koh & Pierson, Annotation alignment | 2406.06369 | venue (EMNLP 2024) taken from the project brief, not stated in the arXiv PDF; the three RQs; r = 0.59 vs 0.51; 5,000-permutation demographic null; underpowered conclusion; models; Likert binarisation; 11 removed annotators; DICES-990 used for prompt engineering |
| Jiang et al., Artificial Hivemind | 2510.22954 | venue line (NeurIPS 2025 Datasets and Benchmarks track); title and authors; INFINITY-CHAT; the quoted calibration finding |

Papers that are often cited near this topic but were **not** opened (for
example, general LLM-as-a-judge benchmarks) are deliberately not cited.
