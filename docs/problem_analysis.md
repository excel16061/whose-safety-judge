# Problem analysis

## The practical problem

LLMs are increasingly used as *judges*: they label content as safe or unsafe
to filter training data, evaluate chatbots, or replace human annotators. That
saves money, but it quietly assumes there is one right answer the judge should
recover. For safety this assumption is shaky. In DICES, the same chatbot reply
is often called unsafe by some raters and safe by others. The split is
sometimes close to 50/50, and it sometimes runs along demographic lines.

If a judge outputs a single verdict, or a probability that doesn't reflect that
spread, two things can go wrong:

1. **Lost disagreement.** On a conversation where 45% of raters say "unsafe",
   a judge that says 95% or 5% misrepresents the human picture, even if its
   majority label is "right".
2. **Whose view wins.** If the judge sits closer to some groups of raters than
   to others, using it at scale amplifies those groups' safety norms.

## The research question

*When a small open LLM is used as a safety judge, does its probability of
"unsafe" match the full spread of human safety ratings? Does it match some
demographic groups worse than others? Does it get worse on the conversations
humans disagree about most?*

This splits into three testable questions (`methodology.md`, section 1):

- **RQ1, overall fit.** How far is the judge's probability from the share of
  raters saying unsafe, compared with simple baselines?
- **RQ2, whose view.** Is the judge further from a demographic group than from
  a random group of the same number of raters?
- **RQ3, disagreement.** Is the judge's skill (relative to a constant
  baseline) lower on the highest-disagreement third of conversations than on
  the lowest?

## Why these measurements

- The **target is a distribution**, not a label: per conversation, the share
  of raters who said "unsafe". A majority label would throw away exactly the
  disagreement under study.
- **Jensen–Shannon divergence** and the **Brier score** compare a predicted
  probability with that share directly. Unlike correlation, they penalise a
  judge whose numbers are systematically too high or too low, or too extreme.
  Calibration error (ECE) asks the same question in bins.
- **Baselines** answer the question "compared with what?". A constant
  prediction (the average share from the other dataset) is the bar for "has
  any skill at all". A keyword rule and a majority-class rule add context.
- **A random-rater-group null for RQ2.** Any group of 20 raters will differ
  from the overall pool somewhat by chance, and individual raters differ in
  strictness. A demographic gap only means something if it is bigger than the
  gap for random groups of the same raters (Amendment 1).
- **Relative skill for RQ3.** High-disagreement conversations have targets
  near 0.5, where Brier errors behave differently than near 0 or 1. Comparing
  the judge with the constant baseline within each tertile (the Brier skill
  score) separates "the judge is worse here" from "this kind of conversation
  is harder for any predictor".

## Scope and boundaries

- The project only classifies conversations that already exist in DICES, and
  never generates text.
- The judges are small open models on CPU, chosen for reproducibility and
  cost, not because they represent deployed judges (`limitations.md`, §1).
- Two public datasets (DICES-350 and DICES-990) are analysed separately. One
  rater pool and one chatbot limit how general the results can be
  (`limitations.md`, §5).
- All planned runs are done: three judges (0.5B–1.7B) on all 1,340
  conversations with prompt P0, and the prompt checks P1/P2 on one judge
  (README, "Deviations from plan"). The conclusions cover only these judges
  and prompts.

## What would count as an answer

A null result is a valid answer to each question. "The judge is no further
from any group than random groups are", or "the judge's skill does not drop
on high-disagreement conversations", are findings worth reporting. Given how
small the groups are, the most likely honest outcome for RQ2 is "can't tell",
and the confidence intervals are there to show how wide that "can't tell" is.
