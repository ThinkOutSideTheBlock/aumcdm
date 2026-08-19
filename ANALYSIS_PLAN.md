# Pre-registration — Adaptive Uncertainty-Aware MCDM under Censored Feedback

Frozen: 2026-08-14. Predictions in §2 are unchanged from first registration.
Every post-freeze modification is logged in §8 with date, reason, and whether any
outcome metric had already been inspected.

## 1. Claim

When an agent may abstain, and abstention yields no criterion outcome, outcome-
driven adaptation of criterion weights is systematically biased against the
criteria that trigger abstention. The safety weight decays; abstention decays
with it; realised tail risk worsens while mean utility improves. We call this the
**Confidence Ratchet**. It is removable by forced exploration plus clipped IPS,
at a cost of O(T^{2/3}) rather than O(T^{1/2}) regret.

The mechanism is not "the learner is badly tuned". It is that the sampling
distribution of observed outcomes is a function of the current weights, and the
map is contractive toward action. This is a property of the loop, not of the
optimiser — which is what P5 exists to test.

## 2. Predictions

| ID | Prediction | Primary statistic | Falsified if |
|----|-----------|-------------------|--------------|
| P1 | Safety weight decays under skip-on-abstain | total OLS drift of w_{t,2}, B4 | 95% CI covers 0, or is positive |
| P2 | Abstention decays, catastrophic rate rises | cat_last − cat_first, B4 | CI covers 0, or is negative |
| P3 | Adaptation is worse than no adaptation on tail risk | CVaR@10(B4) − CVaR@10(B3), paired | CI covers 0, or is positive |
| P4 | Forced exploration + clipped IPS removes the drift | w2 drift and CVaR@10, B5 − B4, paired | B5 not flatter / not safer than B4 |
| P5 | The pathology is the setting, not the optimiser | w2 drift of an online-ridge adapter (B4′) | B4′ flat while B4 drifts |

P5 is the most informative possible negative result. If B4′ shows no drift, the
finding is an exponentiated-gradient artefact and the theory section must be
rewritten around EG specifically. That reframing is publishable but is a
different, smaller paper, and the claim must change before the results are
written up, not after.

## 3. Design

**E1 (Day 1, locked).** Tool selection under harm asymmetry. K=6, M=4 criteria
(task value, safety, cost efficiency, epistemic gain), Beta-Bernoulli beliefs,
action set {Act(k), Gather(k), Ask, Abstain}, information budget 6 steps and 0.30
spend.

Episodes are heterogeneous by construction: each draws a random availability
subset of tools (P_AVAIL=0.5, at least one guaranteed) plus N_INIT=3 initial
observations per available tool. This is load-bearing. With a fixed prior,
`score()` would be identical every episode and the abstain rate would be exactly
0 or 1, making the entire censoring story vacuous. Availability heterogeneity
also makes censoring **correlated with harm**: rounds where only high-harm tools
exist are exactly the rounds that get censored. Assumption (A5) is therefore
instantiated structurally rather than assumed, which is the difference between a
demonstration and a tautology.

**E2 (Week 3).** Identical rule, non-stationary w*: safety weight steps up at
t = T/2. This is the honest steelman for B4 — it is the regime where adaptation
should help, and if B4 fails to track the step while B5 tracks it, P4
strengthens considerably.

**E3 (Week 4, optional).** LLM as black-box proposer of (Q, sigma); decision core
unchanged. Reported as a transfer check, never as a main result. If time is
short, this is the first thing cut.

**Budget.** T = 500 episodes, 20 seeds, seeds 10000–10019, fixed before any run.
w* = make_w_star(3.0). Prior w0 = (0.45, 0.40, 0.10, 0.05), deliberately
misaligned with w* so that some adaptation is genuinely warranted. The drift
claim is therefore not an artefact of initialising at the optimum — a reviewer
will check this first, and it is the single most likely reason for a desk reject.

## 4. Arms

B0 uniform random | B1 greedy task value (no abstention, no information) |
B3 static AU-MCDM | B4 adaptive skip-on-abstain | B5 forced-eps + clipped IPS |
ORACLE (true theta, true w*; normaliser only, never an arm under test).

B4 and B5 share every hyperparameter except the exploration/estimator pair, and
`test_arm_configurations_match_their_roles` asserts this. Any difference between
them is attributable to that pair alone. B4′ (online ridge) enters Week 2 for P5.

B2 (uncertainty-free AU-MCDM, kappa=0) is deferred: it is an ablation of the
decision rule, not of the learning loop, and Day 1 is about the loop.

## 5. Metrics

Primary: CVaR@10 of episode return; catastrophic-action rate (realised harm
>= 0.50); trajectory, total drift, and OLS slope of w_{t,2}; abstention rate,
first half vs second half.

Secondary: mean return, mean regret vs ORACLE, information spend, steps to
terminal, forced-exploration rate, censor rate.

Regret vs ORACLE is context only, not an endpoint: ORACLE sees true theta and is
unachievable by any arm, so its regret has no decision-theoretic interpretation
for the comparisons that matter.

## 6. Analysis

Seed is the unit of analysis (n=20). Cross-arm comparisons are paired within
seed, since arms share the seed and therefore the episode stream. Bootstrap 95%
CIs, 10k resamples over seeds. Holm correction across the five predictions.
No per-episode significance testing. Effect sizes reported as paired differences
with CIs; no p-value appears without its effect size.

The invariant suite asserts structure only — feasibility, simplex membership,
monotonicity in kappa and tau, exact overlap failure under determinism,
estimator bias on a synthetic stream. It deliberately contains **no assertion of
P1–P5**. Encoding the predictions as tests would make them unfalsifiable by
construction and would convert a scientific claim into a build artefact.

## 7. Sweeps and stopping rules

- **tau0 ∈ {0.45, 0.55, 0.65}**, varying abstention probability p_abs. Theory
  says the bias is first-order in p_abs, so P1 must *strengthen monotonically*
  with p_abs. This is the sharpest test available and the one separating "a
  mechanism" from "a bug in a toy". Non-monotonicity here is more damaging than
  a weak main effect.
- **eps ∈ {0, 0.02, 0.05, 0.10, 0.20}**, bracketing the predicted
  eps* ~ T^(−1/3) ≈ 0.13 at T=500. The prediction is a U-shaped CVaR curve:
  too little exploration leaves the bias, too much pays for it in mean return.
  A monotone curve falsifies the rate claim even if P4 holds at eps=0.10.
- **alpha ∈ {0.02, 0.05, 0.10}** as a robustness check only, not a claim.
- **rho ∈ {1, 3, 8}**: at rho=1 there is no harm asymmetry, so P1 should vanish.
  This is the built-in negative control, and it runs before the headline sweep.

Stopping: no sweep dimension is expanded after the outcome metrics are read.
If the tau0 monotonicity check fails, work stops and §1 is rewritten — no
additional environments are added to rescue the effect.

## 8. Change log

| Date | Change | Reason | Outcomes seen first? |
|------|--------|--------|----------------------|
| 2026-08-14 | Added per-episode availability sampling and N_INIT initial draws | Fixed-prior episodes made the abstain rate degenerate (0 or 1); the design could not exhibit censoring at all | No — found by `test_both_regimes_occur_at_the_locked_threshold`, before any outcome metric was computed |
| 2026-08-14 | Set clip M_c = 1/eps in IPSEGAdapter | With behaviour propensity exactly eps in the censored region the clip never binds, so the estimator is unbiased rather than merely low-variance; removes a confound in P4 | No |
| 2026-08-14 | Removed P1–P5 assertions from the invariant suite | Asserting the predictions would make them unfalsifiable | No |
| 2026-08-15 | P1 restated as skip-vs-full paired difference; added B4_full arm. Raw w2 drift cannot separate censoring bias from pull toward misaligned w0. Defect identified in the design, not only in the data. | Yes — Gate-1 T=500 drift sign was inspected first. |
| 2026-08-15 | tau0 to be raised to target 25–35% B3 censoring. 11% leaves insufficient censored mass. | Yes — B3 abstain rate inspected. No P-statistic used to choose the value. |
| 2026-08-15 | rho=1 recorded as an invalid negative control (w*[1] > w0[1] at rho=1). Superseded by aligned-start (w0 = w*). | Yes. |
| 2026-08-16 | tau0 frozen at 0.65 (B3 abstain ≈ 22%). Short search over {0.55,0.60,0.65,0.70}; chosen for band + force|cens ≈ eps. | Yes — abstain rate only; no P-stat used. |
| 2026-08-16 | STOP on original P1 (absolute safety-weight decay). Aligned run shows w2_drift > 0 from w0=w*. Claim rewritten to relative bias: drift(B4_skip)−drift(B4_full) < 0. P2 retired. P3 kept as conditional outcome finding under weight misspecification. | Yes — all Gate-1 and aligned outcomes inspected. |
| 2026-08-16 | tau0 frozen at 0.65 for remaining runs. Short sweep used only for abstain-band targeting. | Yes — abstain rate only. |

## 9. Theory targets (Week 2, stated in advance)

- **Prop. 1 (overlap failure).** A deterministic threshold rule assigns
  probability exactly zero to acting inside the abstain region; hence no
  consistent estimator of the full-information gradient exists from
  skip-on-abstain data. Asserted exactly, not statistically, by
  `test_lemma1_zero_propensity_for_action_under_censoring`.
- **Prop. 2 (bias direction).** The skip-on-abstain gradient has bias
  −p_abs · E[g | censored], and under (A5) — censoring correlated with harm —
  the safety coordinate of that term is negative. First-order in p_abs, which is
  what the tau0 sweep tests.
- **Prop. 3 (repair and price).** Forced exploration at rate eps with clipped
  IPS restores unbiasedness; balancing bias against exploration cost gives
  eps* ~ T^(−1/3) and regret O(T^(2/3)).

If Prop. 2's sign cannot be established without assuming what it concludes, the
paper is downgraded to an empirical finding with a bias decomposition and no
rate claim. That decision is made before writing §5, not during.