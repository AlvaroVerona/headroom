# Headroom

**Dynamic Credit Limit Optimization for a Digital Bank**

> A risk-adjusted credit decision engine combining machine learning, financial
> economics, and mathematical optimization.

A decision-intelligence platform for a digital bank/neobank: it validates messy
customer/transaction data, models probability of default and expected loss,
estimates customer profitability, and optimizes credit limits — per customer and
across the portfolio — subject to risk and exposure constraints, with Monte Carlo
loss simulation and stress testing on top.

> Status: project scaffolding in progress. This README will be filled in with
> architecture, methodology and actual results as each phase is built — no
> fabricated numbers.

## Roadmap

- [x] Phase 1 — foundation (structure, config, logging)
- [x] Phase 1 — synthetic data generator, tests
- [x] Phase 1 — data validation
- [x] Phase 2 — feature engineering
- [x] Phase 2 — EDA
- [x] Phase 2 — data quality reporting
- [x] Phase 3 — Logistic Regression baseline
- [x] Phase 3 — XGBoost advanced model
- [x] Phase 3 — probability calibration
- [x] Phase 3 — SHAP explainability
- [x] Phase 4 — expected loss model
- [x] Phase 4 — revenue model
- [x] Phase 4 — funding cost
- [x] Phase 4 — customer profitability
- [x] Phase 5 — individual credit limit optimization
- [x] Phase 5 — portfolio optimization (OR-Tools)
- [x] Phase 6 — stress testing / scenarios
- [x] Phase 6 — Monte Carlo loss simulation / VaR / CVaR
- [x] Phase 6 — drift monitoring (PSI)
- [ ] Phase 7 — dashboard, documentation, final audit

## Quickstart

```bash
make install
make generate-data
make validate-data
make test
```

## Synthetic data (Phase 1), actual output from `make generate-data` with seed=42

- 50,000 customers (35% prime / 40% near_prime / 25% subprime), 36 months of history
  (2023-01 to 2025-12 — extended from the spec's illustrative 12-24 months so that
  `default_12m`'s forward-looking label has a fully-observed 12-month window at each of
  three time-separated snapshots; see CLAUDE.md), 1,800,000 monthly behavior rows,
  ~4.3-4.5M transactions (~9x the spec's 500,000+ floor), 1,800,000 payments. Full
  generation runs in ~35 seconds.
- **`default_12m` rate: ~4.5-4.8%, stable across train/validation/test snapshots**
  (months 12/18/24), with a realistic, non-degenerate segment gradient — prime ~2.4%,
  near_prime ~4.5%, subprime ~7.7%. Defined as reaching 90+ days-past-due within 12
  months of the snapshot, among customers not already at 90+ DPD at the snapshot itself.
- Utilization by segment (final month): prime ~9-10%, near_prime ~39-40%, subprime ~67%
  — realistic levels, not pinned at the credit ceiling (see below).
- **Three real calibration bugs found and fixed while building this** (full detail in
  CLAUDE.md): (1) the original minimum-payment mechanic couldn't keep pace with ongoing
  spend, so balance never reached equilibrium and 34% of all customer-months ended up
  stuck exactly at the credit limit — fixed by making the actual payment a real fraction
  of balance, calibrated by solving the steady-state balance equation per segment;
  (2) even after that, subprime specifically stayed pinned at the ceiling (65.5% of
  subprime months) because their spend/payoff/limit combination made >100%-of-limit
  balance mathematically inevitable — fixed by constraining spend to available headroom
  (a purchase that would exceed the limit gets declined, the way a real card does) and
  raising their limit multiple; (3) the "distress" AR(1) shock used an EWMA formulation
  that actually *shrinks* variance as persistence increases, so raising persistence
  barely moved the default rate no matter how far it was pushed — fixed with the correct
  AR(1) parametrization, which holds variance constant regardless of persistence.
- Injected data-quality issues (rates in `config/settings.yaml`): missing values (income,
  employment status, optional metadata), duplicate customers/transactions, impossible
  values (age < 18, negative income/limits, utilization > 100%), inconsistent credit
  fields (available_credit ≠ limit − balance, balance > limit), and invalid customer_id
  references in accounts/transactions/payments.
- Reproducible: identical seed produces byte-identical output (verified for the full
  pipeline, not just customers.csv — see `tests/test_data_generation.py`).

## Data validation engine (Phase 1), actual output from `make validate-data`

- **Data Quality Score: 98.6 / 100** — by dataset: Customers 96.4, Credit Accounts 96.6,
  Transactions 98.5, Payments 99.1 (weighted average; every number computed from the
  actual check results, see `reports/outputs/quality_report.json`).
- 395,481 issues found across completeness, duplicate, validity and consistency checks on
  the full 4.37M-row raw dataset (customers + credit_accounts + transactions + payments);
  77,201 records quarantined for a CRITICAL or HIGH-severity issue. Full validation run:
  ~20 seconds.
- RAW → VALIDATED → QUARANTINED lineage is exact for all 4 datasets:
  `len(processed) + distinct(quarantined) == len(raw)`, verified in tests. Each quarantined
  record in `data/quarantine/*.csv` carries `record_id, rule_id, severity, reason,
  detected_at` (§11's exact schema).
- **A real bug found and fixed while building this**: the `by_segment` score breakdown
  originally summed issues from all 4 datasets (customers + their transactions + payments +
  accounts) while dividing by a customer-row-only denominator — a scale mismatch that
  floored every segment's score to 0.0, since a segment's customers can have tens of
  thousands of associated transaction rows. Fixed by scoping `by_segment` to the customers
  dataset's own issues, the only version of "data quality by segment" that's a coherent
  single number — it now reads Prime 96.2, Near-Prime 96.5, Subprime 96.6, consistent with
  the overall Customers score.

## Feature engineering (Phase 2), actual output from `make features`

- 47 features across demographic, income, credit, behavioral, delinquency, liquidity and
  3/6-month trend groups (§13), joined against `labels.csv`'s three time-separated
  snapshot cohorts (train/validation/test, months 12/18/24) — 144,003 total rows (~48,000
  per cohort; ~2,000 customers per cohort are dropped because their account or profile
  record was quarantined in Phase 1, not imputed).
- **No-future-leakage verified, not just asserted**: every trend/rolling feature is
  recomputed independently from the raw monthly table for a random sample and checked for
  exact agreement (`test_no_future_leakage_against_raw_monthly_table`).
- **A real bug found and fixed while building this**: the first version of the 3/6-month
  growth features used a naive `(current - past) / past` formula, which exploded to
  absurd magnitudes whenever `past` was near zero — `credit_utilization` and `end_balance`
  legitimately hit exactly 0 (a customer who pays off in full), and the observed max on
  the real output was ~1,000,000 for `credit_utilization_3m_growth` and ~1.08e10 for
  `end_balance_3m_growth`. Fixed by switching to a symmetric relative-change formula,
  bounded to [-2, 2] whenever both values are non-negative — verified on the real 144,003-
  row output.

## EDA (Phase 2), actual output from `make eda`

Full report with figures: `reports/outputs/eda_report.md` (16 charts in
`reports/figures/eda/`, every number backed by `reports/outputs/eda_stats.json`). Covers
every investigation point in spec §12. Selected findings:

- **Default rate: 4.69% overall** (142,575 eligible rows), prime 2.74% / near_prime 4.45% /
  subprime 7.83%, stable across the three time-separated snapshot cohorts.
- Strong, monotonic default-rate cuts by **utilization** (2% at 0-10% utilization → 21% at
  90-100%), **debt-to-income**, and **delinquency history** (90+ DPD customers default at
  ~12% vs. ~2% for a clean history) — the dataset's real risk signal.
- **Demographics carry almost no default signal**: age, employment status, and tenure are
  all flat within ~1pp across every bucket — the generator's hazard process is driven by
  segment/behavior, not demographics (see CLAUDE.md for what this means for the fairness
  section).
- Linear correlation with `default_12m` tops out at 0.16 even for the strongest predictors
  — expected for a ~4.7%-base-rate binary outcome, and the standard justification for
  using XGBoost/WOE-IV rather than linear correlation to size a variable's real
  predictive power (Phase 3).

## Data quality report (Phase 2), actual output from `make quality-report`

Full report with figures: `reports/outputs/quality_report.md` (7 charts in
`reports/figures/quality/`, sourced entirely from `reports/outputs/quality_report.json`).
Adds the score-by-month breakdown §11 requires (mean 98.67, essentially flat across all 36
months — quality issues were injected at a uniform rate independent of calendar month) and
the RAW → VALIDATED → QUARANTINED lineage funnel.

- **A real bug found and fixed while building this**: `by_field` divided every field's
  issue count by the total row count across *all 4 datasets combined* (~6.27M rows), even
  for a field that only exists in one dataset — e.g. `available_credit` only exists in
  `credit_accounts.csv` (50,000 rows), but scoring it against the full 6.27M-row
  denominator diluted its score from a true ~98.7 to a reported 99.99. Fixed by scoping
  each field's denominator to only the dataset(s) that actually contain that column — the
  same class of bug as the `by_segment` floor-to-0 bug above, milder here but real.

## Risk model — Logistic Regression baseline (Phase 3), actual output from `make train`

Full model card: `reports/model_cards/logistic_regression.md`. Time-based validation
(§15) reuses the existing snapshot cohorts directly (train = month 12, validation = month
18, test = month 24) — the same 50,000 customers at successively later points in time, so
the model is judged on generalizing *forward*, the property that matters for non-stationary
credit risk.

- **ROC-AUC 0.72-0.75** across train/validation/test (PR-AUC 0.15-0.17 against a ~4.7% base
  rate), from 32 features (of 46 candidates — `age` excluded as a protected characteristic,
  14 more dropped by automated correlation pruning at a >93% threshold, see below and the
  Explainability section for why 93% and not the original 95%).
- **A real bug found and fixed while building this**: the first run's top coefficients
  flatly contradicted the EDA — `monthly_income_6m_avg`/`12m_avg` showed up as the top
  RISK-INCREASING coefficients while `monthly_income_3m_avg` was the top RISK-REDUCING one,
  even though EDA found default rate falls monotonically with income. Root cause: the
  income trend windows are r > 0.99 correlated with each other (classic multicollinearity),
  plus several other exact-duplicate feature pairs (`income_stability` is a literal
  `1 - income_volatility` transform; `credit_utilization_Nm_growth` is mathematically
  identical to `end_balance_Nm_growth` in this dataset, since credit limits never change
  over the 36-month history). Fixed with a deterministic, automated correlation-pruning
  step (drop any feature > 95% correlated with an already-kept one, computed on the train
  cohort only) rather than a hand-picked exclusion list — after pruning, top coefficients
  are directionally consistent with EDA (delinquency and utilization risk-increasing,
  income risk-reducing).
- **Predicted probabilities are not yet usable as real PD estimates**: `class_weight=
  'balanced'` (used to handle the ~4.7% base rate) inflates mean predicted probability to
  ~42-48%, roughly 10x the true rate — expected and documented, not a bug; exactly what the
  upcoming Probability Calibration piece (§17) fixes.

## Risk model — XGBoost advanced model (Phase 3), actual output from `make train-xgboost`

Full model card with baseline-vs-advanced comparison table: `reports/model_cards/xgboost.md`.
Reuses the LR baseline's time-based split; uses the full 46-feature set (no correlation
pruning needed — trees handle correlated/redundant features natively).

- **ROC-AUC 0.730 on test (vs. 0.724 for the LR baseline)**, PR-AUC 0.165 (vs. 0.148) — a
  modest, genuine improvement using the complete feature set, the expected trade-off for
  the interpretability the LR baseline provides instead.
- **A real bug found and fixed while tuning this**: the first run used log loss as the
  early-stopping metric, which never triggered early stopping at all (ran the full 500-tree
  budget) and produced train ROC-AUC 0.879 against test 0.702 — a severe overfit that made
  the "advanced" model *worse* than the LR baseline, defeating the entire point of building
  it. Root cause: under reweighted imbalanced training (`scale_pos_weight`), log loss keeps
  rewarding sharper (already-inflated) predicted probabilities long after real ranking
  ability on unseen data stops improving. Fixed by early-stopping on ROC-AUC directly (the
  metric actually reported and compared) plus shallower trees — early stopping now
  triggers at round 124 of 500, with train/validation/test ROC-AUC sitting close together
  (~0.73-0.76) instead of collapsing apart.
- Top gain-based feature importances (`recent_delinquency`, `delinquency_count`,
  `max_days_past_due`, `credit_utilization`) match the EDA's strongest risk drivers.

## Probability calibration (Phase 3), actual output from `make calibrate`

Full report with reliability diagrams: `reports/model_cards/calibration.md`. Fixes exactly
the miscalibration both risk models' cards flagged.

- **Mean predicted probability drops from ~46-48% to ~5.0-5.5%** (actual base rate ~4.8%)
  for both models — Brier score improves from ~0.23-0.26 to ~0.043-0.044, and Expected
  Calibration Error drops from ~0.41-0.44 to ~0.002-0.007 (two orders of magnitude), while
  **ROC-AUC stays within 0.001** of the uncalibrated model — calibration reshapes the
  probability scale only, not the ranking.
- Sigmoid (Platt) won for Logistic Regression, isotonic for XGBoost — chosen per model by
  Brier score on a held-out half of the validation cohort, disjoint from the half used to
  fit the calibrators; the test cohort is touched exactly once, for final reporting only.

## SHAP explainability (Phase 3), actual output from `make explain`

Full report with figures and individual customer examples: `reports/model_cards/explainability.md`.
Global importance, SHAP summary plot, SHAP dependence plots (top 4 features), and 3 full
individual customer explanations (lowest/median/highest predicted risk in the sample) for
both models — `delinquency_count`, `recent_delinquency` and `days_past_due` dominate for
both, consistent with every prior piece's findings.

- **A real, visible bug found and fixed while building this**: the SHAP summary plot showed
  `cash_buffer` (rank #2 by importance) with a POSITIVE SHAP value for HIGH `cash_buffer` —
  more available credit headroom reading as *more* risky, backwards from intuition and from
  `cash_buffer`'s own raw correlation with `default_12m` (-0.083, genuinely protective).
  Root cause: `cash_buffer = credit_exposure - end_balance` is r=0.9465 with
  `credit_exposure` — just under the LR baseline's original 0.95 correlation-pruning cutoff,
  so it survived and became a coefficient artifact large enough to be immediately visible in
  a headline chart (smaller residuals from the same cause were already documented as
  acceptable in the LR baseline's own model card). Fixed by lowering the pruning threshold
  to 0.93 in `risk_model.py` (catching 3 more near-miss pairs found at the same time) and
  retraining the LR baseline, calibration, and this piece in sequence — `cash_buffer` no
  longer appears in the LR model at all.
- Also fixed: stale dependence-plot images from a previous run's top-N features weren't
  being cleaned up between runs (harmless until a feature drops out of the top-N, as
  `cash_buffer` just did) — the output directory is now cleared at the start of each run.

## Expected Loss model (Phase 4), actual output from `make expected-loss`

Full model card: `reports/model_cards/expected_loss.md`. Expected Loss = PD (calibrated
XGBoost, forced to 1.0 for customers already at 90+ DPD) × LGD (segment lookup) × EAD (a
genuine regression model predicting labels.csv's `future_utilization`, R² ~0.58-0.61, then
scaled by credit limit) — not a formula dressed up as a model.

- **Mean Expected Loss by segment: prime €12.49, near_prime €59.13, subprime €122.45** —
  correctly ordered. Validation check: mean EL is €112.68 for customers who actually
  defaulted within 12 months vs. €44.04 for those who didn't (known only from the snapshot,
  before the outcome exists) — a real, out-of-sample confirmation that the whole PD → LGD →
  EAD pipeline points the right direction, not just each piece individually.
- Total portfolio Expected Loss on the test cohort extrapolates to ~€2.93M across the full
  ~50,000-customer portfolio — remarkably close to `config/settings.yaml`'s
  `portfolio_expected_loss_limit` placeholder of €3,000,000, set back in Phase 1 before any
  real computation existed to check it against.
- **A real bug found and fixed while building this**: `credit_exposure` carries real Phase 1
  missing-value injections (~2% of rows) — multiplying a missing credit limit into EAD, then
  Expected Loss, and taking a plain mean/sum silently produced `NaN` for every top-line
  aggregate, even though the by-segment and by-outcome breakdowns looked fine (`pandas`'
  `groupby().mean()` skips NaN by default, masking the problem). Fixed by explicitly
  excluding the affected rows (976 of 48,001 on test) and reporting the exclusion count
  directly, rather than silently dropping or silently producing NaN.

## Revenue model (Phase 4), actual output from `make economics`

Full model card: `reports/model_cards/revenue_model.md`. Interest Revenue = Average Balance
× APR / 12, Interchange = Monthly Spend × 1.2%, Late Fee = €25 in any month with
`days_past_due > 0` (FX/other product fees deliberately not modeled — the generated data is
100% EUR with no basis for either).

- **Formula independently verified against real generated data**: this is the exact same
  formula Phase 1's label generator used to produce `future_interest_revenue`/
  `future_interchange_revenue` — reconstructing those labels from the raw monthly table with
  this module's formula reproduces them to within €0.01 (rounding only) across 147,150 rows.
- **Mean monthly revenue by segment**: prime €23.47, near_prime €40.70, subprime €44.47 —
  subprime generates the most revenue (high APR × high balance) but also the highest
  Expected Loss; annualized net (revenue − EL, before funding/operational cost): prime
  ~€269/year, near_prime ~€429, subprime ~€411 — not the "subprime is simply worse" story a
  risk-only view would suggest, which is exactly why Customer Profitability (§22, next)
  needs the full revenue/loss/cost picture rather than any single piece alone.

## Funding cost (Phase 4), actual output from `make economics`

Full model card: `reports/model_cards/funding_cost.md`. Funding Cost = Average Balance ×
Funding Rate / 12. Rate varies by macroeconomic scenario (base 5%, recession 6%,
high_interest_rate 8%, consumer_stress unchanged at 5%) using config values already
scaffolded for Phase 6's stress testing; deliberately flat across customer segment (a bank's
cost of funds is a treasury-level blended rate, not a credit-risk quantity — segment risk is
already priced via PD/LGD and APR elsewhere).

- **Total portfolio monthly funding cost by scenario**: base €213,178, recession €255,814,
  high_interest_rate €341,085 — a 60% increase in funding cost under the high-interest-rate
  scenario relative to base.
- **Net Interest Margin (base scenario)**: mean €16.67/customer/month, positive for 95.3% of
  customers — verified that every one of the remaining rows is NIM = 0 exactly (zero balance
  that month), never negative, since every segment's APR (16.9-27.9%) comfortably exceeds
  every scenario's funding rate (5-8%).

## Customer profitability (Phase 4), actual output from `make economics` — closes Phase 4

Full model card with per-segment candidate-limit tables and profit curves:
`reports/model_cards/profitability.md`. Expected Customer Profit = Total Revenue − Expected
Loss − Funding Cost − Operational Cost, computed both at each customer's current limit and
across the full €500-€10,000 candidate grid for one representative example customer per
segment.

- **Introduces the balance-response-to-candidate-limit model** the earlier pieces deferred:
  `balance(L) = min(current_balance × (L/current_limit)^0.3, L)`, anchored exactly at each
  customer's own observed point, giving the diminishing-returns concave Limit-vs-Profit curve
  by construction. Deliberately NOT derived from the EAD/PD models — `credit_exposure` had
  ~0 importance (rank 13/46) in the fitted EAD regressor, because this generator never varies
  a customer's credit limit over their history, so neither model ever saw the within-customer
  variation needed to learn a genuine causal limit-response.
- **Mean annual Expected Profit by segment: prime €203.08, near_prime €318.02, subprime
  €299.07** — all positive, portfolio total ~€12.2M/year on the test cohort.
- **A real bug found and fixed while building this**: the first run produced NEGATIVE mean
  profit for every segment (prime -€3.42, near_prime -€44.54, subprime -€102.21/month) — not
  a real economics finding but a units mismatch: Expected Loss is inherently a 12-month
  figure (PD is a 12-month default probability) while revenue/funding cost were left as
  monthly figures, so subtracting one from the other was comparing a year of loss against a
  month of revenue. Caught by cross-checking against the revenue model's own earlier
  "revenue vs. EL" preview, which had already annualized correctly and found every segment
  profitable — the disagreement between the two pieces surfaced the bug. Fixed by putting
  every term on a consistent annual basis.

## Individual credit limit optimization (Phase 5), actual output from `make optimize-customer`

Full model card: `reports/model_cards/customer_optimization.md`. For each customer,
evaluate every candidate limit and select the feasible one maximizing Expected Profit,
subject to income/risk/utilization/debt constraints (§24); no feasible candidate → decline,
not "approve at the smallest grid value." This is an independent per-customer argmax, not an
OR-Tools problem — that's next, for portfolio optimization, where the exposure/loss
constraints genuinely couple every customer's limit to everyone else's.

- **87.0% approved**, decline driven almost entirely by the risk constraint (5,791 of 5,798
  declines) — matches an independent sanity check (~13% of the test cohort has PD above the
  8% threshold) done before building the optimizer.
- **Mean profit uplift vs. current limits is positive for every segment**: prime €4.21,
  near_prime €49.68, subprime €104.46/year; €1.69M/year total across the approved test
  cohort.
- **A real bug found and fixed while building this**: for any customer whose Expected Profit
  doesn't actually depend on the candidate limit (most commonly a zero-balance customer, since
  interest revenue/EL/funding cost are all balance-driven), `pandas.groupby().idxmax()`
  silently picked the SMALLEST candidate — a real "cut this customer's credit line"
  recommendation with zero economic basis, found via an unexplained left-tail spike in the
  limit-change chart. ~40% of ALL "decrease" recommendations were this exact artifact. Fixed
  by breaking ties toward whichever candidate is closest to the customer's current limit
  instead of accepting the arbitrary first-occurrence default — after the fix, 83% of the
  remaining decreases are explained by a real, different cause: the customer's actual current
  limit already exceeds the candidate grid's own maximum.

## Portfolio credit limit optimization (Phase 5), actual output from `make optimize-portfolio` — closes Phase 5

Full model card: `reports/model_cards/portfolio_optimization.md`. OR-Tools CBC MIP:
maximize total Expected Profit across individually-approved customers, subject to portfolio
exposure/loss/average-PD/high-risk-concentration constraints — one binary variable per
customer (fund at their stage-1-optimal limit, or not), ~38,900 variables, solved to
**OPTIMAL in ~1 second**.

- **A genuinely binding problem, not a rubber stamp**: funding every individually-approved
  customer would need €320.5M exposure and 18.4% high-risk concentration, both over the
  €150M / 15% portfolio limits — Expected Loss and average PD both have slack, so exposure
  and risk-concentration are the real trade-offs the solver has to resolve.
- **MIP-optimal funds 17,205 of 38,879 customers (44.3%)**, hitting both binding constraints
  exactly at their limits, for €8.19M/year total profit — **7.77% better than a
  profit-per-exposure greedy heuristic** (€590K/year), because the MIP jointly respects all
  4 constraints at once instead of a single ratio blind to which one actually binds.
- **Funding rate is NOT "safest first"**: prime (safest segment) gets funded the LEAST
  (20%, vs. 72% for near_prime) — exposure is the binding constraint, and prime's
  profit-per-euro-of-exposure (€0.022) is the segment's worst, since prime customers carry
  large individually-optimal limits for comparatively modest absolute profit. Subprime has
  the BEST profit-per-euro (€0.057) but is capped by the also-binding high-risk exposure
  constraint, leaving near_prime with the most headroom under both constraints at once.
- No bug found while building this piece — verified the two ratio constraints (average PD,
  high-risk exposure %) actually linearize exactly (not an approximation) by recomputing the
  TRUE ratio on the solved solution and checking it still holds.

## Stress testing / scenarios (Phase 6), actual output from `make stress`

Full model card: `reports/model_cards/scenarios.md`. Re-evaluates the FUNDED portfolio
(Phase 5's 17,205-customer MIP solution) under each macro scenario in
`config/settings.yaml`. Feature-level shocks (income, spending, income volatility,
delinquency, cash buffer) are re-scored through the existing calibrated XGBoost PD model;
`default_multiplier`/`funding_rate_shift`/`apr_shift` are direct overlays on the downstream
economics.

| Scenario | Mean PD | Total Expected Loss | EL vs. base | Total Expected Profit |
|---|---|---|---|---|
| base | 3.56% | €939,348 | 1.00x | €8,186,017 |
| recession | 5.85% | €1,547,545 | 1.65x | €6,918,281 |
| high_interest_rate | 3.56% | €939,348 | 1.00x | €8,186,017 |
| consumer_stress | 3.62% | €959,011 | 1.02x | €8,166,353 |

- **`high_interest_rate`'s net profit exactly matches base, to the cent** — `funding_rate_shift`
  and `apr_shift` are both +3pp, and since interest revenue and funding cost scale the same
  `balance` figure, the two shifts cancel exactly in net profit even though gross revenue and
  gross cost both moved by real, non-trivial amounts (+€1.03M each). A genuine finding about
  what happens when a bank's own repricing tracks its cost of funds one-for-one, not a bug.
- Recession is the worst scenario on every axis, as expected: default_multiplier 1.6 plus
  income/spending feature shocks push mean PD up 64% and Expected Loss up 65%.

## Monte Carlo loss simulation / VaR / CVaR (Phase 6), actual output from `make simulate`

Full model card: `reports/model_cards/monte_carlo.md`. Single-factor (Vasicek/ASRF) Gaussian
copula default simulation, 10,000 draws, asset correlation 0.04 (Basel's own flat correlation
for Qualifying Revolving Retail Exposures — credit cards — not fit to this data).

| Scenario | Analytical EL | MC mean loss | VaR 99% | CVaR 99% | Economic Capital (99%) |
|---|---|---|---|---|---|
| base | €939,348 | €943,821 | €2,241,452 | €2,557,487 | €1,302,104 |
| recession | €1,547,545 | €1,554,468 | €3,375,848 | €3,797,512 | €1,828,303 |

- **Sanity check passed**: MC mean loss lands within 0.5% of the analytical `PD × LGD × EAD`
  sum on every scenario — the simulation is unbiased, as it must be (both describe the same
  expectation; the simulation adds the TAIL shape analytical EL can't).
- VaR 99% is ~2.4x Expected Loss even at 4% asset correlation — correlated defaults create a
  materially fatter tail than a plain binomial loss count would, the entire reason to run a
  copula simulation instead of just scaling the mean.

## Drift monitoring / PSI (Phase 6), actual output from `make monitor`

Full model card: `reports/model_cards/monitoring.md`. Population Stability Index across the
three real, time-separated snapshot vintages (train=month 12, validation=month 18,
test=month 24) — no synthetic "production" data exists past month 24, so adjacent real
vintages stand in for successive monitoring periods.

- **PD score is STABLE across every period** (PSI 0.01–0.07, well under the 0.10 warning
  band) — the calibrated model's own output distribution hasn't meaningfully shifted.
- **One real, explained exception**: `delinquency_count` (a top-6 SHAP feature) shows
  SIGNIFICANT PSI (0.35, train vs. test) — but this is a vintage-design artifact, not true
  population drift: `train`/`validation`/`test` are the SAME 50,000 customers observed at
  later points in their own history, so a LIFETIME/cumulative counter mechanically
  accumulates more events by a later snapshot. `days_past_due` (point-in-time) and
  `recent_delinquency` (recent-window) both stay STABLE, exactly the pattern that
  explanation predicts — see the model card for the full case.
