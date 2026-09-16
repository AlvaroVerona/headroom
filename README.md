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
- [ ] Phase 2 — data quality reporting
- [ ] Phase 3 — risk models (Logistic Regression, XGBoost, calibration, SHAP)
- [ ] Phase 4 — economics (revenue, funding cost, expected loss, profitability)
- [ ] Phase 5 — optimization (individual, portfolio, decision policy)
- [ ] Phase 6 — risk management (scenarios, Monte Carlo, VaR/CVaR, monitoring)
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
