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
- [ ] Phase 2 — EDA, feature engineering, data quality reporting
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
