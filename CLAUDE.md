# Headroom — engineering notes for Claude Code

Dynamic Credit Limit Optimization for a Digital Bank. Portfolio project: raw,
imperfect banking data → validated data → risk model (PD) → expected loss →
revenue/profitability → per-customer and portfolio credit limit optimization
→ Monte Carlo/stress testing → Streamlit dashboard.

Sibling project: `../breach-point` (Financial Data Quality/Forecasting/
Liquidity Optimization). Same engineering conventions, different domain —
don't assume code is shared, but the same principles apply (never fabricate
a result, config-driven constants, RAW→VALIDATED→QUARANTINED lineage,
recalibrate placeholder thresholds once real data exists).

## Key decisions (do not relitigate without asking)

- **XGBoost is used** (unlike breach-point's sklearn-only forecasting
  choice) — this spec explicitly asks for it as the advanced risk model
  ("Prefer XGBoost unless there is a strong reason otherwise"), confirmed
  with the user. Logistic Regression is the baseline.
- **36 months of history generated, not the spec's illustrative 12-24.**
  `default_12m` is a forward-looking label: it needs a full 12-month
  window observed AFTER each prediction snapshot, and §15's time-based
  train/validation/test split needs three separate snapshot dates, each
  with its own fully-observed forward window. Three non-degenerate
  snapshots each needing 12 forward months is the reason for 36 total.
- **Snapshot design (`config/settings.yaml: snapshots`)**: train snapshot
  at month 12 (label observed through month 24), validation at month 18
  (through month 30), test at month 24 (through month 36). This is the
  standard credit-risk "vintage" out-of-time validation design — the same
  50,000 customers appear in all three cohorts, but each cohort's features
  only use data up to its own snapshot month (no future-feature leakage),
  and the three cohorts are genuinely time-separated for the split.
  Outcome windows between cohorts overlap somewhat (e.g. train's 13-24 vs.
  validation's 19-30) — that's normal for rolling-vintage validation, not
  leakage, since it's the FEATURES that must never see the future, not the
  outcome-window calendar overlap.
- **LGD varies by customer_segment** (prime 0.55 / near_prime 0.70 /
  subprime 0.85, base_lgd 0.70 is the portfolio-average reference from the
  spec's own example) — unsecured revolving credit has no collateral, so
  segment proxies for cure likelihood.
- **Credit limit income constraint uses monthly_income × 4.0**, not annual
  income — matches how real unsecured-revolving-credit policies are
  typically expressed (3-5x monthly income for prime), and keeps the
  constraint's scale sensible against the €500-€10,000 candidate grid.
- **Portfolio exposure/loss constraints in config are placeholders** to be
  recalibrated once the actual generated portfolio's scale is known (the
  same pattern as breach-point's `liquidity.minimum_cash` recalibration —
  an unrecalibrated limit is just as likely to be meaninglessly loose or
  impossibly tight as a guess is to be right).
- **Balance/spend response to credit limit is a real modeling choice the
  spec doesn't specify** (needed for the Limit vs. Profit curve in §30 to
  have a genuine concave shape — more limit should increase utilization
  and revenue with diminishing returns, capped by income-driven spending
  capacity, while increasing exposure/expected loss). Document the exact
  functional form used once built in `src/credit_limit_optimizer/models/
  profitability.py` and here.
- **Accounting identities in `credit_accounts.csv`** (`utilization_rate ≈
  balance/limit`, `available_credit ≈ limit - balance`) hold exactly in
  the clean generative data and are deliberately broken only in the
  injected data-quality subset — mirrors breach-point's Balance Sheet
  approach (clean data reconciles by construction; injected noise is what
  breaks it, on purpose, for the quality engine to catch).

## Phase 1 data generation — real bugs found and fixed

- **Balance never reached equilibrium; 34% of all customer-months ended up
  pinned exactly at the credit ceiling** (`src/credit_limit_optimizer/
  data/behavior_series.py`). The original payment mechanic paid a tiny
  contractual minimum (~3% of balance, floor €25) regardless of ongoing
  spend, which for most non-full-payoff customers is far smaller than
  their monthly spend — balance necessarily grows every month with no
  equilibrium, eventually hitting and sticking at the limit. Fixed by
  making the *actual* payment a real fraction of balance
  (`REVOLVER_PAYOFF_FRACTION_BY_SEGMENT`, replacing the old scheduled-
  amount-based payment), calibrated by solving the steady-state equation
  `balance* = spend*(1-f)/f` for each segment's target utilization, then
  empirically re-checked (full-payoff months periodically zero the
  balance and pull the time-average utilization below the pure-revolver
  formula's prediction, so the analytical solve is a starting point, not
  the final answer).
- **Even after that fix, subprime stayed pinned at the ceiling (65.5% of
  subprime customer-months).** The steady-state formula showed subprime's
  equilibrium balance was mathematically >100% of their limit given their
  spend_ratio/revolver_fraction/limit_multiple combination — no fraction
  of REALISTIC payoff behavior could keep it under the limit; they were
  guaranteed to hit and stay at the ceiling by construction. Fixed with
  two changes: (1) spend is now constrained by available headroom
  (`credit_limit - balance_prev`) before it's added to balance — a real
  card purchase that would exceed the limit gets declined, it doesn't
  silently inflate balance past the ceiling and then get clipped, which
  is what pinned utilization at exactly 100.000% rather than letting it
  fluctuate just under it; (2) `LIMIT_INCOME_MULTIPLE_BY_SEGMENT["subprime"]`
  raised 0.8 -> 1.0. Result: subprime settled at ~67% average utilization,
  10.7% at-cap (down from 65.5%) -- realistic for a segment that
  genuinely does run close to its limit, without being permanently
  pinned there. Both bugs are regression-tested in `test_data_generation.py`
  (`test_utilization_not_pinned_at_ceiling`, `test_utilization_ordered_by_segment`).
- **The AR(1) "distress" shock initially had almost no effect on the
  default rate no matter how much persistence was increased** —
  `rho*prev + (1-rho)*innovation` looks like a standard AR(1) but actually
  *shrinks* variance as rho rises (the innovation weight shrinks faster
  than persistence extends memory), so raising persistence barely moved
  the default_12m rate. Fixed with the correct parametrization
  (`rho*prev + sqrt(1-rho^2)*innovation`), which holds variance constant
  regardless of rho — after that, persistence and the shock coefficient
  actually calibrate the way you'd expect. Landed on `DISTRESS_PERSISTENCE
  = 0.92`, `DISTRESS_SHOCK_COEF = 0.14`, giving a stable ~4.5-4.8%
  default_12m rate across all three snapshot cohorts (train/validation/
  test), with a realistic, non-degenerate segment gradient (prime ~2.4%,
  near_prime ~4.5%, subprime ~7.7%).
- **`payment_ratio` (paid/scheduled) has a heavy right skew** (median
  ~12.8x, 89.5% of months > 5x) once the payoff-fraction fix above was in
  place — this is mathematically expected, not a bug: `scheduled_amount`
  is deliberately a small contractual-minimum floor (3% of balance),
  while most non-distressed customers now genuinely pay a much larger
  real fraction of their balance. Left as-is in the raw data (an honest
  reflection of "how far above the minimum did they pay"); feature
  engineering should cap or log-transform it before using it as a model
  input, not the generator's job to produce an artificially bounded
  distribution.
- **transactions.csv is a deliberate SAMPLE, not an exhaustive replay of
  every purchase** (~4.3-4.5M rows at full 50,000-customer scale, still
  ~9x the spec's 500,000+ floor). A literal "every purchase" interpretation
  at this customer/month scale would run into the tens of millions of
  rows — unworkable for iteration speed in a portfolio project. Each
  customer-month gets up to 3 category-lump transactions instead; the
  behavioral aggregates a real model trains on
  (essential_spend/discretionary_spend/etc.) are generated independently
  in monthly_customer_behavior.csv, not derived from transactions.csv.

## Engineering principles

Business logic lives in `src/`, never in notebooks. All stochastic code
takes its seed from `config/settings.yaml` (`random_seed: 42`). No print
statements for anything that matters; use
`credit_limit_optimizer.utils.logging.get_logger(__name__)`. Never fabricate
a result — every number in the README/dashboard/model card must come from
an actual pipeline run. SHAP summary plots may subsample rows (e.g. 1,000-
2,000) for runtime; document the sample size wherever shown. Never use a
protected characteristic (age, gender proxy, etc.) as a direct model input
for the credit decision without discussing it in the Fairness section.

## Workflow

Work in phases per the spec's own Development Order (§53): foundation →
analytics (EDA/features/quality reporting) → risk (LR/XGBoost/calibration/
SHAP) → economics (revenue/funding/expected loss/profitability) →
optimization (individual/portfolio/decision policy) → risk management
(scenarios/Monte Carlo/VaR/monitoring) → product (dashboard/docs/tests).
After each phase, run the relevant tests and inspect generated output
before moving to the next one.
