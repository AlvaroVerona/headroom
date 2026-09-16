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

## Phase 1 validation — real bugs found and fixed

- **`record_id` is position-based, assigned at ingestion, never a business
  key** (`src/credit_limit_optimizer/data/ingestion.py`, same pattern as
  breach-point's `row_uid`). A business key (`customer_id`, etc.) can
  itself be the thing that's missing or duplicated, which is exactly when
  a row identifier is needed most.
- **`customer_id` is not unique in the raw data** — duplicate customers
  are one of the injected quality issues — so building a `customer_id ->
  segment` lookup via `set_index("customer_id")` raised
  `InvalidIndexError: Reindexing only valid with uniquely valued Index
  objects` the first time `_attach_customer_segment` ran on real data.
  Fixed by deduplicating on `customer_id` before building that lookup (an
  exact duplicate maps to the same segment regardless of which copy
  survives the dedup).
- **`by_segment` score was floored to 0.0 for every segment** — the
  original implementation summed issues from ALL 4 datasets (a segment's
  customers, plus their many thousands of associated transaction/payment
  rows) into the numerator, while dividing by a customer-row-only
  denominator (`customer_segment.value_counts()`). Numerator and
  denominator were at completely different scales — e.g. "prime" pulled in
  136,527 issues (mostly from transactions) against a denominator of only
  ~17,600 prime customers, guaranteeing the penalty rate clips to 1.0 and
  the score floors to 0. Fixed by scoping `by_segment` to the customers
  dataset's own issues only — "data quality by segment" is a coherent
  question for customer-level fields; it isn't a well-defined single
  number once a segment's entire transaction history is folded in too.
  Regression-tested in `test_by_segment_score_is_not_degenerate`.

## Phase 2 feature engineering — real bugs found and fixed

- **`FEATURE_COLUMNS["income"]` duplicated `monthly_income_3m_growth`/
  `6m_growth`**, which the `trend` group already produces (spec §13 lists
  a single generic `income_growth` under Income, already satisfied by the
  trend group's per-window growth columns). Selecting the same column name
  twice via `table[output_cols]` produced two identically-named columns in
  the DataFrame, which surfaced as `.1`-suffixed duplicate columns on a
  CSV round-trip (`monthly_income_3m_growth.1`) — caught by inspecting the
  actual written CSV's column list, not by a passing test (the original
  test suite didn't check for duplicate names). Fixed by removing the two
  entries from the `income` group (49 -> 47 real features). Regression-
  tested (`test_no_duplicate_feature_columns`,
  `test_output_csv_has_no_dot_one_suffixed_columns`).
- **Naive growth features exploded to absurd magnitudes** (`src/
  credit_limit_optimizer/features/engineering.py`). The first version
  computed trend growth as `(current - past) / past`; `credit_utilization`
  and `end_balance` legitimately hit exactly 0 (a customer who pays off in
  full), so the moment `past` was near zero the ratio blew up — observed
  max on the real 144,003-row output: ~1,000,000 for
  `credit_utilization_3m_growth`, ~1.08e10 for `end_balance_3m_growth`.
  Fixed by switching to a symmetric relative-change formula,
  `(current - past) / ((|current| + |past|) / 2)`, which is mathematically
  bounded to `[-2, 2]` whenever both values are non-negative (true for
  every trend variable used here: income, spend, utilization, balance).
  Regression-tested in `test_feature_engineering.py`
  (`test_growth_feature_bounded_even_at_near_zero_base`,
  `test_growth_features_bounded_on_real_data`).
- **No-future-leakage design**: every rolling/trend feature is computed
  with pandas `rolling()`/`shift()`/`cummax()` over each customer's own
  chronologically-sorted 36-month history (sorted by the raw `month`
  string, not assumed pre-sorted), then the single row at each cohort's
  own snapshot month (12/18/24) is selected — a feature for snapshot month
  M is structurally incapable of seeing month M+1 or later. Verified
  against the real data by recomputing trailing averages directly from
  `monthly_customer_behavior.csv` for a random sample and asserting exact
  agreement (`test_no_future_leakage_against_raw_monthly_table`).
- **Quarantined customers/accounts are dropped via inner join**, not
  imputed or kept — a risk model has no business training on a record that
  already failed a Phase 1 quality check. This drops ~2,000 of 50,000
  customers per cohort (feature table: 144,003 rows across train/
  validation/test, vs. 150,000 in `labels.csv`).
- **`credit_exposure`/`cash_buffer` and `employment_tenure_months` carry
  real NaNs** (2,928 and 2,883 rows respectively, out of 144,003) — these
  come from Phase 1's MEDIUM/LOW-severity missing-value injections, which
  stay in the VALIDATED layer (only CRITICAL/HIGH gets quarantined) by
  design. Left as genuine missing values for the risk model to handle
  (e.g. XGBoost's native NaN handling), not silently imputed here.

## Phase 2 EDA — real findings (not bugs, but worth knowing before Phase 3)

`src/credit_limit_optimizer/analysis/eda.py` covers every point in spec
§12 against the real 144,003-row feature table + full 36-month raw
history; see `reports/outputs/eda_report.md` for the full writeup with
numbers. Highlights that will matter for modeling:

- **`age`, `employment_status`, `employment_tenure_months` and
  `customer_tenure_months` carry essentially no default signal** (default
  rate flat within ~1pp across every bucket of each) — the generator's
  hazard process is driven by segment/utilization/delinquency dynamics,
  not demographics. Good news for the fairness section (§spec warns
  against protected characteristics as direct model inputs — this data
  gives no predictive reason to use `age` anyway); bad news if the
  portfolio narrative wants to show demographic features "earning their
  place" in the model.
- **`essential_spending_ratio`/`discretionary_spending_ratio` are close to
  three fixed points per row** (~0.45/0.60/0.72, `ESSENTIAL_SHARE_BY_SEGMENT`
  in `behavior_series.py` fixes the spend *mix* per segment and only
  varies the total spend *amount*) — a real generator simplification, not
  a bug. Their correlation with `default_12m` is mostly a mechanical echo
  of segment membership, not an independent behavioral signal.
- **Monthly spend never exceeds monthly income in this dataset (0.0% of
  144,003 rows)** — `monthly_spend` is drawn as `income * spend_ratio *
  lognormal(...)`, so revolving balances come entirely from customers not
  paying in full, never from an income shortfall funded by credit within
  the same month. A real bank's data would show some negative-cash-flow
  months; worth naming as a modeling simplification if asked.
- **Linear correlation with `default_12m` tops out around 0.16**
  (`delinquency_count`) even for variables with dramatic monotonic
  default-rate spreads in the bar-chart cuts (utilization 0-10% band ~2%
  default vs. 90-100% band ~21%) — expected for a ~4.7%-base-rate binary
  outcome, and the standard justification for XGBoost/WOE-IV over linear
  models picking up on these relationships in Phase 3.
- **Delinquency ramps up over the first ~12-15 months before reaching a
  stable per-segment band** (every customer starts at DPD=0; the AR(1)
  distress process needs time to reach its stationary distribution) — the
  snapshot months (12/18/24) all fall after this window closes.

## Phase 2 data quality reporting — real bugs found and fixed

`src/credit_limit_optimizer/data/quality_report.py` turns
`quality_report.json` (written by `validate-data`) into figures + a
written report (`reports/outputs/quality_report.md`), matching the EDA
report's format. Building it surfaced two real gaps in `validation.py`'s
score computation, both fixed there (not just in the report layer):

- **`by_month` was missing entirely.** §11 explicitly requires the score
  broken down by Dataset, Field, Customer segment, AND Month; only the
  first three existed. Added `compute_quality_score`'s `by_month`, scoped
  to `transactions.csv` + `payments.csv` (the only two datasets with a
  real per-record calendar timestamp — `customers.csv`/
  `credit_accounts.csv` are single-row-per-customer snapshots with no
  comparable monthly axis, the same reasoning that scopes `by_segment` to
  `customers.csv` only). Result: a flat ~98.6-98.7 across all 36 months,
  as expected since issues are injected at a uniform rate independent of
  calendar month.
- **`by_field` divided every field's issue count by the total row count
  across all 4 datasets combined** (~6.27M), even for a field that only
  exists in one dataset — e.g. `available_credit` only exists in
  `credit_accounts.csv` (50,000 rows), but its score was computed against
  the full 6.27M-row denominator, diluting it from a true ~98.7 to a
  reported 99.99. The same denominator-scope bug class as the `by_segment`
  floor-to-0 bug (see Phase 1 validation notes above), just milder here
  because the diluting datasets are still a comparable order of magnitude
  rather than orders larger. Fixed by scoping each field's denominator to
  the sum of row counts of only the dataset(s) that actually contain that
  column (`customer_id`, which genuinely exists in all 4 datasets, is
  correctly unaffected — its scoped denominator equals the old unscoped
  total exactly). Regression-tested in
  `test_by_field_score_scoped_to_owning_datasets_only` and
  `test_by_month_covers_full_history_and_is_not_degenerate`.

## Phase 3 risk model — Logistic Regression baseline (§14) — real bugs found and fixed

`src/credit_limit_optimizer/models/risk_model.py`. Time-based validation
(§15) comes for free from the existing snapshot cohorts (train = month
12, validation = month 18, test = month 24) — no extra split logic
needed, just `cohort == "train"` etc. Feature set excludes `age`
(protected characteristic, ~0 EDA signal anyway) and any feature that's
>95% Pearson-correlated (on train only) with an earlier-kept feature —
found because it was needed, not planned upfront:

- **The first run's top coefficients flatly contradicted the EDA**:
  `monthly_income_6m_avg`/`12m_avg` were the top RISK-INCREASING
  coefficients while `monthly_income_3m_avg` was the top RISK-REDUCING
  one, even though EDA found default rate falls monotonically with
  income. Root cause: the three trend windows of the same variable are
  r > 0.99 correlated with each other (rolling averages of a slow-moving
  series) — textbook multicollinearity, which doesn't hurt ROC-AUC
  (rank-based) but makes coefficients numerically unstable and
  uninterpretable, defeating the entire point of an LR baseline (§14:
  "interpretability, economic intuition"). A first manual fix (keep only
  the 6m window) still left `monthly_income` (base) at r=0.995 with
  `monthly_income_6m_avg`, plus several OTHER exact/near-exact duplicate
  pairs discovered by inspecting the full correlation matrix:
  `income_stability` is a literal `1 - income_volatility` transform
  (r=1.0); `minimum_payment_ratio` is r=1.0 with `payment_ratio`;
  `credit_utilization_Nm_growth` is r=1.0 with `end_balance_Nm_growth`
  (mathematically inevitable — this generator never changes a customer's
  `credit_limit` over the 36-month history, so a *relative*-growth ratio
  of utilization or balance is scale-invariant to that constant limit,
  making them identical). Replaced the hand-picked exclusion list with
  `select_model_features`: a deterministic greedy pruner that walks
  `ALL_FEATURE_COLUMNS` in order and drops any feature more than 95%
  correlated (train cohort only, no leakage) with an already-kept
  feature — found 11 such features automatically, more than manual
  inspection caught. After pruning, top coefficients are directionally
  consistent with EDA (`delinquency_count`, `days_past_due`,
  `credit_utilization` all risk-increasing; `monthly_income` risk-
  reducing). A few small-magnitude coefficients (< 0.03) among
  features still 0.8-0.95 correlated remain counter-intuitive — expected
  multivariate partial-correlation behavior once the >0.95 pairs are
  gone, not pruned further to avoid discarding real information for
  diminishing interpretability gains on a baseline model. Regression-
  tested in `test_kept_features_are_pairwise_below_threshold_on_real_train_data`
  and the `select_model_features` unit tests in `test_risk_model.py`.
- **`class_weight='balanced'` inflates predicted probabilities far above
  the true base rate**: mean predicted probability ~42-48% vs. actual
  base rate ~4.5-4.8% across all three cohorts — improves ranking
  (ROC-AUC) but makes raw probabilities unusable as PD estimates and
  Precision/Recall/F1 at the naive 0.5 threshold look strange (precision
  ~8-10%, recall ~60-70%). Documented as expected, not a bug — this is
  exactly what the upcoming Probability Calibration piece (§17) exists
  to fix. Captured as `mean_predicted_probability` in the metrics dict
  and asserted in `test_probabilities_are_not_naively_assumed_calibrated`
  so a future change to the imbalance-handling strategy doesn't silently
  invalidate this documented behavior.

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
