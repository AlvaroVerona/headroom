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
  impossibly tight as a guess is to be right). Update (Phase 4, expected
  loss): `portfolio_expected_loss_limit: 3,000,000` turns out to already
  be a reasonable estimate — the real, computed test-cohort Expected
  Loss extrapolates to ~€2.93M across the full ~50,000-customer
  portfolio (see `expected_loss.py`'s model card). Left unchanged.
  Update (Phase 5, portfolio optimization): `portfolio_exposure_limit`
  (150,000,000) checked against the real number — funding every
  individually-approved customer at their individually-optimal limit
  would need €320.5M, ~2.14x the configured cap. Left unchanged
  deliberately, not recalibrated up: unlike the EL limit (which just
  happened to already match reality), a exposure cap that's ALWAYS
  slack would make the portfolio optimization piece pointless — the
  point of §25 is to demonstrate a genuine, binding resource allocation
  trade-off, and €150M creates exactly that (see `portfolio_optimizer.py`'s
  model card for the resulting 17,205-of-38,879-customer allocation).
- **The calibrated XGBoost classifier is "the" production PD source**
  for Expected Loss (Phase 4) and everything downstream (optimization,
  simulation) — not the LR baseline. It modestly but genuinely
  outperforms the LR baseline on held-out test data (ROC-AUC 0.730 vs.
  0.724, PR-AUC 0.165 vs. 0.148) and both are now calibrated, so there's
  no accuracy reason to prefer the less powerful model once calibration
  is done; LR's role is the required, explicitly-interpretable baseline
  (§14) and comparison point, not a second production candidate.
- **Balance response to a candidate credit limit** (Phase 4, §22):
  `balance(L) = min(current_balance * (L/current_limit)**elasticity, L)`,
  `elasticity = 0.3` (`optimization.balance_elasticity`, config-driven).
  Anchored exactly at each customer's own observed point, `elasticity <
  1` gives diminishing returns by construction (the concave Limit-vs-
  Profit shape this note originally asked for), and `min(..., L)`
  enforces the physical cap. NOT derived from the EAD/PD models: this
  generator never varies a customer's `credit_limit` over their 36-month
  history, so `credit_exposure` had ~0 importance (rank 13/46) in the
  fitted EAD regressor — no within-customer variation for either model to
  have learned a genuine causal limit-response from, so PD is held
  constant across candidate limits too. See `profitability.py`'s module
  docstring for the full reasoning.
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

## Phase 3 risk model — XGBoost advanced model (§14) — real bug found and fixed

`src/credit_limit_optimizer/models/xgboost_model.py`. Reuses the LR
baseline's data loading and time-based split; uses the full
`ALL_FEATURE_COLUMNS` set (minus `age`) with no correlation pruning and
no imputation/scaling, since trees handle both natively.

- **`eval_metric='logloss'` for early stopping caused severe overfitting
  that made the "advanced" model worse than the baseline.** First run
  (max_depth=4, min_child_weight=10, `eval_metric='logloss'`,
  `early_stopping_rounds=30`) never triggered early stopping
  (`best_iteration=499` of `n_estimators=500` — validation log loss kept
  inching down for the entire run) and produced train ROC-AUC 0.879
  against validation 0.729 and test 0.702 — a textbook overfit, and
  specifically one that scored *worse* on held-out test data than the LR
  baseline's 0.724, defeating the entire point of building an "advanced"
  model. Root cause: under `scale_pos_weight`-reweighted training with
  severe class imbalance, log loss keeps rewarding the model for
  sharpening its (already inflated, see the LR baseline's calibration
  note above) predicted probabilities long after its actual ranking
  ability on unseen data has stopped improving — log loss and ROC-AUC
  diverge in a way they wouldn't on balanced data. Fixed by switching the
  early-stopping metric to `auc` (directly matching what's reported and
  compared against the baseline) and using shallower trees (max_depth 4→3,
  min_child_weight 10→20). Result: train/validation/test ROC-AUC now sit
  close together (~0.73-0.76, best_iteration=124 of 500 — stops well
  before the budget), and the advanced model modestly but genuinely beats
  the LR baseline on the test cohort (0.730 vs. 0.724 ROC-AUC, 0.165 vs.
  0.148 PR-AUC) — the expected, sensible outcome. Regression-tested in
  `test_eval_metric_is_auc_not_logloss`,
  `test_early_stopping_actually_triggers`,
  `test_train_test_roc_auc_gap_is_not_a_severe_overfit`, and
  `test_xgboost_test_roc_auc_beats_or_matches_lr_baseline`.
- Same `class_weight`-style probability inflation as the LR baseline
  (mean predicted probability ~42% vs. ~4.5% actual on train, via
  `scale_pos_weight`) — documented, not a bug, deferred to §17.

## Phase 3 probability calibration (§17)

`src/credit_limit_optimizer/models/calibration.py`. Both models' raw
probabilities are exactly as miscalibrated as their model cards warned
(test-cohort mean predicted probability: LR 48.4%, XGBoost 45.6%, vs.
~4.8% actual) — fitting `CalibratedClassifierCV(FrozenEstimator(model))`
fixes this dramatically: calibrated mean predicted probability lands at
5.5% (LR, sigmoid) and 5.0% (XGBoost, isotonic), Brier score drops from
~0.23-0.26 to ~0.043-0.044, and Expected Calibration Error drops from
~0.41-0.44 to ~0.002-0.007 (two orders of magnitude), all while ROC-AUC
stays within 0.001 of the raw model's — confirming calibration only
reshapes the probability *scale*, not the model's ranking ability, as it
should. Reliability diagrams (`reports/figures/calibration/*.png`) show
this visually: the raw curve sits far below the diagonal (systematic
overconfidence), the calibrated curve hugs it closely.

Split design, so no validation row is ever used for two purposes: the
validation cohort is split in half into `calib_fit` (fits both sigmoid
and isotonic calibrators) and `calib_select` (scores them by Brier score
to choose the better one per model, purely — sigmoid won for LR, isotonic
for XGBoost, both by a razor-thin margin at the calib_select stage even
though the margin widens somewhat on test). The test cohort is touched
exactly once, for the final numbers reported in the model card — no
model-selection decision is based on it. No bug found while building
this piece; it worked essentially as expected on the first working run.
Uses `sklearn.frozen.FrozenEstimator` (sklearn 1.9's supported mechanism
for calibrating an already-fitted estimator) rather than the older
`CalibratedClassifierCV(cv="prefit")` pattern.

## Phase 3 SHAP explainability (§18) — real bug found and fixed

`src/credit_limit_optimizer/models/explain.py`. `shap.LinearExplainer`
(LR, on the imputed+scaled feature space, exact) and `shap.TreeExplainer`
with `feature_perturbation="tree_path_dependent"` (XGBoost, exact,
handles NaN natively) on a 2,000-row sample of the test cohort per
CLAUDE.md's documented sampling convention. SHAP explains each model's
own raw decision function, not the calibrated probability (post-hoc,
non-linear, no well-defined per-feature attribution) — the "Predicted
Default Probability" shown for each individual customer example comes
from the calibrated model instead, the actually business-meaningful PD;
this is a deliberate, documented split of sources, not an inconsistency.

- **Found a real, visible multicollinearity artifact the LR baseline's
  correlation-pruning threshold (0.95) missed**: the SHAP summary plot
  showed `cash_buffer` (rank #2 by importance) with a POSITIVE SHAP value
  for HIGH `cash_buffer` — i.e. more available headroom reading as MORE
  risky, backwards from intuition and from `cash_buffer`'s own raw
  correlation with `default_12m` (-0.083, genuinely protective). Unlike
  the earlier small-magnitude residuals documented in the LR baseline
  (accepted as normal partial-correlation behavior), this one was large
  and prominent enough to be immediately visible in a headline chart.
  Root cause: `cash_buffer = credit_exposure - end_balance` is r=0.9465
  with `credit_exposure` — just under the original 0.95 pruning cutoff,
  so it survived and became a coefficient artifact like the others, just
  a bigger one. Fixed in `risk_model.py`, not here: lowered
  `CORRELATION_PRUNE_THRESHOLD` to 0.93 (catches this pair plus three
  more found at the same time in the 0.93-0.95 gap:
  `credit_utilization_3m_avg`/`6m_avg` 0.948, `transaction_count`/
  `monthly_spend_3m_avg` 0.941, `end_balance_12m_avg`/`6m_avg` 0.936),
  retrained the LR baseline (35→32 features), and re-ran calibration.py
  and explain.py downstream since both depend on the LR model artifact.
  `cash_buffer` no longer appears in the LR SHAP summary at all.
  Regression-tested in `test_cash_buffer_dropped_for_correlation_with_
  credit_exposure` (risk_model) and `test_cash_buffer_not_among_lr_top_
  features` (explain).
- **Stale dependence-plot PNGs left over between runs**: `explain.py`
  wrote dependence plots for the current top-N features but never
  removed a PNG for a feature that dropped out of the top-N on a
  subsequent run (exactly what happened to `cash_buffer`'s own
  now-obsolete dependence plot after the fix above) — the file just sat
  in `reports/figures/shap/` unreferenced by the model card, silently
  wrong if anyone browsed the folder directly instead of the card. Fixed
  by clearing every PNG in `FIG_DIR` at the start of each run (the module
  owns that directory exclusively). Regression-tested in
  `test_no_stale_dependence_plots_left_over`.

## Phase 4 expected loss (§19) — real bug found and fixed

`src/credit_limit_optimizer/models/expected_loss.py`. Expected Loss = PD
× LGD × EAD. PD from the calibrated XGBoost classifier (forced to 1.0
for customers already at 90+ DPD at their snapshot — they're not a
forecast target); LGD from the existing segment lookup; EAD from a
genuine regression model (XGBoost) predicting labels.csv's
`future_utilization` from the same feature table and time-based split as
the risk models (R² ~0.58-0.61 — the scatter shows the model capturing
*segment*-level utilization equilibria well but less of the within-
segment variation, consistent with the generator's steady-state design,
not a modeling shortfall), then `EAD = clip(prediction, 0, 1) ×
credit_exposure`.

- **`mean_ead`/`mean_expected_loss`/`total_portfolio_expected_loss` came
  out `NaN`** on the first run, while the by-segment and by-outcome
  breakdowns looked fine. Root cause: `credit_exposure` carries real
  Phase 1 injected missing-value issues (~2% of rows, documented since
  Phase 2) — multiplying a NaN credit_exposure into EAD, then into
  Expected Loss, and taking a plain `.mean()`/`.sum()` over the whole
  array propagates that NaN to every aggregate; `pandas.groupby().mean()`
  silently skips NaN by default, which is why the segment/outcome-
  conditional numbers looked correct while the top-line ones didn't —
  easy to miss if you only check the grouped tables. Fixed by explicitly
  excluding rows with missing `credit_exposure` from the euro-denominated
  Expected Loss calculation (976 of 48,001 test rows, ~2.0%) and
  reporting the exclusion count directly in both the log output and the
  JSON report, rather than silently dropping or silently producing NaN.
  The EAD model's own R²/MAE are unaffected (predicting
  `future_utilization` doesn't need `credit_exposure`) and are still
  evaluated on the full cohort. Regression-tested in
  `test_expected_loss_has_no_nan`.
- **Validated, not just computed**: mean EL is higher for customers who
  actually defaulted within 12 months than for those who didn't (€112.68
  vs. €44.04, known only from the snapshot before the outcome exists) and
  ordered prime < near_prime < subprime as expected — both checks pass on
  the real numbers, not asserted by construction.

## Phase 4 revenue model (§20)

`src/credit_limit_optimizer/models/revenue_model.py`. Interest Revenue =
Average Balance × APR / 12, Interchange = Monthly Spend × interchange
rate, Late Fee = a new config value (`economics.late_fee_amount: 25`,
charged in any month with `days_past_due > 0`) — §20's optional FX fees
and other product fees are NOT modeled, since the generated transaction
data is verified 100% EUR (checked directly against `transactions.csv`)
with no basis for either.

This is deliberately the exact same formula `labels.py` already used to
generate `future_interest_revenue`/`future_interchange_revenue` back in
Phase 1 — not incidental, it means correctness can be checked against
real generated numbers instead of just asserted.
`validate_against_labels()` reconstructs each label's target future
month's `average_balance`/`monthly_spend` directly from the raw monthly
table, applies this module's formula, and confirms it reproduces
labels.csv's real values to within €0.01 (labels.csv's own 2-decimal
rounding) across 147,150 rows. No bug found while building this piece —
the validation passed on the first working run, which itself confirms
the formula and the underlying `average_balance`/`monthly_spend` column
semantics were both understood correctly.

Revenue by segment reveals a real, non-obvious tension worth carrying
into Customer Profitability (§22, next): subprime generates the highest
mean revenue (€44.47/month, mostly interest — high APR × high balance)
but also the highest Expected Loss (€122.45, from the Phase 4 EL piece);
annualized, subprime nets ~€411/customer/year before funding/operational
cost vs. prime's ~€269 and near_prime's ~€429 — subprime is not
obviously the least profitable segment once volume is accounted for,
which is exactly the kind of number a "just cap subprime limits" policy
would get wrong without actually computing it.

## Phase 4 funding cost (§21)

`src/credit_limit_optimizer/models/funding_cost.py`. Funding Cost =
Average Balance × Funding Rate / 12, same monthly-rate treatment as
Interest Revenue. Rate varies by macroeconomic scenario using
`config/settings.yaml`'s already-scaffolded
`scenarios.*.funding_rate_shift` (base 5%, recession 6%,
high_interest_rate 8%, consumer_stress 5% unchanged — it stresses
income/spending/delinquency, not cost of funds). Deliberately does NOT
vary by customer segment: a bank's cost of funds is a treasury-level
blended rate, not a credit-risk quantity like LGD — segment risk is
already priced through PD/LGD (Expected Loss) and APR (Revenue), so
segment-varying funding cost would double-count it. Full time-varying/
Monte-Carlo funding-rate paths are explicitly Phase 6's job, not this
piece's — a scenario here stands in for an alternate macro state, not a
simulated path.

Also computes Net Interest Margin (Interest Revenue − Funding Cost,
base scenario) as a sanity cross-check with the revenue model: positive
for 95.3% of test-cohort customers, and verified (not just assumed) that
every one of the remaining rows is NIM = 0 exactly from
`average_balance == 0.0`, never negative — every segment's APR
(16.9-27.9%) comfortably exceeds every scenario's funding rate (5-8%).
No bug found while building this piece.

## Phase 4 customer profitability (§22) — real bugs found and fixed

`src/credit_limit_optimizer/models/profitability.py`. Expected Customer
Profit = Total Revenue − Expected Loss − Funding Cost − Operational Cost,
computed both at each customer's current/observed limit (portfolio
summary) and across the full candidate limit grid for one representative
example customer per segment (§22's own illustrative table format). See
the "Key decisions" entry above for the balance-response-to-limit model
this piece introduced.

- **Every segment's mean Expected Profit came out negative on the first
  run** (prime -€3.42, near_prime -€44.54, subprime -€102.21/month) — a
  units/period mismatch, not a real economics finding. `expected_loss =
  PD × LGD × EAD` is inherently a 12-MONTH figure (PD is `default_12m`'s
  12-month default probability), but revenue and funding cost were left
  as the MONTHLY figures `revenue_model.py`/`funding_cost.py` compute —
  subtracting a 12-month loss from a 1-month revenue is guaranteed to
  look catastrophically unprofitable regardless of the underlying
  economics. Caught by comparing against `revenue_model.py`'s own
  "Preview: revenue vs Expected Loss" section, which had already
  annualized revenue (× 12) before comparing to EL and found every
  segment solidly profitable — the two pieces disagreed, which is what
  surfaced the bug. Fixed by putting every term in `evaluate_at_limit` on
  a consistent ANNUAL basis (revenue/funding cost × 12, "run-rate"
  annualization of the customer's current-month observed rate; EL and
  `operational_cost_per_customer`, already annual, used as-is). Result:
  prime €203.08, near_prime €318.02, subprime €299.07/year — all
  positive, all in the same ballpark the revenue model's own preview
  predicted. Regression-tested in
  `test_evaluate_at_limit_all_components_are_annual` and
  `test_mean_profit_positive_for_every_segment`.
- **Stale example-customer `profit_curve_*.png` files left over between
  runs** — same bug class already fixed once in `explain.py`'s SHAP
  dependence plots: this module's `FIG_DIR` wasn't cleared before writing
  new figures, so a customer no longer selected as a segment's example
  left an orphaned, unreferenced chart behind. Fixed the same way:
  clear every PNG in `FIG_DIR` at the start of each run. Regression-
  tested in `test_no_stale_profit_curve_files_left_over`.
- **The first chosen "example customer" per segment was whichever row
  happened to be first in the test cohort** — degenerate for prime
  (€0.00 average_balance, current limit €11,400 outside the candidate
  grid entirely, so the profit curve was flat and uninformative).
  Switched to filtering for a within-grid current limit and positive
  balance, then picking the MEDIAN balance within that pool — a
  representative, not cherry-picked or degenerate, illustrative example.

## Phase 5 individual credit limit optimization (§23-24) — real bug found and fixed

`src/credit_limit_optimizer/optimization/customer_optimizer.py`. For
each customer, evaluate the full candidate grid (reusing
`profitability.py`'s machinery), apply four constraints
(`config/settings.yaml`'s `risk.*`: income, risk/PD, utilization, debt/
DSR), and select the feasible candidate maximizing Expected Profit; no
feasible candidate → decline, not "approve at the smallest grid value."
DSR reuses the generator's own scheduled-minimum-payment formula
(`max(0.03 × balance, 25 if balance > 0 else 0) / monthly_income`, from
`behavior_series.py`) applied to `balance(L)`, not a new assumption. This
is deliberately NOT an OR-Tools problem: individual optimization is an
independent per-customer argmax over ~20 candidates, no coupling between
customers, so a full solver would be pure overhead — OR-Tools is for the
NEXT piece, portfolio optimization, where the exposure/loss constraints
genuinely couple every customer's limit to everyone else's.

Result on the test cohort: **87.0% approved**, decline driven almost
entirely by the risk constraint (5,791 of 5,798 declines — PD is held
constant across candidates per Phase 4's documented simplification, so
`PD <= max_pd` is either satisfiable for every candidate or none, no
limit-specific trade-off possible). Mean profit uplift vs. current limits
is positive for every segment (prime €4.21, near_prime €49.68, subprime
€104.46/year).

- **`groupby().idxmax()` silently picked the SMALLEST candidate for any
  customer with a flat (tied) profit curve across the whole grid** — most
  commonly a zero-`average_balance` customer, whose profit doesn't depend
  on the limit at all in this model (interest revenue, EL, and funding
  cost are all balance-driven, hence all zero). `idxmax()` returns the
  first-occurring maximum, which for a flat series is just "whichever
  candidate sorts first" — i.e. €500, producing a real, misleading "cut
  this customer's credit line" recommendation with zero economic basis
  behind it. Found by inspecting the limit-change distribution chart: an
  unexplained left-tail spike down to -€12,500 that had no business
  reading. Verified the scope before fixing: ~40% of ALL "decrease"
  recommendations were this exact artifact. Fixed with
  `_select_optimal`: among each customer's candidates within a small
  tolerance of the max profit, break the tie toward whichever is CLOSEST
  to the customer's current limit (least-disruption principle) instead
  of accepting pandas' arbitrary first-occurrence default. After the fix,
  the remaining "decrease" recommendations are 83% explained by a
  genuinely different, real cause: the customer's actual current limit
  already exceeds the candidate grid's own maximum (€10,000) — capped by
  the grid range, not a model judgment that their limit should shrink.
  Regression-tested in `test_select_optimal_breaks_ties_toward_current_limit`
  and `test_decrease_recommendations_mostly_explained_by_grid_cap`.

## Phase 5 portfolio optimization (§25) — closes Phase 5

`src/credit_limit_optimizer/optimization/portfolio_optimizer.py`, OR-Tools
CBC MIP. Two-stage design: stage 1 (`customer_optimizer.py`) already
picked each customer's individually-optimal (limit, profit, EAD, PD)
tuple under customer-level constraints; stage 2 is a portfolio-level 0/1
knapsack — one binary variable per already-approved customer (not per
customer × candidate), deciding which to actually FUND so the aggregate
exposure/loss/risk-concentration budget holds while maximizing total
profit. ~38,900 variables, 4 constraints, solves to OPTIMAL in ~1 second
— a deliberate simplification (loses the ability to shrink a customer's
limit to fit them in under budget; it's fund-at-optimal-or-not) that
keeps the problem tractable and mirrors real risk-based portfolio triage
(score individually, then allocate scarce budget).

The two ratio constraints (average PD, high-risk exposure %) both have a
denominator that depends on the decision variables — not linear as
written, but since PD and limit are fixed per customer from stage 1,
multiplying through gives an exactly equivalent linear form (see module
docstring), not an approximation — verified directly in
`test_average_pd_linearization_matches_true_ratio` and
`test_high_risk_exposure_linearization_matches_true_ratio` (solve with a
tight limit, then recompute the TRUE, non-linearized ratio on the
selected set and check it still holds).

Real, non-trivial result: funding every individually-approved customer
would need €320.5M exposure and 18.4% high-risk concentration, both over
the €150M / 15% limits — Expected Loss and average PD both have slack.
MIP-optimal funds 17,205 of 38,879 (44.3%), hitting exposure and
high-risk exposure exactly at their limits, €8.19M/year total profit —
7.77% better than a profit-per-exposure greedy heuristic (both solve the
same problem; the MIP's advantage is jointly respecting all 4 constraints
at once rather than a single ratio blind to which one actually binds).
**Funding rate is NOT "safest first"**: prime (safest, 20% funded) loses
out to near_prime (72% funded) because exposure is the binding
constraint and prime's profit-per-euro-of-exposure (€0.022) is the
segment's worst — prime customers carry large individually-optimal
limits for comparatively modest absolute profit. Subprime has the BEST
profit-per-euro (€0.057) but is capped by the also-binding high-risk
exposure constraint, leaving near_prime with the most headroom under
both constraints simultaneously. No bug found while building this
piece — the MIP solved correctly and the result made sense once the
segment-level profit-efficiency numbers were checked directly.

## Phase 6 stress testing / scenarios (§26-27)

`src/credit_limit_optimizer/simulation/scenarios.py`. Re-evaluates the FUNDED portfolio
(Phase 5's MIP solution, 17,205 customers) -- not the raw approved-but-unfunded pool -- under
each macro scenario already scaffolded in `config/settings.yaml`'s `scenarios` block.

**Two shock mechanisms, chosen per shock's own nature, not one blanket approach**: (1)
feature-level shocks (`income_growth_shift`, `spending_shift`, `income_volatility_multiplier`,
`delinquency_multiplier`, `cash_balance_shift`) perturb genuine PD-model input features and
are re-scored through the ALREADY-FITTED, calibrated XGBoost classifier -- no retraining,
which would be a different kind of leakage (fitting to a hypothetical future); this is the
same "shock inputs, re-score the model" design the prior session verified before starting
this piece, confirming `income`/`spend`/`income_volatility` all carry real PD signal (unlike
`credit_exposure`, already known to have ~0 importance per `profitability.py`). (2) Direct
macro overlays on the downstream economics formula: `default_multiplier` (a flat PD overlay
-- the standard "management overlay" every real stress-testing framework layers on top of a
model score), `funding_rate_shift` (reuses `funding_cost.py`'s own
`funding_rate_for_scenario`), `apr_shift` (added to each customer's own APR).

**Deliberately NOT shocked**: `average_balance` (no causal balance-response-to-macro-shock
model exists, the same reasoning CLAUDE.md already gives for why `credit_exposure` can't be
used to shock EAD) and LGD (no scenario key for it in the spec's own scaffolded config --
same "don't invent what the spec didn't ask for" principle as `funding_cost.py`'s flat
funding rate).

**A real, non-obvious finding, not a bug**: `high_interest_rate`'s total Expected Profit is
IDENTICAL to base to the cent (€8,186,016.98 both times). It has no feature-level shock at
all (PD unchanged from base, verified), only `funding_rate_shift: +0.03` and `apr_shift:
+0.03` -- and because `interest_revenue = balance × apr` and `funding_cost = balance ×
funding_rate` scale the exact same `balance` figure, two EQUAL-MAGNITUDE shifts cancel
exactly in net profit even though gross revenue AND gross cost both moved by
+€1,028,505 each. Illustrates a real risk: if a bank's own repricing tracks its cost of
funds one-for-one, a rate-rise scenario looks like a bottom-line non-event even though NIM
composition changed substantially -- invisible from a single "total profit" number without
the revenue/cost breakdown. No bug found otherwise; recession (default_multiplier 1.6 +
income/spend shocks) is the worst scenario on every axis as expected (mean PD +64%,
Expected Loss +65% vs. base).

## Phase 6 Monte Carlo loss simulation / VaR / CVaR (§28)

`src/credit_limit_optimizer/simulation/monte_carlo.py`. Single-factor (Vasicek/ASRF)
Gaussian copula default simulation over the same funded book, one run per scenario (not just
base) so stress scenarios' TAIL risk -- not just their mean EL, already in `scenarios.py` --
is visible too. `asset_correlation` (`simulation.asset_correlation`, 0.04) is Basel's own
flat correlation for Qualifying Revolving Retail Exposures (credit cards) -- the actual
regulatory constant for this exact product type, not fit to this data or invented.

**Sanity check, not just computed**: Monte Carlo mean loss lands within 0.5% of the
analytical `PD × LGD × EAD` sum (already computed in `scenarios.py`) on every scenario --
both describe the same expectation, so a real divergence here would mean a bug in either the
copula simulation or the LGD-recovery step (`LGD = expected_loss / (PD × balance)`, chosen
deliberately over a second independent LGD lookup so analytical/MC consistency can never be
an artifact of two sources of truth drifting apart). VaR 99% (€2.24M base) is ~2.4x Expected
Loss even at a modest 4% asset correlation -- correlated defaults genuinely fatten the tail
past what a plain binomial loss count would show, the whole reason to simulate rather than
just scale the mean. No bug found while building this piece.

## Phase 6 drift monitoring / PSI (§29) -- closes Phase 6

`src/credit_limit_optimizer/monitoring/drift.py`. No true "production" data exists past the
test cohort's month-24 snapshot, so monitoring compares each ADJACENT pair of the three real,
genuinely time-separated snapshot vintages (train=month 12, validation=month 18, test=month
24) already built for time-based validation -- a genuine "how much did the real population
move in the ~6 real months between two actual snapshots" measurement, not a placeholder.
Monitored features are the calibrated XGBoost model's own top-6 SHAP global-importance
features (`reports/outputs/shap_report.json`, Phase 3's `explain.py`) -- reusing the model's
own data-driven ranking rather than hand-picking, the same principle as `risk_model.py`'s
correlation pruning. PSI thresholds (0.10/0.25) are the standard industry bands, not
invented.

- **PD score is STABLE across every period** (PSI 0.01-0.07, well under 0.10) -- the
  calibrated model's own output hasn't meaningfully shifted.
- **A real, fully-explained exception, not a bug**: `delinquency_count` shows SIGNIFICANT
  PSI (0.345, train vs. test) -- mean value climbs 0.99 -> 1.54 -> 2.10 across the three
  vintages. Root cause: `train`/`validation`/`test` are the SAME 50,000 customers observed
  at increasingly later points in their OWN 36-month history (not three independent,
  non-overlapping populations), so a LIFETIME/cumulative counter mechanically has more
  elapsed time to accumulate delinquency events at a later snapshot for the exact same
  customers -- consistent with Phase 2's EDA finding that "delinquency ramps up over the
  first ~12-15 months before reaching a stable per-segment band." Confirmed by checking the
  two other delinquency-family monitored features: `days_past_due` (point-in-time, not
  cumulative) and `recent_delinquency` (recent-window indicator, not cumulative) both stay
  STABLE (PSI ~0.0005 and 0.0000) across every period -- exactly the pattern this
  explanation predicts, and evidence against an alternative "the population is genuinely
  destabilizing" reading. A real production deployment (genuinely new customers each
  period, no vintage reuse) would not have this artifact; flagged prominently in the model
  card so a future PSI alert on a cumulative feature here isn't misread as the population
  instability PSI monitoring exists to catch.

## Phase 7 dashboard, documentation, final audit -- closes Phase 7

`app/` -- an 8-page Streamlit dashboard (Executive Overview + Data Quality, Risk
Models, Economics, Optimization, Stress Testing, Monte Carlo Risk, Monitoring),
mirroring `../breach-point`'s own dashboard conventions: `app/__init__.py` (required
for `app.pages.*` to import `from app.components...` -- without it,
`streamlit run app/app.py` raises `ModuleNotFoundError`), `app/components/data_loader.py`
(`st.cache_data`-wrapped readers of `reports/outputs/*.json` -- never recomputes the
pipeline live, since Phase 3's training and Phase 6's Monte Carlo alone would make
the UI unusable per-click), `app/components/style.py` (fixed per-segment colors,
status/severity badges). Tested headlessly with Streamlit's `AppTest` harness
(`tests/test_dashboard.py`, 12 tests, ~1.5s for all 8 pages) rather than a live
browser under pytest -- and additionally walked through live in a real browser
(`streamlit run app/app.py`) to confirm actual rendering, since AppTest alone only
proves "didn't crash," not "looks right."

**Documentation**: README.md rewritten from the phase-by-phase build log it was
during Phases 1-6 into a polished, portfolio-facing document (results table,
condensed methodology, dashboard description, verification) -- this file (CLAUDE.md)
keeps the detailed, dated engineering log with every decision and bug.

**Final audit**: full `make all` (data generation through drift monitoring) re-run
end to end, seed=42 -- every headline figure reproduced exactly (Data Quality 98.64,
17,205 funded customers, €8,186,017/year MIP profit, 87.04% approval rate, base VaR
99% €2,241,452, `any_significant_drift=True`), confirming determinism holds across
the full pipeline, not just within individual phases. **227/227 tests passed**
(215 pipeline + 12 dashboard). If a future change to any upstream phase shifts these
numbers, that's a real signal — check whether it's an intentional code change
(update the README) or a regression (fix it), never just overwrite the README to
match without understanding why it moved.

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
