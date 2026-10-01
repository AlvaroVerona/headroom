# `delinquency_count`'s SIGNIFICANT PSI is a vintage-design artifact, not real drift

> [!bug] What the monitor reported
> `delinquency_count` (a top-6 SHAP-importance feature) shows SIGNIFICANT PSI — **0.345** train vs. test, above the 0.25 critical threshold — while the PD score itself stays STABLE (PSI 0.01-0.07) across every period. Mean value climbs 0.99 (train) → 1.54 (validation) → 2.10 (test).

> [!tip] Root cause
> `train`/`validation`/`test` are NOT three independent, non-overlapping populations — they're the SAME 50,000 customers observed at increasingly later points in their own 36-month history (months 12/18/24, the standard credit-risk vintage design). A LIFETIME/cumulative counter like `delinquency_count` mechanically has more elapsed time to accumulate events at a later snapshot, for the exact same customers, even with zero true change in underlying credit-risk behavior. Consistent with Phase 2's EDA finding that "delinquency ramps up over the first ~12-15 months before reaching a stable per-segment band."

> [!success] How this was confirmed, not just theorized
> Checked the other two delinquency-family monitored features: `days_past_due` (point-in-time DPD, not cumulative — PSI ~0.0005) and `recent_delinquency` (a recent-window indicator, not cumulative — PSI ~0.0000) both stay STABLE across every period. Exactly the pattern the vintage-reuse explanation predicts, and good evidence against the alternative "the population is genuinely destabilizing" reading — if it were real drift, the non-cumulative delinquency features should show it too.

> [!warning] Why this matters going forward
> A real production deployment (genuinely new customers each monitoring period, no vintage reuse) would NOT have this artifact — a future PSI alert on a cumulative feature in a real deployment should be taken at face value. This note exists so a future reader of this specific project's PSI numbers doesn't misread the artifact as a real signal.

> [!example] Where this shows up
> `monitoring/drift.py`'s `write_model_card` — the full explanation and the confirming numbers are written into `reports/model_cards/monitoring.md` on every `make monitor` run. Regression-tested in `tests/test_drift.py::test_delinquency_count_flagged_as_vintage_artifact` so a future feature-engineering change that accidentally "fixes" this artifact away doesn't go unnoticed.

See also: [[Home]]
