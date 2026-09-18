# PD source: calibrated XGBoost, not the LR baseline

> [!note] Decision
> The calibrated XGBoost classifier is "the" production PD source for Expected Loss and everything downstream — optimization, stress testing, Monte Carlo. The Logistic Regression baseline is never used past its own model card.

> [!tip] Why
> XGBoost modestly but genuinely outperforms the LR baseline on held-out test data (ROC-AUC 0.730 vs. 0.724, PR-AUC 0.165 vs. 0.148), and both are now calibrated (Brier ~0.043, ECE ~0.0016 for XGBoost) — so there's no accuracy reason left to prefer the less powerful model once calibration is done.

> [!warning] Trade-off accepted
> LR still gets trained, calibrated and explained on every `make all` run even though nothing downstream reads its output. Its role is the required, explicitly-interpretable baseline (§14) and comparison point — not a second production candidate that would need to stay in sync with XGBoost's re-scoring paths.

> [!example] Where this shows up in the code
> - `models/expected_loss.py`'s `PD_MODEL_PATH` points at `xgboost_calibrated.joblib`, never the LR artifact
> - `models/profitability.py`'s `compute_pd`, `optimization/customer_optimizer.py`, `simulation/scenarios.py`'s `compute_pd` — every downstream PD read is XGBoost-only
> - `models/risk_model.py` (LR) and `models/calibration.py`/`models/explain.py` (both models) exist purely for the comparison and the interpretability requirement

See CLAUDE.md's Phase 3/4 entries for the exact metrics and the early-stopping-metric bug that made the first XGBoost attempt score *worse* than LR before this decision was even reachable.

See also: [[Home]]
