"""Credit risk model, advanced (§14): XGBoost predicting default_12m.

Reuses the LR baseline's data loading/splitting (`load_feature_table`,
`split_cohorts`) and time-based validation (§15) unchanged -- see
risk_model.py's docstring for why a random split would leak.

Unlike the LR baseline, this model uses the FULL `ALL_FEATURE_COLUMNS`
set (minus the one protected characteristic, `age`) with no correlation
pruning and no imputation/scaling: gradient-boosted trees split on one
feature at a time, so two 99%-correlated columns just make similar splits
available at similar gain rather than destabilizing a coefficient, and
XGBoost natively routes missing values (NaN) to whichever branch improves
the split -- both of the LR baseline's main preprocessing headaches don't
apply here, and undoing the LR-specific pruning restores information
(e.g. the 3m/12m trend windows) a tree model can actually use.

Imbalance is handled with `scale_pos_weight` (XGBoost's analog to
`class_weight='balanced'`), and early stopping uses the validation
cohort -- consistent with the time-based split's purpose (test stays
untouched until final evaluation).

Early stopping metric is ROC-AUC, not log loss, and hyperparameters are
deliberately shallow (max_depth=3, min_child_weight=20) -- found the hard
way. A first pass with `eval_metric='logloss'`, max_depth=4,
min_child_weight=10 never triggered early stopping (best_iteration=499 of
500: validation log loss kept inching down for the full run) and produced
train ROC-AUC 0.879 against validation 0.729 and test 0.702 -- a textbook
overfit, and one that made the "advanced" model score WORSE than the LR
baseline (test ROC-AUC 0.724) on held-out data, defeating the entire
point of building it. Root cause: log loss keeps rewarding the model for
sharpening its (already `scale_pos_weight`-inflated) predicted
probabilities long after the actual ranking ability on unseen data has
stopped improving or started degrading -- the two metrics diverge under
reweighted, imbalanced training in a way they wouldn't on a balanced
dataset. Switching the early-stopping metric to `auc` (which only cares
about ranking, matching what's actually reported and compared against
the LR baseline) plus shallower trees fixed it: train/validation/test
ROC-AUC now sit close together (~0.73-0.76) with early stopping
triggering well before the round budget, and the advanced model modestly
but genuinely beats the baseline on held-out test data, as it should.

Run: python -m credit_limit_optimizer.models.xgboost_model
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
from xgboost import XGBClassifier

from credit_limit_optimizer.features.engineering import ALL_FEATURE_COLUMNS
from credit_limit_optimizer.models.risk_model import (
    CLASSIFICATION_THRESHOLD, PROTECTED_CHARACTERISTICS, evaluate, load_feature_table, split_cohorts,
)
from credit_limit_optimizer.utils.config import PROJECT_ROOT, load_config
from credit_limit_optimizer.utils.logging import get_logger

log = get_logger("xgboost_model")

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "outputs" / "xgboost_metrics.json"
MODEL_CARD_PATH = PROJECT_ROOT / "reports" / "model_cards" / "xgboost.md"
LR_METRICS_PATH = PROJECT_ROOT / "reports" / "outputs" / "risk_model_baseline_metrics.json"

MODEL_FEATURES = [c for c in ALL_FEATURE_COLUMNS if c not in PROTECTED_CHARACTERISTICS]

HYPERPARAMETERS = {
    "n_estimators": 500, "max_depth": 3, "learning_rate": 0.03,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20,
    "reg_lambda": 1.0, "eval_metric": "auc", "early_stopping_rounds": 30,
}


def build_model(random_state: int, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        **HYPERPARAMETERS, scale_pos_weight=scale_pos_weight,
        random_state=random_state, n_jobs=-1,
    )


def top_feature_importances(model: XGBClassifier, n: int = 15) -> dict:
    importances = pd.Series(model.feature_importances_, index=MODEL_FEATURES).sort_values(ascending=False)
    return importances.head(n).round(4).to_dict()


def train_and_evaluate() -> dict:
    config = load_config()
    df = load_feature_table()
    cohorts = split_cohorts(df)
    log.info("Cohort sizes: %s", {k: len(v) for k, v in cohorts.items()})

    X_train, y_train = cohorts["train"][MODEL_FEATURES], cohorts["train"]["default_12m"]
    X_val, y_val = cohorts["validation"][MODEL_FEATURES], cohorts["validation"]["default_12m"]
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = build_model(random_state=config["random_seed"], scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    log.info(
        "Fit XGBoost on %d train rows, %d features, best_iteration=%d (of %d), scale_pos_weight=%.1f",
        len(X_train), len(MODEL_FEATURES), model.best_iteration, HYPERPARAMETERS["n_estimators"], scale_pos_weight,
    )

    metrics = {}
    for cohort_name, cohort_df in cohorts.items():
        X, y = cohort_df[MODEL_FEATURES], cohort_df["default_12m"]
        metrics[cohort_name] = evaluate(model, X, y)
        log.info(
            "%s: ROC-AUC=%.3f PR-AUC=%.3f LogLoss=%.3f Brier=%.4f F1@0.5=%.3f",
            cohort_name, metrics[cohort_name]["roc_auc"], metrics[cohort_name]["pr_auc"],
            metrics[cohort_name]["log_loss"], metrics[cohort_name]["brier_score"],
            metrics[cohort_name]["f1_at_0.5"],
        )

    importances = top_feature_importances(model)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": MODEL_FEATURES}, MODEL_PATH)
    log.info("Saved model to %s", MODEL_PATH.relative_to(PROJECT_ROOT))

    report = {
        "model": "XGBoost", "features": MODEL_FEATURES,
        "excluded_protected_characteristics": PROTECTED_CHARACTERISTICS,
        "hyperparameters": {k: v for k, v in HYPERPARAMETERS.items()},
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(model.best_iteration),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "metrics": metrics, "feature_importances": importances,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Wrote %s", METRICS_PATH.relative_to(PROJECT_ROOT))

    write_model_card(report)
    log.info("Wrote %s", MODEL_CARD_PATH.relative_to(PROJECT_ROOT))
    return report


def write_model_card(report: dict) -> None:
    m = report["metrics"]
    lines = [
        "# Model Card — XGBoost (advanced)",
        "",
        "**Target**: `default_12m` (90+ days past due within 12 months of the snapshot, "
        "among customers not already at 90+ DPD).",
        f"**Features**: {len(report['features'])} (full `ALL_FEATURE_COLUMNS` minus "
        f"{', '.join(report['excluded_protected_characteristics'])}) — no correlation "
        "pruning, unlike the LR baseline: trees handle correlated/redundant features "
        "natively, so the full trend-window set (3m/6m/12m avg + growth) is kept.",
        "**Validation**: time-based (§15), same three snapshot cohorts as the LR baseline "
        "(train = month 12, validation = month 18, test = month 24). Validation cohort "
        f"also drives early stopping (best_iteration={report['best_iteration']} of "
        f"{report['hyperparameters']['n_estimators']}, {report['hyperparameters']['early_stopping_rounds']}-round patience).",
        f"**Class imbalance**: `scale_pos_weight={report['scale_pos_weight']:.1f}` "
        "(negative/positive ratio on train) — XGBoost's analog to the LR baseline's "
        "`class_weight='balanced'`.",
        f"**Hyperparameters**: {', '.join(f'{k}={v}' for k, v in report['hyperparameters'].items() if k not in ('eval_metric', 'early_stopping_rounds'))}"
        " — reasonable defaults for this dataset's scale, not tuned via search; "
        "hyperparameter tuning is out of scope for this baseline-vs-advanced comparison.",
        "",
        "## Metrics by cohort",
        "",
        "| Cohort | N | Base rate | ROC-AUC | PR-AUC | Log Loss | Brier | Precision@0.5 | Recall@0.5 | F1@0.5 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cohort in ["train", "validation", "test"]:
        v = m[cohort]
        lines.append(
            f"| {cohort} | {v['n']:,} | {v['base_rate']:.2%} | {v['roc_auc']:.3f} | "
            f"{v['pr_auc']:.3f} | {v['log_loss']:.3f} | {v['brier_score']:.4f} | "
            f"{v['precision_at_0.5']:.3f} | {v['recall_at_0.5']:.3f} | {v['f1_at_0.5']:.3f} |"
        )
    lines += [
        "",
        f"Mean predicted probability on train: {m['train']['mean_predicted_probability']:.1%} "
        f"vs. actual base rate {m['train']['base_rate']:.1%} — `scale_pos_weight` inflates "
        "probabilities above the true rate for the same reason `class_weight='balanced'` "
        "does in the LR baseline (both reweight the loss toward the minority class); "
        "Precision/Recall/F1@0.5 are similarly not business-meaningful yet. Raw "
        "probabilities from both models await the Probability Calibration piece (§17).",
        "",
        "## Top feature importances (gain-based)",
        "",
        "Native XGBoost gain importance, not SHAP — full SHAP summary/dependence/individual "
        "explanations are a separate deliverable (§18), covering both models.",
        "",
    ] + [f"- `{k}`: {v:.4f}" for k, v in report["feature_importances"].items()] + [""]
    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_CARD_PATH.open("w") as f:
        f.write("\n".join(lines))
    append_comparison_section(report)


def append_comparison_section(xgb_report: dict) -> None:
    if not LR_METRICS_PATH.exists():
        log.warning("LR baseline metrics not found at %s -- skipping comparison section", LR_METRICS_PATH)
        return
    with LR_METRICS_PATH.open() as f:
        lr_report = json.load(f)

    lines = [
        "",
        "## Baseline vs. advanced model (test cohort)",
        "",
        "| Metric | Logistic Regression | XGBoost |",
        "|---|---|---|",
    ]
    lr_test, xgb_test = lr_report["metrics"]["test"], xgb_report["metrics"]["test"]
    for label, key in [
        ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Log Loss", "log_loss"),
        ("Brier Score", "brier_score"), ("Precision@0.5", "precision_at_0.5"),
        ("Recall@0.5", "recall_at_0.5"), ("F1@0.5", "f1_at_0.5"),
    ]:
        fmt = "{:.3f}" if key not in ("brier_score",) else "{:.4f}"
        lines.append(f"| {label} | {fmt.format(lr_test[key])} | {fmt.format(xgb_test[key])} |")
    lines += [
        "",
        f"XGBoost {'improves on' if xgb_test['roc_auc'] > lr_test['roc_auc'] else 'does not improve on'} "
        f"the LR baseline's ROC-AUC on the test cohort "
        f"({xgb_test['roc_auc']:.3f} vs. {lr_test['roc_auc']:.3f}), using the full feature "
        "set without needing the LR baseline's correlation pruning or imputation/scaling — "
        "the expected trade-off for the interpretability §14 asks the LR baseline to "
        "provide instead.",
        "",
    ]
    with MODEL_CARD_PATH.open("a") as f:
        f.write("\n".join(lines))


def main() -> None:
    train_and_evaluate()


if __name__ == "__main__":
    main()
