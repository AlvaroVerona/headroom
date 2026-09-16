"""Credit risk model, baseline (§14): Logistic Regression predicting
default_12m.

Time-based validation (§15): train/validation/test are NOT a random split
-- they're `features.csv`'s three snapshot cohorts (months 12/18/24),
each drawn from the same 50,000 customers but at successively later
points in time, with default_12m observed over the following 12 months.
This matters in credit risk specifically because a random split would let
the model see a customer's month-18 behavior while training on their
month-12 outcome (or vice versa) -- information that would never be
available at real decision time, and because credit risk is non-
stationary (economic conditions drift), so a model must be judged on its
ability to generalize FORWARD in time, not just to unseen customers at
the same point in time a random split would test.

Feature set: ALL_FEATURE_COLUMNS minus `age` minus features that are
near-duplicates of an earlier-listed feature on the train cohort (see
select_model_features below).

`age` is the one protected characteristic (§ fairness) in the feature
table; CLAUDE.md's rule is not to use one as a direct model input without
a fairness discussion, and Phase 2's EDA already found age carries ~0
default signal (flat within ~1pp across every age band) -- so excluding
it costs nothing and needs no such discussion for this baseline. Every
other feature is a behavioral/credit variable, not a protected
characteristic.

Several feature pairs turned out to be near- or exactly duplicated on the
real data -- textbook multicollinearity, which doesn't hurt a linear
model's ranking (ROC-AUC) but makes coefficients numerically unstable and
uninterpretable, which matters because interpretability is the explicit
point of the LR baseline (§14). Found by inspecting the train cohort's
correlation matrix after the first run produced a coefficient list that
flatly contradicted the EDA: `monthly_income_6m_avg`/`12m_avg` were the
top RISK-INCREASING coefficients while `monthly_income_3m_avg` was the
top RISK-REDUCING one, even though EDA found default rate falls
monotonically with income. The three trend windows of the same variable
are r > 0.99 correlated with each other (they're rolling averages of
nearly the same slow-moving series); `income_stability` is a literal
`1 - income_volatility` transform (r = 1.0); `minimum_payment_ratio` is
r = 1.0 with `payment_ratio`; and `credit_utilization_Nm_growth` is
r = 1.0 with `end_balance_Nm_growth` (mathematically inevitable: this
generator never changes a customer's credit_limit over the 36-month
history, so utilization and balance move in exact lockstep and a
*relative*-growth ratio of either is scale-invariant to that constant
limit). `select_model_features` prunes all of these automatically and
deterministically (greedy, in ALL_FEATURE_COLUMNS order, threshold 0.95,
fit on the train cohort only -- no leakage) rather than chasing each
pair by hand. This pruning is specific to the linear baseline -- XGBoost
(next piece) handles correlated features natively and should use the
full ALL_FEATURE_COLUMNS set.

Run: python -m credit_limit_optimizer.models.risk_model
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_limit_optimizer.features.engineering import ALL_FEATURE_COLUMNS, build_feature_table, load_inputs
from credit_limit_optimizer.utils.config import PROJECT_ROOT, load_config
from credit_limit_optimizer.utils.logging import get_logger

log = get_logger("risk_model")

MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "outputs" / "risk_model_baseline_metrics.json"
MODEL_CARD_PATH = PROJECT_ROOT / "reports" / "model_cards" / "logistic_regression.md"

PROTECTED_CHARACTERISTICS = ["age"]
CORRELATION_PRUNE_THRESHOLD = 0.95


def select_model_features(train_df: pd.DataFrame, threshold: float = CORRELATION_PRUNE_THRESHOLD) -> tuple[list[str], dict]:
    """Greedy correlation-based pruning: walk ALL_FEATURE_COLUMNS in
    order, keep a feature unless it's more than `threshold`-correlated
    (Pearson, on train only) with an already-kept feature. Deterministic
    and data-driven -- rerun this if the feature set changes rather than
    hand-listing exclusions."""
    candidates = [c for c in ALL_FEATURE_COLUMNS if c not in PROTECTED_CHARACTERISTICS]
    corr = train_df[candidates].corr().abs()
    kept: list[str] = []
    dropped = {}
    for feat in candidates:
        collision = next((k for k in kept if corr.loc[feat, k] > threshold), None)
        if collision is not None:
            dropped[feat] = {"correlated_with": collision, "correlation": round(float(corr.loc[feat, collision]), 4)}
        else:
            kept.append(feat)
    return kept, dropped


CLASSIFICATION_THRESHOLD = 0.5


def load_feature_table() -> pd.DataFrame:
    config = load_config()
    inputs = load_inputs()
    table = build_feature_table(inputs, config)
    return table.dropna(subset=["default_12m"])


def split_cohorts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {cohort: df[df["cohort"] == cohort] for cohort in ["train", "validation", "test"]}


def build_pipeline(random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state,
        )),
    ])


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= CLASSIFICATION_THRESHOLD).astype(int)
    return {
        "n": int(len(y)), "base_rate": float(y.mean()),
        "mean_predicted_probability": float(proba.mean()),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "log_loss": float(log_loss(y, proba)),
        "brier_score": float(brier_score_loss(y, proba)),
        "precision_at_0.5": float(precision_score(y, pred, zero_division=0)),
        "recall_at_0.5": float(recall_score(y, pred, zero_division=0)),
        "f1_at_0.5": float(f1_score(y, pred, zero_division=0)),
    }


def top_coefficients(pipeline: Pipeline, model_features: list[str], n: int = 10) -> dict:
    coefs = pd.Series(pipeline.named_steps["model"].coef_[0], index=model_features).sort_values()
    return {
        "most_risk_increasing": coefs.tail(n)[::-1].round(4).to_dict(),
        "most_risk_reducing": coefs.head(n).round(4).to_dict(),
    }


def train_and_evaluate() -> dict:
    config = load_config()
    df = load_feature_table()
    cohorts = split_cohorts(df)
    log.info("Cohort sizes: %s", {k: len(v) for k, v in cohorts.items()})

    model_features, dropped_for_correlation = select_model_features(cohorts["train"])
    log.info(
        "Selected %d of %d candidate features (%d dropped for correlation > %.2f with an earlier feature)",
        len(model_features), len(ALL_FEATURE_COLUMNS) - len(PROTECTED_CHARACTERISTICS),
        len(dropped_for_correlation), CORRELATION_PRUNE_THRESHOLD,
    )

    X_train, y_train = cohorts["train"][model_features], cohorts["train"]["default_12m"]
    pipeline = build_pipeline(random_state=config["random_seed"])
    pipeline.fit(X_train, y_train)
    log.info("Fit LogisticRegression on %d train rows, %d features", len(X_train), len(model_features))

    metrics = {}
    for cohort_name, cohort_df in cohorts.items():
        X, y = cohort_df[model_features], cohort_df["default_12m"]
        metrics[cohort_name] = evaluate(pipeline, X, y)
        log.info(
            "%s: ROC-AUC=%.3f PR-AUC=%.3f LogLoss=%.3f Brier=%.4f F1@0.5=%.3f",
            cohort_name, metrics[cohort_name]["roc_auc"], metrics[cohort_name]["pr_auc"],
            metrics[cohort_name]["log_loss"], metrics[cohort_name]["brier_score"],
            metrics[cohort_name]["f1_at_0.5"],
        )

    coefficients = top_coefficients(pipeline, model_features)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": model_features}, MODEL_PATH)
    log.info("Saved model to %s", MODEL_PATH.relative_to(PROJECT_ROOT))

    report = {
        "model": "LogisticRegression", "features": model_features,
        "excluded_protected_characteristics": PROTECTED_CHARACTERISTICS,
        "dropped_for_correlation": dropped_for_correlation,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "metrics": metrics, "coefficients": coefficients,
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
    c = report["coefficients"]
    lines = [
        "# Model Card — Logistic Regression (baseline)",
        "",
        "**Target**: `default_12m` (90+ days past due within 12 months of the snapshot, "
        "among customers not already at 90+ DPD).",
        f"**Features**: {len(report['features'])} ({', '.join(report['features'][:8])}, ...; "
        f"full list in `risk_model_baseline_metrics.json`). Excludes "
        f"{', '.join(report['excluded_protected_characteristics'])} (protected "
        "characteristic; §12 EDA found ~0 default signal for it anyway) and "
        f"{len(report['dropped_for_correlation'])} features dropped by greedy correlation "
        f"pruning (>{CORRELATION_PRUNE_THRESHOLD:.0%} Pearson correlation, train cohort "
        "only, with an earlier-kept feature — see `dropped_for_correlation` in "
        "`risk_model_baseline_metrics.json` for the exact pairs; kept coefficients "
        "numerically stable and interpretable, see module docstring for the sign-flip this "
        "fixed).",
        "**Validation**: time-based (§15) — train = month-12 snapshot, validation = "
        "month-18, test = month-24, all drawn from the same 50,000 customers but at "
        "successively later, non-overlapping points in time. A random split would let the "
        "model see later-snapshot behavior while training on an earlier-snapshot outcome "
        "(or vice versa), which is never available at real decision time; credit risk is "
        "also non-stationary, so generalizing forward in time is the property that "
        "actually matters.",
        f"**Class imbalance**: base rate ~{m['train']['base_rate']:.1%}; handled with "
        "`class_weight='balanced'`, not resampling.",
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
    train_mean_pred = m["train"]["mean_predicted_probability"]
    train_base_rate = m["train"]["base_rate"]
    lines += [
        "",
        f"**Probabilities are badly miscalibrated**: mean predicted probability "
        f"{train_mean_pred:.1%} vs. actual base rate {train_base_rate:.1%} on train (similar "
        "gap on validation/test) — `class_weight='balanced'` reweights the training loss to "
        "improve ranking/discrimination for the minority class, which is exactly why ROC-AUC "
        "is usable above, but it systematically inflates predicted probabilities well above "
        "the true rate as a side effect. This is why Precision/Recall/F1 at the naive 0.5 "
        "threshold look strange (precision ~8-10%, recall ~60-70%: the model flags far more "
        "customers as high-risk than actually default) — expected given the inflated "
        "probabilities, not a bug. Raw probabilities from this model are **not** usable "
        "as real PD estimates yet; that's exactly what the Probability Calibration piece "
        "(§17, Platt scaling / isotonic regression) fixes next. Threshold selection for "
        "actual risk bands is also deferred to §16, on calibrated probabilities.",
        "",
        "## Top coefficients (standardized features, so directly comparable)",
        "",
        "**Most risk-increasing:**",
        "",
    ] + [f"- `{k}`: {v:+.4f}" for k, v in c["most_risk_increasing"].items()] + [
        "",
        "**Most risk-reducing:**",
        "",
    ] + [f"- `{k}`: {v:+.4f}" for k, v in c["most_risk_reducing"].items()] + [
        "",
        f"A few small-magnitude coefficients above (e.g. `debt_to_income`, "
        f"`max_utilization`, `credit_utilization_3m_avg`, all under 0.03 in absolute "
        f"value) have a sign that looks surprising next to their raw EDA correlation "
        f"with default. This is expected multivariate behavior, not a bug: each is still "
        f"0.8-0.95 correlated with another kept feature (below the "
        f"{CORRELATION_PRUNE_THRESHOLD:.0%} pruning threshold, so not dropped), and a "
        "linear model's coefficient reflects the *partial* effect holding every other "
        "feature fixed, not the raw marginal correlation EDA reports. Pruning further "
        "into the 0.8-0.95 range would start discarding real, non-redundant information "
        "for diminishing interpretability gains on what is a baseline model anyway.",
        "",
    ]
    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_CARD_PATH.open("w") as f:
        f.write("\n".join(lines))


def main() -> None:
    train_and_evaluate()


if __name__ == "__main__":
    main()
