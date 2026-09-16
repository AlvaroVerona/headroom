"""SHAP explainability (§18), for both risk models: global feature
importance, SHAP summary plot, SHAP dependence plots, and individual
customer explanations.

SHAP explains each model's own raw decision function (the log-odds/margin
that drives ranking), not the calibrated probability -- calibration
(sigmoid/isotonic, from calibration.py) is a post-hoc, non-linear
reshaping of the output that doesn't have a well-defined per-feature
attribution. So SHAP values here answer "why did the model's underlying
score come out this way", while the "Predicted Default Probability"
shown in each individual customer explanation is the CALIBRATED
probability from calibration.py -- the actually business-meaningful PD.
These two numbers come from different (documented) sources on purpose;
they should tell the same directional story but won't match a raw
sigmoid(margin) computation.

Explainer choice:
- LogisticRegression: `shap.LinearExplainer` on the imputed+scaled
  feature space (matching the pipeline's own preprocessing) -- exact for
  a linear model, no sampling approximation needed.
- XGBoost: `shap.TreeExplainer` with `feature_perturbation=
  "tree_path_dependent"` directly on the raw (NaN-containing) features --
  exact, fast, and handles missing values the same way the model itself
  does, no background dataset needed.

Per CLAUDE.md convention, summary/dependence plots subsample rows for
runtime (documented sample size below) -- individual customer
explanations use the customer's own single row, not a sample.

Run: python -m credit_limit_optimizer.models.explain
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from credit_limit_optimizer.models import risk_model, xgboost_model
from credit_limit_optimizer.utils.config import PROJECT_ROOT, load_config
from credit_limit_optimizer.utils.logging import get_logger

log = get_logger("explain")

FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "shap"
REPORT_PATH = PROJECT_ROOT / "reports" / "outputs" / "shap_report.json"
MODEL_CARD_PATH = PROJECT_ROOT / "reports" / "model_cards" / "explainability.md"

SUMMARY_SAMPLE_SIZE = 2000
N_DEPENDENCE_PLOTS = 4
N_EXAMPLE_CUSTOMERS = 3

MODEL_SPECS = [
    {
        "name": "logistic_regression", "raw_path": risk_model.MODEL_PATH,
        "calibrated_path": PROJECT_ROOT / "models" / "logistic_regression_calibrated.joblib",
    },
    {
        "name": "xgboost", "raw_path": xgboost_model.MODEL_PATH,
        "calibrated_path": PROJECT_ROOT / "models" / "xgboost_calibrated.joblib",
    },
]


LINEAR_BACKGROUND_SIZE = 200


def build_linear_explainer(pipeline, X_background: pd.DataFrame):
    preprocessor = pipeline[:-1]
    linear_model = pipeline.named_steps["model"]
    background_transformed = preprocessor.transform(X_background)
    # Explicit max_samples avoids SHAP silently subsampling the masker to
    # its default of 100 with a warning; 200 rows is already more than
    # enough to estimate feature means for a linear model's interventional
    # SHAP values (unlike the tree explainer, this isn't the sample being
    # explained -- just the background distribution SHAP compares against).
    masker = shap.maskers.Independent(background_transformed, max_samples=LINEAR_BACKGROUND_SIZE)
    explainer = shap.LinearExplainer(linear_model, masker)
    return explainer, preprocessor


def compute_shap_values(name: str, estimator, features: list[str], X: pd.DataFrame, rng: np.random.Generator):
    if name == "logistic_regression":
        explainer, preprocessor = build_linear_explainer(estimator, X)
        X_transformed = pd.DataFrame(preprocessor.transform(X), columns=features, index=X.index)
        shap_values = explainer(X_transformed)
        return shap_values, X_transformed
    else:
        explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
        shap_values = explainer(X)
        return shap_values, X


def global_importance(shap_values, features: list[str]) -> dict:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    return pd.Series(mean_abs, index=features).sort_values(ascending=False).round(4).to_dict()


def plot_summary(shap_values, X_display: pd.DataFrame, name: str) -> None:
    fig = plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values.values, X_display, show=False, max_display=15)
    plt.title(f"SHAP summary — {name} (n={len(X_display)} sampled rows)")
    fig.savefig(FIG_DIR / f"{name}_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_dependence(shap_values, X_display: pd.DataFrame, features: list[str], name: str, top_features: list[str]) -> None:
    for feat in top_features:
        fig = plt.figure(figsize=(6, 4.5))
        shap.dependence_plot(feat, shap_values.values, X_display, feature_names=features, show=False, ax=plt.gca())
        plt.title(f"SHAP dependence — {name}: {feat}")
        safe_name = feat.replace("/", "_")
        fig.savefig(FIG_DIR / f"{name}_dependence_{safe_name}.png", bbox_inches="tight")
        plt.close(fig)


def explain_customer(
    customer_row: pd.Series, features: list[str], shap_row: np.ndarray, base_value: float,
    calibrated_proba: float, n_factors: int = 5,
) -> dict:
    contributions = pd.Series(shap_row, index=features).sort_values()
    risk_reducing = contributions.head(n_factors)
    risk_increasing = contributions.tail(n_factors)[::-1]
    return {
        "customer_id": customer_row["customer_id"],
        "predicted_default_probability": float(calibrated_proba),
        "model_base_value": float(base_value),
        "risk_increasing_factors": [
            {"feature": f, "value": float(customer_row[f]) if pd.notna(customer_row[f]) else None, "shap_contribution": round(float(v), 4)}
            for f, v in risk_increasing.items()
        ],
        "risk_reducing_factors": [
            {"feature": f, "value": float(customer_row[f]) if pd.notna(customer_row[f]) else None, "shap_contribution": round(float(v), 4)}
            for f, v in risk_reducing.items()
        ],
    }


def pick_example_customers(cohort_df: pd.DataFrame, calibrated_proba: np.ndarray, n: int) -> np.ndarray:
    order = np.argsort(calibrated_proba)
    if n >= len(order):
        return order
    # Lowest, highest, and median predicted risk -- one clearly-safe,
    # one clearly-risky, one ambiguous example.
    idx = [order[0], order[len(order) // 2], order[-1]]
    return np.array(idx[:n])


def explain_one_model(spec: dict, cohorts: dict, rng: np.random.Generator) -> dict:
    name = spec["name"]
    raw_saved = joblib.load(spec["raw_path"])
    estimator = raw_saved.get("pipeline", raw_saved.get("model"))
    features = raw_saved["features"]
    calibrated_saved = joblib.load(spec["calibrated_path"])
    calibrated_model = calibrated_saved["model"]

    test = cohorts["test"]
    sample_n = min(SUMMARY_SAMPLE_SIZE, len(test))
    sample_idx = rng.choice(test.index, size=sample_n, replace=False)
    X_sample = test.loc[sample_idx, features]

    shap_values, X_display = compute_shap_values(name, estimator, features, X_sample, rng)
    importance = global_importance(shap_values, features)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_summary(shap_values, X_display, name)
    top_features = list(importance)[:N_DEPENDENCE_PLOTS]
    plot_dependence(shap_values, X_display, features, name, top_features)
    log.info("%s: global importance top-5: %s", name, dict(list(importance.items())[:5]))

    calibrated_proba_sample = calibrated_model.predict_proba(X_sample)[:, 1]
    example_positions = pick_example_customers(test.loc[sample_idx], calibrated_proba_sample, N_EXAMPLE_CUSTOMERS)

    base_value = float(np.asarray(shap_values.base_values).reshape(-1)[0]) if hasattr(shap_values, "base_values") else 0.0
    examples = []
    for pos in example_positions:
        row_index = sample_idx[pos]
        customer_row = test.loc[row_index]
        examples.append(explain_customer(
            customer_row, features, shap_values.values[pos], base_value,
            calibrated_proba_sample[pos],
        ))
        log.info(
            "%s example customer %s: predicted PD=%.1f%%",
            name, customer_row["customer_id"], calibrated_proba_sample[pos] * 100,
        )

    return {
        "model": name, "sample_size": sample_n, "global_importance": importance,
        "top_dependence_features": top_features, "examples": examples,
    }


def run_explainability() -> dict:
    config = load_config()
    rng = np.random.default_rng(config["random_seed"])
    df = risk_model.load_feature_table()
    cohorts = risk_model.split_cohorts(df)

    # This module owns FIG_DIR exclusively -- clear it first so a feature
    # that's no longer in the top-N dependence plots (e.g. after a
    # feature-selection change upstream) doesn't leave a stale, orphaned
    # PNG behind that no longer matches the model card's references.
    if FIG_DIR.exists():
        for stale in FIG_DIR.glob("*.png"):
            stale.unlink()

    results = {}
    for spec in MODEL_SPECS:
        results[spec["name"]] = explain_one_model(spec, cohorts, rng)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Wrote %s", REPORT_PATH.relative_to(PROJECT_ROOT))

    write_model_card(results)
    log.info("Wrote %s", MODEL_CARD_PATH.relative_to(PROJECT_ROOT))
    return results


def write_model_card(results: dict) -> None:
    lines = [
        "# Model Card — SHAP Explainability",
        "",
        "SHAP explains each model's own raw decision function, not the calibrated "
        "probability (a post-hoc, non-linear reshaping with no well-defined per-feature "
        "attribution) — see module docstring. `Predicted Default Probability` in each "
        "individual example below is the CALIBRATED probability "
        "(`calibration.py`), the business-meaningful PD; SHAP factors explain the "
        "direction and relative weight of what drove the model there, not that exact "
        "number.",
        "",
    ]
    for name, r in results.items():
        lines += [
            f"## {name}",
            "",
            f"Summary/dependence plots computed on {r['sample_size']} sampled test-cohort "
            "rows (full customer explanations below use each customer's own single row, no "
            "sampling).",
            "",
            "### Global feature importance (top 10, mean |SHAP value|)",
            "",
        ]
        for feat, val in list(r["global_importance"].items())[:10]:
            lines.append(f"- `{feat}`: {val:.4f}")
        lines += [
            "",
            f"![SHAP summary](../figures/shap/{name}_summary.png)",
            "",
            "### SHAP dependence plots",
            "",
        ]
        for feat in r["top_dependence_features"]:
            lines.append(f"![SHAP dependence {feat}](../figures/shap/{name}_dependence_{feat}.png)")
        lines += ["", "### Individual customer explanations", ""]
        for ex in r["examples"]:
            lines += [
                f"**Customer: {ex['customer_id']}**",
                "",
                f"Predicted Default Probability: {ex['predicted_default_probability']:.1%}",
                "",
                "Main risk-increasing factors:",
            ]
            for f in ex["risk_increasing_factors"]:
                val = f"{f['value']:.3f}" if f["value"] is not None else "NaN"
                lines.append(f"- `{f['feature']}` = {val} (SHAP {f['shap_contribution']:+.4f})")
            lines += ["", "Main risk-reducing factors:"]
            for f in ex["risk_reducing_factors"]:
                val = f"{f['value']:.3f}" if f["value"] is not None else "NaN"
                lines.append(f"- `{f['feature']}` = {val} (SHAP {f['shap_contribution']:+.4f})")
            lines.append("")
    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_CARD_PATH.open("w") as f:
        f.write("\n".join(lines))


def main() -> None:
    run_explainability()


if __name__ == "__main__":
    main()
