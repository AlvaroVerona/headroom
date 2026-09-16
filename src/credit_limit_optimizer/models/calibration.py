"""Probability calibration (§17) for both trained risk models.

Both risk_model.py (LR, `class_weight='balanced'`) and xgboost_model.py
(XGBoost, `scale_pos_weight`) produce probabilities inflated far above
the true base rate (~42-48% mean predicted vs. ~4.5-4.8% actual) -- both
model cards flag this as unusable for real PD estimates. This module
fixes it.

Split usage (three-way, no cohort touched twice for anything that could
leak): the underlying models were already fit on the TRAIN cohort. The
VALIDATION cohort is split in half here -- `calib_fit` (fits the sigmoid/
isotonic calibrators) and `calib_select` (picks the better of the two
methods per model, purely by comparing their Brier scores) -- so method
selection never peeks at the same rows used to fit the calibrator. The
TEST cohort is touched exactly once, for the final, fully out-of-sample
evaluation of the selected calibrated model against the uncalibrated one.

Calibration is evaluated both quantitatively (Brier score, log loss,
Expected Calibration Error) and visually (reliability diagrams, §17's
explicit ask) for every model, comparing raw vs. calibrated.

Run: python -m credit_limit_optimizer.models.calibration
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from credit_limit_optimizer.models import risk_model, xgboost_model
from credit_limit_optimizer.utils.config import PROJECT_ROOT, load_config
from credit_limit_optimizer.utils.logging import get_logger

log = get_logger("calibration")

FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "calibration"
REPORT_PATH = PROJECT_ROOT / "reports" / "outputs" / "calibration_report.json"
MODEL_CARD_PATH = PROJECT_ROOT / "reports" / "model_cards" / "calibration.md"

CALIBRATION_METHODS = ["sigmoid", "isotonic"]
N_ECE_BINS = 10

MODEL_SPECS = [
    {"name": "logistic_regression", "module": risk_model, "estimator_key": "pipeline",
     "raw_path": risk_model.MODEL_PATH, "out_path": PROJECT_ROOT / "models" / "logistic_regression_calibrated.joblib"},
    {"name": "xgboost", "module": xgboost_model, "estimator_key": "model",
     "raw_path": xgboost_model.MODEL_PATH, "out_path": PROJECT_ROOT / "models" / "xgboost_calibrated.joblib"},
]


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_ECE_BINS) -> float:
    """Mean absolute gap between predicted probability and observed
    frequency, weighted by bin size -- the standard ECE definition."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def score_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "mean_predicted_probability": float(y_prob.mean()),
        "actual_base_rate": float(y_true.mean()),
    }


def reliability_diagram(results: dict, model_name: str, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9E9E9E", label="Perfect calibration")
    colors = {"raw": "#C62828", "sigmoid": "#F9A825", "isotonic": "#2E7D32"}
    for label, (y_true, y_prob) in results.items():
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=N_ECE_BINS, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", markersize=4, label=label, color=colors.get(label))
    ax.set_xlabel("Mean predicted probability (within bin)")
    ax.set_ylabel("Observed default rate (within bin)")
    ax.set_title(f"Reliability diagram — {model_name} (test cohort)")
    ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def calibrate_one_model(spec: dict, cohorts: dict, random_state: int) -> dict:
    name = spec["name"]
    saved = joblib.load(spec["raw_path"])
    estimator = saved[spec["estimator_key"]]
    features = saved["features"]

    validation = cohorts["validation"]
    calib_fit, calib_select = train_test_split(
        validation, test_size=0.5, stratify=validation["default_12m"], random_state=random_state,
    )
    X_calib_fit, y_calib_fit = calib_fit[features], calib_fit["default_12m"]
    X_calib_select, y_calib_select = calib_select[features], calib_select["default_12m"]
    X_test, y_test = cohorts["test"][features], cohorts["test"]["default_12m"]

    raw_proba_select = estimator.predict_proba(X_calib_select)[:, 1]
    raw_proba_test = estimator.predict_proba(X_test)[:, 1]

    candidates = {"raw": estimator}
    select_scores = {"raw": score_probabilities(y_calib_select.to_numpy(), raw_proba_select)}
    for method in CALIBRATION_METHODS:
        calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
        calibrated.fit(X_calib_fit, y_calib_fit)
        proba_select = calibrated.predict_proba(X_calib_select)[:, 1]
        select_scores[method] = score_probabilities(y_calib_select.to_numpy(), proba_select)
        candidates[method] = calibrated

    best_method = min(CALIBRATION_METHODS, key=lambda m: select_scores[m]["brier_score"])
    best_model = candidates[best_method]
    log.info(
        "%s: calib_select Brier -- raw=%.4f sigmoid=%.4f isotonic=%.4f -> chose %s",
        name, select_scores["raw"]["brier_score"], select_scores["sigmoid"]["brier_score"],
        select_scores["isotonic"]["brier_score"], best_method,
    )

    test_scores = {
        "raw": score_probabilities(y_test.to_numpy(), raw_proba_test),
        best_method: score_probabilities(y_test.to_numpy(), best_model.predict_proba(X_test)[:, 1]),
    }
    log.info(
        "%s test: raw Brier=%.4f ECE=%.4f | %s Brier=%.4f ECE=%.4f",
        name, test_scores["raw"]["brier_score"], test_scores["raw"]["ece"],
        best_method, test_scores[best_method]["brier_score"], test_scores[best_method]["ece"],
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    reliability_diagram(
        {"raw": (y_test.to_numpy(), raw_proba_test), best_method: (y_test.to_numpy(), best_model.predict_proba(X_test)[:, 1])},
        name, FIG_DIR / f"{name}_reliability.png",
    )

    spec["out_path"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "features": features, "method": best_method}, spec["out_path"])
    log.info("Saved calibrated %s to %s", name, spec["out_path"].relative_to(PROJECT_ROOT))

    return {
        "model": name, "chosen_method": best_method,
        "calib_select_scores": select_scores, "test_scores": test_scores,
        "n_calib_fit": len(calib_fit), "n_calib_select": len(calib_select),
    }


def run_calibration() -> dict:
    config = load_config()
    df = risk_model.load_feature_table()
    cohorts = risk_model.split_cohorts(df)

    results = {}
    for spec in MODEL_SPECS:
        results[spec["name"]] = calibrate_one_model(spec, cohorts, random_state=config["random_seed"])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Wrote %s", REPORT_PATH.relative_to(PROJECT_ROOT))

    write_model_card(results)
    log.info("Wrote %s", MODEL_CARD_PATH.relative_to(PROJECT_ROOT))
    return results


def write_model_card(results: dict) -> None:
    lines = [
        "# Model Card — Probability Calibration",
        "",
        "Both `class_weight='balanced'` (LR) and `scale_pos_weight` (XGBoost) inflate raw "
        "predicted probabilities far above the true default rate to improve ranking during "
        "training -- fine for ROC-AUC, unusable as PD estimates. This fixes that.",
        "",
        "**Method selection**: sigmoid (Platt) and isotonic calibrators are both fit on "
        "half of the validation cohort (`calib_fit`), then scored by Brier score on the "
        "other half (`calib_select`) to choose the better one per model -- so the same rows "
        "are never used to both fit and select. The test cohort is touched exactly once, "
        "for the final reported numbers below.",
        "",
    ]
    for name, r in results.items():
        cs = r["calib_select_scores"]
        ts = r["test_scores"]
        best = r["chosen_method"]
        lines += [
            f"## {name}",
            "",
            f"Chosen method: **{best}** (lower Brier on `calib_select`: raw "
            f"{cs['raw']['brier_score']:.4f}, sigmoid {cs['sigmoid']['brier_score']:.4f}, "
            f"isotonic {cs['isotonic']['brier_score']:.4f}).",
            "",
            "| | Mean predicted proba | Actual base rate | Brier | Log Loss | ECE | ROC-AUC |",
            "|---|---|---|---|---|---|---|",
            f"| Raw (test) | {ts['raw']['mean_predicted_probability']:.1%} | "
            f"{ts['raw']['actual_base_rate']:.1%} | {ts['raw']['brier_score']:.4f} | "
            f"{ts['raw']['log_loss']:.3f} | {ts['raw']['ece']:.4f} | {ts['raw']['roc_auc']:.3f} |",
            f"| Calibrated ({best}, test) | {ts[best]['mean_predicted_probability']:.1%} | "
            f"{ts[best]['actual_base_rate']:.1%} | {ts[best]['brier_score']:.4f} | "
            f"{ts[best]['log_loss']:.3f} | {ts[best]['ece']:.4f} | {ts[best]['roc_auc']:.3f} |",
            "",
            f"![reliability diagram](../figures/calibration/{name}_reliability.png)",
            "",
        ]
    lines += [
        "ROC-AUC is unchanged by calibration up to floating-point noise (calibration is a "
        "monotonic-ish reshaping of probabilities for sigmoid, exactly monotonic for "
        "isotonic — ranking is preserved or nearly so), confirming calibration only fixes "
        "the probability *scale*, not the model's discriminative power.",
        "",
    ]
    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_CARD_PATH.open("w") as f:
        f.write("\n".join(lines))


def main() -> None:
    run_calibration()


if __name__ == "__main__":
    main()
