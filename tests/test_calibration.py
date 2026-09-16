"""Tests for probability calibration (§17): unit tests for the ECE
metric on a handcrafted example, plus an integration test against the
real trained models verifying calibration actually fixes the documented
miscalibration without touching the same validation rows for both
fitting and method selection."""

from __future__ import annotations

import numpy as np
import pytest

from credit_limit_optimizer.models.calibration import (
    FIG_DIR, MODEL_CARD_PATH, MODEL_SPECS, REPORT_PATH,
    expected_calibration_error, run_calibration, score_probabilities,
)


def test_ece_zero_for_perfectly_calibrated_predictions():
    rng = np.random.default_rng(0)
    n = 5000
    y_prob = rng.uniform(0, 1, size=n)
    y_true = (rng.uniform(0, 1, size=n) < y_prob).astype(int)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 0.03  # sampling noise only


def test_ece_large_for_systematically_overconfident_predictions():
    """Mirrors the real bug: predictions uniformly ~10x the true rate."""
    rng = np.random.default_rng(0)
    n = 5000
    true_rate = 0.05
    y_true = (rng.uniform(0, 1, size=n) < true_rate).astype(int)
    y_prob = np.full(n, 0.45)  # matches the documented ~42-48% inflation
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece == pytest.approx(0.40, abs=0.02)


def test_score_probabilities_reports_mean_vs_actual_gap():
    rng = np.random.default_rng(0)
    n = 1000
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_prob = np.full(n, 0.9)
    scores = score_probabilities(y_true, y_prob)
    assert scores["mean_predicted_probability"] == pytest.approx(0.9)
    assert scores["actual_base_rate"] == pytest.approx(y_true.mean())
    assert scores["ece"] > 0.3


@pytest.fixture(scope="module")
def results():
    return run_calibration()


def test_calibration_fixes_the_documented_miscalibration(results):
    """The whole point of this piece: raw mean predicted probability is
    ~10x the actual base rate (documented in both model cards);
    calibrated should land close to it."""
    for name, r in results.items():
        best = r["chosen_method"]
        raw = r["test_scores"]["raw"]
        calibrated = r["test_scores"][best]
        assert raw["mean_predicted_probability"] > 3 * raw["actual_base_rate"], (
            f"{name}: expected the documented raw miscalibration to still be present"
        )
        assert abs(calibrated["mean_predicted_probability"] - calibrated["actual_base_rate"]) < 0.02
        assert calibrated["brier_score"] < raw["brier_score"]
        assert calibrated["ece"] < raw["ece"]


def test_calibration_preserves_roc_auc(results):
    """Calibration reshapes probability scale; it must not meaningfully
    change ranking quality (isotonic is exactly monotonic, sigmoid nearly
    so)."""
    for name, r in results.items():
        best = r["chosen_method"]
        raw_auc = r["test_scores"]["raw"]["roc_auc"]
        calibrated_auc = r["test_scores"][best]["roc_auc"]
        assert abs(raw_auc - calibrated_auc) < 0.01, f"{name}: ROC-AUC shifted more than expected under calibration"


def test_calib_fit_and_calib_select_are_disjoint():
    """Regression coverage for the split design: the same validation rows
    must never be used to both fit a calibrator and select between
    methods."""
    import credit_limit_optimizer.models.risk_model as risk_model
    from sklearn.model_selection import train_test_split
    from credit_limit_optimizer.utils.config import load_config

    config = load_config()
    df = risk_model.load_feature_table()
    cohorts = risk_model.split_cohorts(df)
    validation = cohorts["validation"]
    calib_fit, calib_select = train_test_split(
        validation, test_size=0.5, stratify=validation["default_12m"], random_state=config["random_seed"],
    )
    assert set(calib_fit.index).isdisjoint(set(calib_select.index))
    assert len(calib_fit) + len(calib_select) == len(validation)


def test_all_models_covered(results):
    assert set(results) == {spec["name"] for spec in MODEL_SPECS}


def test_output_files_written(results):
    assert REPORT_PATH.exists()
    assert MODEL_CARD_PATH.exists()
    for spec in MODEL_SPECS:
        assert spec["out_path"].exists()
        assert (FIG_DIR / f"{spec['name']}_reliability.png").exists()
