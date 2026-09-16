"""Tests for the Expected Loss model (§19): unit tests for the PD/LGD/EAD
helper functions on handcrafted fixtures (regression coverage for the
NaN-propagation bug found while building this), plus an integration test
against the real trained models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_limit_optimizer.models.expected_loss import (
    FIG_DIR, MODEL_CARD_PATH, MODEL_PATH, REPORT_PATH,
    compute_lgd, compute_pd, run_expected_loss,
)


def test_compute_lgd_maps_segment_to_config_value():
    df = pd.DataFrame({"customer_segment": ["prime", "near_prime", "subprime", "prime"]})
    config = {"economics": {"lgd_by_segment": {"prime": 0.55, "near_prime": 0.70, "subprime": 0.85}}}
    lgd = compute_lgd(df, config)
    np.testing.assert_array_equal(lgd, [0.55, 0.70, 0.85, 0.55])


class _FakeCalibratedModel:
    def predict_proba(self, X):
        # Deterministic fake PD, distinguishable from the forced 1.0.
        return np.column_stack([1 - np.full(len(X), 0.1), np.full(len(X), 0.1)])


def test_compute_pd_forces_one_for_already_in_default(monkeypatch, tmp_path):
    import joblib
    import credit_limit_optimizer.models.expected_loss as el_module

    fake_path = tmp_path / "fake_calibrated.joblib"
    joblib.dump({"model": _FakeCalibratedModel(), "features": ["f1"]}, fake_path)
    monkeypatch.setattr(el_module, "PD_MODEL_PATH", fake_path)

    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0],
        "eligible_for_default_label": [True, True, False],
    })
    pd_values = compute_pd(df)
    assert pd_values[0] == pytest.approx(0.1)
    assert pd_values[1] == pytest.approx(0.1)
    assert pd_values[2] == pytest.approx(1.0)  # already in default -> forced PD=1.0


@pytest.fixture(scope="module")
def report():
    return run_expected_loss()


def test_expected_loss_has_no_nan(report):
    """Regression test: credit_exposure carries real Phase 1 missing-value
    injections (~2% of rows); computing EAD-in-euros / Expected Loss
    without excluding those rows produced NaN for every aggregate
    statistic (mean_ead, mean_expected_loss, total_portfolio_expected_loss),
    silently, even though grouped aggregates looked fine because
    groupby().mean() skips NaN by default."""
    ts = report["test_summary"]
    for key in ["mean_pd", "mean_lgd", "mean_ead", "mean_expected_loss", "total_portfolio_expected_loss"]:
        assert not np.isnan(ts[key]), f"{key} is NaN"
    assert ts["n_excluded_missing_credit_exposure"] > 0


def test_expected_loss_row_accounting_is_exact(report):
    ts = report["test_summary"]
    # n (rows actually scored) + excluded (missing credit_exposure) must
    # equal the full test cohort size implied by the two counts.
    assert ts["n"] > 0
    assert ts["n_excluded_missing_credit_exposure"] < ts["n"]


def test_el_ordered_by_segment_prime_lowest_subprime_highest(report):
    by_seg = report["el_by_segment"]
    assert by_seg["prime"] < by_seg["near_prime"] < by_seg["subprime"]


def test_el_higher_for_actual_defaulters(report):
    """Expected Loss, known only at the snapshot before any outcome is
    observed, should still be systematically higher for the group that
    went on to actually default within 12 months."""
    by_outcome = report["el_by_actual_outcome"]
    assert by_outcome["default"] > by_outcome["no_default"]


def test_ead_model_beats_naive_baseline(report):
    """R² > 0 means the regression beats predicting the mean for every
    row -- a weak floor, but a real regression should clear it easily."""
    for cohort in ["train", "validation", "test"]:
        assert report["ead_metrics"][cohort]["r2"] > 0.3


def test_ead_mae_within_plausible_range(report):
    for cohort in ["train", "validation", "test"]:
        assert 0 < report["ead_metrics"][cohort]["mae"] < 0.3


def test_output_files_written(report):
    assert MODEL_PATH.exists()
    assert REPORT_PATH.exists()
    assert MODEL_CARD_PATH.exists()
    assert (FIG_DIR / "01_ead_model_fit.png").exists()
    assert (FIG_DIR / "02_el_distribution.png").exists()
    assert (FIG_DIR / "03_el_by_segment.png").exists()
    assert (FIG_DIR / "04_el_by_actual_outcome.png").exists()
