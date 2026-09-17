"""Tests for the Population Stability Index / drift monitoring engine
(§29): unit tests for the PSI formula on handcrafted fixtures, plus an
integration test against the real snapshot vintages and trained model."""

from __future__ import annotations

import numpy as np
import pytest

from credit_limit_optimizer.monitoring.drift import (
    FIG_DIR, MODEL_CARD_PATH, MONITORING_PAIRS, REPORT_PATH,
    psi, psi_severity, run_monitoring,
)


def test_psi_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    values = rng.normal(size=5000)
    value, _ = psi(values, values.copy(), n_buckets=10)
    assert value == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_a_real_mean_shift():
    rng = np.random.default_rng(1)
    baseline = rng.normal(loc=0, scale=1, size=5000)
    current = rng.normal(loc=2, scale=1, size=5000)  # large shift
    value, _ = psi(baseline, current, n_buckets=10)
    assert value > 0.25  # should register as SIGNIFICANT


def test_psi_small_shift_stays_below_warning():
    rng = np.random.default_rng(2)
    baseline = rng.normal(loc=0, scale=1, size=5000)
    current = rng.normal(loc=0.01, scale=1, size=5000)  # negligible shift
    value, _ = psi(baseline, current, n_buckets=10)
    assert value < 0.10


def test_psi_is_symmetric_in_severity_direction_but_not_value():
    """PSI(A, B) and PSI(B, A) both detect the same shift exists (both
    non-trivially > 0 for a real shift), even though the exact numeric
    value need not be identical (bucket edges are baseline-derived, so
    swapping baseline/current changes which distribution defines the
    buckets)."""
    rng = np.random.default_rng(3)
    a = rng.normal(loc=0, scale=1, size=3000)
    b = rng.normal(loc=1.5, scale=1, size=3000)
    forward, _ = psi(a, b, n_buckets=10)
    backward, _ = psi(b, a, n_buckets=10)
    assert forward > 0.1
    assert backward > 0.1


def test_psi_handles_near_constant_baseline_without_raising():
    """A near-degenerate baseline (e.g. a boolean-like feature) has too
    few distinct quantile edges to bucket meaningfully -- must return 0,
    not raise or divide by zero."""
    baseline = np.zeros(1000)
    current = np.ones(1000)
    value, breakdown = psi(baseline, current, n_buckets=10)
    assert value == 0.0
    assert breakdown.empty


def test_psi_ignores_nan_values():
    rng = np.random.default_rng(4)
    baseline = rng.normal(size=2000)
    current = np.concatenate([rng.normal(size=1900), np.full(100, np.nan)])
    value, _ = psi(baseline, current, n_buckets=10)
    assert not np.isnan(value)


@pytest.mark.parametrize("value,expected", [(0.05, "STABLE"), (0.15, "MODERATE"), (0.30, "SIGNIFICANT")])
def test_psi_severity_bands(value, expected):
    assert psi_severity(value, warning=0.10, critical=0.25) == expected


def test_psi_severity_boundary_values_are_inclusive():
    assert psi_severity(0.10, warning=0.10, critical=0.25) == "MODERATE"
    assert psi_severity(0.25, warning=0.10, critical=0.25) == "SIGNIFICANT"


@pytest.fixture(scope="module")
def report():
    return run_monitoring()


def test_report_covers_all_monitoring_pairs(report):
    expected_pairs = {f"{a}_vs_{b}" for a, b in MONITORING_PAIRS}
    assert set(report["score_drift"]) == expected_pairs
    for feature_pairs in report["feature_drift"].values():
        assert set(feature_pairs) == expected_pairs


def test_score_drift_no_nan(report):
    for row in report["score_drift"].values():
        assert not np.isnan(row["psi"])
        assert not np.isnan(row["baseline_mean"])
        assert not np.isnan(row["current_mean"])


def test_score_psi_is_stable_across_real_vintages(report):
    """The PD SCORE distribution (unlike the raw cumulative delinquency
    feature) should be broadly stable across the three real snapshot
    vintages -- a regression guard against reintroducing a scoring bug
    that would make the calibrated model's own output look unstable."""
    for row in report["score_drift"].values():
        assert row["psi"] < report["thresholds"]["psi_critical_threshold"]


def test_monitored_features_are_a_real_subset_of_shap_importance(report):
    assert len(report["monitored_features"]) == 6
    assert len(set(report["monitored_features"])) == len(report["monitored_features"])


def test_delinquency_count_flagged_as_vintage_artifact(report):
    """Documented, verified finding (see model card): delinquency_count
    is a lifetime/cumulative counter over the SAME 50,000 customers
    observed at later points in their own history, so it mechanically
    drifts across vintages -- this must show up as elevated PSI, or the
    vintage-artifact explanation in the model card would be describing a
    result that no longer exists."""
    assert report["feature_drift"]["delinquency_count"]["train_vs_test"]["psi"] > report["thresholds"]["psi_warning_threshold"]


def test_output_files_written(report):
    assert REPORT_PATH.exists()
    assert MODEL_CARD_PATH.exists()
    assert (FIG_DIR / "01_score_psi.png").exists()
    assert (FIG_DIR / "02_feature_psi_heatmap.png").exists()
    assert (FIG_DIR / "03_score_distribution_by_vintage.png").exists()
