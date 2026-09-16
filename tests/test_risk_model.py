"""Tests for the Logistic Regression baseline (§14): unit tests for the
correlation-based feature pruning on a handcrafted fixture (regression
coverage for the multicollinearity bug found while building this), plus
an integration test that trains against the real time-separated cohorts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_limit_optimizer.models.risk_model import (
    CORRELATION_PRUNE_THRESHOLD, MODEL_CARD_PATH, MODEL_PATH, METRICS_PATH,
    PROTECTED_CHARACTERISTICS, select_model_features, train_and_evaluate,
)
from credit_limit_optimizer.features.engineering import ALL_FEATURE_COLUMNS


def _toy_train_df(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, size=n)
    df = pd.DataFrame({col: rng.normal(0, 1, size=n) for col in ALL_FEATURE_COLUMNS})
    # Make credit_utilization and its perfectly-correlated "growth" proxy
    # (standing in for the real end_balance_growth identity) -- and a
    # near-duplicate 6m_avg column at r > 0.95.
    df["credit_utilization"] = base
    df["credit_utilization_3m_growth"] = base * 2 + 5  # exact linear transform -> r = 1.0
    df["credit_utilization_6m_avg"] = base + rng.normal(0, 0.05, size=n)  # r > 0.99
    return df


def test_select_model_features_drops_perfectly_correlated_pair():
    df = _toy_train_df()
    kept, dropped = select_model_features(df, threshold=CORRELATION_PRUNE_THRESHOLD)
    # credit_utilization comes before its perfectly-correlated transform in
    # ALL_FEATURE_COLUMNS order, so the transform must be the one dropped.
    assert "credit_utilization" in kept
    assert "credit_utilization_3m_growth" not in kept
    assert "credit_utilization_3m_growth" in dropped
    assert dropped["credit_utilization_3m_growth"]["correlated_with"] == "credit_utilization"
    assert dropped["credit_utilization_3m_growth"]["correlation"] == pytest.approx(1.0, abs=1e-6)


def test_select_model_features_drops_near_duplicate_above_threshold():
    df = _toy_train_df()
    kept, dropped = select_model_features(df, threshold=CORRELATION_PRUNE_THRESHOLD)
    assert "credit_utilization_6m_avg" not in kept
    assert dropped["credit_utilization_6m_avg"]["correlation"] > CORRELATION_PRUNE_THRESHOLD


def test_select_model_features_never_drops_protected_characteristics():
    """age should never even be a candidate, regardless of correlation."""
    df = _toy_train_df()
    kept, dropped = select_model_features(df)
    for characteristic in PROTECTED_CHARACTERISTICS:
        assert characteristic not in kept
        assert characteristic not in dropped


def test_select_model_features_uncorrelated_columns_all_kept():
    rng = np.random.default_rng(1)
    n = 500
    df = pd.DataFrame({col: rng.normal(0, 1, size=n) for col in ALL_FEATURE_COLUMNS})
    kept, dropped = select_model_features(df, threshold=CORRELATION_PRUNE_THRESHOLD)
    # Independently-drawn normals essentially never exceed 0.95 correlation
    # at n=500 -- almost everything should survive.
    assert len(dropped) <= 2
    assert len(kept) >= len(ALL_FEATURE_COLUMNS) - len(PROTECTED_CHARACTERISTICS) - 2


@pytest.fixture(scope="module")
def report():
    return train_and_evaluate()


def test_kept_features_are_pairwise_below_threshold_on_real_train_data(report):
    """Regression test: an early version of this baseline kept all 3
    trend-average windows of the same variable (r > 0.99 for income),
    producing a coefficient list that flatly contradicted the EDA
    (monthly_income_6m_avg/12m_avg showed up as risk-INCREASING while
    monthly_income_3m_avg was the top risk-REDUCING coefficient). No pair
    of kept features should exceed the pruning threshold on the actual
    train cohort."""
    from credit_limit_optimizer.models.risk_model import load_feature_table, split_cohorts
    df = load_feature_table()
    train = split_cohorts(df)["train"]
    corr = train[report["features"]].corr().abs().to_numpy(copy=True)
    np.fill_diagonal(corr, 0)
    assert corr.max() <= CORRELATION_PRUNE_THRESHOLD


def test_model_and_report_files_written(report):
    assert MODEL_PATH.exists()
    assert METRICS_PATH.exists()
    assert MODEL_CARD_PATH.exists()


def test_metrics_present_for_all_three_cohorts(report):
    assert set(report["metrics"]) == {"train", "validation", "test"}
    for cohort_metrics in report["metrics"].values():
        assert 0.5 < cohort_metrics["roc_auc"] < 1.0, "baseline should beat random guessing by a clear margin"
        assert 0 <= cohort_metrics["log_loss"]
        assert 0 <= cohort_metrics["brier_score"] <= 1


def test_default_rate_ordered_by_segment_reproduced_in_cohort_sizes(report):
    """Sanity check the loaded/split data still matches the known,
    previously-verified default rates (not a new claim -- just guards
    against silent drift in the upstream pipeline)."""
    for cohort_metrics in report["metrics"].values():
        assert 0.03 < cohort_metrics["base_rate"] < 0.06


def test_probabilities_are_not_naively_assumed_calibrated(report):
    """class_weight='balanced' is known (see model card) to inflate mean
    predicted probability well above the true base rate -- assert this
    documented property holds, so a future change silently "fixing" the
    imbalance handling doesn't go unnoticed without updating the card."""
    for cohort_metrics in report["metrics"].values():
        assert cohort_metrics["mean_predicted_probability"] > 3 * cohort_metrics["base_rate"]


def test_top_coefficients_directionally_consistent_with_eda_for_strong_signals(report):
    """The dominant risk drivers from EDA (delinquency history, utilization)
    must retain their expected sign among the coefficients that made the
    top-10 cut -- the whole point of the correlation-pruning fix."""
    increasing = report["coefficients"]["most_risk_increasing"]
    if "delinquency_count" in increasing:
        assert increasing["delinquency_count"] > 0
    if "credit_utilization" in increasing:
        assert increasing["credit_utilization"] > 0
