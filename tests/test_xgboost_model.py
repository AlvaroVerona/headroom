"""Tests for the XGBoost advanced model (§14): integration test against
the real time-separated cohorts, plus regression coverage for the
overfitting bug found while tuning early stopping (logloss-based early
stopping never triggered and produced train ROC-AUC 0.879 vs. test 0.702,
worse than the LR baseline)."""

from __future__ import annotations

import json

import pytest

from credit_limit_optimizer.models.xgboost_model import (
    HYPERPARAMETERS, LR_METRICS_PATH, MODEL_CARD_PATH, MODEL_PATH, METRICS_PATH,
    MODEL_FEATURES, train_and_evaluate,
)


@pytest.fixture(scope="module")
def report():
    return train_and_evaluate()


def test_eval_metric_is_auc_not_logloss():
    """Regression test: 'logloss' as the early-stopping metric let
    training run the full estimator budget while overfitting badly on
    ranking quality. 'auc' directly matches the metric this model is
    judged and compared against the LR baseline on."""
    assert HYPERPARAMETERS["eval_metric"] == "auc"


def test_early_stopping_actually_triggers(report):
    """Regression test: an earlier hyperparameter set never triggered
    early stopping (best_iteration=499 of n_estimators=500)."""
    assert report["best_iteration"] < HYPERPARAMETERS["n_estimators"] - 1


def test_train_test_roc_auc_gap_is_not_a_severe_overfit(report):
    """Regression test: the bad run had train ROC-AUC 0.879 vs. test
    0.702, a 0.177 gap. After the fix, the gap should be modest."""
    train_auc = report["metrics"]["train"]["roc_auc"]
    test_auc = report["metrics"]["test"]["roc_auc"]
    assert train_auc - test_auc < 0.08


def test_xgboost_test_roc_auc_beats_or_matches_lr_baseline(report):
    """The whole point of building an 'advanced' model (§14) is that it
    should do at least as well as the baseline on held-out data -- the
    overfit version failed this (0.702 vs. LR's 0.724)."""
    if not LR_METRICS_PATH.exists():
        pytest.skip("LR baseline metrics not present -- run risk_model training first")
    with LR_METRICS_PATH.open() as f:
        lr_report = json.load(f)
    lr_test_auc = lr_report["metrics"]["test"]["roc_auc"]
    xgb_test_auc = report["metrics"]["test"]["roc_auc"]
    assert xgb_test_auc >= lr_test_auc - 0.01, (
        f"XGBoost test ROC-AUC {xgb_test_auc:.3f} is meaningfully worse than "
        f"the LR baseline's {lr_test_auc:.3f} -- likely overfitting regression"
    )


def test_model_and_report_files_written(report):
    assert MODEL_PATH.exists()
    assert METRICS_PATH.exists()
    assert MODEL_CARD_PATH.exists()


def test_metrics_present_for_all_three_cohorts(report):
    assert set(report["metrics"]) == {"train", "validation", "test"}
    for cohort_metrics in report["metrics"].values():
        assert 0.5 < cohort_metrics["roc_auc"] < 1.0


def test_uses_full_feature_set_no_correlation_pruning(report):
    """Unlike the LR baseline, XGBoost should use the complete
    ALL_FEATURE_COLUMNS set (minus only the protected characteristic)."""
    from credit_limit_optimizer.features.engineering import ALL_FEATURE_COLUMNS
    assert len(MODEL_FEATURES) == len(ALL_FEATURE_COLUMNS) - 1  # minus age
    assert "age" not in MODEL_FEATURES
    assert "monthly_income_3m_avg" in MODEL_FEATURES  # pruned in the LR baseline, kept here


def test_top_feature_importance_directionally_sensible(report):
    top_feature = next(iter(report["feature_importances"]))
    assert top_feature in {
        "recent_delinquency", "delinquency_count", "max_days_past_due",
        "days_past_due", "credit_utilization",
    }, f"unexpected top feature importance: {top_feature}"


def test_model_card_contains_comparison_section(report):
    if not LR_METRICS_PATH.exists():
        pytest.skip("LR baseline metrics not present -- run risk_model training first")
    text = MODEL_CARD_PATH.read_text()
    assert "Baseline vs. advanced model" in text
    assert "Logistic Regression" in text
