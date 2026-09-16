"""Tests for SHAP explainability (§18): unit tests for the small pure
helpers on handcrafted data, plus an integration test against the real
trained models -- including regression coverage for the stale-figure bug
(dependence plots for a feature no longer in the top-N weren't cleaned up
between runs) and the cash_buffer multicollinearity artifact this piece
surfaced in the LR baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_limit_optimizer.models.explain import (
    FIG_DIR, MODEL_CARD_PATH, MODEL_SPECS, N_DEPENDENCE_PLOTS, REPORT_PATH,
    explain_customer, global_importance, pick_example_customers, run_explainability,
)


def test_pick_example_customers_picks_low_median_high():
    proba = np.array([0.9, 0.01, 0.5, 0.3, 0.05, 0.02])
    idx = pick_example_customers(pd.DataFrame(index=range(len(proba))), proba, n=3)
    order = np.argsort(proba)
    assert idx[0] == order[0]  # lowest
    assert idx[-1] == order[-1]  # highest
    assert proba[idx[1]] == pytest.approx(sorted(proba)[len(proba) // 2])  # median


def test_pick_example_customers_handles_n_gte_population():
    proba = np.array([0.1, 0.2])
    idx = pick_example_customers(pd.DataFrame(index=range(2)), proba, n=5)
    assert len(idx) == 2


def test_global_importance_is_mean_absolute_shap():
    class FakeShapValues:
        values = np.array([[1.0, -2.0], [-1.0, 2.0], [3.0, 0.0]])

    result = global_importance(FakeShapValues(), features=["a", "b"])
    assert result["a"] == pytest.approx(np.mean([1.0, 1.0, 3.0]), abs=1e-4)
    assert result["b"] == pytest.approx(np.mean([2.0, 2.0, 0.0]), abs=1e-4)
    # sorted descending
    assert list(result.keys())[0] == "a"


def test_explain_customer_sorts_and_labels_factors_correctly():
    features = ["income", "utilization", "delinquency"]
    row = pd.Series({"customer_id": "C1", "income": 3000.0, "utilization": 0.8, "delinquency": 2})
    shap_row = np.array([-0.5, 0.3, 0.9])  # income reduces risk, delinquency increases it most
    result = explain_customer(row, features, shap_row, base_value=0.1, calibrated_proba=0.037, n_factors=2)
    assert result["customer_id"] == "C1"
    assert result["predicted_default_probability"] == pytest.approx(0.037)
    assert result["risk_increasing_factors"][0]["feature"] == "delinquency"
    assert result["risk_increasing_factors"][0]["shap_contribution"] == pytest.approx(0.9)
    assert result["risk_reducing_factors"][0]["feature"] == "income"
    assert result["risk_reducing_factors"][0]["shap_contribution"] == pytest.approx(-0.5)


def test_explain_customer_handles_nan_feature_value():
    features = ["a", "b"]
    row = pd.Series({"customer_id": "C2", "a": np.nan, "b": 1.0})
    shap_row = np.array([0.2, -0.1])
    result = explain_customer(row, features, shap_row, base_value=0.0, calibrated_proba=0.05, n_factors=2)
    a_entry = next(f for f in result["risk_increasing_factors"] if f["feature"] == "a")
    assert a_entry["value"] is None


@pytest.fixture(scope="module")
def results():
    return run_explainability()


def test_both_models_explained(results):
    assert set(results) == {spec["name"] for spec in MODEL_SPECS}


def test_delinquency_count_is_a_top_driver_for_both_models(results):
    """Matches every prior piece's finding (EDA, both model cards) --
    delinquency history dominates the risk signal in this dataset."""
    for name, r in results.items():
        top5 = list(r["global_importance"])[:5]
        assert "delinquency_count" in top5, f"{name}: delinquency_count not in top-5 SHAP importance ({top5})"


def test_cash_buffer_not_among_lr_top_features(results):
    """Regression test: cash_buffer (r=0.9465 with credit_exposure) used
    to rank #2 in the LR SHAP summary with a backwards sign (higher
    buffer read as riskier) before the correlation-pruning threshold was
    tightened in risk_model.py. It must no longer be a candidate feature
    at all for the LR model."""
    top10 = list(results["logistic_regression"]["global_importance"])[:10]
    assert "cash_buffer" not in top10


def test_no_stale_dependence_plots_left_over(results):
    """Regression test: explain.py used to leave old *_dependence_*.png
    files behind when the top-N feature set changed between runs. Every
    PNG in FIG_DIR must correspond to a summary plot or a currently-listed
    dependence feature."""
    expected = set()
    for spec in MODEL_SPECS:
        name = spec["name"]
        expected.add(f"{name}_summary.png")
        for feat in results[name]["top_dependence_features"]:
            expected.add(f"{name}_dependence_{feat}.png")
    actual = {p.name for p in FIG_DIR.glob("*.png")}
    assert actual == expected


def test_dependence_plot_count_matches_config(results):
    for r in results.values():
        assert len(r["top_dependence_features"]) == N_DEPENDENCE_PLOTS


def test_example_customer_probabilities_are_plausible(results):
    for r in results.values():
        probs = [ex["predicted_default_probability"] for ex in r["examples"]]
        assert probs == sorted(probs)  # low, median, high by construction
        assert all(0 <= p <= 1 for p in probs)


def test_output_files_written(results):
    assert REPORT_PATH.exists()
    assert MODEL_CARD_PATH.exists()
