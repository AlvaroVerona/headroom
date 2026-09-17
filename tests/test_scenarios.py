"""Tests for the Stress Testing / Scenario engine (§26-27): unit tests for
the shock-application logic on handcrafted fixtures, plus an integration
test against the real funded portfolio and trained models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_limit_optimizer.simulation.scenarios import (
    FIG_DIR, MODEL_CARD_PATH, REPORT_PATH, SCENARIOS,
    apply_shocks, portfolio_summary, run_stress_test,
)
from credit_limit_optimizer.utils.config import load_config


def _toy_book() -> pd.DataFrame:
    return pd.DataFrame({
        "monthly_income": [3000.0], "monthly_income_3m_avg": [3000.0],
        "monthly_income_6m_avg": [3000.0], "monthly_income_12m_avg": [3000.0],
        "monthly_income_3m_growth": [0.02], "monthly_income_6m_growth": [0.01],
        "monthly_spend": [1500.0], "monthly_spend_3m_avg": [1500.0],
        "monthly_spend_6m_avg": [1500.0], "monthly_spend_12m_avg": [1500.0],
        "monthly_spend_3m_growth": [0.0], "monthly_spend_6m_growth": [0.0],
        "income_volatility": [0.2], "income_stability": [0.8],
        "delinquency_count": [2.0], "days_past_due": [10.0], "max_days_past_due": [30.0],
        "cash_buffer": [1000.0], "apr": [0.20],
    })


def test_apply_shocks_income_growth_shift_scales_level_and_shifts_growth():
    shocked = apply_shocks(_toy_book(), {"income_growth_shift": -0.03})
    assert shocked["monthly_income"].iloc[0] == pytest.approx(3000.0 * 0.97)
    assert shocked["monthly_income_3m_avg"].iloc[0] == pytest.approx(3000.0 * 0.97)
    assert shocked["monthly_income_3m_growth"].iloc[0] == pytest.approx(0.02 - 0.03)


def test_apply_shocks_spending_shift_scales_spend_only():
    shocked = apply_shocks(_toy_book(), {"spending_shift": -0.10})
    assert shocked["monthly_spend"].iloc[0] == pytest.approx(1500.0 * 0.9)
    assert shocked["monthly_income"].iloc[0] == pytest.approx(3000.0)  # untouched


def test_apply_shocks_income_volatility_keeps_stability_consistent():
    """income_stability = 1 - income_volatility (r=1.0 per risk_model.py) --
    the shocked pair must stay internally consistent."""
    shocked = apply_shocks(_toy_book(), {"income_volatility_multiplier": 1.5})
    assert shocked["income_volatility"].iloc[0] == pytest.approx(0.3)
    assert shocked["income_stability"].iloc[0] == pytest.approx(0.7)


def test_apply_shocks_income_volatility_clipped_to_valid_range():
    book = _toy_book()
    book["income_volatility"] = [0.9]
    shocked = apply_shocks(book, {"income_volatility_multiplier": 2.0})
    assert shocked["income_volatility"].iloc[0] <= 1.0


def test_apply_shocks_delinquency_multiplier_scales_all_three_columns():
    shocked = apply_shocks(_toy_book(), {"delinquency_multiplier": 1.4})
    assert shocked["delinquency_count"].iloc[0] == pytest.approx(2.8)
    assert shocked["days_past_due"].iloc[0] == pytest.approx(14.0)
    assert shocked["max_days_past_due"].iloc[0] == pytest.approx(42.0)


def test_apply_shocks_cash_balance_shift():
    shocked = apply_shocks(_toy_book(), {"cash_balance_shift": -0.20})
    assert shocked["cash_buffer"].iloc[0] == pytest.approx(800.0)


def test_apply_shocks_apr_shift_is_additive():
    shocked = apply_shocks(_toy_book(), {"apr_shift": 0.03})
    assert shocked["apr"].iloc[0] == pytest.approx(0.23)


def test_apply_shocks_base_scenario_is_a_no_op():
    book = _toy_book()
    shocked = apply_shocks(book, {})
    pd.testing.assert_frame_equal(book, shocked)


def test_apply_shocks_does_not_mutate_input():
    book = _toy_book()
    original = book.copy()
    apply_shocks(book, {"spending_shift": -0.10, "income_growth_shift": -0.03})
    pd.testing.assert_frame_equal(book, original)


def test_portfolio_summary_matches_hand_calculation():
    economics = pd.DataFrame({
        "candidate_limit": [1000.0, 2000.0], "total_revenue": [100.0, 200.0],
        "funding_cost": [10.0, 20.0], "expected_loss": [5.0, 40.0],
        "expected_profit": [85.0, 140.0], "pd": [0.02, 0.08],
    })
    summary = portfolio_summary(economics)
    assert summary["n_customers"] == 2
    assert summary["mean_pd"] == pytest.approx(0.05)
    assert summary["total_exposure"] == pytest.approx(3000.0)
    assert summary["total_expected_loss"] == pytest.approx(45.0)
    assert summary["total_expected_profit"] == pytest.approx(225.0)


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_real_config_has_all_four_scenarios_defined(config):
    for scenario in SCENARIOS:
        assert scenario in config["scenarios"]


@pytest.fixture(scope="module")
def report():
    return run_stress_test()


def test_report_covers_all_scenarios(report):
    assert set(report["summary_by_scenario"]) == set(SCENARIOS)


def test_base_scenario_expected_loss_multiplier_is_one(report):
    assert report["expected_loss_multiplier_vs_base"]["base"] == pytest.approx(1.0)


def test_recession_is_the_worst_scenario_for_expected_loss(report):
    """Regression test: recession stacks a default_multiplier on top of
    feature-level income/spend shocks, so it must produce the highest
    Expected Loss of the four scenarios."""
    el = {s: report["summary_by_scenario"][s]["total_expected_loss"] for s in SCENARIOS}
    assert el["recession"] == max(el.values())


def test_stress_scenarios_never_reduce_expected_loss_vs_base(report):
    base_el = report["summary_by_scenario"]["base"]["total_expected_loss"]
    for scenario in ["recession", "consumer_stress"]:
        assert report["summary_by_scenario"][scenario]["total_expected_loss"] >= base_el


def test_high_interest_rate_pd_unchanged_from_base(report):
    """high_interest_rate has no feature-level shock -- PD must be
    identical to base, only the funding-rate/APR economics differ."""
    assert report["summary_by_scenario"]["high_interest_rate"]["mean_pd"] == pytest.approx(
        report["summary_by_scenario"]["base"]["mean_pd"]
    )


def test_report_has_no_nan(report):
    for summary in report["summary_by_scenario"].values():
        for v in summary.values():
            if isinstance(v, float):
                assert not np.isnan(v)


def test_output_files_written(report):
    assert REPORT_PATH.exists()
    assert MODEL_CARD_PATH.exists()
    assert (FIG_DIR / "01_expected_loss_by_scenario.png").exists()
    assert (FIG_DIR / "02_expected_profit_by_scenario.png").exists()
    assert (FIG_DIR / "03_pd_distribution_by_scenario.png").exists()
