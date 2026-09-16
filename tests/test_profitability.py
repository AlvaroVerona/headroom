"""Tests for Customer Profitability (§22): unit tests for the balance-
response function and evaluate_at_limit's annual-unit formula on
handcrafted fixtures (regression coverage for the period-mismatch bug
found while building this), plus an integration test against the real
trained models and generated data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_limit_optimizer.models.profitability import (
    FIG_DIR, MODEL_CARD_PATH, REPORT_PATH,
    balance_at_limit, evaluate_at_limit, run_profitability,
)


def test_balance_at_limit_anchor_point_exact():
    """balance(current_limit) must reproduce current_balance exactly --
    the response function's self-consistency guarantee."""
    balance = balance_at_limit(
        current_balance=np.array([1200.0]), current_limit=np.array([4000.0]),
        candidate_limit=4000.0, elasticity=0.3,
    )
    assert balance[0] == pytest.approx(1200.0)


def test_balance_at_limit_increases_with_candidate_limit():
    current_balance, current_limit = np.array([1000.0]), np.array([2000.0])
    low = balance_at_limit(current_balance, current_limit, 1000.0, 0.3)
    high = balance_at_limit(current_balance, current_limit, 8000.0, 0.3)
    assert high[0] > low[0]


def test_balance_at_limit_diminishing_returns():
    """elasticity < 1 must give diminishing returns: for EQUAL-SIZED
    steps in candidate limit, the balance gain per step must shrink."""
    current_balance, current_limit = np.array([500.0]), np.array([1000.0])
    b2000 = balance_at_limit(current_balance, current_limit, 2000.0, 0.3)[0]
    b4000 = balance_at_limit(current_balance, current_limit, 4000.0, 0.3)[0]
    b6000 = balance_at_limit(current_balance, current_limit, 6000.0, 0.3)[0]
    assert (b4000 - b2000) > (b6000 - b4000)  # marginal gain per EUR2000 step shrinks


def test_balance_at_limit_never_exceeds_candidate_limit():
    """A customer can never carry more balance than the limit itself."""
    current_balance, current_limit = np.array([100.0]), np.array([200.0])
    balance = balance_at_limit(current_balance, current_limit, 50000.0, 0.9)
    assert balance[0] <= 50000.0


def _toy_row() -> pd.DataFrame:
    return pd.DataFrame({
        "average_balance": [1200.0], "credit_exposure": [4000.0], "apr": [0.20],
        "monthly_spend": [500.0], "days_past_due": [30], "customer_segment": ["prime"],
    })


def _toy_config() -> dict:
    return {
        "economics": {
            "interchange_rate": 0.012, "late_fee_amount": 25,
            "funding_rate": 0.05, "lgd_by_segment": {"prime": 0.55},
            "operational_cost_per_customer": 25,
        },
        "optimization": {"balance_elasticity": 0.3},
    }


def test_evaluate_at_limit_all_components_are_annual():
    """Regression test: the first version computed expected_loss (already
    annual, since PD is a 12-month probability) but left revenue/funding
    cost as monthly figures, producing negative "profit" for every
    segment -- a units/period mismatch, not a real economic finding.
    Every returned euro figure must be on a consistent annual basis."""
    row = _toy_row()
    config = _toy_config()
    pd_values = np.array([0.05])
    result = evaluate_at_limit(row, row["credit_exposure"].to_numpy(), pd_values, config)

    # At the anchor point, balance == observed average_balance exactly.
    assert result["balance"].iloc[0] == pytest.approx(1200.0)
    # Interest revenue: balance x ANNUAL apr (no /12).
    assert result["interest_revenue"].iloc[0] == pytest.approx(1200.0 * 0.20)
    # Interchange: monthly_spend x rate x 12.
    assert result["interchange_revenue"].iloc[0] == pytest.approx(500.0 * 0.012 * 12)
    # Late fee: amount x 12 (DPD > 0 this month, annualized run-rate).
    assert result["late_fee_revenue"].iloc[0] == pytest.approx(25 * 12)
    # Expected loss: PD (12-month) x LGD x balance -- already annual.
    assert result["expected_loss"].iloc[0] == pytest.approx(0.05 * 0.55 * 1200.0)
    # Funding cost: balance x ANNUAL funding_rate (no /12).
    assert result["funding_cost"].iloc[0] == pytest.approx(1200.0 * 0.05)
    # Operational cost: already annual in config, used as-is.
    assert result["operational_cost"].iloc[0] == pytest.approx(25.0)

    expected_profit = (
        result["total_revenue"].iloc[0] - result["expected_loss"].iloc[0]
        - result["funding_cost"].iloc[0] - result["operational_cost"].iloc[0]
    )
    assert result["expected_profit"].iloc[0] == pytest.approx(expected_profit)


def test_evaluate_at_limit_profit_is_not_trivially_negative():
    """With realistic inputs (APR >> funding rate, low PD), annual profit
    must come out positive -- guards against reintroducing the period-
    mismatch bug that made every segment's mean profit negative."""
    row = _toy_row()
    config = _toy_config()
    result = evaluate_at_limit(row, row["credit_exposure"].to_numpy(), np.array([0.03]), config)
    assert result["expected_profit"].iloc[0] > 0


@pytest.fixture(scope="module")
def report():
    return run_profitability()


def test_anchor_point_sanity_check_passes_on_real_data(report):
    assert report["anchor_point_max_abs_diff"] < 0.01


def test_mean_profit_positive_for_every_segment(report):
    """Regression test for the period-mismatch bug: an early run produced
    negative mean profit for prime/near_prime/subprime alike."""
    for seg, profit in report["profit_by_segment"].items():
        assert profit > 0, f"{seg}: mean profit {profit} is not positive"


def test_portfolio_summary_has_no_nan(report):
    cs = report["current_state_summary"]
    for key in ["mean_expected_profit", "total_portfolio_expected_profit", "mean_total_revenue", "mean_expected_loss", "mean_funding_cost"]:
        assert not np.isnan(cs[key]), f"{key} is NaN"


def test_example_customers_cover_all_three_segments(report):
    segments = {ex["segment"] for ex in report["examples"]}
    assert segments == {"prime", "near_prime", "subprime"}


def test_example_tables_are_monotonically_increasing_profit(report):
    """Diminishing returns, not decreasing profit: each example customer's
    profit should rise (or at least not fall) as candidate limit rises."""
    for ex in report["examples"]:
        profits = [row["expected_profit"] for row in ex["table"]]
        assert all(b >= a - 0.01 for a, b in zip(profits, profits[1:])), f"{ex['customer_id']}: profit not monotonic"


def test_example_tables_respect_utilization_bounds(report):
    for ex in report["examples"]:
        for row in ex["table"]:
            assert 0 <= row["utilization"] <= 1


def test_no_stale_profit_curve_files_left_over(report):
    """Regression test: an earlier version left a previous run's example
    customers' profit_curve_*.png files behind when a different customer
    was selected on a subsequent run (same bug class already fixed once
    in explain.py)."""
    expected = {"00_profit_by_segment.png"} | {f"profit_curve_{ex['customer_id']}.png" for ex in report["examples"]}
    actual = {p.name for p in FIG_DIR.glob("*.png")}
    assert actual == expected


def test_output_files_written(report):
    assert REPORT_PATH.exists()
    assert MODEL_CARD_PATH.exists()
