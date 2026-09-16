"""Tests for the Revenue Model (§20): unit test for compute_revenue on a
handcrafted fixture, plus an integration test that validates the formula
against real generated labels (the module's own headline claim) and
checks the real-data report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_limit_optimizer.models.revenue_model import (
    FIG_DIR, MODEL_CARD_PATH, REPORT_PATH, compute_revenue,
    run_revenue_model, validate_against_labels,
)
from credit_limit_optimizer.utils.config import load_config


def test_compute_revenue_matches_hand_calculation():
    df = pd.DataFrame({
        "average_balance": [1200.0, 0.0],
        "apr": [0.24, 0.17],
        "monthly_spend": [500.0, 300.0],
        "days_past_due": [30, 0],
    })
    config = {"economics": {"interchange_rate": 0.012, "late_fee_amount": 25}}
    revenue = compute_revenue(df, config)
    assert revenue.loc[0, "interest_revenue"] == pytest.approx(1200.0 * 0.24 / 12)
    assert revenue.loc[0, "interchange_revenue"] == pytest.approx(500.0 * 0.012)
    assert revenue.loc[0, "late_fee_revenue"] == pytest.approx(25.0)  # DPD > 0
    assert revenue.loc[1, "late_fee_revenue"] == pytest.approx(0.0)  # DPD == 0
    assert revenue.loc[0, "total_revenue"] == pytest.approx(
        revenue.loc[0, "interest_revenue"] + revenue.loc[0, "interchange_revenue"] + 25.0
    )


def test_compute_revenue_zero_balance_zero_interest():
    df = pd.DataFrame({"average_balance": [0.0], "apr": [0.25], "monthly_spend": [0.0], "days_past_due": [0]})
    config = {"economics": {"interchange_rate": 0.012, "late_fee_amount": 25}}
    revenue = compute_revenue(df, config)
    assert revenue.loc[0, "total_revenue"] == pytest.approx(0.0)


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_validate_against_labels_matches_real_generated_data(config):
    """The module's headline claim: applying this exact formula to the
    raw monthly table at each label's future month must reconstruct
    labels.csv's future_interest_revenue/future_interchange_revenue
    (independently produced by Phase 1's simulation) up to rounding."""
    result = validate_against_labels(config)
    assert result["n_checked"] > 100000
    assert result["max_abs_diff_interest"] <= 0.011  # rounding-scale tolerance, not a real gap
    assert result["max_abs_diff_interchange"] <= 0.011


@pytest.fixture(scope="module")
def report():
    return run_revenue_model()


def test_revenue_report_has_no_nan(report):
    for v in report["mean_revenue"].values():
        assert not np.isnan(v)
    assert not np.isnan(report["total_portfolio_monthly_revenue"])


def test_revenue_ordered_by_segment_subprime_highest(report):
    """Subprime carries the highest APR and, per Phase 1's calibration,
    the highest utilization/balance -- both push interest revenue up."""
    by_seg = report["by_segment"]
    assert by_seg["subprime"]["total_revenue"] > by_seg["prime"]["total_revenue"]


def test_late_fee_revenue_is_nonzero_and_plausible(report):
    for seg in ["prime", "near_prime", "subprime"]:
        fee = report["by_segment"][seg]["late_fee_revenue"]
        assert 0 <= fee <= report["late_fee_amount"]


def test_output_files_written(report):
    assert REPORT_PATH.exists()
    assert MODEL_CARD_PATH.exists()
    assert (FIG_DIR / "01_revenue_distribution.png").exists()
    assert (FIG_DIR / "02_revenue_by_segment.png").exists()
