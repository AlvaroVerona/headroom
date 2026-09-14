"""Tests for feature engineering: a handcrafted-fixture unit test for the
rolling-window/growth logic (leakage-safety and the bounded-growth-formula
regression), plus an integration test against the real VALIDATED data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_limit_optimizer.features.engineering import (
    ALL_FEATURE_COLUMNS, GROWTH_WINDOWS, TREND_VARIABLES,
    add_rolling_features, build_feature_table, load_inputs,
)
from credit_limit_optimizer.utils.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


def _toy_monthly() -> pd.DataFrame:
    # Two customers, 4 months each, deliberately out of chronological order
    # in the input to exercise the sort-before-cumcount step. Customer C2's
    # credit_utilization starts at exactly 0.0 -- the case that broke the
    # naive (current-past)/past growth formula.
    rows = [
        {"customer_id": "C1", "month": "2023-02", "monthly_income": 2000.0, "monthly_spend": 500.0,
         "credit_utilization": 0.2, "end_balance": 200.0, "delinquency_count": 0, "days_past_due": 0},
        {"customer_id": "C1", "month": "2023-01", "monthly_income": 1000.0, "monthly_spend": 400.0,
         "credit_utilization": 0.1, "end_balance": 100.0, "delinquency_count": 0, "days_past_due": 0},
        {"customer_id": "C1", "month": "2023-04", "monthly_income": 4000.0, "monthly_spend": 700.0,
         "credit_utilization": 0.4, "end_balance": 400.0, "delinquency_count": 1, "days_past_due": 30},
        {"customer_id": "C1", "month": "2023-03", "monthly_income": 3000.0, "monthly_spend": 600.0,
         "credit_utilization": 0.3, "end_balance": 300.0, "delinquency_count": 0, "days_past_due": 0},
        {"customer_id": "C2", "month": "2023-01", "monthly_income": 500.0, "monthly_spend": 100.0,
         "credit_utilization": 0.0, "end_balance": 0.0, "delinquency_count": 0, "days_past_due": 0},
        {"customer_id": "C2", "month": "2023-02", "monthly_income": 500.0, "monthly_spend": 100.0,
         "credit_utilization": 0.1, "end_balance": 10.0, "delinquency_count": 0, "days_past_due": 0},
        {"customer_id": "C2", "month": "2023-03", "monthly_income": 500.0, "monthly_spend": 100.0,
         "credit_utilization": 0.2, "end_balance": 20.0, "delinquency_count": 0, "days_past_due": 0},
        {"customer_id": "C2", "month": "2023-04", "monthly_income": 500.0, "monthly_spend": 100.0,
         "credit_utilization": 0.6, "end_balance": 60.0, "delinquency_count": 0, "days_past_due": 0},
    ]
    return pd.DataFrame(rows)


def test_month_index_assigned_in_chronological_order():
    result = add_rolling_features(_toy_monthly())
    c1 = result[result.customer_id == "C1"].sort_values("month_index")
    assert list(c1["month"]) == ["2023-01", "2023-02", "2023-03", "2023-04"]
    assert list(c1["month_index"]) == [1, 2, 3, 4]


def test_rolling_avg_uses_only_trailing_months_not_future():
    result = add_rolling_features(_toy_monthly())
    c1 = result[result.customer_id == "C1"].set_index("month_index")
    # At month 2, the 3m avg can only see months 1-2 (income 1000, 2000).
    assert c1.loc[2, "monthly_income_3m_avg"] == pytest.approx(1500.0)
    # At month 4, it sees months 2-4 (income 2000, 3000, 4000) -- NOT month 1.
    assert c1.loc[4, "monthly_income_3m_avg"] == pytest.approx(3000.0)


def test_growth_feature_bounded_even_at_near_zero_base():
    """Regression test: the original (current-past)/past growth formula
    exploded to ~1e6-1e10 whenever `past` was near zero (which
    credit_utilization and end_balance legitimately are, e.g. a customer
    who pays off in full). The symmetric formula is bounded to [-2, 2]."""
    result = add_rolling_features(_toy_monthly())
    growth_cols = [f"{var}_{w}m_growth" for var in TREND_VARIABLES for w in GROWTH_WINDOWS]
    values = result[growth_cols].to_numpy(dtype=float)
    assert not np.isinf(values).any()
    finite = values[~np.isnan(values)]
    assert (finite >= -2.0).all() and (finite <= 2.0).all()
    # C2 utilization goes 0.0 (month 1) -> 0.6 (month 4) -- a near-zero base
    # 3 months back, the case that broke the naive (current-past)/past
    # formula (division by ~0). Symmetric growth caps it at +2.
    c2 = result[result.customer_id == "C2"].set_index("month_index")
    assert c2.loc[4, "credit_utilization_3m_growth"] == pytest.approx(2.0)


def test_max_days_past_due_is_expanding_max_not_reset():
    result = add_rolling_features(_toy_monthly())
    c1 = result[result.customer_id == "C1"].set_index("month_index")
    assert list(c1["max_days_past_due"]) == [0, 0, 0, 30]


@pytest.fixture(scope="module")
def real_inputs():
    return load_inputs()


@pytest.fixture(scope="module")
def real_feature_table(real_inputs, config):
    return build_feature_table(real_inputs, config)


def test_feature_table_has_all_expected_columns(real_feature_table):
    missing = set(ALL_FEATURE_COLUMNS) - set(real_feature_table.columns)
    assert not missing, f"missing feature columns: {missing}"


def test_feature_table_row_count_matches_cohorts(real_feature_table):
    counts = real_feature_table["cohort"].value_counts()
    assert set(counts.index) == {"train", "validation", "test"}
    # Every cohort starts from the same 50,000 customers; only customers
    # whose account/profile record was quarantined in Phase 1 get dropped,
    # so the three cohort sizes should match each other closely.
    assert counts.max() - counts.min() < 500


def test_growth_features_bounded_on_real_data(real_feature_table):
    growth_cols = [f"{var}_{w}m_growth" for var in TREND_VARIABLES for w in GROWTH_WINDOWS]
    values = real_feature_table[growth_cols].to_numpy(dtype=float)
    assert not np.isinf(values).any()
    assert np.nanmax(np.abs(values)) <= 2.0 + 1e-9


def test_no_future_leakage_against_raw_monthly_table(real_inputs, real_feature_table):
    """For a random sample of real rows, recompute the 3-month trailing
    average directly from the raw monthly table using ONLY months up to
    and including the snapshot month, and check it matches exactly."""
    raw = real_inputs["monthly"].copy()
    raw = raw.sort_values(["customer_id", "month"]).reset_index(drop=True)
    raw["month_index"] = raw.groupby("customer_id").cumcount() + 1

    sample = real_feature_table.sample(50, random_state=42)
    for _, row in sample.iterrows():
        snap_month = row["snapshot_month"]
        window = raw[
            (raw["customer_id"] == row["customer_id"])
            & (raw["month_index"] <= snap_month)
            & (raw["month_index"] > snap_month - 3)
        ]
        assert window["month_index"].max() == snap_month, "window must not stop before the snapshot month"
        expected = window["monthly_income"].mean()
        assert row["monthly_income_3m_avg"] == pytest.approx(expected, rel=1e-6)


def test_debt_to_income_and_ratios_are_finite_and_nonnegative(real_feature_table):
    for col in ["debt_to_income", "essential_spending_ratio", "discretionary_spending_ratio", "cash_withdrawal_ratio"]:
        values = real_feature_table[col]
        assert np.isfinite(values).all()
        assert (values >= 0).all()
