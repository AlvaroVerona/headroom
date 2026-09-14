"""Phase 1 tests: correct customer/transaction counts, reproducibility,
valid IDs (pre-injection), and calibration sanity checks (default rate in
a believable range, utilization not degenerately pinned at the credit
ceiling -- regression coverage for the two real calibration bugs found
while building this, see CLAUDE.md) -- plus injected-issue presence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_limit_optimizer.data import (
    behavior_series, credit_accounts as credit_accounts_mod, generator, labels as labels_mod,
    master_data, monthly_table, quality_injection,
)
from credit_limit_optimizer.utils.config import load_config

SEED = 42
N_CUSTOMERS_SMALL = 3000
N_MONTHS = 36
START_DATE = pd.Timestamp("2023-01-01")


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def small_pipeline():
    rng = np.random.default_rng(SEED)
    customers = master_data.generate_customers(rng, N_CUSTOMERS_SMALL, START_DATE)
    limits = behavior_series.assign_initial_credit_limits(rng, customers)
    beh = behavior_series.simulate_behavior(rng, customers, limits, N_MONTHS, START_DATE)
    accounts = credit_accounts_mod.build_credit_accounts(rng, customers, limits, beh)
    monthly = monthly_table.build_monthly_table(customers, beh, START_DATE)
    return {"customers": customers, "limits": limits, "beh": beh, "accounts": accounts, "monthly": monthly}


def test_customer_count(small_pipeline):
    assert len(small_pipeline["customers"]) == N_CUSTOMERS_SMALL
    assert small_pipeline["customers"]["customer_id"].is_unique


def test_monthly_table_row_count(small_pipeline):
    assert len(small_pipeline["monthly"]) == N_CUSTOMERS_SMALL * N_MONTHS


def test_credit_accounts_valid_ids(small_pipeline):
    accounts = small_pipeline["accounts"]
    customers = small_pipeline["customers"]
    assert accounts["account_id"].is_unique
    assert accounts["customer_id"].isin(customers["customer_id"]).all()


def test_reproducibility_customers():
    rng_a = np.random.default_rng(SEED)
    rng_b = np.random.default_rng(SEED)
    a = master_data.generate_customers(rng_a, 1000, START_DATE)
    b = master_data.generate_customers(rng_b, 1000, START_DATE)
    pd.testing.assert_frame_equal(a, b)


def test_reproducibility_full_generate_all(config):
    rng_a = np.random.default_rng(SEED)
    rng_b = np.random.default_rng(SEED)
    small_config = {**config, "data": {**config["data"], "n_customers": 500, "n_months": 36}}
    a = generator.generate_all(rng_a, small_config)
    b = generator.generate_all(rng_b, small_config)
    pd.testing.assert_frame_equal(a["customers"], b["customers"])
    pd.testing.assert_frame_equal(a["monthly_customer_behavior"], b["monthly_customer_behavior"])


def test_default_rate_within_believable_range(small_pipeline, config):
    accounts = small_pipeline["accounts"]
    label_table = labels_mod.build_all_labels(small_pipeline["customers"], small_pipeline["beh"], accounts, config)
    rates = label_table.dropna(subset=["default_12m"]).groupby("cohort")["default_12m"].mean()
    # Not trivially rare (needs enough positive examples to model) and not
    # implausibly common for a real unsecured-credit portfolio.
    assert (rates > 0.01).all()
    assert (rates < 0.15).all()


def test_default_rate_ordered_by_segment(small_pipeline, config):
    """Regression check for the spec's explicit "not trivially predictable
    but economically meaningful" requirement (§8): subprime should default
    more than prime, but the gap shouldn't be so extreme the segments are
    perfectly separable."""
    accounts = small_pipeline["accounts"]
    label_table = labels_mod.build_all_labels(small_pipeline["customers"], small_pipeline["beh"], accounts, config)
    train = label_table[(label_table["cohort"] == "train") & label_table["eligible_for_default_label"]]
    merged = train.merge(small_pipeline["customers"][["customer_id", "customer_segment"]], on="customer_id")
    by_segment = merged.groupby("customer_segment")["default_12m"].mean()
    assert by_segment["subprime"] > by_segment["near_prime"] > by_segment["prime"]
    assert by_segment["prime"] > 0  # not perfectly separable -- some prime customers still default


def test_utilization_not_pinned_at_ceiling(small_pipeline):
    """Regression test for the balance-equilibrium bug: an early version
    of the payment mechanic made 34% of all customer-months sit exactly at
    the credit limit (utilization >= 0.999) because minimum payments
    couldn't keep pace with ongoing spend. Real portfolios have some
    maxed-out accounts, not a third of them pinned at exactly 100%."""
    utilization_final_month = small_pipeline["beh"]["utilization"][:, -1]
    pct_at_cap = (utilization_final_month >= 0.999).mean()
    assert pct_at_cap < 0.15


def test_utilization_ordered_by_segment(small_pipeline):
    customers = small_pipeline["customers"]
    util = small_pipeline["beh"]["utilization"][:, -1]
    by_segment = pd.DataFrame({"segment": customers["customer_segment"], "u": util}).groupby("segment")["u"].mean()
    assert by_segment["subprime"] > by_segment["near_prime"] > by_segment["prime"]


def test_credit_account_identities_hold_on_clean_data(small_pipeline):
    accounts = small_pipeline["accounts"]
    implied_available = accounts["current_credit_limit"] - accounts["current_balance"]
    assert np.allclose(accounts["available_credit"], implied_available, atol=0.01)
    implied_utilization = accounts["current_balance"] / accounts["current_credit_limit"]
    assert np.allclose(accounts["utilization_rate"], implied_utilization, atol=0.001)
    assert (accounts["current_balance"] <= accounts["current_credit_limit"] + 0.01).all()


def test_quality_injection_missing_values():
    df = pd.DataFrame({"a": range(1000), "b": range(1000)})
    rng = np.random.default_rng(SEED)
    result = quality_injection.inject_missing_values(df, rng, {"a": 0.1})
    assert result["a"].isna().sum() == 100
    assert result["b"].isna().sum() == 0


def test_quality_injection_duplicates():
    df = pd.DataFrame({"a": range(1000)})
    rng = np.random.default_rng(SEED)
    result = quality_injection.inject_exact_duplicates(df, rng, 0.05)
    assert len(result) == 1050
    assert result.duplicated().sum() == 50


def test_quality_injection_impossible_values_int_dtype():
    """Regression test: assigning a float draw into an int64 column used
    to raise TypeError in pandas >= 2 instead of silently upcasting."""
    df = pd.DataFrame({"age": np.random.default_rng(0).integers(18, 80, size=1000)})
    rng = np.random.default_rng(SEED)
    result = quality_injection.inject_impossible_values(df, rng, 0.1, "age", "below_min")
    assert (result["age"] < 18).sum() == 100


def test_quality_injection_inconsistency_skips_already_null_limit():
    """Regression test: inject_inconsistent_credit_fields used to cascade
    NaN into current_balance/utilization_rate when it happened to sample a
    row whose current_credit_limit was already NaN from an earlier
    injection step."""
    df = pd.DataFrame({
        "current_credit_limit": [np.nan] + [1000.0] * 999,
        "current_balance": [500.0] * 1000,
        "available_credit": [500.0] * 1000,
        "utilization_rate": [0.5] * 1000,
    })
    rng = np.random.default_rng(SEED)
    result = quality_injection.inject_inconsistent_credit_fields(df, rng, 0.5)
    assert result.loc[0, "current_balance"] == 500.0  # untouched, not cascaded to NaN
    assert not result["current_balance"].isna().any()


def test_quality_injection_invalid_references():
    df = pd.DataFrame({"customer_id": [f"C{i:07d}" for i in range(1000)]})
    valid_ids = set(df["customer_id"])
    rng = np.random.default_rng(SEED)
    result = quality_injection.inject_invalid_references(df, rng, 0.1, "customer_id")
    assert (~result["customer_id"].isin(valid_ids)).sum() == 100
