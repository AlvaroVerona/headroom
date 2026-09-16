"""Tests for the Funding Cost model (§21): unit tests for the scenario
rate lookup and cost formula on handcrafted fixtures, plus an integration
test against the real trained/generated data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_limit_optimizer.models.funding_cost import (
    FIG_DIR, MODEL_CARD_PATH, REPORT_PATH, SCENARIOS,
    compute_funding_cost, funding_rate_for_scenario, run_funding_cost,
)
from credit_limit_optimizer.utils.config import load_config


def test_funding_rate_for_scenario_applies_shift():
    config = {
        "economics": {"funding_rate": 0.05},
        "scenarios": {"base": {}, "recession": {"funding_rate_shift": 0.01}},
    }
    assert funding_rate_for_scenario(config, "base") == pytest.approx(0.05)
    assert funding_rate_for_scenario(config, "recession") == pytest.approx(0.06)


def test_funding_rate_for_scenario_without_shift_key_uses_base():
    config = {
        "economics": {"funding_rate": 0.05},
        "scenarios": {"consumer_stress": {"delinquency_multiplier": 1.4}},
    }
    assert funding_rate_for_scenario(config, "consumer_stress") == pytest.approx(0.05)


def test_compute_funding_cost_matches_hand_calculation():
    df = pd.DataFrame({"average_balance": [1200.0, 0.0]})
    cost = compute_funding_cost(df, rate=0.05)
    assert cost[0] == pytest.approx(1200.0 * 0.05 / 12)
    assert cost[1] == pytest.approx(0.0)


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_real_config_has_all_four_scenarios_defined(config):
    for scenario in SCENARIOS:
        assert scenario in config["scenarios"]


def test_stress_scenarios_never_cheaper_than_base(config):
    """Recession and high_interest_rate must raise, not lower, the
    funding rate -- a regression test against a sign-flip in the config
    shift or the lookup logic."""
    base_rate = funding_rate_for_scenario(config, "base")
    for scenario in ["recession", "high_interest_rate"]:
        assert funding_rate_for_scenario(config, scenario) >= base_rate


@pytest.fixture(scope="module")
def report():
    return run_funding_cost()


def test_funding_cost_report_has_no_nan(report):
    for v in report["mean_funding_cost_by_scenario"].values():
        assert not np.isnan(v)
    for v in report["total_portfolio_funding_cost_by_scenario"].values():
        assert not np.isnan(v)
    assert not np.isnan(report["base_scenario"]["mean_net_interest_margin"])


def test_funding_cost_ordered_by_scenario_severity(report):
    costs = report["mean_funding_cost_by_scenario"]
    assert costs["high_interest_rate"] > costs["recession"] > costs["base"]
    assert costs["consumer_stress"] == pytest.approx(costs["base"])


def test_net_interest_margin_is_mostly_positive_and_never_negative(report):
    """Verified finding: every segment's APR exceeds every scenario's
    funding rate, so NIM is either positive (balance > 0) or exactly zero
    (balance == 0) -- never negative."""
    base = report["base_scenario"]
    assert base["fraction_positive_nim"] > 0.9
    assert base["mean_net_interest_margin"] > 0
    assert base["n_negative_nim"] == 0


def test_nim_present_for_all_segments(report):
    assert set(report["nim_by_segment"]) == {"prime", "near_prime", "subprime"}
    for v in report["nim_by_segment"].values():
        assert v > 0


def test_output_files_written(report):
    assert REPORT_PATH.exists()
    assert MODEL_CARD_PATH.exists()
    assert (FIG_DIR / "01_funding_cost_by_scenario.png").exists()
    assert (FIG_DIR / "02_nim_by_segment.png").exists()
