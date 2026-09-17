"""Tests for portfolio credit limit optimization (§25): a small
handcrafted MIP verifying the linearized ratio constraints (average PD,
high-risk exposure %) actually behave like the ratios they stand in for,
plus an integration test against the real candidate pool and OR-Tools
solver."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_limit_optimizer.optimization.portfolio_optimizer import (
    FIG_DIR, MODEL_CARD_PATH, REPORT_PATH,
    build_candidate_pool, greedy_baseline, portfolio_metrics,
    profit_efficiency_by_segment, run_portfolio_optimization, solve_portfolio,
)


def _toy_pool() -> pd.DataFrame:
    # 4 customers: two cheap+profitable, two expensive+less efficient.
    return pd.DataFrame({
        "customer_id": ["A", "B", "C", "D"],
        "customer_segment": ["prime", "subprime", "prime", "subprime"],
        "limit": [1000.0, 1000.0, 5000.0, 5000.0],
        "expected_profit": [100.0, 120.0, 200.0, 600.0],
        "expected_loss": [10.0, 40.0, 20.0, 300.0],
        "pd_value": [0.02, 0.06, 0.02, 0.09],
        "is_high_risk": [0.0, 1.0, 0.0, 1.0],
    })


def _toy_config(**overrides) -> dict:
    base = {
        "optimization": {
            "portfolio_exposure_limit": 6000.0,
            "portfolio_expected_loss_limit": 1000.0,
            "max_average_pd": 0.10,
            "max_high_risk_exposure_pct": 1.0,
        }
    }
    base["optimization"].update(overrides)
    return base


def test_solve_portfolio_respects_exposure_constraint():
    pool = _toy_pool()
    config = _toy_config(portfolio_exposure_limit=6000.0)
    result = solve_portfolio(pool, config)
    metrics = portfolio_metrics(pool, result["selected"])
    assert metrics["total_exposure"] <= 6000.0 + 1e-6


def test_solve_portfolio_maximizes_profit_within_budget():
    """With a tight exposure budget that can only fit one 5000-limit
    customer, the solver must pick D (profit 600) over C (profit 200)."""
    pool = _toy_pool()
    config = _toy_config(portfolio_exposure_limit=5000.0, max_high_risk_exposure_pct=1.0, max_average_pd=1.0, portfolio_expected_loss_limit=10000.0)
    result = solve_portfolio(pool, config)
    selected_ids = pool.loc[result["selected"], "customer_id"].tolist()
    assert "D" in selected_ids
    assert "C" not in selected_ids


def test_average_pd_linearization_matches_true_ratio():
    """Regression coverage for the linearization itself: solving with a
    tight max_average_pd must produce a selected set whose TRUE (non-
    linearized) average PD is actually <= the limit -- not just satisfy
    the linear proxy by coincidence."""
    pool = _toy_pool()
    config = _toy_config(
        portfolio_exposure_limit=100000.0, portfolio_expected_loss_limit=100000.0,
        max_high_risk_exposure_pct=1.0, max_average_pd=0.05,
    )
    result = solve_portfolio(pool, config)
    selected = pool[result["selected"]]
    if len(selected) > 0:
        assert selected["pd_value"].mean() <= 0.05 + 1e-9


def test_high_risk_exposure_linearization_matches_true_ratio():
    pool = _toy_pool()
    config = _toy_config(
        portfolio_exposure_limit=100000.0, portfolio_expected_loss_limit=100000.0,
        max_average_pd=1.0, max_high_risk_exposure_pct=0.20,
    )
    result = solve_portfolio(pool, config)
    selected = pool[result["selected"]]
    if selected["limit"].sum() > 0:
        true_pct = (selected["limit"] * selected["is_high_risk"]).sum() / selected["limit"].sum()
        assert true_pct <= 0.20 + 1e-9


def test_greedy_baseline_respects_all_constraints():
    pool = _toy_pool()
    config = _toy_config(portfolio_exposure_limit=6000.0, max_high_risk_exposure_pct=0.5, max_average_pd=0.05)
    selected = greedy_baseline(pool, config)
    metrics = portfolio_metrics(pool, selected)
    opt = config["optimization"]
    assert metrics["total_exposure"] <= opt["portfolio_exposure_limit"] + 1e-6
    assert metrics["total_expected_loss"] <= opt["portfolio_expected_loss_limit"] + 1e-6
    if metrics["n_funded"] > 0:
        assert metrics["average_pd"] <= opt["max_average_pd"] + 1e-9
        assert metrics["high_risk_exposure_pct"] <= opt["max_high_risk_exposure_pct"] + 1e-9


def test_profit_efficiency_by_segment_matches_hand_calculation():
    pool = _toy_pool()
    eff = profit_efficiency_by_segment(pool)
    expected_prime = np.mean([100.0 / 1000.0, 200.0 / 5000.0])
    assert eff["prime"] == pytest.approx(expected_prime)


@pytest.fixture(scope="module")
def real_pool():
    from credit_limit_optimizer.utils.config import load_config
    return build_candidate_pool(load_config())


def test_real_candidate_pool_shape_and_columns(real_pool):
    assert len(real_pool) > 1000
    for col in ["customer_id", "customer_segment", "limit", "expected_profit", "expected_loss", "pd_value", "is_high_risk"]:
        assert col in real_pool.columns
    assert real_pool["limit"].min() > 0
    assert set(real_pool["is_high_risk"].unique()) <= {0.0, 1.0}


@pytest.fixture(scope="module")
def report():
    return run_portfolio_optimization()


def test_solver_reaches_optimal(report):
    assert report["solver_status"] == "OPTIMAL"


def test_mip_solution_respects_every_constraint(report):
    m, c = report["mip_optimal_metrics"], report["constraints"]
    assert m["total_exposure"] <= c["portfolio_exposure_limit"] + 1.0
    assert m["total_expected_loss"] <= c["portfolio_expected_loss_limit"] + 1.0
    assert m["average_pd"] <= c["max_average_pd"] + 1e-6
    assert m["high_risk_exposure_pct"] <= c["max_high_risk_exposure_pct"] + 1e-6


def test_mip_at_least_as_good_as_greedy(report):
    """The MIP solves the problem exactly; it must never do worse than a
    feasible heuristic solution."""
    assert report["mip_optimal_metrics"]["total_expected_profit"] >= report["greedy_baseline_metrics"]["total_expected_profit"] - 1.0


def test_exposure_constraint_is_genuinely_binding(report):
    """Regression sanity: on the real data, unconstrained demand for
    exposure must exceed the limit -- otherwise this whole piece would be
    solving a trivial, non-binding problem."""
    assert report["unconstrained_metrics"]["total_exposure"] > report["constraints"]["portfolio_exposure_limit"]


def test_prime_funded_rate_lower_than_near_prime(report):
    """Regression coverage for the documented, non-obvious finding: the
    exposure-constrained MIP favors profit-per-euro efficiency, not raw
    safety -- prime (lowest profit-per-exposure) should be funded at a
    lower rate than near_prime."""
    rates = report["funded_rate_by_segment"]
    assert rates["prime"] < rates["near_prime"]


def test_output_files_written(report):
    assert REPORT_PATH.exists()
    assert MODEL_CARD_PATH.exists()
    assert (FIG_DIR / "04_portfolio_funded_by_segment.png").exists()
    assert (FIG_DIR / "05_constraint_utilization.png").exists()
