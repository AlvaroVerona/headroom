"""Tests for individual credit limit optimization (§23-24): unit tests
for the DSR formula and constraint/tie-breaking logic on handcrafted
fixtures (regression coverage for the tie-breaking bug found while
building this), plus an integration test against real trained models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_limit_optimizer.optimization.customer_optimizer import (
    CONSTRAINT_COLUMNS, FIG_DIR, MODEL_CARD_PATH, REPORT_PATH,
    _select_optimal, dsr_at_limit, run_customer_optimization,
)


def test_dsr_matches_generator_scheduled_payment_formula():
    """DSR must reuse behavior_series.py's exact scheduled-payment
    formula: max(0.03 x balance, 25 if balance > 0 else 0) / income."""
    balance = np.array([1000.0, 0.0, 100.0])
    income = np.array([2000.0, 2000.0, 2000.0])
    dsr = dsr_at_limit(balance, income)
    assert dsr[0] == pytest.approx(max(1000.0 * 0.03, 25) / 2000.0)  # 30 > 25 -> 0.03x
    assert dsr[1] == pytest.approx(0.0)  # zero balance -> zero payment
    assert dsr[2] == pytest.approx(25 / 2000.0)  # 3 < 25 -> floor applies


def test_dsr_income_floor_prevents_division_explosion():
    dsr = dsr_at_limit(np.array([1000.0]), np.array([0.0]))
    assert np.isfinite(dsr[0])


def _toy_feasible_table(profits, limits, customer_id="C1", current_limit=2000.0) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [customer_id] * len(profits),
        "candidate_limit": limits,
        "expected_profit": profits,
    })


def test_select_optimal_picks_strict_max_when_not_tied():
    table = _toy_feasible_table(profits=[100.0, 150.0, 120.0], limits=[500.0, 1000.0, 1500.0])
    current = pd.Series({"C1": 2000.0})
    result = _select_optimal(table, current)
    assert result.loc["C1", "candidate_limit"] == 1000.0


def test_select_optimal_breaks_ties_toward_current_limit():
    """Regression test: the first version used groupby().idxmax(), which
    silently picks the SMALLEST candidate for a flat (tied) profit curve
    -- e.g. a zero-balance customer, whose profit doesn't depend on the
    limit at all -- producing a misleading "cut this customer's limit"
    recommendation with no economic basis. ~40% of all "decrease"
    recommendations on the real test cohort were exactly this artifact."""
    table = _toy_feasible_table(profits=[200.0, 200.0, 200.0, 200.0], limits=[500.0, 1000.0, 1500.0, 2000.0])
    current = pd.Series({"C1": 1500.0})
    result = _select_optimal(table, current)
    assert result.loc["C1", "candidate_limit"] == 1500.0  # closest to current, not 500


def test_select_optimal_ties_pick_closest_even_when_current_outside_grid():
    """A customer whose real current limit exceeds the candidate grid's
    max should tie-break toward the LARGEST tied candidate (closest to
    an out-of-range current limit), not the smallest."""
    table = _toy_feasible_table(profits=[50.0, 50.0, 50.0], limits=[500.0, 1000.0, 1500.0])
    current = pd.Series({"C1": 11400.0})
    result = _select_optimal(table, current)
    assert result.loc["C1", "candidate_limit"] == 1500.0


def test_select_optimal_near_tie_within_tolerance_still_ties():
    table = _toy_feasible_table(profits=[200.0, 200.005, 199.999], limits=[500.0, 1000.0, 1500.0])
    current = pd.Series({"C1": 1500.0})
    result = _select_optimal(table, current, tolerance=0.01)
    assert result.loc["C1", "candidate_limit"] == 1500.0


@pytest.fixture(scope="module")
def report():
    return run_customer_optimization()


def test_approval_rate_matches_pd_constraint_sanity_check(report):
    """Sanity floor: ~13% of the test cohort has PD > max_pd (0.08),
    verified independently before building the optimizer -- decline rate
    should land in that ballpark, not be near 0% or near 100%."""
    decline_rate = 1 - report["approval_rate"]
    assert 0.08 < decline_rate < 0.20


def test_risk_is_the_dominant_decline_reason(report):
    """PD is held constant across candidates (Phase 4's documented
    simplification), so the risk constraint is either satisfiable for
    every candidate or none -- it should dominate decline reasons."""
    reasons = report["decline_reason_counts"]
    assert reasons.get("risk", 0) > 0
    assert reasons.get("risk", 0) >= sum(v for k, v in reasons.items() if k != "risk")


def test_profit_uplift_nonnegative_by_construction(report):
    """The optimizer only recommends a change when it (weakly) improves
    Expected Profit vs. the current limit -- uplift should never be
    substantially negative in aggregate."""
    for seg, uplift in report["uplift_by_segment"].items():
        assert uplift >= -0.01, f"{seg}: negative mean uplift {uplift}"


def test_decrease_recommendations_mostly_explained_by_grid_cap(report):
    """Regression test for the tie-breaking bug: after the fix, most
    remaining "decrease" recommendations should be explained by the
    candidate grid's max being below the customer's real current limit,
    not an unexplained cut."""
    assert report["fraction_decrease_from_grid_cap"] > 0.5


def test_no_nan_in_report(report):
    assert not np.isnan(report["mean_profit_uplift"])
    assert not np.isnan(report["total_profit_uplift"])
    for v in report["uplift_by_segment"].values():
        assert not np.isnan(v)


def test_output_files_written(report):
    assert REPORT_PATH.exists()
    assert MODEL_CARD_PATH.exists()
    assert (FIG_DIR / "01_approval_and_decline_reasons.png").exists()
    assert (FIG_DIR / "02_limit_change_distribution.png").exists()
    assert (FIG_DIR / "03_profit_uplift_by_segment.png").exists()
