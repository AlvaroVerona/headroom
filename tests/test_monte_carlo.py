"""Tests for the Monte Carlo loss simulation / VaR-CVaR engine (§28): unit
tests for the simulation kernel and VaR/CVaR math on handcrafted/synthetic
fixtures, plus an integration test against the real funded portfolio."""

from __future__ import annotations

import numpy as np
import pytest

from credit_limit_optimizer.simulation.monte_carlo import (
    FIG_DIR, MODEL_CARD_PATH, REPORT_PATH,
    compute_var_cvar, run_monte_carlo, simulate_portfolio_losses,
)
from credit_limit_optimizer.simulation.scenarios import SCENARIOS


def test_simulate_portfolio_losses_shape():
    pd_values = np.full(50, 0.05)
    lgd_values = np.full(50, 0.70)
    ead_values = np.full(50, 1000.0)
    losses = simulate_portfolio_losses(pd_values, lgd_values, ead_values, rho=0.04, n_simulations=1000, random_state=42)
    assert losses.shape == (1000,)
    assert (losses >= 0).all()


def test_simulate_portfolio_losses_mean_matches_analytical_el():
    """With enough draws, mean simulated loss must converge to the
    closed-form PD x LGD x EAD sum -- the single most important sanity
    check for a Monte Carlo engine (a biased simulator would fail this
    even if it "looked" reasonable)."""
    rng_pd = 0.06
    n, lgd, ead = 2000, 0.70, 1000.0
    pd_values = np.full(n, rng_pd)
    lgd_values = np.full(n, lgd)
    ead_values = np.full(n, ead)
    analytical_el = n * rng_pd * lgd * ead

    losses = simulate_portfolio_losses(pd_values, lgd_values, ead_values, rho=0.04, n_simulations=20000, random_state=7)
    assert losses.mean() == pytest.approx(analytical_el, rel=0.03)


def test_simulate_portfolio_losses_zero_pd_gives_zero_loss():
    pd_values = np.zeros(10)
    lgd_values = np.full(10, 0.70)
    ead_values = np.full(10, 1000.0)
    losses = simulate_portfolio_losses(pd_values, lgd_values, ead_values, rho=0.04, n_simulations=500, random_state=1)
    assert (losses == 0).all()


def test_higher_correlation_produces_fatter_tail():
    """Higher asset correlation must widen the loss distribution (more
    correlated defaults cluster together in bad draws) -- the entire
    reason to run a copula simulation instead of a plain binomial sum."""
    n = 500
    pd_values = np.full(n, 0.05)
    lgd_values = np.full(n, 0.70)
    ead_values = np.full(n, 1000.0)
    low_rho = simulate_portfolio_losses(pd_values, lgd_values, ead_values, rho=0.01, n_simulations=5000, random_state=3)
    high_rho = simulate_portfolio_losses(pd_values, lgd_values, ead_values, rho=0.30, n_simulations=5000, random_state=3)
    assert high_rho.std() > low_rho.std()


def test_compute_var_cvar_ordering():
    """CVaR (mean of the tail) must be >= VaR (the tail's own threshold)
    by construction, and a higher confidence level must give a higher
    VaR/CVaR."""
    losses = np.concatenate([np.zeros(950), np.linspace(1000, 100000, 50)])
    result = compute_var_cvar(losses, [0.90, 0.99])
    assert result[0.90]["cvar"] >= result[0.90]["var"]
    assert result[0.99]["cvar"] >= result[0.99]["var"]
    assert result[0.99]["var"] >= result[0.90]["var"]


def test_compute_var_cvar_matches_hand_calculation():
    losses = np.arange(1, 101, dtype=float)  # 1..100
    result = compute_var_cvar(losses, [0.95])
    assert result[0.95]["var"] == pytest.approx(np.quantile(losses, 0.95))
    assert result[0.95]["cvar"] == pytest.approx(losses[losses >= result[0.95]["var"]].mean())


@pytest.fixture(scope="module")
def report():
    return run_monte_carlo()


def test_report_covers_all_scenarios(report):
    assert set(report["summary_by_scenario"]) == set(SCENARIOS)


def test_mc_mean_close_to_analytical_el_on_real_data(report):
    """The core sanity check, on the real funded portfolio: MC mean loss
    within a few percent of the analytical Expected Loss for every
    scenario, not just in the synthetic unit test above."""
    for scenario, row in report["summary_by_scenario"].items():
        rel_diff = abs(row["mc_mean_loss"] / row["expected_loss"] - 1)
        assert rel_diff < 0.05, f"{scenario}: MC mean diverges {rel_diff:.1%} from analytical EL"


def test_var_exceeds_expected_loss_for_every_scenario(report):
    """VaR at any reasonable confidence level must exceed the mean
    (Expected Loss) -- it's a tail quantile of a right-skewed
    distribution, not the mean itself."""
    for scenario, row in report["summary_by_scenario"].items():
        for level_key, vc in row["var_cvar"].items():
            assert vc["var"] > row["expected_loss"], f"{scenario} @ {level_key}: VaR does not exceed EL"


def test_cvar_exceeds_var_for_every_scenario(report):
    for scenario, row in report["summary_by_scenario"].items():
        for level_key, vc in row["var_cvar"].items():
            assert vc["cvar"] >= vc["var"]


def test_economic_capital_is_positive(report):
    for scenario, row in report["summary_by_scenario"].items():
        for level_key, ec in row["economic_capital"].items():
            assert ec > 0, f"{scenario} @ {level_key}: non-positive Economic Capital"


def test_recession_var_exceeds_base_var(report):
    base_var99 = report["summary_by_scenario"]["base"]["var_cvar"]["0.99"]["var"]
    recession_var99 = report["summary_by_scenario"]["recession"]["var_cvar"]["0.99"]["var"]
    assert recession_var99 > base_var99


def test_output_files_written(report):
    assert REPORT_PATH.exists()
    assert MODEL_CARD_PATH.exists()
    assert (FIG_DIR / "01_loss_distribution.png").exists()
    assert (FIG_DIR / "02_var_by_scenario.png").exists()
