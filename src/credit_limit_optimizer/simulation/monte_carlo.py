"""Monte Carlo portfolio loss simulation + VaR/CVaR (§28, Phase 6):
single-factor (Vasicek/ASRF) Gaussian copula default simulation over the
FUNDED portfolio (Phase 5's MIP solution, same book `scenarios.py`
stress-tests), the standard regulatory-capital model for retail credit
portfolios.

```
Z ~ N(0, 1)                              # one systemic (macro) factor per draw
eps_i ~ N(0, 1)                          # one idiosyncratic factor per customer per draw
X_i = sqrt(rho) * Z + sqrt(1 - rho) * eps_i
default_i = 1[X_i <= Phi^-1(PD_i)]       # threshold model, calibrated so
                                          # P(default_i) = PD_i exactly
loss_s = sum_i(default_i * LGD_i * EAD_i)
```

`rho` (`simulation.asset_correlation` in config, 0.04) is Basel's own flat
asset correlation for Qualifying Revolving Retail Exposures (credit
cards) -- not fit to this data, the actual regulatory constant for this
exact product type. Sharing `Z` across every customer in a draw is what
makes losses fat-tailed (correlated defaults cluster in bad draws) rather
than a plain binomial sum, which is the entire point of running a
simulation instead of just using PD x LGD x EAD: the ANALYTICAL Expected
Loss (already computed per-customer in `expected_loss.py`/`scenarios.py`)
is exactly `E[loss_s]`, but says nothing about the loss distribution's
tail -- VaR/CVaR are the shape of that tail.

**PD/LGD/EAD source**: reuses `scenarios.py`'s base-scenario re-scored PD
and `profitability.evaluate_at_limit`'s `balance` (= EAD in this project,
same convention `portfolio_optimizer.py` already uses) at each customer's
funded limit -- one MC run per scenario in `config/settings.yaml`, not
just base, so stress scenarios' tail risk (not just their mean EL,
already in `scenarios.py`) is visible too.

**VaR/CVaR**: `VaR_alpha` = the alpha-quantile of the simulated loss
distribution (e.g. the loss level exceeded only (1-alpha) of draws).
`CVaR_alpha` (Expected Shortfall) = mean loss AMONG draws at or beyond
VaR_alpha -- the standard, coherent (subadditive, unlike VaR) tail-risk
measure. `Economic Capital = VaR_alpha - Expected Loss` (the standard
definition: EL is already provisioned for/priced in; capital exists for
losses ABOVE that expectation).

Run: python -m credit_limit_optimizer.simulation.monte_carlo
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from credit_limit_optimizer.simulation.scenarios import (
    SCENARIOS, build_funded_book, run_scenario,
)
from credit_limit_optimizer.utils.config import PROJECT_ROOT, load_config
from credit_limit_optimizer.utils.logging import get_logger

log = get_logger("monte_carlo")

FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "monte_carlo"
REPORT_PATH = PROJECT_ROOT / "reports" / "outputs" / "monte_carlo_report.json"
MODEL_CARD_PATH = PROJECT_ROOT / "reports" / "model_cards" / "monte_carlo.md"


def simulate_portfolio_losses(pd_values: np.ndarray, lgd_values: np.ndarray, ead_values: np.ndarray, rho: float, n_simulations: int, random_state: int) -> np.ndarray:
    """Vectorized single-factor Gaussian copula simulation. Returns one
    total portfolio loss per simulation (shape (n_simulations,))."""
    rng = np.random.default_rng(random_state)
    n = len(pd_values)
    default_threshold = stats.norm.ppf(np.clip(pd_values, 1e-9, 1 - 1e-9))  # Phi^-1(PD_i)

    systemic = rng.standard_normal(n_simulations)  # Z, one per draw
    idiosyncratic = rng.standard_normal((n_simulations, n))  # eps_i, one per (draw, customer)
    latent = np.sqrt(rho) * systemic[:, None] + np.sqrt(1 - rho) * idiosyncratic

    defaulted = latent <= default_threshold[None, :]
    losses = defaulted @ (lgd_values * ead_values)
    return losses


def compute_var_cvar(losses: np.ndarray, confidence_levels: list[float]) -> dict:
    result = {}
    for level in confidence_levels:
        var = float(np.quantile(losses, level))
        tail = losses[losses >= var]
        cvar = float(tail.mean()) if len(tail) else var
        result[level] = {"var": var, "cvar": cvar}
    return result


def plot_loss_distribution(losses: np.ndarray, var_cvar: dict, expected_loss: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(losses, bins=80, color="#1565C0", alpha=0.75)
    ax.axvline(expected_loss, color="#333333", linestyle="-", linewidth=1.5, label=f"Expected Loss (€{expected_loss:,.0f})")
    colors = {0.95: "#F9A825", 0.99: "#C62828"}
    for level, stats_ in var_cvar.items():
        color = colors.get(level, "#6A1B9A")
        ax.axvline(stats_["var"], color=color, linestyle="--", linewidth=1.5, label=f"VaR {level:.0%} (€{stats_['var']:,.0f})")
        ax.axvline(stats_["cvar"], color=color, linestyle=":", linewidth=1.5, label=f"CVaR {level:.0%} (€{stats_['cvar']:,.0f})")
    ax.set_xlabel("Simulated portfolio loss (EUR/year)")
    ax.set_ylabel("Simulations")
    ax.set_title("Monte Carlo portfolio loss distribution (base scenario)")
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "01_loss_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_var_by_scenario(summaries: dict, level: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    scenarios = list(summaries.keys())
    el = [summaries[s]["expected_loss"] for s in scenarios]
    var = [summaries[s]["var_cvar"][str(level)]["var"] for s in scenarios]
    x = np.arange(len(scenarios))
    width = 0.35
    ax.bar(x - width / 2, el, width, label="Expected Loss", color="#1565C0")
    ax.bar(x + width / 2, var, width, label=f"VaR {level:.0%}", color="#C62828")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("EUR/year")
    ax.set_title(f"Expected Loss vs. VaR {level:.0%} by scenario")
    ax.legend()
    fig.savefig(FIG_DIR / "02_var_by_scenario.png", bbox_inches="tight")
    plt.close(fig)


def run_monte_carlo() -> dict:
    config = load_config()
    sim_cfg = config["simulation"]
    rho = sim_cfg["asset_correlation"]
    n_sim = sim_cfg["n_simulations"]
    confidence_levels = sim_cfg["var_confidence_levels"]

    book = build_funded_book(config)
    required = ["monthly_income", "monthly_spend", "income_volatility", "delinquency_count", "days_past_due", "cash_buffer", "apr", "average_balance"]
    book = book.dropna(subset=required)

    summaries = {}
    base_losses = None
    for scenario_name in SCENARIOS:
        economics = run_scenario(book, scenario_name, config)
        pd_values = economics["pd"].to_numpy()
        lgd_values = economics["expected_loss"].to_numpy() / np.maximum(pd_values * economics["balance"].to_numpy(), 1e-9)
        # LGD recovered from EL = PD x LGD x balance rather than
        # re-deriving it here -- guarantees analytical-EL/MC-mean
        # consistency isn't an artifact of two independent LGD lookups
        # drifting apart, only ever one source of truth.
        ead_values = economics["balance"].to_numpy()

        losses = simulate_portfolio_losses(pd_values, lgd_values, ead_values, rho, n_sim, random_state=config["random_seed"])
        if scenario_name == "base":
            base_losses = losses
        analytical_el = float(economics["expected_loss"].sum())
        var_cvar = compute_var_cvar(losses, confidence_levels)

        summaries[scenario_name] = {
            "expected_loss": analytical_el,
            "mc_mean_loss": float(losses.mean()),
            "mc_std_loss": float(losses.std()),
            "var_cvar": {str(level): v for level, v in var_cvar.items()},
            "economic_capital": {str(level): v["var"] - analytical_el for level, v in var_cvar.items()},
        }
        log.info(
            "%s: analytical_EL=€%.0f MC_mean=€%.0f (diff %.2f%%) VaR99=€%.0f CVaR99=€%.0f",
            scenario_name, analytical_el, summaries[scenario_name]["mc_mean_loss"],
            100 * (summaries[scenario_name]["mc_mean_loss"] / analytical_el - 1),
            var_cvar[0.99]["var"], var_cvar[0.99]["cvar"],
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for stale in FIG_DIR.glob("*.png"):
        stale.unlink()
    plot_loss_distribution(base_losses, {level: v for level, v in compute_var_cvar(base_losses, confidence_levels).items()}, summaries["base"]["expected_loss"])
    plot_var_by_scenario(summaries, confidence_levels[-1])

    report = {
        "n_funded": int(len(book)), "asset_correlation": rho, "n_simulations": n_sim,
        "confidence_levels": confidence_levels,
        "summary_by_scenario": summaries,
        "portfolio_expected_loss_limit": config["optimization"]["portfolio_expected_loss_limit"],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Wrote %s", REPORT_PATH.relative_to(PROJECT_ROOT))

    write_model_card(report)
    log.info("Wrote %s", MODEL_CARD_PATH.relative_to(PROJECT_ROOT))
    return report


def write_model_card(report: dict) -> None:
    s = report["summary_by_scenario"]
    levels = report["confidence_levels"]
    lines = [
        "# Model Card — Monte Carlo Loss Simulation / VaR-CVaR (§28)",
        "",
        f"Single-factor (Vasicek/ASRF) Gaussian copula default simulation, "
        f"{report['n_simulations']:,} draws, asset correlation "
        f"{report['asset_correlation']} (Basel's flat Qualifying Revolving Retail Exposure "
        "constant, not fit to this data). See module docstring for the full model.",
        "",
        "## Analytical Expected Loss vs. Monte Carlo mean (sanity check)",
        "",
        "The simulated loss distribution's mean must closely match the analytical "
        "`PD x LGD x EAD` sum already computed in `scenarios.py` -- both describe the SAME "
        "expectation, just one is a closed form and the other a Monte Carlo estimate of it.",
        "",
        "| Scenario | Analytical EL | MC mean loss | Difference |",
        "|---|---|---|---|",
    ]
    for scenario, row in s.items():
        diff_pct = 100 * (row["mc_mean_loss"] / row["expected_loss"] - 1)
        lines.append(f"| {scenario} | €{row['expected_loss']:,.0f} | €{row['mc_mean_loss']:,.0f} | {diff_pct:+.2f}% |")
    lines += [
        "",
        "## VaR / CVaR / Economic Capital by scenario",
        "",
        "`Economic Capital = VaR - Expected Loss` -- EL is already priced in via the pricing/"
        "provisioning models built in Phase 4; capital is held against loss ABOVE that "
        "expectation, the standard regulatory-capital definition.",
        "",
    ]
    for level in levels:
        lines += [f"### {level:.0%} confidence", "", "| Scenario | VaR | CVaR (Expected Shortfall) | Economic Capital |", "|---|---|---|---|"]
        for scenario, row in s.items():
            vc = row["var_cvar"][str(level)]
            ec = row["economic_capital"][str(level)]
            lines.append(f"| {scenario} | €{vc['var']:,.0f} | €{vc['cvar']:,.0f} | €{ec:,.0f} |")
        lines.append("")
    lines += [
        "![loss distribution](../figures/monte_carlo/01_loss_distribution.png)",
        "",
        f"![VaR by scenario](../figures/monte_carlo/02_var_by_scenario.png)",
        "",
    ]
    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_CARD_PATH.open("w") as f:
        f.write("\n".join(lines))


def main() -> None:
    run_monte_carlo()


if __name__ == "__main__":
    main()
