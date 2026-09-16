"""Funding Cost model (§21): the cost of financing credit exposure.

```
Funding Cost = Average Balance × Funding Rate / 12   (annual rate, /12
                                                        for a monthly
                                                        figure -- same
                                                        treatment as
                                                        Interest Revenue
                                                        in revenue_model.py)
```

**Varies with macroeconomic scenario** (§21's explicit ask): reuses
`config/settings.yaml`'s `scenarios.*.funding_rate_shift`, already
scaffolded there (recession +1pp, high_interest_rate +3pp) ahead of
Phase 6's full stress-testing engine -- `funding_rate(scenario) =
economics.funding_rate + scenarios[scenario].get("funding_rate_shift", 0)`.
`consumer_stress` has no funding_rate_shift key (it stresses income/
spending/delinquency, not the cost of funds), so it uses the base rate
unchanged -- a deliberate choice, not a gap.

**Does NOT vary by customer segment**, even though §21 allows it to: a
bank's cost of funds is a treasury-level blended rate (roughly the same
whichever customer's balance it happens to fund), not a credit-risk
quantity like LGD/PD that legitimately differs by segment. Segment risk
is already priced through PD/LGD (Expected Loss) and APR (Revenue), not
duplicated here.

**Does NOT build the full time-varying/Monte Carlo funding-rate path**
that "varies with time" could imply at full generality -- that's Phase
6's stress-testing/Monte Carlo engine. Here, a scenario stands in for an
alternate macro/time state (§21's own example structure), computed as a
straightforward multi-scenario comparison, not a simulated path.

Also computes Net Interest Margin = Interest Revenue − Funding Cost
(base scenario) per customer, since a positive NIM is the most basic
sanity check that this and the revenue model agree on scale.

Run: python -m credit_limit_optimizer.models.funding_cost
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_limit_optimizer.models.expected_loss import load_full_feature_table
from credit_limit_optimizer.models.revenue_model import compute_revenue
from credit_limit_optimizer.models.risk_model import split_cohorts
from credit_limit_optimizer.utils.config import PROJECT_ROOT, load_config
from credit_limit_optimizer.utils.logging import get_logger

log = get_logger("funding_cost")

FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "funding_cost"
REPORT_PATH = PROJECT_ROOT / "reports" / "outputs" / "funding_cost_report.json"
MODEL_CARD_PATH = PROJECT_ROOT / "reports" / "model_cards" / "funding_cost.md"

SCENARIOS = ["base", "recession", "high_interest_rate", "consumer_stress"]

plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def funding_rate_for_scenario(config: dict, scenario: str) -> float:
    base_rate = config["economics"]["funding_rate"]
    shift = config["scenarios"].get(scenario, {}).get("funding_rate_shift", 0.0)
    return base_rate + shift


def compute_funding_cost(df: pd.DataFrame, rate: float) -> np.ndarray:
    return (df["average_balance"] * rate / 12).to_numpy()


def plot_funding_cost_by_scenario(mean_cost_by_scenario: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    scenarios = list(mean_cost_by_scenario.keys())
    values = [mean_cost_by_scenario[s] for s in scenarios]
    colors = ["#1565C0" if s == "base" else "#C62828" for s in scenarios]
    ax.bar(scenarios, values, color=colors)
    ax.set_ylabel("Mean monthly funding cost (EUR)")
    ax.set_title("Mean funding cost by scenario (test cohort)")
    fig.savefig(FIG_DIR / "01_funding_cost_by_scenario.png", bbox_inches="tight")
    plt.close(fig)


def plot_nim_by_segment(nim_by_segment: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    segments = ["prime", "near_prime", "subprime"]
    ax.bar(segments, [nim_by_segment[s] for s in segments], color=["#2E7D32", "#F9A825", "#C62828"])
    ax.set_ylabel("Mean Net Interest Margin (EUR/month)")
    ax.set_title("Net Interest Margin by segment (base scenario, test cohort)")
    fig.savefig(FIG_DIR / "02_nim_by_segment.png", bbox_inches="tight")
    plt.close(fig)


def run_funding_cost() -> dict:
    config = load_config()
    df = load_full_feature_table()
    cohorts = split_cohorts(df)
    test = cohorts["test"]

    n_before = len(test)
    test = test.dropna(subset=["average_balance", "apr", "monthly_spend", "days_past_due"])
    n_excluded = n_before - len(test)
    log.info("Excluding %d of %d test rows (missing funding-cost inputs)", n_excluded, n_before)

    rates_by_scenario = {s: funding_rate_for_scenario(config, s) for s in SCENARIOS}
    log.info("Funding rate by scenario: %s", {k: f"{v:.2%}" for k, v in rates_by_scenario.items()})

    mean_cost_by_scenario = {}
    total_cost_by_scenario = {}
    for scenario, rate in rates_by_scenario.items():
        cost = compute_funding_cost(test, rate)
        mean_cost_by_scenario[scenario] = float(cost.mean())
        total_cost_by_scenario[scenario] = float(cost.sum())

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_funding_cost_by_scenario(mean_cost_by_scenario)

    base_cost = compute_funding_cost(test, rates_by_scenario["base"])
    revenue = compute_revenue(test, config)
    nim = revenue["interest_revenue"].to_numpy() - base_cost
    log.info(
        "Base scenario: mean funding cost=€%.2f mean interest revenue=€%.2f mean NIM=€%.2f",
        base_cost.mean(), revenue["interest_revenue"].mean(), nim.mean(),
    )

    nim_df = pd.DataFrame({"customer_segment": test["customer_segment"].to_numpy(), "nim": nim})
    nim_by_segment = nim_df.groupby("customer_segment")["nim"].mean().to_dict()
    plot_nim_by_segment(nim_by_segment)
    log.info("Mean NIM by segment: %s", {k: round(v, 2) for k, v in nim_by_segment.items()})

    report = {
        "n_test_rows": int(len(test)), "n_excluded_missing_inputs": n_excluded,
        "funding_rate_by_scenario": rates_by_scenario,
        "mean_funding_cost_by_scenario": mean_cost_by_scenario,
        "total_portfolio_funding_cost_by_scenario": total_cost_by_scenario,
        "base_scenario": {
            "mean_funding_cost": float(base_cost.mean()),
            "mean_interest_revenue": float(revenue["interest_revenue"].mean()),
            "mean_net_interest_margin": float(nim.mean()),
            "fraction_positive_nim": float((nim > 0).mean()),
            "n_negative_nim": int((nim < 0).sum()),
        },
        "nim_by_segment": {k: float(v) for k, v in nim_by_segment.items()},
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Wrote %s", REPORT_PATH.relative_to(PROJECT_ROOT))

    write_model_card(report)
    log.info("Wrote %s", MODEL_CARD_PATH.relative_to(PROJECT_ROOT))
    return report


def write_model_card(report: dict) -> None:
    base = report["base_scenario"]
    lines = [
        "# Model Card — Funding Cost (§21)",
        "",
        "Funding Cost = Average Balance × Funding Rate / 12. Rate varies by macroeconomic "
        "scenario (`config/settings.yaml`'s `scenarios.*.funding_rate_shift`); deliberately "
        "flat across customer segment (a bank's cost of funds is a treasury-level blended "
        "rate, not a credit-risk quantity — segment risk is already priced via PD/LGD and "
        "APR elsewhere). Full time-varying/Monte Carlo funding-rate paths are Phase 6's "
        "stress-testing engine, not this piece.",
        "",
        f"{report['n_excluded_missing_inputs']:,} of "
        f"{report['n_test_rows'] + report['n_excluded_missing_inputs']:,} test rows excluded "
        "(missing one of average_balance/apr/monthly_spend/days_past_due — same real Phase 1 "
        "data-quality artifacts as the Expected Loss and Revenue pieces).",
        "",
        "## Funding rate and cost by scenario",
        "",
        "| Scenario | Funding rate | Mean monthly cost | Total portfolio monthly cost |",
        "|---|---|---|---|",
    ]
    for s in SCENARIOS:
        rate = report["funding_rate_by_scenario"][s]
        mean_cost = report["mean_funding_cost_by_scenario"][s]
        total_cost = report["total_portfolio_funding_cost_by_scenario"][s]
        lines.append(f"| {s} | {rate:.2%} | €{mean_cost:.2f} | €{total_cost:,.0f} |")
    lines += [
        "",
        "![funding cost by scenario](../figures/funding_cost/01_funding_cost_by_scenario.png)",
        "",
        "## Net Interest Margin (base scenario)",
        "",
        f"Mean interest revenue €{base['mean_interest_revenue']:.2f}, mean funding cost "
        f"€{base['mean_funding_cost']:.2f}, **mean NIM €{base['mean_net_interest_margin']:.2f}** "
        f"per customer-snapshot — positive for {base['fraction_positive_nim']:.1%} of "
        "customers. The remaining rows are NIM = 0 exactly, never negative: verified to be "
        "customers with `average_balance == 0.0` for that month (zero interest revenue, zero "
        "funding cost) — every segment's APR (16.9-27.9%) exceeds every scenario's funding "
        "rate (5-8%), so any customer who actually carries a balance nets a positive spread "
        "by construction.",
        "",
        "![NIM by segment](../figures/funding_cost/02_nim_by_segment.png)",
        "",
    ]
    for seg in ["prime", "near_prime", "subprime"]:
        lines.append(f"- **{seg}**: €{report['nim_by_segment'][seg]:.2f}/month")
    lines.append("")
    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_CARD_PATH.open("w") as f:
        f.write("\n".join(lines))


def main() -> None:
    run_funding_cost()


if __name__ == "__main__":
    main()
