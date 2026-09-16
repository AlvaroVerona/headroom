"""Customer Profitability (§22):

```
Expected Revenue
- Expected Credit Loss
- Funding Cost
- Operational Cost
=
Expected Customer Profit
```

computed both at each customer's observed/current limit (portfolio-level
summary) and across a candidate limit grid for a handful of example
customers (§22's own illustrative table format).

**Balance response to a candidate limit** -- the modeling choice
CLAUDE.md flagged as needed and undefined until now:

```
balance(L) = min(current_balance * (L / current_limit) ** elasticity, L)
```

anchored exactly at the customer's own observed (current_limit,
current_balance) point (so `balance(current_limit) == current_balance`
by construction), with `elasticity` a config value (`optimization.
balance_elasticity`, default 0.3) `< 1` so balance grows slower than the
candidate limit -- diminishing returns by construction, the concave
Limit-vs-Profit shape CLAUDE.md's note asked for. The `min(..., L)` term
enforces the physical constraint that balance can never exceed the limit
that bounds it.

**Why not reuse the EAD regression model (or a PD-vs-limit model) for
this instead of a hand-specified formula**: `credit_exposure` (the EAD
model's stand-in for credit limit) had rank 13 of 46 and importance
0.0048 in the fitted EAD regressor (`expected_loss.py`) -- because this
generator never varies a customer's `credit_limit` over their 36-month
history, the model never saw within-customer limit variation to learn a
genuine credit_exposure → utilization *causal* relationship from; it
only ever saw *between*-customer correlation (confounded by segment/
income, which drive both the originally-assigned limit and behavior).
Substituting a different `credit_exposure` value into that model's
feature vector would extrapolate noise, not a real counterfactual --
using it anyway would be a subtler, harder-to-notice version of exactly
the mistake the LR baseline's multicollinearity bugs already were. Same
reasoning rules out a candidate-limit-varying PD, so **PD is held
constant** at each customer's observed, calibrated value here -- an
explicit, documented simplification, not an oversight.

**Also held constant across candidate limits**: interchange revenue and
late-fee revenue (driven by spending behavior / payment history, not
credit limit, in this model) -- only the revolving balance, and
everything computed from it (interest revenue, EAD, funding cost),
responds to the candidate limit.

Run: python -m credit_limit_optimizer.models.profitability
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_limit_optimizer.models.expected_loss import PD_MODEL_PATH, load_full_feature_table
from credit_limit_optimizer.models.risk_model import split_cohorts
from credit_limit_optimizer.utils.config import PROJECT_ROOT, load_config
from credit_limit_optimizer.utils.logging import get_logger

log = get_logger("profitability")

FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "profitability"
REPORT_PATH = PROJECT_ROOT / "reports" / "outputs" / "profitability_report.json"
MODEL_CARD_PATH = PROJECT_ROOT / "reports" / "model_cards" / "profitability.md"

REQUIRED_COLUMNS = ["average_balance", "apr", "monthly_spend", "days_past_due", "credit_exposure"]
N_EXAMPLE_CUSTOMERS_PER_SEGMENT = 1

plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def balance_at_limit(current_balance: np.ndarray, current_limit: np.ndarray, candidate_limit, elasticity: float) -> np.ndarray:
    power_law = current_balance * (candidate_limit / current_limit) ** elasticity
    return np.minimum(power_law, candidate_limit)


def evaluate_at_limit(df: pd.DataFrame, candidate_limit, pd_values: np.ndarray, config: dict) -> pd.DataFrame:
    """All returned euro figures are ANNUAL. `expected_loss` is
    inherently annual (PD is a 12-month default probability, per
    default_12m) -- revenue and funding cost, computed monthly in
    revenue_model.py/funding_cost.py from the customer's CURRENT-month
    balance/spend/delinquency status, are annualized here the standard
    "run-rate x 12" way (assume the current month's observed rate holds
    for the year) so every term in Expected Profit is on the same annual
    basis. `operational_cost_per_customer` is already an annual figure in
    config, used as-is."""
    econ = config["economics"]
    balance = balance_at_limit(df["average_balance"].to_numpy(), df["credit_exposure"].to_numpy(), candidate_limit, config["optimization"]["balance_elasticity"])
    utilization = np.clip(balance / np.asarray(candidate_limit, dtype=float), 0, 1)

    interest_revenue = balance * df["apr"].to_numpy()  # apr is already annual
    interchange_revenue = df["monthly_spend"].to_numpy() * econ["interchange_rate"] * 12
    late_fee_revenue = np.where(df["days_past_due"].to_numpy() > 0, econ["late_fee_amount"], 0.0) * 12
    total_revenue = interest_revenue + interchange_revenue + late_fee_revenue

    lgd = df["customer_segment"].map(econ["lgd_by_segment"]).to_numpy()
    expected_loss = pd_values * lgd * balance  # already annual (12-month PD)
    funding_cost = balance * econ["funding_rate"]  # funding_rate is already annual
    operational_cost = econ["operational_cost_per_customer"]  # already annual

    expected_profit = total_revenue - expected_loss - funding_cost - operational_cost
    return pd.DataFrame({
        "candidate_limit": candidate_limit if np.ndim(candidate_limit) else np.full(len(df), candidate_limit),
        "balance": balance, "utilization": utilization,
        "interest_revenue": interest_revenue, "interchange_revenue": interchange_revenue,
        "late_fee_revenue": late_fee_revenue, "total_revenue": total_revenue,
        "expected_loss": expected_loss, "funding_cost": funding_cost,
        "operational_cost": np.full(len(df), operational_cost, dtype=float),
        "expected_profit": expected_profit,
    }, index=df.index)


def compute_pd(test_df: pd.DataFrame) -> np.ndarray:
    saved = joblib.load(PD_MODEL_PATH)
    calibrated_model, features = saved["model"], saved["features"]
    pd_values = calibrated_model.predict_proba(test_df[features])[:, 1]
    already_in_default = ~test_df["eligible_for_default_label"].astype(bool)
    return np.where(already_in_default, 1.0, pd_values)


def candidate_limit_grid(config: dict) -> list[int]:
    opt = config["optimization"]
    return list(range(opt["minimum_limit"], opt["maximum_limit"] + opt["step"], opt["step"]))


def plot_profit_curve(customer_id: str, segment: str, table: pd.DataFrame, current_limit: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["candidate_limit"], table["total_revenue"], label="Revenue", color="#2E7D32")
    ax.plot(table["candidate_limit"], table["expected_loss"], label="Expected Loss", color="#C62828")
    ax.plot(table["candidate_limit"], table["expected_profit"], label="Expected Profit", color="#1565C0", linewidth=2.5)
    ax.axvline(current_limit, color="#9E9E9E", linestyle="--", linewidth=1, label="Current limit")
    ax.set_xlabel("Candidate credit limit (EUR)")
    ax.set_ylabel("Annual EUR")
    ax.set_title(f"Profitability vs. limit — {customer_id} ({segment})")
    ax.legend()
    fig.savefig(FIG_DIR / f"profit_curve_{customer_id}.png", bbox_inches="tight")
    plt.close(fig)


def plot_portfolio_profit_by_segment(profit_by_segment: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    segments = ["prime", "near_prime", "subprime"]
    ax.bar(segments, [profit_by_segment[s] for s in segments], color=["#2E7D32", "#F9A825", "#C62828"])
    ax.set_ylabel("Mean Expected Customer Profit (EUR/year)")
    ax.set_title("Mean Expected Profit by segment, at current limits (test cohort)")
    fig.savefig(FIG_DIR / "00_profit_by_segment.png", bbox_inches="tight")
    plt.close(fig)


def run_profitability() -> dict:
    config = load_config()
    df = load_full_feature_table()
    cohorts = split_cohorts(df)
    test = cohorts["test"]

    n_before = len(test)
    test = test.dropna(subset=REQUIRED_COLUMNS)
    n_excluded = n_before - len(test)
    log.info("Excluding %d of %d test rows (missing profitability inputs)", n_excluded, n_before)

    pd_values = compute_pd(test)

    # Sanity check: evaluating "at the candidate limit == current limit"
    # must reproduce the customer's own observed balance exactly.
    current_state = evaluate_at_limit(test, test["credit_exposure"].to_numpy(), pd_values, config)
    max_balance_diff = float((current_state["balance"] - test["average_balance"]).abs().max())
    log.info("Anchor-point sanity check: max|balance(current_limit) - observed balance| = €%.6f", max_balance_diff)

    profit_by_segment = current_state.join(test["customer_segment"]).groupby("customer_segment")["expected_profit"].mean().to_dict()
    log.info("Mean Expected Profit by segment (current limits): %s", {k: round(v, 2) for k, v in profit_by_segment.items()})
    log.info(
        "Portfolio (current limits, test cohort): mean profit=€%.2f total=€%.0f",
        current_state["expected_profit"].mean(), current_state["expected_profit"].sum(),
    )

    # This module owns FIG_DIR exclusively -- clear stale files first
    # (e.g. a previous run's example-customer profit_curve_*.png for a
    # customer no longer selected), the same fix already applied to
    # explain.py for the same reason.
    if FIG_DIR.exists():
        for stale in FIG_DIR.glob("*.png"):
            stale.unlink()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_portfolio_profit_by_segment(profit_by_segment)

    grid = candidate_limit_grid(config)
    examples = []
    opt = config["optimization"]
    for segment in ["prime", "near_prime", "subprime"]:
        seg_rows = test[
            (test["customer_segment"] == segment)
            & (test["average_balance"] > 0)
            & test["credit_exposure"].between(opt["minimum_limit"], opt["maximum_limit"])
        ]
        if seg_rows.empty:
            seg_rows = test[test["customer_segment"] == segment]
        if seg_rows.empty:
            continue
        # Median balance within the filtered pool -- a representative,
        # not cherry-picked or degenerate (e.g. zero-balance), example.
        row_idx = seg_rows["average_balance"].sort_values().index[len(seg_rows) // 2]
        row = test.loc[[row_idx]]
        row_pd = compute_pd(row)
        table_rows = [evaluate_at_limit(row, L, row_pd, config).iloc[0] for L in grid]
        table = pd.DataFrame(table_rows).reset_index(drop=True)
        plot_profit_curve(row["customer_id"].iloc[0], segment, table, row["credit_exposure"].iloc[0])
        examples.append({
            "customer_id": row["customer_id"].iloc[0], "segment": segment,
            "current_limit": float(row["credit_exposure"].iloc[0]),
            "table": table[["candidate_limit", "balance", "utilization", "total_revenue", "expected_loss", "funding_cost", "expected_profit"]].round(2).to_dict(orient="records"),
        })

    report = {
        "n_test_rows": int(len(test)), "n_excluded_missing_inputs": n_excluded,
        "anchor_point_max_abs_diff": max_balance_diff,
        "current_state_summary": {
            "mean_expected_profit": float(current_state["expected_profit"].mean()),
            "total_portfolio_expected_profit": float(current_state["expected_profit"].sum()),
            "mean_total_revenue": float(current_state["total_revenue"].mean()),
            "mean_expected_loss": float(current_state["expected_loss"].mean()),
            "mean_funding_cost": float(current_state["funding_cost"].mean()),
            "operational_cost_annual": float(current_state["operational_cost"].iloc[0]),
        },
        "profit_by_segment": {k: float(v) for k, v in profit_by_segment.items()},
        "balance_elasticity": config["optimization"]["balance_elasticity"],
        "candidate_limit_grid": grid,
        "examples": examples,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Wrote %s", REPORT_PATH.relative_to(PROJECT_ROOT))

    write_model_card(report)
    log.info("Wrote %s", MODEL_CARD_PATH.relative_to(PROJECT_ROOT))
    return report


def write_model_card(report: dict) -> None:
    cs = report["current_state_summary"]
    lines = [
        "# Model Card — Customer Profitability (§22)",
        "",
        "Expected Customer Profit = Total Revenue − Expected Loss − Funding Cost − "
        "Operational Cost, computed per customer-snapshot. Balance response to a candidate "
        f"limit uses elasticity {report['balance_elasticity']} (see module docstring for the "
        "formula and why the EAD/PD models can't be reused for this).",
        "",
        f"{report['n_excluded_missing_inputs']:,} of "
        f"{report['n_test_rows'] + report['n_excluded_missing_inputs']:,} test rows excluded "
        "(missing revenue/loss/funding inputs — same real Phase 1 data-quality artifacts as "
        "the other Phase 4 pieces).",
        "",
        f"**Anchor-point sanity check**: evaluating every customer's own current limit as "
        f"the \"candidate\" reproduces their observed balance to within "
        f"€{report['anchor_point_max_abs_diff']:.6f} — the response function is "
        "self-consistent by construction, not just close.",
        "",
        "## Current-limit portfolio summary (test cohort)",
        "",
        "All figures ANNUAL: `expected_loss` is inherently a 12-month figure (PD is "
        "`default_12m`'s 12-month probability); revenue and funding cost are annualized "
        "from the customer's current-month observed rate (× 12, standard run-rate practice) "
        "so every term in Expected Profit is on the same basis.",
        "",
        f"Mean per customer-snapshot: revenue €{cs['mean_total_revenue']:.2f}, expected loss "
        f"€{cs['mean_expected_loss']:.2f}, funding cost €{cs['mean_funding_cost']:.2f}, "
        f"operational cost €{cs['operational_cost_annual']:.2f} — **mean Expected Profit "
        f"€{cs['mean_expected_profit']:.2f}/year**. Total portfolio annual Expected Profit on "
        f"the {report['n_test_rows']:,}-row test cohort: €{cs['total_portfolio_expected_profit']:,.0f}.",
        "",
        "![profit by segment](../figures/profitability/00_profit_by_segment.png)",
        "",
    ]
    for seg in ["prime", "near_prime", "subprime"]:
        lines.append(f"- **{seg}**: €{report['profit_by_segment'][seg]:.2f}/year")
    lines += ["", "## Example customers across the candidate limit grid", ""]
    for ex in report["examples"]:
        lines += [
            f"### {ex['customer_id']} ({ex['segment']}, current limit €{ex['current_limit']:,.0f})",
            "",
            "Revenue/Loss/Funding Cost/Profit are all annual.",
            "",
            "| Limit | Balance | Utilization | Revenue | Loss | Funding Cost | Profit |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in ex["table"]:
            lines.append(
                f"| €{row['candidate_limit']:,.0f} | €{row['balance']:,.2f} | "
                f"{row['utilization']:.1%} | €{row['total_revenue']:.2f} | "
                f"€{row['expected_loss']:.2f} | €{row['funding_cost']:.2f} | "
                f"€{row['expected_profit']:.2f} |"
            )
        lines += ["", f"![profit curve {ex['customer_id']}](../figures/profitability/profit_curve_{ex['customer_id']}.png)", ""]
    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_CARD_PATH.open("w") as f:
        f.write("\n".join(lines))


def main() -> None:
    run_profitability()


if __name__ == "__main__":
    main()
