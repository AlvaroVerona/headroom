"""Individual credit limit optimization (§23-24): for each customer,
evaluate every candidate limit (reusing profitability.py's evaluate_at_limit
and balance-response model), apply §24's four customer-level constraints,
and select the feasible candidate that maximizes Expected Profit.

```
Income constraint:       Candidate Limit <= monthly_income x income_multiple
Risk constraint:         PD <= max_pd
Utilization constraint:  predicted utilization(L) <= max_utilization
Debt constraint:         DSR(L) <= max_dsr
```

(§24's fifth, "Exposure constraint", is portfolio-level -- next piece,
`portfolio_optimizer.py`, which is also where OR-Tools comes in: this
piece is an independent per-customer argmax over a small grid, with no
coupling between customers, so a full solver isn't needed here. The
portfolio piece's constraints genuinely couple customers together (one
customer's limit affects whether the total exposure/loss constraint is
still satisfiable for everyone else), which is what actually needs one.)

**DSR (Debt Service Ratio)** has no existing feature-table column, so
it's estimated by reusing the generator's own scheduled-minimum-payment
formula (`behavior_series.py`: `max(0.03 x balance, 25 if balance > 0
else 0)`) applied to `balance(L)`, divided by `monthly_income` -- not a
new, independent assumption, the same formula this dataset's actual
payment behavior already follows.

**PD is held constant across candidates** (Phase 4's documented
simplification -- `credit_exposure` had ~0 importance in the fitted PD/EAD
models). This has a real, worth-naming implication for the risk
constraint specifically: since PD doesn't vary with the candidate limit,
`PD <= max_pd` is either satisfied for EVERY candidate or for NONE --
there's no candidate limit that "fixes" a too-high PD by being smaller.
About 13% of the test cohort fails this constraint outright (PD > 8%)
before any limit-specific trade-off is even considered.

A customer with no feasible candidate is DECLINED, not assigned the
smallest grid value -- an infeasible customer at every candidate limit is
a real, legitimate business outcome (matches how real underwriting works:
some applicants get declined, not "approved for the minimum by default").

Run: python -m credit_limit_optimizer.optimization.customer_optimizer
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_limit_optimizer.models.expected_loss import load_full_feature_table
from credit_limit_optimizer.models.profitability import (
    REQUIRED_COLUMNS, candidate_limit_grid, compute_pd, evaluate_at_limit,
)
from credit_limit_optimizer.models.risk_model import split_cohorts
from credit_limit_optimizer.utils.config import PROJECT_ROOT, load_config
from credit_limit_optimizer.utils.logging import get_logger

log = get_logger("customer_optimizer")

FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "optimization"
REPORT_PATH = PROJECT_ROOT / "reports" / "outputs" / "customer_optimization_report.json"
MODEL_CARD_PATH = PROJECT_ROOT / "reports" / "model_cards" / "customer_optimization.md"

OPTIMIZER_REQUIRED_COLUMNS = REQUIRED_COLUMNS + ["monthly_income"]
CONSTRAINT_COLUMNS = ["satisfies_income", "satisfies_risk", "satisfies_utilization", "satisfies_debt"]

plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def dsr_at_limit(balance: np.ndarray, monthly_income: np.ndarray) -> np.ndarray:
    min_payment = np.maximum(balance * 0.03, np.where(balance > 0, 25.0, 0.0))
    return min_payment / np.maximum(monthly_income, 1.0)


def evaluate_candidates(df: pd.DataFrame, config: dict, pd_values: np.ndarray) -> pd.DataFrame:
    """Long table: one row per (customer, candidate_limit)."""
    risk = config["risk"]
    grid = candidate_limit_grid(config)
    frames = []
    for L in grid:
        result = evaluate_at_limit(df, L, pd_values, config)
        result["candidate_limit"] = L
        result["dsr"] = dsr_at_limit(result["balance"].to_numpy(), df["monthly_income"].to_numpy())
        result["customer_id"] = df["customer_id"].to_numpy()
        result["monthly_income"] = df["monthly_income"].to_numpy()
        result["pd"] = pd_values
        frames.append(result)
    table = pd.concat(frames, ignore_index=True)

    table["satisfies_income"] = table["candidate_limit"] <= table["monthly_income"] * risk["income_multiple"]
    table["satisfies_risk"] = table["pd"] <= risk["max_pd"]
    table["satisfies_utilization"] = table["utilization"] <= risk["max_utilization"]
    table["satisfies_debt"] = table["dsr"] <= risk["max_dsr"]
    table["feasible"] = table[CONSTRAINT_COLUMNS].all(axis=1)
    return table


def _select_optimal(feasible: pd.DataFrame, current_limit_by_customer: pd.Series, tolerance: float = 0.01) -> pd.DataFrame:
    """Among each customer's feasible candidates, pick the one maximizing
    Expected Profit -- but when several candidates are tied (within
    `tolerance` EUR of the max, e.g. a zero-balance customer whose profit
    doesn't depend on the limit at all), break the tie toward whichever
    tied candidate is closest to the customer's CURRENT limit, not
    whichever happens to sort first. `groupby().idxmax()` picks the
    first-occurring max, which for a genuinely flat profit curve silently
    means "smallest candidate in the grid" -- a real, misleading "cut
    this customer's limit" recommendation with zero economic basis behind
    it, found while inspecting the limit-change distribution (~40% of all
    "decrease" recommendations turned out to be exactly this artifact)."""
    max_profit = feasible.groupby("customer_id")["expected_profit"].transform("max")
    tied = feasible[feasible["expected_profit"] >= max_profit - tolerance].copy()
    tied["current_limit"] = tied["customer_id"].map(current_limit_by_customer)
    tied["distance_to_current"] = (tied["candidate_limit"] - tied["current_limit"]).abs()
    best_idx = tied.groupby("customer_id")["distance_to_current"].idxmin()
    return feasible.loc[best_idx].set_index("customer_id")


def optimize_individual(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    pd_values = compute_pd(df)
    long_table = evaluate_candidates(df, config, pd_values)

    feasible = long_table[long_table["feasible"]]
    current_limit_by_customer = df.set_index("customer_id")["credit_exposure"]
    optimal = _select_optimal(feasible, current_limit_by_customer)

    declined_ids = set(df["customer_id"]) - set(optimal.index)
    never_satisfiable = long_table.groupby("customer_id")[CONSTRAINT_COLUMNS].any()
    decline_reasons = {
        cid: [c.replace("satisfies_", "") for c in CONSTRAINT_COLUMNS if not never_satisfiable.loc[cid, c]]
        for cid in declined_ids
    }

    result = df.set_index("customer_id")[["customer_segment", "credit_exposure", "monthly_income"]].copy()
    result["optimal_limit"] = optimal["candidate_limit"]
    result["expected_profit_at_optimal"] = optimal["expected_profit"]
    result["declined"] = result["optimal_limit"].isna()
    result["decline_reasons"] = result.index.map(lambda cid: decline_reasons.get(cid, []))
    result = result.reset_index()
    return result, long_table


def plot_approval_and_decline_reasons(result: pd.DataFrame) -> None:
    declined = result[result["declined"]]
    reason_counts = pd.Series([r for reasons in declined["decline_reasons"] for r in reasons]).value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    approved_n, declined_n = (~result["declined"]).sum(), result["declined"].sum()
    axes[0].bar(["Approved", "Declined"], [approved_n, declined_n], color=["#2E7D32", "#C62828"])
    axes[0].set_title(f"Approval outcome (n={len(result):,})")
    axes[0].set_ylabel("Customers")
    if len(reason_counts) > 0:
        axes[1].bar(reason_counts.index, reason_counts.values, color="#C62828")
    axes[1].set_title("Decline reasons (constraint never satisfiable)")
    axes[1].set_ylabel("Customers")
    fig.savefig(FIG_DIR / "01_approval_and_decline_reasons.png", bbox_inches="tight")
    plt.close(fig)


def plot_limit_change(result: pd.DataFrame) -> None:
    approved = result[~result["declined"]].copy()
    approved["change"] = approved["optimal_limit"] - approved["credit_exposure"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(approved["change"], bins=40, color="#1565C0")
    ax.axvline(0, color="#9E9E9E", linestyle="--", linewidth=1)
    ax.set_xlabel("Optimal limit − current limit (EUR)")
    ax.set_ylabel("Approved customers")
    ax.set_title("Change from current to optimized limit")
    fig.savefig(FIG_DIR / "02_limit_change_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_profit_uplift_by_segment(uplift_by_segment: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    segments = ["prime", "near_prime", "subprime"]
    ax.bar(segments, [uplift_by_segment[s] for s in segments], color=["#2E7D32", "#F9A825", "#C62828"])
    ax.set_ylabel("Mean Expected Profit uplift (EUR/year)")
    ax.set_title("Optimized vs. current-limit profit, by segment (approved customers)")
    fig.savefig(FIG_DIR / "03_profit_uplift_by_segment.png", bbox_inches="tight")
    plt.close(fig)


def run_customer_optimization() -> dict:
    config = load_config()
    df = load_full_feature_table()
    cohorts = split_cohorts(df)
    test = cohorts["test"]

    n_before = len(test)
    test = test.dropna(subset=OPTIMIZER_REQUIRED_COLUMNS)
    n_excluded = n_before - len(test)
    log.info("Excluding %d of %d test rows (missing optimizer inputs)", n_excluded, n_before)

    result, long_table = optimize_individual(test, config)
    approval_rate = float((~result["declined"]).mean())
    log.info("Approval rate: %.1f%%", approval_rate * 100)

    # Compare against Expected Profit AT THE CUSTOMER'S CURRENT LIMIT, so
    # "uplift" isolates the optimizer's contribution from the elasticity
    # response model itself (both numbers come from the same model).
    pd_values_all = compute_pd(test)
    current_state = evaluate_at_limit(test, test["credit_exposure"].to_numpy(), pd_values_all, config)
    current_profit = current_state.set_index(test["customer_id"])["expected_profit"]

    approved = result[~result["declined"]].copy()
    approved["current_profit"] = approved["customer_id"].map(current_profit)
    approved["uplift"] = approved["expected_profit_at_optimal"] - approved["current_profit"]

    uplift_by_segment = approved.groupby("customer_segment")["uplift"].mean().to_dict()
    log.info("Mean profit uplift by segment: %s", {k: round(v, 2) for k, v in uplift_by_segment.items()})
    log.info(
        "Portfolio: mean uplift=€%.2f/year, total=€%.0f/year (approved customers only)",
        approved["uplift"].mean(), approved["uplift"].sum(),
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_approval_and_decline_reasons(result)
    plot_limit_change(approved)
    plot_profit_uplift_by_segment({k: float(v) for k, v in uplift_by_segment.items()})

    decline_reason_counts = pd.Series(
        [r for reasons in result.loc[result["declined"], "decline_reasons"] for r in reasons]
    ).value_counts().to_dict()

    decreased = approved[approved["optimal_limit"] < approved["credit_exposure"]]
    grid_max = config["optimization"]["maximum_limit"]
    fraction_decrease_from_grid_cap = float((decreased["credit_exposure"] > grid_max).mean()) if len(decreased) else 0.0

    report = {
        "n_customers": int(len(result)), "n_excluded_missing_inputs": n_excluded,
        "approval_rate": approval_rate,
        "n_approved": int((~result["declined"]).sum()), "n_declined": int(result["declined"].sum()),
        "decline_reason_counts": {k: int(v) for k, v in decline_reason_counts.items()},
        "mean_profit_uplift": float(approved["uplift"].mean()),
        "total_profit_uplift": float(approved["uplift"].sum()),
        "uplift_by_segment": {k: float(v) for k, v in uplift_by_segment.items()},
        "fraction_limit_increased": float((approved["optimal_limit"] > approved["credit_exposure"]).mean()),
        "fraction_limit_decreased": float((approved["optimal_limit"] < approved["credit_exposure"]).mean()),
        "fraction_limit_unchanged": float((approved["optimal_limit"] == approved["credit_exposure"]).mean()),
        "n_decreased": int(len(decreased)),
        "fraction_decrease_from_grid_cap": fraction_decrease_from_grid_cap,
        "candidate_grid_max": grid_max,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Wrote %s", REPORT_PATH.relative_to(PROJECT_ROOT))

    write_model_card(report)
    log.info("Wrote %s", MODEL_CARD_PATH.relative_to(PROJECT_ROOT))
    return report


def write_model_card(report: dict) -> None:
    lines = [
        "# Model Card — Individual Credit Limit Optimization (§23-24)",
        "",
        "For each customer, evaluate every candidate limit and select the feasible one "
        "maximizing Expected Profit, subject to income/risk/utilization/debt constraints "
        "(`config/settings.yaml`'s `risk.*`). A customer with no feasible candidate is "
        "declined, not assigned the smallest grid value.",
        "",
        f"{report['n_excluded_missing_inputs']:,} of "
        f"{report['n_customers'] + report['n_excluded_missing_inputs']:,} test rows excluded "
        "(missing optimizer inputs — same real Phase 1 data-quality artifacts as the other "
        "Phase 4/5 pieces).",
        "",
        "## Approval outcome",
        "",
        f"**{report['approval_rate']:.1%} approved** ({report['n_approved']:,} of "
        f"{report['n_customers']:,}), {report['n_declined']:,} declined. Decline reasons "
        "(a decline can have more than one — the constraint is never satisfiable at ANY "
        "candidate limit, not just tight at the chosen one):",
        "",
    ]
    for reason, count in report["decline_reason_counts"].items():
        lines.append(f"- **{reason}**: {count:,} customers")
    lines += [
        "",
        "![approval and decline reasons](../figures/optimization/01_approval_and_decline_reasons.png)",
        "",
        "## Limit change vs. current",
        "",
        f"Of approved customers: {report['fraction_limit_increased']:.1%} get a higher limit, "
        f"{report['fraction_limit_decreased']:.1%} a lower one, "
        f"{report['fraction_limit_unchanged']:.1%} unchanged. Expected Profit is monotonically "
        "increasing in the candidate limit under this balance-response model (§22), so the "
        "optimizer generally pushes toward the largest FEASIBLE candidate — a direct "
        "consequence of that model's shape, not a nuanced trade-off discovery; ties (e.g. a "
        "zero-balance customer, whose profit doesn't depend on the limit at all) are broken "
        "toward the candidate closest to the customer's current limit, not an arbitrary one "
        "(see module docstring for a real bug this fixed). Of the "
        f"{report['n_decreased']:,} customers who DO get a lower limit, "
        f"{report['fraction_decrease_from_grid_cap']:.1%} already have a current limit above "
        f"the candidate grid's own maximum (€{report['candidate_grid_max']:,}) — their "
        "\"decrease\" is the grid range capping the recommendation, not the model concluding "
        "their limit should shrink; the remainder are small, mostly single-grid-step "
        "differences from the current limit not landing exactly on a €500 candidate.",
        "",
        "![limit change distribution](../figures/optimization/02_limit_change_distribution.png)",
        "",
        "## Expected Profit uplift (approved customers, vs. current limit)",
        "",
        f"Mean uplift €{report['mean_profit_uplift']:.2f}/year per approved customer; total "
        f"€{report['total_profit_uplift']:,.0f}/year across the approved test cohort.",
        "",
    ]
    for seg in ["prime", "near_prime", "subprime"]:
        lines.append(f"- **{seg}**: €{report['uplift_by_segment'][seg]:.2f}/year")
    lines += ["", "![profit uplift by segment](../figures/optimization/03_profit_uplift_by_segment.png)", ""]
    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_CARD_PATH.open("w") as f:
        f.write("\n".join(lines))


def main() -> None:
    run_customer_optimization()


if __name__ == "__main__":
    main()
