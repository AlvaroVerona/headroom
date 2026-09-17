"""Portfolio credit limit optimization (§25): maximize total expected
portfolio profit subject to portfolio-level constraints, using OR-Tools.

```
Maximize   sum(expected_profit_i * x_i)
subject to sum(limit_i * x_i)          <= portfolio_exposure_limit
           sum(expected_loss_i * x_i)  <= portfolio_expected_loss_limit
           sum(PD_i * x_i) / sum(x_i)  <= max_average_pd
           high_risk_exposure_i * x_i / sum(limit_i * x_i) <= max_high_risk_exposure_pct
           x_i in {0, 1}
```

**Two-stage design, not a joint (customer x candidate) MIP**: stage 1
(`customer_optimizer.py`, already run) picks each customer's individually
-optimal limit subject to customer-level constraints (income/risk/
utilization/debt) -- one (limit, profit, EAD, PD) tuple per approved
customer. Stage 2 (this module) is a portfolio-level 0/1 knapsack: given
those already-fixed tuples, decide WHICH approved customers to actually
fund so the AGGREGATE portfolio respects its own budget constraints,
maximizing total profit. This mirrors how real risk-based portfolio
triage actually works (score individually first, then allocate scarce
risk budget across the approved population) and keeps the problem to one
binary variable per customer (~38,900) rather than one per
(customer, candidate) pair (~500,000+) -- the latter would also let the
solver pick a DIFFERENT, smaller limit to help a customer fit the
budget, which this simplification deliberately doesn't support. A
documented modeling choice, not an oversight.

**Why the ratio constraints (avg PD, high-risk exposure %) are still
linear**: both have a denominator that depends on x (sum(x_i) or
sum(limit_i * x_i)) -- not linear as written, but PD_i and limit_i are
fixed per customer (from stage 1), so multiplying through gives an
exactly equivalent linear form:
```
sum(PD_i * x_i) <= max_average_pd * sum(x_i)
  =>  sum((PD_i - max_average_pd) * x_i) <= 0

sum(limit_i * x_i for i in high_risk) <= max_high_risk_exposure_pct * sum(limit_i * x_i for all i)
  =>  sum(limit_i * (is_high_risk_i - max_high_risk_exposure_pct) * x_i) <= 0
```
both genuinely linear in x, not an approximation.

**"High-risk" definition**: the `subprime` segment -- config has no
separate high-risk PD threshold, and `subprime` is the categorical risk
tier already used consistently everywhere else in this project (EDA,
Expected Loss, Revenue, Profitability), so reusing it here keeps the
constraint auditable rather than introducing a new, unexplained cutoff.

Run: python -m credit_limit_optimizer.optimization.portfolio_optimizer
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from credit_limit_optimizer.models.expected_loss import load_full_feature_table
from credit_limit_optimizer.models.risk_model import split_cohorts
from credit_limit_optimizer.optimization.customer_optimizer import OPTIMIZER_REQUIRED_COLUMNS, optimize_individual
from credit_limit_optimizer.utils.config import PROJECT_ROOT, load_config
from credit_limit_optimizer.utils.logging import get_logger

log = get_logger("portfolio_optimizer")

FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "optimization"
REPORT_PATH = PROJECT_ROOT / "reports" / "outputs" / "portfolio_optimization_report.json"
MODEL_CARD_PATH = PROJECT_ROOT / "reports" / "model_cards" / "portfolio_optimization.md"

plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def build_candidate_pool(config: dict) -> pd.DataFrame:
    """Each individually-approved customer's single chosen (limit,
    profit, EL, PD) tuple from stage 1 -- the pool stage 2 selects from."""
    df = load_full_feature_table()
    test = split_cohorts(df)["test"].dropna(subset=OPTIMIZER_REQUIRED_COLUMNS)
    result, long_table = optimize_individual(test, config)
    approved = result[~result["declined"]][["customer_id", "customer_segment", "optimal_limit"]]
    chosen = long_table.merge(approved, on="customer_id").query("candidate_limit == optimal_limit")
    pool = chosen[["customer_id", "customer_segment", "candidate_limit", "expected_profit", "expected_loss", "pd"]].copy()
    pool = pool.rename(columns={"candidate_limit": "limit", "pd": "pd_value"})
    pool["is_high_risk"] = (pool["customer_segment"] == "subprime").astype(float)
    return pool.reset_index(drop=True)


def solve_portfolio(pool: pd.DataFrame, config: dict) -> dict:
    opt = config["optimization"]
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("OR-Tools CBC solver unavailable")

    n = len(pool)
    limit = pool["limit"].to_numpy()
    profit = pool["expected_profit"].to_numpy()
    el = pool["expected_loss"].to_numpy()
    pd_value = pool["pd_value"].to_numpy()
    high_risk = pool["is_high_risk"].to_numpy()

    x = [solver.BoolVar(f"x_{i}") for i in range(n)]

    solver.Add(solver.Sum(limit[i] * x[i] for i in range(n)) <= opt["portfolio_exposure_limit"])
    solver.Add(solver.Sum(el[i] * x[i] for i in range(n)) <= opt["portfolio_expected_loss_limit"])
    solver.Add(solver.Sum((pd_value[i] - opt["max_average_pd"]) * x[i] for i in range(n)) <= 0)
    solver.Add(solver.Sum(limit[i] * (high_risk[i] - opt["max_high_risk_exposure_pct"]) * x[i] for i in range(n)) <= 0)

    solver.Maximize(solver.Sum(profit[i] * x[i] for i in range(n)))

    start = time.monotonic()
    status = solver.Solve()
    elapsed = time.monotonic() - start

    status_name = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE", pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    }.get(status, f"UNKNOWN({status})")
    log.info("OR-Tools status=%s in %.1fs (n=%d binary variables, 4 constraints)", status_name, elapsed, n)
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"Portfolio MIP did not solve: status={status_name}")

    selected = np.array([x[i].solution_value() > 0.5 for i in range(n)])
    return {"selected": selected, "status": status_name, "solve_seconds": elapsed, "objective_value": float(solver.Objective().Value())}


def greedy_baseline(pool: pd.DataFrame, config: dict) -> np.ndarray:
    """Classic knapsack greedy: sort by profit-per-euro-of-exposure
    descending, add while every constraint stays satisfied. Used only to
    verify the MIP is at least as good, not as the real policy."""
    opt = config["optimization"]
    order = (pool["expected_profit"] / pool["limit"].clip(lower=1)).sort_values(ascending=False).index
    selected = np.zeros(len(pool), dtype=bool)
    exposure = el = pd_sum = high_risk_exposure = n_selected = 0.0
    for i in order:
        row = pool.loc[i]
        new_exposure = exposure + row["limit"]
        new_el = el + row["expected_loss"]
        new_pd_sum = pd_sum + row["pd_value"]
        new_n = n_selected + 1
        new_high_risk_exposure = high_risk_exposure + row["limit"] * row["is_high_risk"]
        if (
            new_exposure <= opt["portfolio_exposure_limit"]
            and new_el <= opt["portfolio_expected_loss_limit"]
            and new_pd_sum / new_n <= opt["max_average_pd"]
            and (new_high_risk_exposure / new_exposure if new_exposure > 0 else 0) <= opt["max_high_risk_exposure_pct"]
        ):
            selected[pool.index.get_loc(i)] = True
            exposure, el, pd_sum, high_risk_exposure, n_selected = new_exposure, new_el, new_pd_sum, new_high_risk_exposure, new_n
    return selected


def portfolio_metrics(pool: pd.DataFrame, selected: np.ndarray) -> dict:
    sub = pool[selected]
    total_exposure = float(sub["limit"].sum())
    return {
        "n_funded": int(selected.sum()), "n_pool": int(len(pool)),
        "total_exposure": total_exposure,
        "total_expected_loss": float(sub["expected_loss"].sum()),
        "total_expected_profit": float(sub["expected_profit"].sum()),
        "average_pd": float(sub["pd_value"].mean()) if len(sub) else 0.0,
        "high_risk_exposure_pct": float((sub["limit"] * sub["is_high_risk"]).sum() / total_exposure) if total_exposure > 0 else 0.0,
    }


def plot_funded_by_segment(pool: pd.DataFrame, selected: np.ndarray) -> None:
    pool = pool.copy()
    pool["funded"] = selected
    counts = pool.groupby(["customer_segment", "funded"]).size().unstack(fill_value=0)
    segments = ["prime", "near_prime", "subprime"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(segments))
    for funded, color, label in [(True, "#2E7D32", "Funded"), (False, "#C62828", "Not funded (budget)")]:
        values = [counts.loc[s, funded] if (s in counts.index and funded in counts.columns) else 0 for s in segments]
        ax.bar(segments, values, bottom=bottom, label=label, color=color)
        bottom += np.array(values)
    ax.set_ylabel("Customers")
    ax.set_title("Portfolio funding outcome by segment")
    ax.legend()
    fig.savefig(FIG_DIR / "04_portfolio_funded_by_segment.png", bbox_inches="tight")
    plt.close(fig)


def plot_constraint_utilization(mip_metrics: dict, config: dict) -> None:
    opt = config["optimization"]
    constraints = [
        ("Exposure", mip_metrics["total_exposure"], opt["portfolio_exposure_limit"]),
        ("Expected Loss", mip_metrics["total_expected_loss"], opt["portfolio_expected_loss_limit"]),
        ("Avg PD", mip_metrics["average_pd"], opt["max_average_pd"]),
        ("High-risk exposure %", mip_metrics["high_risk_exposure_pct"], opt["max_high_risk_exposure_pct"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [c[0] for c in constraints]
    utilization = [c[1] / c[2] for c in constraints]
    colors = ["#C62828" if u >= 0.999 else "#F9A825" if u >= 0.9 else "#2E7D32" for u in utilization]
    ax.barh(labels, utilization, color=colors)
    ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1, label="Limit")
    ax.set_xlabel("Fraction of portfolio limit used")
    ax.set_title("Constraint utilization at the MIP-optimal solution")
    ax.legend()
    fig.savefig(FIG_DIR / "05_constraint_utilization.png", bbox_inches="tight")
    plt.close(fig)


def profit_efficiency_by_segment(pool: pd.DataFrame) -> dict:
    """Mean profit-per-euro-of-exposure by segment -- the quantity the
    exposure-constrained MIP is implicitly maximizing, which explains WHY
    it favors some segments over others (see write_model_card)."""
    eff = pool.assign(profit_per_exposure=pool["expected_profit"] / pool["limit"])
    return eff.groupby("customer_segment")["profit_per_exposure"].mean().to_dict()


def run_portfolio_optimization() -> dict:
    config = load_config()
    pool = build_candidate_pool(config)
    log.info("Candidate pool: %d individually-approved customers", len(pool))

    unconstrained = portfolio_metrics(pool, np.ones(len(pool), dtype=bool))
    log.info(
        "Unconstrained (fund everyone individually approved): exposure=€%.0f EL=€%.0f avg_pd=%.3f high_risk_pct=%.3f",
        unconstrained["total_exposure"], unconstrained["total_expected_loss"],
        unconstrained["average_pd"], unconstrained["high_risk_exposure_pct"],
    )

    solution = solve_portfolio(pool, config)
    selected = solution["selected"]
    mip_metrics = portfolio_metrics(pool, selected)
    log.info(
        "MIP-optimal: funded=%d/%d profit=€%.0f exposure=€%.0f EL=€%.0f avg_pd=%.3f high_risk_pct=%.3f",
        mip_metrics["n_funded"], mip_metrics["n_pool"], mip_metrics["total_expected_profit"],
        mip_metrics["total_exposure"], mip_metrics["total_expected_loss"],
        mip_metrics["average_pd"], mip_metrics["high_risk_exposure_pct"],
    )

    greedy_selected = greedy_baseline(pool, config)
    greedy_metrics = portfolio_metrics(pool, greedy_selected)
    log.info(
        "Greedy baseline: funded=%d/%d profit=€%.0f (MIP improvement: €%.0f, %.2f%%)",
        greedy_metrics["n_funded"], greedy_metrics["n_pool"], greedy_metrics["total_expected_profit"],
        mip_metrics["total_expected_profit"] - greedy_metrics["total_expected_profit"],
        100 * (mip_metrics["total_expected_profit"] / greedy_metrics["total_expected_profit"] - 1) if greedy_metrics["total_expected_profit"] else 0,
    )

    efficiency = profit_efficiency_by_segment(pool)
    funded_rate_by_segment = pool.assign(funded=selected).groupby("customer_segment")["funded"].mean().to_dict()
    log.info("Profit-per-exposure-euro by segment: %s", {k: round(v, 4) for k, v in efficiency.items()})
    log.info("Funded rate by segment: %s", {k: round(v, 3) for k, v in funded_rate_by_segment.items()})

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_funded_by_segment(pool, selected)
    plot_constraint_utilization(mip_metrics, config)

    report = {
        "solver_status": solution["status"], "solve_seconds": solution["solve_seconds"],
        "profit_efficiency_by_segment": {k: float(v) for k, v in efficiency.items()},
        "funded_rate_by_segment": {k: float(v) for k, v in funded_rate_by_segment.items()},
        "unconstrained_metrics": unconstrained,
        "mip_optimal_metrics": mip_metrics,
        "greedy_baseline_metrics": greedy_metrics,
        "mip_vs_greedy_profit_improvement": mip_metrics["total_expected_profit"] - greedy_metrics["total_expected_profit"],
        "constraints": {
            "portfolio_exposure_limit": config["optimization"]["portfolio_exposure_limit"],
            "portfolio_expected_loss_limit": config["optimization"]["portfolio_expected_loss_limit"],
            "max_average_pd": config["optimization"]["max_average_pd"],
            "max_high_risk_exposure_pct": config["optimization"]["max_high_risk_exposure_pct"],
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Wrote %s", REPORT_PATH.relative_to(PROJECT_ROOT))

    write_model_card(report)
    log.info("Wrote %s", MODEL_CARD_PATH.relative_to(PROJECT_ROOT))
    return report


def write_model_card(report: dict) -> None:
    u, m, g, c = report["unconstrained_metrics"], report["mip_optimal_metrics"], report["greedy_baseline_metrics"], report["constraints"]
    lines = [
        "# Model Card — Portfolio Credit Limit Optimization (§25)",
        "",
        "Maximize total Expected Profit across individually-approved customers (stage 1: "
        "`customer_optimizer.py`), subject to portfolio-level exposure/loss/risk "
        "constraints, using OR-Tools' CBC MIP solver. See module docstring for the exact "
        "linearized formulation and the two-stage design rationale.",
        "",
        f"Solved to **{report['solver_status']}** in {report['solve_seconds']:.1f}s "
        f"({m['n_pool']:,} binary variables, 4 constraints).",
        "",
        "## Why this constraint set is not trivial",
        "",
        f"Funding every individually-approved customer at their individually-optimal limit "
        f"would need €{u['total_exposure']:,.0f} of exposure and reach "
        f"{u['high_risk_exposure_pct']:.1%} high-risk exposure — both **over** the portfolio "
        f"limits (€{c['portfolio_exposure_limit']:,.0f} and "
        f"{c['max_high_risk_exposure_pct']:.0%} respectively). Expected Loss "
        f"(€{u['total_expected_loss']:,.0f} vs. €{c['portfolio_expected_loss_limit']:,.0f} "
        f"limit) and average PD ({u['average_pd']:.2%} vs. {c['max_average_pd']:.0%} limit) "
        "both have slack — the binding constraints are exposure and high-risk concentration, "
        "not credit risk per se, so the solver has a genuine trade-off to resolve.",
        "",
        "## MIP-optimal solution",
        "",
        "| Metric | Unconstrained (fund all approved) | MIP-optimal | Limit |",
        "|---|---|---|---|",
        f"| Customers funded | {u['n_funded']:,} | {m['n_funded']:,} | — |",
        f"| Total exposure | €{u['total_exposure']:,.0f} | €{m['total_exposure']:,.0f} | €{c['portfolio_exposure_limit']:,.0f} |",
        f"| Total Expected Loss | €{u['total_expected_loss']:,.0f} | €{m['total_expected_loss']:,.0f} | €{c['portfolio_expected_loss_limit']:,.0f} |",
        f"| Average PD | {u['average_pd']:.2%} | {m['average_pd']:.2%} | {c['max_average_pd']:.0%} |",
        f"| High-risk exposure % | {u['high_risk_exposure_pct']:.1%} | {m['high_risk_exposure_pct']:.1%} | {c['max_high_risk_exposure_pct']:.0%} |",
        f"| **Total Expected Profit/year** | €{u['total_expected_profit']:,.0f} | **€{m['total_expected_profit']:,.0f}** | — |",
        "",
        "![constraint utilization](../figures/optimization/05_constraint_utilization.png)",
        "",
        "## Why funding rate isn't just \"safest first\"",
        "",
        "![funded by segment](../figures/optimization/04_portfolio_funded_by_segment.png)",
        "",
        f"Funded rate: prime {report['funded_rate_by_segment']['prime']:.0%}, near_prime "
        f"{report['funded_rate_by_segment']['near_prime']:.0%}, subprime "
        f"{report['funded_rate_by_segment']['subprime']:.0%} — prime, the LOWEST-risk "
        "segment, gets funded the LEAST. Exposure is the binding constraint here, not risk, "
        "so the solver implicitly maximizes profit *per euro of scarce exposure budget*: "
        f"prime averages only €{report['profit_efficiency_by_segment']['prime']:.3f} profit "
        f"per €1 of limit, vs. €{report['profit_efficiency_by_segment']['near_prime']:.3f} "
        f"for near_prime and €{report['profit_efficiency_by_segment']['subprime']:.3f} for "
        "subprime (prime customers tend toward large individually-optimal limits with "
        "comparatively modest absolute profit — see `profitability.py`). Subprime, despite "
        "being the MOST profit-efficient per euro, doesn't win the largest funded share "
        "either — its concentration is directly capped by the high-risk exposure "
        "constraint, which is *also* binding. Near_prime ends up with the most headroom "
        "under both binding constraints at once, hence the highest funded rate. A real, "
        "non-obvious three-way interaction the constraints were built to create.",
        "",
        "## MIP vs. greedy baseline",
        "",
        f"A greedy heuristic (sort by profit-per-euro-of-exposure, add while every "
        f"constraint holds) reaches €{g['total_expected_profit']:,.0f}/year funding "
        f"{g['n_funded']:,} customers. The MIP reaches "
        f"€{m['total_expected_profit']:,.0f}/year "
        f"(+€{report['mip_vs_greedy_profit_improvement']:,.0f}, "
        f"{'a real improvement' if report['mip_vs_greedy_profit_improvement'] > 0 else 'matching the greedy result'}) "
        "funding the AGGREGATE budget jointly across all four constraints at once, rather "
        "than a single per-customer ratio that ignores which constraint actually binds.",
        "",
    ]
    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_CARD_PATH.open("w") as f:
        f.write("\n".join(lines))


def main() -> None:
    run_portfolio_optimization()


if __name__ == "__main__":
    main()
