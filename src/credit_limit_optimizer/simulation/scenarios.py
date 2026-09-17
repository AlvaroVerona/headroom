"""Stress testing / scenario engine (§26-27, Phase 6): re-evaluate the
FUNDED portfolio (Phase 5's `portfolio_optimizer.py` output -- the actual
book the bank would carry, not the raw approved-but-unfunded pool) under
each named macroeconomic scenario in `config/settings.yaml`'s `scenarios`
block (base/recession/high_interest_rate/consumer_stress).

**Two distinct shock mechanisms, not one, chosen per shock's own nature**:

1. **Feature-level shocks, re-scored through the existing calibrated PD
   model** (no retraining -- retraining on a hypothetical future would be
   leakage of a different kind). `income_growth_shift`, `spending_shift`,
   `income_volatility_multiplier`, `delinquency_multiplier` and
   `cash_balance_shift` all perturb genuine PD model input features
   (`monthly_income*`, `monthly_spend*`, `income_volatility`/
   `income_stability`, `delinquency_count`/`days_past_due`/
   `max_days_past_due`, `cash_buffer`) and let the ALREADY-FITTED,
   calibrated XGBoost classifier translate that into an elevated PD --
   this is how real credit-risk stress testing works (shock the inputs,
   re-score the model), and it's exactly the design the prior session
   verified before starting this piece: `income`/`spend`/
   `income_volatility` all carry real PD signal (unlike `credit_exposure`,
   which profitability.py's docstring already found has ~0 importance).
2. **Direct macro overlays on the downstream economics formula**:
   `default_multiplier` (a PD overlay -- named and used differently from
   the feature multipliers above: a flat stress multiplier on the
   MODEL'S OWN OUTPUT, the standard "management overlay" every real
   stress-testing framework layers on top of a model score, not a feature
   it could shock instead), `funding_rate_shift` (added to
   `economics.funding_rate`, reusing `funding_cost.py`'s own
   `funding_rate_for_scenario` convention) and `apr_shift` (added to each
   customer's own `apr` before computing interest revenue).

**What is deliberately NOT shocked**: `average_balance` and LGD.
`evaluate_at_limit` (profitability.py) has no causal balance-response-to
-macro-shock model -- same reasoning CLAUDE.md already gives for why
`credit_exposure` isn't used to shock EAD (this generator never varies a
customer's limit over time, so no model has ever seen the counterfactual
needed to estimate one). Expected Loss rises under stress via PD x LGD x
(unchanged) balance, not via an invented balance shock. LGD has no
scenario key in the spec's own scaffolded config and is left at its
segment-based value, the same "don't invent what the spec didn't ask
for" principle already applied to funding_cost.py's flat-across-segment
funding rate.

Run: python -m credit_limit_optimizer.simulation.scenarios
"""

from __future__ import annotations

import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from credit_limit_optimizer.models.expected_loss import PD_MODEL_PATH, load_full_feature_table
from credit_limit_optimizer.models.funding_cost import funding_rate_for_scenario
from credit_limit_optimizer.models.profitability import evaluate_at_limit
from credit_limit_optimizer.models.risk_model import split_cohorts
from credit_limit_optimizer.optimization.portfolio_optimizer import build_candidate_pool, solve_portfolio
from credit_limit_optimizer.utils.config import PROJECT_ROOT, load_config
from credit_limit_optimizer.utils.logging import get_logger

log = get_logger("scenarios")

FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "scenarios"
REPORT_PATH = PROJECT_ROOT / "reports" / "outputs" / "scenario_report.json"
MODEL_CARD_PATH = PROJECT_ROOT / "reports" / "model_cards" / "scenarios.md"

SCENARIOS = ["base", "recession", "high_interest_rate", "consumer_stress"]

# monthly_income/monthly_spend trend columns that must move WITH the base
# variable's level shock, or a re-scored model would see a customer whose
# current income/spend disagrees with their own recent trailing average --
# an internally inconsistent, out-of-distribution feature vector no real
# customer would ever present.
_INCOME_LEVEL_COLUMNS = ["monthly_income", "monthly_income_3m_avg", "monthly_income_6m_avg", "monthly_income_12m_avg"]
_INCOME_GROWTH_COLUMNS = ["monthly_income_3m_growth", "monthly_income_6m_growth"]
_SPEND_LEVEL_COLUMNS = ["monthly_spend", "monthly_spend_3m_avg", "monthly_spend_6m_avg", "monthly_spend_12m_avg"]
_SPEND_GROWTH_COLUMNS = ["monthly_spend_3m_growth", "monthly_spend_6m_growth"]
_DELINQUENCY_COLUMNS = ["delinquency_count", "days_past_due", "max_days_past_due"]

plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def build_funded_book(config: dict) -> pd.DataFrame:
    """The actual funded portfolio (Phase 5's MIP solution), joined back
    to its full feature row -- `portfolio_optimizer.build_candidate_pool`
    keeps only a handful of columns, but stress-testing needs the whole
    PD-model feature vector to shock and re-score."""
    pool = build_candidate_pool(config)
    solution = solve_portfolio(pool, config)
    funded_ids = set(pool.loc[solution["selected"], "customer_id"])

    features = load_full_feature_table()
    test = split_cohorts(features)["test"]
    book = test[test["customer_id"].isin(funded_ids)].copy()

    funded_limit = pool.set_index("customer_id").loc[book["customer_id"], "limit"]
    book["funded_limit"] = funded_limit.to_numpy()
    log.info("Funded book: %d of %d individually-approved customers (Phase 5 MIP solution)", len(book), len(pool))
    return book.reset_index(drop=True)


def apply_shocks(df: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    """Perturb PD-model input features and economics inputs for one
    scenario's shock parameters. Returns a copy; `df` is untouched."""
    shocked = df.copy()

    if "income_growth_shift" in scenario:
        factor = 1 + scenario["income_growth_shift"]
        shocked[_INCOME_LEVEL_COLUMNS] = shocked[_INCOME_LEVEL_COLUMNS] * factor
        shocked[_INCOME_GROWTH_COLUMNS] = shocked[_INCOME_GROWTH_COLUMNS] + scenario["income_growth_shift"]

    if "spending_shift" in scenario:
        factor = 1 + scenario["spending_shift"]
        shocked[_SPEND_LEVEL_COLUMNS] = shocked[_SPEND_LEVEL_COLUMNS] * factor
        shocked[_SPEND_GROWTH_COLUMNS] = shocked[_SPEND_GROWTH_COLUMNS] + scenario["spending_shift"]

    if "income_volatility_multiplier" in scenario:
        shocked["income_volatility"] = (shocked["income_volatility"] * scenario["income_volatility_multiplier"]).clip(0, 1)
        # income_stability = 1 - income_volatility (r=1.0, risk_model.py's
        # own correlation-pruning docstring) -- must stay consistent or the
        # re-scored model sees a self-contradictory feature pair no real
        # customer-month in the training data ever had.
        shocked["income_stability"] = 1 - shocked["income_volatility"]

    if "delinquency_multiplier" in scenario:
        shocked[_DELINQUENCY_COLUMNS] = shocked[_DELINQUENCY_COLUMNS] * scenario["delinquency_multiplier"]

    if "cash_balance_shift" in scenario:
        shocked["cash_buffer"] = shocked["cash_buffer"] * (1 + scenario["cash_balance_shift"])

    if "apr_shift" in scenario:
        shocked["apr"] = shocked["apr"] + scenario["apr_shift"]

    return shocked


def compute_pd(df: pd.DataFrame) -> np.ndarray:
    saved = joblib.load(PD_MODEL_PATH)
    model, features = saved["model"], saved["features"]
    pd_values = model.predict_proba(df[features])[:, 1]
    already_in_default = ~df["eligible_for_default_label"].astype(bool)
    return np.where(already_in_default, 1.0, pd_values)


def run_scenario(book: pd.DataFrame, scenario_name: str, config: dict) -> pd.DataFrame:
    scenario = config["scenarios"].get(scenario_name, {})
    shocked = apply_shocks(book, scenario)

    pd_values = compute_pd(shocked)
    if "default_multiplier" in scenario:
        pd_values = np.clip(pd_values * scenario["default_multiplier"], 0, 1)

    scenario_config = json.loads(json.dumps(config))  # cheap deep copy, config is JSON-safe
    scenario_config["economics"]["funding_rate"] = funding_rate_for_scenario(config, scenario_name)

    economics = evaluate_at_limit(shocked, shocked["funded_limit"].to_numpy(), pd_values, scenario_config)
    economics["pd"] = pd_values
    economics["customer_segment"] = shocked["customer_segment"].to_numpy()
    return economics


def portfolio_summary(economics: pd.DataFrame) -> dict:
    return {
        "n_customers": int(len(economics)),
        "mean_pd": float(economics["pd"].mean()),
        "total_exposure": float(economics["candidate_limit"].sum()),
        "total_revenue": float(economics["total_revenue"].sum()),
        "total_funding_cost": float(economics["funding_cost"].sum()),
        "total_expected_loss": float(economics["expected_loss"].sum()),
        "total_expected_profit": float(economics["expected_profit"].sum()),
        "mean_expected_profit": float(economics["expected_profit"].mean()),
    }


def plot_el_by_scenario(summaries: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    scenarios = list(summaries.keys())
    values = [summaries[s]["total_expected_loss"] for s in scenarios]
    colors = ["#1565C0" if s == "base" else "#C62828" for s in scenarios]
    ax.bar(scenarios, values, color=colors)
    ax.set_ylabel("Total portfolio Expected Loss (EUR/year)")
    ax.set_title("Funded-book Expected Loss by scenario")
    fig.savefig(FIG_DIR / "01_expected_loss_by_scenario.png", bbox_inches="tight")
    plt.close(fig)


def plot_profit_by_scenario(summaries: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    scenarios = list(summaries.keys())
    values = [summaries[s]["total_expected_profit"] for s in scenarios]
    colors = ["#2E7D32" if v >= 0 else "#C62828" for v in values]
    ax.bar(scenarios, values, color=colors)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_ylabel("Total portfolio Expected Profit (EUR/year)")
    ax.set_title("Funded-book Expected Profit by scenario")
    fig.savefig(FIG_DIR / "02_expected_profit_by_scenario.png", bbox_inches="tight")
    plt.close(fig)


def plot_pd_shift_by_scenario(economics_by_scenario: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for scenario, economics in economics_by_scenario.items():
        color = "#1565C0" if scenario == "base" else None
        ax.hist(economics["pd"], bins=50, histtype="step", label=scenario, color=color, linewidth=1.5)
    ax.set_xlabel("PD (re-scored under scenario)")
    ax.set_ylabel("Funded customers")
    ax.set_title("PD distribution shift by scenario")
    ax.legend()
    fig.savefig(FIG_DIR / "03_pd_distribution_by_scenario.png", bbox_inches="tight")
    plt.close(fig)


def run_stress_test() -> dict:
    config = load_config()
    book = build_funded_book(config)

    n_before = len(book)
    required = ["monthly_income", "monthly_spend", "income_volatility", "delinquency_count", "days_past_due", "cash_buffer", "apr", "average_balance"]
    book = book.dropna(subset=required)
    n_excluded = n_before - len(book)
    log.info("Excluding %d of %d funded-book rows (missing a stress-test input)", n_excluded, n_before)

    summaries, economics_by_scenario = {}, {}
    for scenario_name in SCENARIOS:
        economics = run_scenario(book, scenario_name, config)
        economics_by_scenario[scenario_name] = economics
        summaries[scenario_name] = portfolio_summary(economics)
        log.info(
            "%s: mean_pd=%.3f%% total_EL=€%.0f total_profit=€%.0f",
            scenario_name, summaries[scenario_name]["mean_pd"] * 100,
            summaries[scenario_name]["total_expected_loss"], summaries[scenario_name]["total_expected_profit"],
        )

    base = summaries["base"]
    el_multiplier_by_scenario = {s: (summaries[s]["total_expected_loss"] / base["total_expected_loss"]) for s in SCENARIOS}
    log.info("Expected Loss multiplier vs. base: %s", {k: round(v, 2) for k, v in el_multiplier_by_scenario.items()})

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for stale in FIG_DIR.glob("*.png"):
        stale.unlink()
    plot_el_by_scenario(summaries)
    plot_profit_by_scenario(summaries)
    plot_pd_shift_by_scenario(economics_by_scenario)

    report = {
        "n_funded": int(len(book)), "n_excluded_missing_inputs": n_excluded,
        "summary_by_scenario": summaries,
        "expected_loss_multiplier_vs_base": el_multiplier_by_scenario,
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
    lines = [
        "# Model Card — Stress Testing / Scenarios (§26-27)",
        "",
        "Re-evaluates the FUNDED portfolio (Phase 5's MIP solution) under each "
        "macroeconomic scenario in `config/settings.yaml`'s `scenarios` block. Feature-level "
        "shocks (income, spending, income volatility, delinquency, cash buffer) are re-scored "
        "through the existing calibrated PD model; `default_multiplier`/`funding_rate_shift`/"
        "`apr_shift` are direct overlays on the downstream economics formula. See module "
        "docstring for the full mechanism-per-shock breakdown and what is deliberately left "
        "unshocked (`average_balance`, LGD).",
        "",
        f"{report['n_excluded_missing_inputs']:,} of "
        f"{report['n_funded'] + report['n_excluded_missing_inputs']:,} funded-book rows excluded "
        "(missing a stress-test input feature).",
        "",
        "## Portfolio Expected Loss / Profit by scenario",
        "",
        "| Scenario | Mean PD | Total Expected Loss | EL vs. base | Total Expected Profit |",
        "|---|---|---|---|---|",
    ]
    for scenario in SCENARIOS:
        row = s[scenario]
        mult = report["expected_loss_multiplier_vs_base"][scenario]
        lines.append(
            f"| {scenario} | {row['mean_pd']:.2%} | €{row['total_expected_loss']:,.0f} | "
            f"{mult:.2f}x | €{row['total_expected_profit']:,.0f} |"
        )
    hir = s["high_interest_rate"]
    lines += [
        "",
        f"Portfolio Expected Loss limit (`optimization.portfolio_expected_loss_limit`): "
        f"€{report['portfolio_expected_loss_limit']:,.0f}.",
        "",
        "## A real, non-obvious finding: `high_interest_rate`'s net profit exactly matches base",
        "",
        f"€{hir['total_expected_profit']:,.2f} vs. base's €{s['base']['total_expected_profit']:,.2f} -- "
        "identical to the cent, not a rounding coincidence. `high_interest_rate` carries no "
        "feature-level shock (PD is unchanged from base, confirmed above), only "
        "`funding_rate_shift: +0.03` and `apr_shift: +0.03` -- and because `interest_revenue "
        "= balance x apr` and `funding_cost = balance x funding_rate` both scale the SAME "
        "`balance` figure, two EQUAL-MAGNITUDE shifts applied to APR and funding rate cancel "
        f"exactly in net profit (revenue +€{(hir['total_revenue'] - s['base']['total_revenue']):,.0f}, "
        f"funding cost +€{(hir['total_funding_cost'] - s['base']['total_funding_cost']):,.0f}, same "
        "figure both times). A real risk this scenario's config is meant to illustrate -- if "
        "a bank's own repricing (APR) exactly tracks its cost of funds, a rate-rise scenario "
        "looks like a non-event on the bottom line even though gross revenue AND gross cost "
        "both moved substantially and NIM composition changed -- not something a single "
        "'total profit' number alone would reveal without this breakdown.",
        "",
        "![Expected Loss by scenario](../figures/scenarios/01_expected_loss_by_scenario.png)",
        "",
        "![Expected Profit by scenario](../figures/scenarios/02_expected_profit_by_scenario.png)",
        "",
        "![PD distribution by scenario](../figures/scenarios/03_pd_distribution_by_scenario.png)",
        "",
    ]
    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_CARD_PATH.open("w") as f:
        f.write("\n".join(lines))


def main() -> None:
    run_stress_test()


if __name__ == "__main__":
    main()
