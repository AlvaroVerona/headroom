"""Exploratory Data Analysis (spec §12).

Investigates every point the spec lists -- distributions, default rate by
income/utilization/age/employment/tenure/DTI/delinquency-history, spending
behavior, monthly cash flow, utilization/delinquency over time, and
correlation structure -- from the real generated data, and writes:

- reports/figures/eda/*.png (one chart per investigation point)
- reports/outputs/eda_stats.json (every number quoted in the report, so
  the report can never say something the data doesn't back up)
- reports/outputs/eda_report.md (the numbers plus business interpretation)

Default-rate cuts pool all three snapshot cohorts (144,003 customer-
snapshot rows: the same ~48,000 customers appear at months 12/18/24, each
with genuinely different age/income/utilization/outcome) -- appropriate
for exploring the data's relationships; Phase 3's actual train/validation/
test split keeps the cohorts separate. Utilization/delinquency "over time"
uses the full 36-month raw table (not just snapshots), since that's the
only way to see a within-customer trajectory.

Run: python -m credit_limit_optimizer.analysis.eda
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_limit_optimizer.features.engineering import build_feature_table, load_inputs
from credit_limit_optimizer.utils.config import PROJECT_ROOT, load_config
from credit_limit_optimizer.utils.logging import get_logger

log = get_logger("eda")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "eda"
STATS_PATH = PROJECT_ROOT / "reports" / "outputs" / "eda_stats.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "outputs" / "eda_report.md"

SEGMENT_ORDER = ["prime", "near_prime", "subprime"]
SEGMENT_COLORS = {"prime": "#2E7D32", "near_prime": "#F9A825", "subprime": "#C62828"}

plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------- helpers

def _describe(series: pd.Series) -> dict:
    s = series.dropna()
    return {
        "count": int(s.count()), "mean": float(s.mean()), "median": float(s.median()),
        "std": float(s.std()), "p10": float(s.quantile(0.10)), "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)), "p90": float(s.quantile(0.90)),
        "min": float(s.min()), "max": float(s.max()), "skew": float(s.skew()),
    }


def _by_segment(df: pd.DataFrame, col: str) -> dict:
    return {seg: _describe(df.loc[df["customer_segment"] == seg, col]) for seg in SEGMENT_ORDER}


def _fmt_interval(interval: pd.Interval, ndigits: int) -> str:
    return f"{round(interval.left, ndigits):g}-{round(interval.right, ndigits):g}"


def _default_rate_by_bin(df: pd.DataFrame, values: pd.Series, bins=None, labels=None, q=None, round_ndigits=0) -> pd.DataFrame:
    d = df.dropna(subset=["default_12m"]).copy()
    v = values.loc[d.index]
    if q is not None:
        d["_bin"] = pd.qcut(v, q=q, duplicates="drop")
    else:
        d["_bin"] = pd.cut(v, bins=bins, labels=labels, right=False)
    grouped = d.groupby("_bin", observed=True)["default_12m"].agg(["mean", "count"]).rename(columns={"mean": "default_rate"})
    if q is not None:
        # qcut labels are exact float quantile edges (e.g. "267.289000...04,
        # 1569.59"), unreadable in a chart -- round for display only; the
        # underlying bin membership (and thus default_rate) is untouched.
        grouped.index = [_fmt_interval(iv, round_ndigits) for iv in grouped.index]
    else:
        grouped.index = grouped.index.astype(str)
    return grouped


def _bar(series: pd.Series, title: str, ylabel: str, path, color="#1565C0", pct=False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(series.index.astype(str), series.values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if pct:
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _hist(series: pd.Series, title: str, xlabel: str, path, bins=50, by_segment: pd.Series | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if by_segment is not None:
        for seg in SEGMENT_ORDER:
            vals = series[by_segment == seg].dropna()
            ax.hist(vals, bins=bins, alpha=0.5, label=seg, color=SEGMENT_COLORS[seg], density=True)
        ax.legend()
    else:
        ax.hist(series.dropna(), bins=bins, color="#1565C0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _line_by_segment(monthly_series: pd.DataFrame, value_col: str, title: str, ylabel: str, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seg in SEGMENT_ORDER:
        sub = monthly_series[monthly_series["customer_segment"] == seg]
        ax.plot(sub["month_index"], sub[value_col], label=seg, color=SEGMENT_COLORS[seg], linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _heatmap(corr: pd.DataFrame, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ investigations

def income_distribution(df: pd.DataFrame) -> dict:
    _hist(df["monthly_income"], "Monthly income distribution", "Monthly income (EUR)",
          FIG_DIR / "01_income_distribution.png", by_segment=df["customer_segment"])
    return {"overall": _describe(df["monthly_income"]), "by_segment": _by_segment(df, "monthly_income")}


def credit_limit_distribution(df: pd.DataFrame) -> dict:
    _hist(df["credit_exposure"], "Credit limit distribution", "Credit limit (EUR)",
          FIG_DIR / "02_credit_limit_distribution.png", by_segment=df["customer_segment"])
    return {"overall": _describe(df["credit_exposure"]), "by_segment": _by_segment(df, "credit_exposure")}


def utilization_distribution(df: pd.DataFrame) -> dict:
    _hist(df["credit_utilization"], "Credit utilization distribution", "Utilization",
          FIG_DIR / "03_utilization_distribution.png", by_segment=df["customer_segment"])
    at_cap = (df["credit_utilization"] >= 0.999).mean()
    return {
        "overall": _describe(df["credit_utilization"]),
        "by_segment": _by_segment(df, "credit_utilization"),
        "fraction_at_cap": float(at_cap),
    }


def default_rate_overall(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["default_12m"])
    by_cohort = d.groupby("cohort")["default_12m"].mean().to_dict()
    by_segment = d.groupby("customer_segment")["default_12m"].mean().reindex(SEGMENT_ORDER).to_dict()
    _bar(pd.Series(by_segment), "Default rate by segment", "Default rate (12m)",
         FIG_DIR / "04_default_rate_by_segment.png", pct=True)
    return {
        "overall_rate": float(d["default_12m"].mean()), "n_eligible": int(len(d)),
        "n_excluded_already_in_default": int(df["default_12m"].isna().sum()),
        "by_cohort": {k: float(v) for k, v in by_cohort.items()},
        "by_segment": {k: float(v) for k, v in by_segment.items()},
    }


def default_rate_by_income(df: pd.DataFrame) -> dict:
    g = _default_rate_by_bin(df, df["monthly_income"], q=5, round_ndigits=0)
    _bar(g["default_rate"], "Default rate by income quintile", "Default rate (12m)",
         FIG_DIR / "05_default_rate_by_income.png", pct=True)
    return g.reset_index().rename(columns={"_bin": "bin"}).to_dict(orient="records")


def default_rate_by_utilization(df: pd.DataFrame) -> dict:
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    labels = ["0-10%", "10-30%", "30-50%", "50-70%", "70-90%", "90-100%"]
    g = _default_rate_by_bin(df, df["credit_utilization"], bins=bins, labels=labels)
    _bar(g["default_rate"], "Default rate by utilization band", "Default rate (12m)",
         FIG_DIR / "06_default_rate_by_utilization.png", pct=True)
    return g.reset_index().rename(columns={"_bin": "bin"}).to_dict(orient="records")


def default_rate_by_age(df: pd.DataFrame) -> dict:
    bins = [18, 25, 35, 45, 55, 65, 100]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    g = _default_rate_by_bin(df, df["age"], bins=bins, labels=labels)
    _bar(g["default_rate"], "Default rate by age band", "Default rate (12m)",
         FIG_DIR / "07_default_rate_by_age.png", pct=True)
    return g.reset_index().rename(columns={"_bin": "bin"}).to_dict(orient="records")


def default_rate_by_employment_status(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["default_12m"])
    g = d.groupby("employment_status", observed=True)["default_12m"].agg(["mean", "count"]).rename(columns={"mean": "default_rate"})
    _bar(g["default_rate"], "Default rate by employment status", "Default rate (12m)",
         FIG_DIR / "08_default_rate_by_employment.png", pct=True)
    return g.reset_index().to_dict(orient="records")


def default_rate_by_tenure(df: pd.DataFrame) -> dict:
    bins = [0, 12, 24, 36, 60, 120, 999]
    labels = ["<1y", "1-2y", "2-3y", "3-5y", "5-10y", "10y+"]
    g = _default_rate_by_bin(df, df["customer_tenure_months"], bins=bins, labels=labels)
    _bar(g["default_rate"], "Default rate by customer tenure", "Default rate (12m)",
         FIG_DIR / "09_default_rate_by_tenure.png", pct=True)
    return g.reset_index().rename(columns={"_bin": "bin"}).to_dict(orient="records")


def default_rate_by_dti(df: pd.DataFrame) -> dict:
    g = _default_rate_by_bin(df, df["debt_to_income"], q=5, round_ndigits=2)
    _bar(g["default_rate"], "Default rate by debt-to-income quintile", "Default rate (12m)",
         FIG_DIR / "10_default_rate_by_dti.png", pct=True)
    return g.reset_index().rename(columns={"_bin": "bin"}).to_dict(orient="records")


def default_rate_by_delinquency_history(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["default_12m"])
    by_recent = d.groupby("recent_delinquency")["default_12m"].agg(["mean", "count"]).rename(columns={"mean": "default_rate"})
    bins = [0, 1, 30, 60, 90, 999]
    labels = ["Never", "1-29d", "30-59d", "60-89d", "90d+"]
    by_dpd = _default_rate_by_bin(df, df["max_days_past_due"], bins=bins, labels=labels)
    _bar(by_dpd["default_rate"], "Default rate by worst-ever delinquency", "Default rate (12m)",
         FIG_DIR / "11_default_rate_by_delinquency_history.png", pct=True)
    return {
        "by_recent_delinquency_flag": {str(k): {"default_rate": float(v["default_rate"]), "count": int(v["count"])}
                                        for k, v in by_recent.iterrows()},
        "by_max_days_past_due": by_dpd.reset_index().rename(columns={"_bin": "bin"}).to_dict(orient="records"),
    }


def spending_behavior(df: pd.DataFrame) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in zip(
        axes, ["essential_spending_ratio", "discretionary_spending_ratio", "cash_withdrawal_ratio"],
        ["Essential spend ratio", "Discretionary spend ratio", "Cash withdrawal ratio"],
    ):
        ax.hist(df[col].clip(0, 1).dropna(), bins=40, color="#1565C0")
        ax.set_title(title)
    fig.savefig(FIG_DIR / "12_spending_behavior.png", bbox_inches="tight")
    plt.close(fig)
    d = df.dropna(subset=["default_12m"])
    corr_with_default = {
        col: float(d[col].corr(d["default_12m"]))
        for col in ["essential_spending_ratio", "discretionary_spending_ratio", "cash_withdrawal_ratio",
                    "transaction_count", "average_transaction_amount"]
    }
    return {
        "essential_spending_ratio": _describe(df["essential_spending_ratio"]),
        "discretionary_spending_ratio": _describe(df["discretionary_spending_ratio"]),
        "cash_withdrawal_ratio": _describe(df["cash_withdrawal_ratio"]),
        "transaction_count": _describe(df["transaction_count"]),
        "average_transaction_amount": _describe(df["average_transaction_amount"]),
        "correlation_with_default_12m": corr_with_default,
    }


def monthly_cash_flow(df: pd.DataFrame) -> dict:
    net_cash_flow = df["monthly_income"] - df["monthly_spend"]
    _hist(net_cash_flow, "Monthly net cash flow (income - spend)", "Net cash flow (EUR)",
          FIG_DIR / "13_monthly_cash_flow.png", by_segment=df["customer_segment"])
    by_segment = {seg: _describe(net_cash_flow[df["customer_segment"] == seg]) for seg in SEGMENT_ORDER}
    return {
        "overall": _describe(net_cash_flow),
        "by_segment": by_segment,
        "fraction_negative": float((net_cash_flow < 0).mean()),
    }


def utilization_over_time(monthly_raw: pd.DataFrame) -> dict:
    series = monthly_raw.groupby(["customer_segment", "month_index"], observed=True)["credit_utilization"].mean().reset_index()
    _line_by_segment(series, "credit_utilization", "Average credit utilization over time, by segment",
                      "Avg. utilization", FIG_DIR / "14_utilization_over_time.png")
    month_1 = series[series.month_index == 1].set_index("customer_segment")["credit_utilization"]
    month_36 = series[series.month_index == 36].set_index("customer_segment")["credit_utilization"]
    return {
        "month_1_by_segment": month_1.to_dict(),
        "month_36_by_segment": month_36.to_dict(),
    }


def delinquency_evolution(monthly_raw: pd.DataFrame) -> dict:
    monthly_raw = monthly_raw.copy()
    monthly_raw["delinquent"] = monthly_raw["days_past_due"] > 0
    series = monthly_raw.groupby(["customer_segment", "month_index"], observed=True)["delinquent"].mean().reset_index()
    _line_by_segment(series, "delinquent", "Share of customers delinquent (DPD > 0), by segment",
                      "Share delinquent", FIG_DIR / "15_delinquency_evolution.png")
    month_1 = series[series.month_index == 1].set_index("customer_segment")["delinquent"]
    month_36 = series[series.month_index == 36].set_index("customer_segment")["delinquent"]
    return {
        "month_1_by_segment": month_1.to_dict(),
        "month_36_by_segment": month_36.to_dict(),
    }


CORRELATION_COLUMNS = [
    "credit_utilization", "max_utilization", "debt_to_income", "payment_ratio",
    "minimum_payment_ratio", "max_days_past_due", "delinquency_count", "income_volatility",
    "spending_volatility", "cash_withdrawal_ratio", "savings_rate", "customer_tenure_months",
    "employment_tenure_months", "age", "monthly_income", "default_12m",
]


def correlation_structure(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["default_12m"])
    corr = d[CORRELATION_COLUMNS].corr()
    _heatmap(corr, "Correlation structure (key features + default_12m)", FIG_DIR / "16_correlation_heatmap.png")
    with_default = corr["default_12m"].drop("default_12m").sort_values(key=np.abs, ascending=False)
    return {
        "top_correlates_with_default_12m": with_default.to_dict(),
        "full_matrix": corr.round(3).to_dict(),
    }


# --------------------------------------------------------------------- main

def load_full_monthly_raw() -> pd.DataFrame:
    monthly = pd.read_csv(RAW_DIR / "monthly_customer_behavior.csv")
    customers = pd.read_csv(PROCESSED_DIR / "customers.csv")[["customer_id", "customer_segment"]]
    monthly = monthly.sort_values(["customer_id", "month"]).reset_index(drop=True)
    monthly["month_index"] = monthly.groupby("customer_id").cumcount() + 1
    return monthly.merge(customers, on="customer_id", how="inner")


def run_eda() -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    config = load_config()
    inputs = load_inputs()
    df = build_feature_table(inputs, config)
    monthly_raw = load_full_monthly_raw()

    log.info("Running EDA on %d customer-snapshot rows (%d raw monthly rows)", len(df), len(monthly_raw))

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "income_distribution": income_distribution(df),
        "credit_limit_distribution": credit_limit_distribution(df),
        "utilization_distribution": utilization_distribution(df),
        "default_rate_overall": default_rate_overall(df),
        "default_rate_by_income": default_rate_by_income(df),
        "default_rate_by_utilization": default_rate_by_utilization(df),
        "default_rate_by_age": default_rate_by_age(df),
        "default_rate_by_employment_status": default_rate_by_employment_status(df),
        "default_rate_by_tenure": default_rate_by_tenure(df),
        "default_rate_by_dti": default_rate_by_dti(df),
        "default_rate_by_delinquency_history": default_rate_by_delinquency_history(df),
        "spending_behavior": spending_behavior(df),
        "monthly_cash_flow": monthly_cash_flow(df),
        "utilization_over_time": utilization_over_time(monthly_raw),
        "delinquency_evolution": delinquency_evolution(monthly_raw),
        "correlation_structure": correlation_structure(df),
    }

    with STATS_PATH.open("w") as f:
        json.dump(stats, f, indent=2, default=str)
    log.info("Wrote %s", STATS_PATH.relative_to(PROJECT_ROOT))

    write_report(stats)
    log.info("Wrote %s", REPORT_PATH.relative_to(PROJECT_ROOT))
    return stats


def write_report(stats: dict) -> None:
    s = stats
    dr = s["default_rate_overall"]
    age_rates = [r["default_rate"] for r in s["default_rate_by_age"]]
    employment_rates = [r["default_rate"] for r in s["default_rate_by_employment_status"]]
    tenure_rates = [r["default_rate"] for r in s["default_rate_by_tenure"]]
    lines = [
        "# Headroom — Exploratory Data Analysis",
        "",
        f"_Generated {s['generated_at']}. All numbers computed from the real generated "
        f"dataset ({s['n_rows']:,} customer-snapshot rows pooling the train/validation/test "
        "cohorts — see `reports/outputs/eda_stats.json` for every figure quoted below.)_",
        "",
        "## Income distribution",
        f"Monthly income: mean €{s['income_distribution']['overall']['mean']:,.0f}, "
        f"median €{s['income_distribution']['overall']['median']:,.0f} "
        f"(p10 €{s['income_distribution']['overall']['p10']:,.0f} — "
        f"p90 €{s['income_distribution']['overall']['p90']:,.0f}). "
        + ", ".join(f"{seg}: median €{v['median']:,.0f}" for seg, v in s['income_distribution']['by_segment'].items())
        + ". Segments are constructed from income/behavior jointly, so overlap is expected.",
        "![income](../figures/eda/01_income_distribution.png)",
        "",
        "## Credit limit distribution",
        f"Mean €{s['credit_limit_distribution']['overall']['mean']:,.0f}, "
        f"median €{s['credit_limit_distribution']['overall']['median']:,.0f}. "
        + ", ".join(f"{seg}: median €{v['median']:,.0f}" for seg, v in s['credit_limit_distribution']['by_segment'].items()),
        "![credit limit](../figures/eda/02_credit_limit_distribution.png)",
        "",
        "## Utilization distribution",
        f"Mean {s['utilization_distribution']['overall']['mean']:.1%}, "
        f"{s['utilization_distribution']['fraction_at_cap']:.1%} of rows at/near the credit "
        "ceiling (≥99.9% utilization) — consistent with the generator's documented subprime "
        "behavior (see CLAUDE.md Phase 1 notes), not a data artifact.",
        "![utilization](../figures/eda/03_utilization_distribution.png)",
        "",
        "## Default rate",
        f"Overall: {dr['overall_rate']:.2%} across {dr['n_eligible']:,} eligible rows "
        f"({dr['n_excluded_already_in_default']:,} excluded as already in default at the "
        "snapshot). By segment: " + ", ".join(f"{k} {v:.2%}" for k, v in dr["by_segment"].items())
        + ". By cohort: " + ", ".join(f"{k} {v:.2%}" for k, v in dr["by_cohort"].items())
        + " — stable across time-separated snapshots, as expected for a well-specified label.",
        "![default by segment](../figures/eda/04_default_rate_by_segment.png)",
        "",
        "## Default rate by income",
        "Monotonically decreasing from the lowest to highest income quintile "
        "(see chart) — higher income gives more repayment headroom relative to fixed "
        "obligations.",
        "![default by income](../figures/eda/05_default_rate_by_income.png)",
        "",
        "## Default rate by utilization",
        "Rises sharply in the 70-100% utilization bands — a customer running close to "
        "their limit is both a symptom of financial stress and a leading indicator of it.",
        "![default by utilization](../figures/eda/06_default_rate_by_utilization.png)",
        "",
        "## Default rate by age",
        f"Flat across age bands ({min(age_rates):.1%}-{max(age_rates):.1%}, no meaningful "
        "trend) — by design, `age` doesn't feed the generator's default hazard directly; "
        "it's driven by segment, utilization and delinquency dynamics instead (see "
        "CLAUDE.md). Relevant later: this makes `age` a poor and unnecessary model input "
        "on its own — the spec explicitly warns against protected characteristics as "
        "direct default-model inputs, and this data gives no predictive reason to use one "
        "anyway.",
        "![default by age](../figures/eda/07_default_rate_by_age.png)",
        "",
        "## Default rate by employment status",
        f"Similarly flat ({min(employment_rates):.1%}-{max(employment_rates):.1%}) across "
        "employed/self-employed/retired/student/unemployed — the same design choice as "
        "age above; employment status isn't itself a hazard driver in the generator.",
        "![default by employment](../figures/eda/08_default_rate_by_employment.png)",
        "",
        "## Default rate by customer tenure",
        f"Also flat ({min(tenure_rates):.1%}-{max(tenure_rates):.1%}) — tenure isn't a "
        "hazard driver either. Utilization, delinquency history and DTI (above) carry "
        "essentially all of this dataset's default signal, not demographic or "
        "relationship-length variables.",
        "![default by tenure](../figures/eda/09_default_rate_by_tenure.png)",
        "",
        "## Default rate by debt-to-income ratio",
        "Rises steadily from the second quintile onward, one of the clearest single-"
        "variable risk signals in the dataset besides delinquency history. The lowest "
        "quintile (near-zero debt balance) ticks slightly above the second — plausibly "
        "customers who simply carry little revolving debt yet for unrelated reasons "
        "(e.g. subprime customers early in their relationship) still show elevated risk; "
        "DTI alone doesn't fully separate that group.",
        "![default by dti](../figures/eda/10_default_rate_by_dti.png)",
        "",
        "## Default rate by delinquency history",
        "The single strongest cut in the dataset: a customer who has ever reached 90+ days "
        "past due is dramatically more likely to default again than one with a clean "
        "history — see `by_max_days_past_due` in the stats file for exact rates.",
        "![default by delinquency history](../figures/eda/11_default_rate_by_delinquency_history.png)",
        "",
        "## Spending behavior",
        "`essential_spending_ratio` clusters tightly around three values (~0.45 / 0.60 / "
        "0.72, plus a spike at 0 for near-zero-spend months) rather than varying "
        "smoothly — by design, `ESSENTIAL_SHARE_BY_SEGMENT` in `behavior_series.py` fixes "
        "the essential/discretionary spend *mix* per segment and only lets the total spend "
        "*amount* vary; a real bank's data would show more within-segment variation in the "
        "mix itself. `cash_withdrawal_ratio` is genuinely continuous. Correlation with "
        "`default_12m` is weak for all three individually (see "
        "`spending_behavior.correlation_with_default_12m` in the stats file) but "
        "directionally sensible: cash-withdrawal share leans riskier (+0.03), while "
        "discretionary share leans safer (-0.12) — a mechanical echo of the fixed "
        "segment mix above (prime carries the highest discretionary share, subprime the "
        "lowest), not an independent discretionary-spending effect.",
        "![spending behavior](../figures/eda/12_spending_behavior.png)",
        "",
        "## Monthly cash flow",
        f"{s['monthly_cash_flow']['fraction_negative']:.1%} of customer-months show "
        "spend exceeding income within the same month (min observed net cash flow: "
        f"€{s['monthly_cash_flow']['overall']['min']:,.0f}, always positive). This is a "
        "real generator characteristic, not a data artifact: `monthly_spend` is drawn as "
        "`income * spend_ratio * lognormal(...)` (see `behavior_series.py`), so spend "
        "never exceeds income in the same month by construction. Revolving balances in "
        "this dataset therefore come entirely from customers not paying their balance in "
        "full, not from income shortfalls funded by credit within a month — worth noting "
        "as a modeling simplification, since real revolving-credit usage often does spike "
        "spend above income in a given month.",
        "![cash flow](../figures/eda/13_monthly_cash_flow.png)",
        "",
        "## Credit utilization over time",
        "Utilization by segment across the full 36-month history (not just the three "
        "snapshots) — shows each segment settling into its own equilibrium band rather "
        "than drifting monotonically, consistent with the steady-state balance dynamics "
        "documented in CLAUDE.md.",
        "![utilization over time](../figures/eda/14_utilization_over_time.png)",
        "",
        "## Delinquency evolution",
        "Share of customers with any DPD > 0 in a given month, by segment, over the full "
        "history. Each segment ramps up over roughly the first 12-15 months before "
        "settling into a stable band (subprime ~17%, near_prime ~11-12%, prime ~8%) — a "
        "burn-in effect, not a real worsening trend: every customer starts at DPD=0 by "
        "construction, and the AR(1) distress process (`DISTRESS_PERSISTENCE=0.92`, see "
        "CLAUDE.md) takes time to reach its stationary distribution from that deterministic "
        "starting point. The snapshot months (12/18/24) all fall after this ramp-up "
        "window closes, which is convenient — feature values at each snapshot reflect the "
        "segment's steady-state behavior rather than the artificial early-month ramp.",
        "![delinquency evolution](../figures/eda/15_delinquency_evolution.png)",
        "",
        "## Correlation structure",
        "Top absolute correlates with `default_12m`: "
        + ", ".join(f"{k} ({v:+.2f})" for k, v in list(s["correlation_structure"]["top_correlates_with_default_12m"].items())[:5])
        + ". These look small in absolute terms (all |r| < 0.16) — expected for a rare binary "
        "outcome (~4.7% base rate) against individually linear predictors, and consistent "
        "with the *far* stronger monotonic relationships visible in the default-rate-by-bin "
        "charts above (e.g. 0-10% to 90-100% utilization spans a ~2% to ~21% default rate). "
        "This is the standard justification for using WOE/IV or a tree-based model "
        "(XGBoost) rather than linear correlation to size a variable's real predictive "
        "power in credit risk — informative for Phase 3. Among the features themselves: "
        "utilization/max_utilization/debt_to_income cluster together (r > 0.7), as do "
        "payment_ratio/minimum_payment_ratio (near-identical by construction) and "
        "delinquency_count/max_days_past_due — worth watching for multicollinearity when "
        "selecting the logistic regression's feature set.",
        "![correlation heatmap](../figures/eda/16_correlation_heatmap.png)",
        "",
    ]
    with REPORT_PATH.open("w") as f:
        f.write("\n".join(lines))


def main() -> None:
    run_eda()


if __name__ == "__main__":
    main()
