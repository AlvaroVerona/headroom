"""Feature engineering (§13), computed on the VALIDATED layer.

Static customer/account attributes come from data/processed/ (quarantined
customers are dropped -- we don't build risk models on records that failed
quality checks). monthly_customer_behavior.csv and labels.csv are treated
as clean, derived ground-truth tables (not raw operational data subject to
the same injected issues) and read directly from data/raw/.

Leakage prevention (§42): every trailing-window/rolling feature is computed
with pandas' rolling() over a customer's OWN chronologically-sorted history,
which by construction only ever looks backward from each row -- then, for
each snapshot cohort, we select the single row AT that snapshot's own
month. A feature computed this way for snapshot month M is mathematically
incapable of depending on month M+1 or later, regardless of which cohort
(train/validation/test) it ends up in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_limit_optimizer.utils.config import PROJECT_ROOT, load_config
from credit_limit_optimizer.utils.logging import get_logger

log = get_logger("feature_engineering")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "outputs" / "features.csv"

ROLLING_WINDOWS = (3, 6, 12)
GROWTH_WINDOWS = (3, 6)
TREND_VARIABLES = ["monthly_income", "monthly_spend", "credit_utilization", "end_balance"]


def load_inputs() -> dict[str, pd.DataFrame]:
    customers = pd.read_csv(PROCESSED_DIR / "customers.csv")
    accounts = pd.read_csv(PROCESSED_DIR / "credit_accounts.csv")
    monthly = pd.read_csv(RAW_DIR / "monthly_customer_behavior.csv")
    labels = pd.read_csv(RAW_DIR / "labels.csv")
    return {"customers": customers, "credit_accounts": accounts, "monthly": monthly, "labels": labels}


def add_rolling_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """One pass over the full 36-month table, computed once and reused by
    every snapshot cohort -- far cheaper than re-filtering and
    re-aggregating per cohort."""
    monthly = monthly.sort_values(["customer_id", "month"]).reset_index(drop=True)
    monthly["month_index"] = monthly.groupby("customer_id").cumcount() + 1
    grouped = monthly.groupby("customer_id")

    for var in TREND_VARIABLES:
        for window in ROLLING_WINDOWS:
            monthly[f"{var}_{window}m_avg"] = grouped[var].transform(
                lambda s, w=window: s.rolling(w, min_periods=1).mean()
            )
        for window in GROWTH_WINDOWS:
            shifted = grouped[var].shift(window)
            # Symmetric ("relative change") growth, not naive pct-change:
            # credit_utilization and end_balance legitimately hit exactly 0
            # (a customer who pays off in full), and naive
            # (current-past)/past blows up to six- and ten-digit values the
            # moment `past` is near zero -- found while inspecting the first
            # full run (max observed: ~1e6 for utilization, ~1e10 for
            # balance). This formula is bounded to [-2, 2] whenever current
            # and past are both non-negative (true for every TREND_VARIABLE
            # here), which utilization/balance/income/spend all are.
            denom = (monthly[var].abs() + shifted.abs()) / 2
            monthly[f"{var}_{window}m_growth"] = (monthly[var] - shifted) / denom.clip(lower=1e-6)

    monthly["min_balance_12m"] = grouped["end_balance"].transform(lambda s: s.rolling(12, min_periods=1).min())
    delinquency_count_3m_ago = grouped["delinquency_count"].shift(3)
    monthly["recent_delinquency"] = (
        monthly["delinquency_count"] - delinquency_count_3m_ago.fillna(monthly["delinquency_count"])
    ) > 0
    # Expanding max over the customer's own trailing history (up to and
    # including this row) -- not a fixed window.
    monthly["max_days_past_due"] = grouped["days_past_due"].cummax()

    return monthly


def build_feature_table(inputs: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    customers, accounts = inputs["customers"], inputs["credit_accounts"]
    monthly, labels = inputs["monthly"], inputs["labels"]

    monthly = add_rolling_features(monthly)
    snap_cfg = config["snapshots"]
    cohort_month = {"train": snap_cfg["train_month"], "validation": snap_cfg["validation_month"], "test": snap_cfg["test_month"]}

    frames = []
    for cohort, month in cohort_month.items():
        cohort_labels = labels[labels["cohort"] == cohort]
        snapshot_rows = monthly[monthly["month_index"] == month]
        merged = cohort_labels.merge(snapshot_rows, on="customer_id", how="inner", suffixes=("", "_monthly"))
        frames.append(merged)
    behavior_features = pd.concat(frames, ignore_index=True)

    # Inner join to static attributes -- quarantined customers/accounts
    # (failed a Phase 1 quality check) are dropped, not modeled on.
    table = behavior_features.merge(customers, on="customer_id", how="inner", suffixes=("", "_cust"))
    table = table.merge(accounts[["customer_id", "current_credit_limit", "apr"]], on="customer_id", how="inner")

    snapshot_date = pd.to_datetime(table["snapshot_period"])
    customer_since = pd.to_datetime(table["customer_since"])
    table["customer_tenure_months"] = ((snapshot_date - customer_since).dt.days / 30.44).round().astype(int)
    table["employment_stable"] = (
        table["employment_status"].isin(["employed", "self_employed"]) & (table["employment_tenure_months"] >= 12)
    ).astype(int)

    table["debt_to_income"] = table["debt_balance"] / table["monthly_income"].clip(lower=1.0)
    table["income_stability"] = 1 - table["income_volatility"].clip(0, 1)
    table["essential_spending_ratio"] = table["essential_spend"] / table["monthly_spend"].clip(lower=1.0)
    table["discretionary_spending_ratio"] = table["discretionary_spend"] / table["monthly_spend"].clip(lower=1.0)
    table["cash_withdrawal_ratio"] = table["cash_withdrawal_amount"] / table["monthly_spend"].clip(lower=1.0)
    table["credit_exposure"] = table["current_credit_limit"]
    # "Cash buffer": remaining borrowing headroom on the revolving line --
    # the closest analog to a liquidity cushion available in a single-
    # product (no separate deposit/savings account) dataset. Documented,
    # not literally a bank balance.
    table["cash_buffer"] = table["current_credit_limit"] - table["end_balance"]

    return table


FEATURE_COLUMNS = {
    "demographic": ["age", "employment_tenure_months", "customer_tenure_months", "employment_stable"],
    # income_growth (§13) is covered by monthly_income_3m_growth/6m_growth
    # in the "trend" group below, not repeated here -- listing it in both
    # groups produced duplicate-named columns in the output (visible as
    # ".1"-suffixed columns after a CSV round-trip). Found and fixed while
    # building this.
    "income": ["monthly_income", "income_volatility", "income_stability"],
    "credit": ["credit_utilization", "max_utilization", "credit_exposure", "debt_to_income", "payment_ratio", "minimum_payment_ratio"],
    "behavioral": ["transaction_count", "average_transaction_amount", "spending_volatility",
                   "essential_spending_ratio", "discretionary_spending_ratio", "cash_withdrawal_ratio"],
    "delinquency": ["days_past_due", "delinquency_count", "recent_delinquency", "max_days_past_due"],
    "liquidity": ["average_balance", "min_balance_12m", "cash_buffer", "savings_rate"],
    "trend": [f"{var}_{w}m_avg" for var in TREND_VARIABLES for w in ROLLING_WINDOWS]
             + [f"{var}_{w}m_growth" for var in TREND_VARIABLES for w in GROWTH_WINDOWS],
}

ALL_FEATURE_COLUMNS = [c for group in FEATURE_COLUMNS.values() for c in group]
ID_COLUMNS = ["customer_id", "cohort", "snapshot_month", "snapshot_period", "customer_segment"]
TARGET_COLUMNS = ["default_12m", "delinquent_90d", "eligible_for_default_label",
                   "future_utilization", "future_monthly_spend", "future_interest_revenue", "future_interchange_revenue"]


def main() -> None:
    config = load_config()
    inputs = load_inputs()
    table = build_feature_table(inputs, config)

    output_cols = ID_COLUMNS + ALL_FEATURE_COLUMNS + TARGET_COLUMNS
    result = table[output_cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    log.info("Wrote %d feature rows (%d features) to %s", len(result), len(ALL_FEATURE_COLUMNS), OUTPUT_PATH.relative_to(PROJECT_ROOT))
    log.info("Rows by cohort: %s", result["cohort"].value_counts().to_dict())
    log.info("Missing-value counts (top 5): %s", result[ALL_FEATURE_COLUMNS].isna().sum().sort_values(ascending=False).head(5).to_dict())


if __name__ == "__main__":
    main()
