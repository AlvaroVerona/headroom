"""Assembles the (n_customers, n_months) simulation arrays from
behavior_series.py into the long-format monthly_customer_behavior.csv
schema (§7), adding the trailing-window statistics (income_volatility,
spending_volatility, 3/6/12-month averages belong in features/engineering.py,
not here -- this module only produces the raw monthly facts)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_monthly_table(customers: pd.DataFrame, beh: dict[str, np.ndarray], start_date: pd.Timestamp) -> pd.DataFrame:
    n_customers, n_months = beh["income"].shape
    months = pd.period_range(start_date, periods=n_months, freq="M")

    customer_id = np.repeat(customers["customer_id"].to_numpy(), n_months)
    month = np.tile(months.astype(str), n_customers)

    # Trailing 3-month realized income volatility (coefficient of
    # variation) -- computed only from CURRENT and PAST months at each t,
    # never future ones, so it's safe to use as a model feature later.
    income = beh["income"]
    income_volatility = np.full_like(income, np.nan)
    spending_volatility = np.full_like(income, np.nan)
    spend = beh["spend"]
    for t in range(2, n_months):
        window_income = income[:, t - 2:t + 1]
        window_spend = spend[:, t - 2:t + 1]
        income_volatility[:, t] = window_income.std(axis=1) / np.maximum(window_income.mean(axis=1), 1.0)
        spending_volatility[:, t] = window_spend.std(axis=1) / np.maximum(window_spend.mean(axis=1), 1.0)

    def flat(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(-1)

    transaction_count = np.maximum(1, np.round(spend / np.maximum(spend.mean(), 1) * 6)).astype(int)
    average_transaction_amount = np.divide(spend, np.maximum(transaction_count, 1))

    table = pd.DataFrame({
        "customer_id": customer_id,
        "month": month,
        "monthly_spend": flat(np.round(spend, 2)),
        "monthly_income": flat(np.round(income, 2)),
        "average_balance": flat(np.round(beh["avg_balance"], 2)),
        "end_balance": flat(np.round(beh["balance"], 2)),
        "credit_utilization": flat(np.round(beh["utilization"], 4)),
        "max_utilization": flat(np.round(beh["max_utilization"], 4)),
        "payment_ratio": flat(np.round(beh["payment_ratio"], 4)),
        "minimum_payment_ratio": flat(np.round(beh["minimum_payment_ratio"], 4)),
        "days_past_due": flat(beh["dpd"]).astype(int),
        "delinquency_count": flat(beh["delinquency_count"]).astype(int),
        "transaction_count": flat(transaction_count),
        "average_transaction_amount": flat(np.round(average_transaction_amount, 2)),
        "cash_withdrawal_amount": flat(np.round(beh["cash_withdrawal"], 2)),
        "essential_spend": flat(np.round(beh["essential_spend"], 2)),
        "discretionary_spend": flat(np.round(beh["discretionary_spend"], 2)),
        "subscription_spend": flat(np.round(beh["subscription_spend"], 2)),
        "debt_balance": flat(np.round(beh["balance"], 2)),
        "total_credit_exposure": flat(np.round(beh["balance"], 2)),
        "savings_rate": flat(np.round(beh["savings_rate"], 4)),
        "income_volatility": flat(np.round(income_volatility, 4)),
        "spending_volatility": flat(np.round(spending_volatility, 4)),
    })
    return table
