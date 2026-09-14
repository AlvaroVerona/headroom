"""Generates transactions.csv and payments.csv from the already-simulated
monthly spend/payment aggregates in behavior_series.py.

transactions.csv is a deliberately illustrative SAMPLE of spending
activity, not an exhaustive replay of every purchase: each customer-month
gets up to 3 category-lump transactions (essential / discretionary+
subscriptions / cash withdrawal) rather than dozens of individual line
items. The behavioral aggregates a real model would train on (essential_
spend, discretionary_spend, etc.) live directly in monthly_customer_
behavior.csv, generated independently -- transactions.csv exists for
transaction-level EDA/dashboard drill-down, not as the source of those
aggregates. This keeps total row count in the single-digit millions
(comfortably over the spec's 500,000+ floor) instead of tens of millions,
which would make the whole pipeline unworkably slow for a portfolio
project. See CLAUDE.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ESSENTIAL_CATEGORIES = ["groceries", "utilities", "rent"]
DISCRETIONARY_CATEGORIES = ["restaurants", "shopping", "travel", "transportation", "subscriptions"]
CASH_WITHDRAWAL_INCLUDE_PROB = 0.5

PAYMENT_TYPES_ON_TIME = ["autopay_full", "autopay_minimum", "manual"]


def _month_timestamps(rng: np.random.Generator, month: pd.Period, n: int, day_lo: int = 1, day_hi: int = 28) -> pd.DatetimeIndex:
    days = rng.integers(day_lo, day_hi, size=n)
    start = month.to_timestamp()
    return pd.DatetimeIndex([start + pd.Timedelta(days=int(d)) for d in days])


def generate_transactions(rng: np.random.Generator, customers: pd.DataFrame, beh: dict[str, np.ndarray], start_date: pd.Timestamp) -> pd.DataFrame:
    n_customers, n_months = beh["income"].shape
    months = pd.period_range(start_date, periods=n_months, freq="M")
    customer_ids = customers["customer_id"].to_numpy()

    rows = []
    txn_seq = 0
    for t, month in enumerate(months):
        essential = beh["essential_spend"][:, t]
        discretionary = beh["discretionary_spend"][:, t] + beh["subscription_spend"][:, t]
        cash = beh["cash_withdrawal"][:, t]

        for amounts, categories, direction, channel in [
            (essential, ESSENTIAL_CATEGORIES, "debit", "card"),
            (discretionary, DISCRETIONARY_CATEGORIES, "debit", "card"),
        ]:
            active = amounts > 1.0
            n_active = int(active.sum())
            if n_active == 0:
                continue
            cat = rng.choice(categories, size=n_active)
            ts = _month_timestamps(rng, month, n_active)
            ids = np.arange(txn_seq + 1, txn_seq + n_active + 1)
            txn_seq += n_active
            rows.append(pd.DataFrame({
                "transaction_id": [f"T{i:08d}" for i in ids],
                "customer_id": customer_ids[active],
                "timestamp": ts,
                "transaction_type": "purchase",
                "merchant_category": cat,
                "amount": np.round(amounts[active], 2),
                "currency": "EUR",
                "direction": direction,
                "channel": channel,
            }))

        cash_active = (cash > 1.0) & (rng.random(n_customers) < CASH_WITHDRAWAL_INCLUDE_PROB)
        n_cash = int(cash_active.sum())
        if n_cash > 0:
            ts = _month_timestamps(rng, month, n_cash)
            ids = np.arange(txn_seq + 1, txn_seq + n_cash + 1)
            txn_seq += n_cash
            rows.append(pd.DataFrame({
                "transaction_id": [f"T{i:08d}" for i in ids],
                "customer_id": customer_ids[cash_active],
                "timestamp": ts,
                "transaction_type": "cash_withdrawal",
                "merchant_category": "cash_withdrawal",
                "amount": np.round(cash[cash_active], 2),
                "currency": "EUR",
                "direction": "debit",
                "channel": "atm",
            }))

    return pd.concat(rows, ignore_index=True)


def generate_payments(rng: np.random.Generator, customers: pd.DataFrame, beh: dict[str, np.ndarray], start_date: pd.Timestamp) -> pd.DataFrame:
    n_customers, n_months = beh["income"].shape
    months = pd.period_range(start_date, periods=n_months, freq="M")
    customer_ids = customers["customer_id"].to_numpy()

    rows = []
    pay_seq = 0
    for t, month in enumerate(months):
        scheduled = beh["scheduled_amount"][:, t]
        paid = beh["payment_amount"][:, t]
        missed = beh["missed_payment"][:, t]
        active = scheduled > 1.0
        n_active = int(active.sum())
        if n_active == 0:
            continue

        ts = _month_timestamps(rng, month, n_active, day_lo=1, day_hi=25)
        payment_type = np.where(
            missed[active], "missed",
            rng.choice(PAYMENT_TYPES_ON_TIME, size=n_active),
        )
        days_late = np.where(missed[active], 30, 0)
        ids = np.arange(pay_seq + 1, pay_seq + n_active + 1)
        pay_seq += n_active

        rows.append(pd.DataFrame({
            "payment_id": [f"P{i:08d}" for i in ids],
            "customer_id": customer_ids[active],
            "timestamp": ts,
            "amount": np.round(paid[active], 2),
            "payment_type": payment_type,
            "scheduled_amount": np.round(scheduled[active], 2),
            "paid_amount": np.round(paid[active], 2),
            "days_late": days_late,
        }))

    return pd.concat(rows, ignore_index=True)
