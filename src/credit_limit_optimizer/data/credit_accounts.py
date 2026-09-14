"""Builds credit_accounts.csv as a point-in-time snapshot at the latest
generated month -- the CURRENT state used for dashboard display and as the
"current_limit" baseline the optimizer compares its recommendation
against. All historical modeling uses monthly_customer_behavior.csv, not
a re-derived history of this table (there isn't one -- credit_limit is
held constant through history per behavior_series.py).

The identities utilization_rate = balance/limit and
available_credit = limit - balance hold EXACTLY here (by construction) --
breaking them is one of the injected data-quality issues (§10), applied
downstream in quality_injection.py, never in this clean-data step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

APR_BY_SEGMENT = {"prime": 0.169, "near_prime": 0.219, "subprime": 0.279}


def build_credit_accounts(
    rng: np.random.Generator, customers: pd.DataFrame, credit_limit: np.ndarray, beh: dict[str, np.ndarray],
) -> pd.DataFrame:
    n = len(customers)
    final_balance = beh["balance"][:, -1]
    final_dpd = beh["dpd"][:, -1]
    final_delinquency = beh["delinquency_count"][:, -1]

    segment = customers["customer_segment"].to_numpy()
    apr_base = np.array([APR_BY_SEGMENT[s] for s in segment])
    apr = np.round(np.clip(apr_base + rng.normal(0, 0.015, size=n), 0.05, 0.35), 4)

    available_credit = np.round(credit_limit - final_balance, 2)
    utilization_rate = np.round(final_balance / credit_limit, 4)
    minimum_payment = np.round(np.maximum(final_balance * 0.03, np.where(final_balance > 0, 25.0, 0.0)), 2)

    account_open_date = customers["customer_since"]

    return pd.DataFrame({
        "customer_id": customers["customer_id"].to_numpy(),
        "account_id": [f"A{i:07d}" for i in range(1, n + 1)],
        "account_open_date": account_open_date.to_numpy(),
        "current_credit_limit": credit_limit,
        "current_balance": np.round(final_balance, 2),
        "available_credit": available_credit,
        "utilization_rate": utilization_rate,
        "apr": apr,
        "minimum_payment": minimum_payment,
        "days_past_due": final_dpd.astype(int),
        "delinquency_count": final_delinquency.astype(int),
        "total_credit_exposure": np.round(final_balance, 2),
    })
