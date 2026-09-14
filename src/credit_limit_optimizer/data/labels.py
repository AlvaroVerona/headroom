"""Builds the supervised targets (§8-9) from the already-simulated
trajectory -- default_12m, delinquent_90d, and the "future_*" revenue/
behavior targets -- for each of the three time-separated snapshot cohorts
(§15). This module only ever READS the trajectory at or after the
snapshot month to build the LABEL; feature engineering (a separate module)
is responsible for making sure no feature ever does the same for months
after its own snapshot. See CLAUDE.md for the snapshot design and why 36
months of history exist at all.

default_12m definition: reaches 90+ days-past-due at any point in the 12
months strictly after the snapshot, given the customer was NOT already at
90+ DPD at the snapshot itself (those customers are already in default --
scoring "will they default" doesn't apply, so they're excluded via
`eligible_for_default_label`, not silently mislabeled as 0).

delinquent_90d is a CURRENT-state indicator (DPD >= 90 at the snapshot
itself) -- no forward-looking window, so it carries no leakage risk by
construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_DPD_THRESHOLD = 90


def build_labels_for_snapshot(
    customers: pd.DataFrame, beh: dict[str, np.ndarray], credit_accounts: pd.DataFrame,
    snapshot_month: int, horizon_months: int, interchange_rate: float, cohort_name: str,
) -> pd.DataFrame:
    """snapshot_month is 1-indexed into the simulated array (month 1 = index 0)."""
    idx = snapshot_month - 1
    dpd = beh["dpd"]
    n_months = dpd.shape[1]
    horizon_end = min(idx + horizon_months, n_months - 1)

    already_in_default = dpd[:, idx] >= DEFAULT_DPD_THRESHOLD
    horizon_window = dpd[:, idx + 1: horizon_end + 1]
    reaches_default = (horizon_window >= DEFAULT_DPD_THRESHOLD).any(axis=1) if horizon_window.shape[1] > 0 else np.zeros(len(customers), dtype=bool)
    default_12m = np.where(already_in_default, np.nan, reaches_default.astype(float))

    delinquent_90d = (dpd[:, idx] >= DEFAULT_DPD_THRESHOLD).astype(int)

    apr = credit_accounts.set_index("customer_id").loc[customers["customer_id"], "apr"].to_numpy()
    future_idx = min(idx + horizon_months, n_months - 1)
    future_utilization = beh["utilization"][:, future_idx]
    future_monthly_spend = beh["spend"][:, future_idx]
    future_avg_balance = beh["avg_balance"][:, future_idx]
    future_interest_revenue = future_avg_balance * apr / 12
    future_interchange_revenue = future_monthly_spend * interchange_rate

    return pd.DataFrame({
        "customer_id": customers["customer_id"].to_numpy(),
        "cohort": cohort_name,
        "snapshot_month": snapshot_month,
        "snapshot_period": str(pd.period_range("2023-01", periods=n_months, freq="M")[idx]),
        "eligible_for_default_label": ~already_in_default,
        "default_12m": default_12m,
        "delinquent_90d": delinquent_90d,
        "days_past_due_at_snapshot": dpd[:, idx].astype(int),
        "utilization_at_snapshot": beh["utilization"][:, idx],
        "future_utilization": future_utilization,
        "future_monthly_spend": np.round(future_monthly_spend, 2),
        "future_interest_revenue": np.round(future_interest_revenue, 2),
        "future_interchange_revenue": np.round(future_interchange_revenue, 2),
    })


def build_all_labels(
    customers: pd.DataFrame, beh: dict[str, np.ndarray], credit_accounts: pd.DataFrame, config: dict,
) -> pd.DataFrame:
    snap_cfg = config["snapshots"]
    interchange_rate = config["economics"]["interchange_rate"]
    cohorts = [
        ("train", snap_cfg["train_month"]),
        ("validation", snap_cfg["validation_month"]),
        ("test", snap_cfg["test_month"]),
    ]
    frames = [
        build_labels_for_snapshot(
            customers, beh, credit_accounts, month, snap_cfg["horizon_months"], interchange_rate, name,
        )
        for name, month in cohorts
    ]
    return pd.concat(frames, ignore_index=True)
