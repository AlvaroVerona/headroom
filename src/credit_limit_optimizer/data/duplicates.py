"""Duplicate detection (§10): exact full-row duplicates for customers and
transactions -- the two cases the spec explicitly calls out."""

from __future__ import annotations

import pandas as pd

ISSUE_COLUMNS = ["record_id", "dataset", "rule_id", "severity", "reason", "field"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=ISSUE_COLUMNS)


def check_exact_duplicates(df: pd.DataFrame, dataset: str, severity: str) -> pd.DataFrame:
    content_cols = [c for c in df.columns if c != "record_id"]
    mask = df.duplicated(subset=content_cols, keep=False)
    if not mask.any():
        return _empty()
    return pd.DataFrame({
        "record_id": df.loc[mask, "record_id"], "dataset": dataset,
        "rule_id": f"duplicate_{dataset[:-1] if dataset.endswith('s') else dataset}",
        "severity": severity, "reason": "Full record is byte-identical to another record in the file",
        "field": None,
    })


def check_duplicates(datasets: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    severity_map = config["quality"]["severity"]
    frames = [
        check_exact_duplicates(datasets["customers"], "customers", severity_map["duplicate_customer"]),
        check_exact_duplicates(datasets["transactions"], "transactions", severity_map["duplicate_transaction"]),
    ]
    return pd.concat(frames, ignore_index=True)
