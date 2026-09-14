"""Completeness checks (§10): required fields must not be null, with
severity driven by config/settings.yaml's quality.severity map. Fields not
explicitly listed there fall back to LOW (optional metadata)."""

from __future__ import annotations

import pandas as pd

ISSUE_COLUMNS = ["record_id", "dataset", "rule_id", "severity", "reason", "field"]
DEFAULT_SEVERITY = "LOW"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=ISSUE_COLUMNS)


def check_missing(df: pd.DataFrame, dataset: str, field: str, severity: str) -> pd.DataFrame:
    if field not in df.columns:
        return _empty()
    mask = df[field].isna() | (df[field].astype(str).str.strip() == "")
    if not mask.any():
        return _empty()
    return pd.DataFrame({
        "record_id": df.loc[mask, "record_id"], "dataset": dataset,
        "rule_id": f"missing_{field}", "severity": severity,
        "reason": f"Required field '{field}' is missing", "field": field,
    })


def check_completeness_dataset(df: pd.DataFrame, dataset: str, fields: list[str], severity_map: dict) -> pd.DataFrame:
    frames = [
        check_missing(df, dataset, field, severity_map.get(f"missing_{field}", DEFAULT_SEVERITY))
        for field in fields
    ]
    return pd.concat(frames, ignore_index=True) if frames else _empty()


def check_completeness(datasets: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    severity_map = config["quality"]["severity"]
    field_map = {
        "customers": ["customer_id", "age", "annual_income", "employment_status", "employment_tenure_months", "country"],
        "credit_accounts": ["customer_id", "account_id", "current_credit_limit", "current_balance", "apr"],
        "transactions": ["transaction_id", "customer_id", "amount", "merchant_category", "channel"],
        "payments": ["payment_id", "customer_id", "amount", "payment_type"],
    }
    frames = [
        check_completeness_dataset(datasets[name], name, fields, severity_map)
        for name, fields in field_map.items()
    ]
    return pd.concat(frames, ignore_index=True)
