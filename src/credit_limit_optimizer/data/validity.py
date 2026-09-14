"""Impossible-value checks (§10): age < 18, income < 0, credit_limit < 0,
utilization_rate > 1, payment_amount < 0 -- the spec's own examples."""

from __future__ import annotations

import pandas as pd

ISSUE_COLUMNS = ["record_id", "dataset", "rule_id", "severity", "reason", "field"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=ISSUE_COLUMNS)


def _issues_for_mask(df: pd.DataFrame, dataset: str, mask: pd.Series, rule: str, severity: str, reason: str, field: str) -> pd.DataFrame:
    if not mask.any():
        return _empty()
    return pd.DataFrame({
        "record_id": df.loc[mask, "record_id"], "dataset": dataset,
        "rule_id": rule, "severity": severity, "reason": reason, "field": field,
    })


def check_below_minimum(df: pd.DataFrame, dataset: str, field: str, minimum: float, rule: str, severity: str) -> pd.DataFrame:
    values = pd.to_numeric(df[field], errors="coerce")
    mask = values.notna() & (values < minimum)
    return _issues_for_mask(df, dataset, mask, rule, severity, f"{field} is below the minimum valid value ({minimum})", field)


def check_negative(df: pd.DataFrame, dataset: str, field: str, rule: str, severity: str) -> pd.DataFrame:
    values = pd.to_numeric(df[field], errors="coerce")
    mask = values.notna() & (values < 0)
    return _issues_for_mask(df, dataset, mask, rule, severity, f"{field} is negative, which is not a valid value for this field", field)


def check_above_maximum(df: pd.DataFrame, dataset: str, field: str, maximum: float, rule: str, severity: str) -> pd.DataFrame:
    values = pd.to_numeric(df[field], errors="coerce")
    mask = values.notna() & (values > maximum)
    return _issues_for_mask(df, dataset, mask, rule, severity, f"{field} exceeds the maximum valid value ({maximum})", field)


def check_validity(datasets: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    severity_map = config["quality"]["severity"]
    customers, accounts = datasets["customers"], datasets["credit_accounts"]
    transactions, payments = datasets["transactions"], datasets["payments"]

    frames = [
        check_below_minimum(customers, "customers", "age", 18, "invalid_age", severity_map["invalid_age"]),
        check_negative(customers, "customers", "annual_income", "invalid_income", severity_map["invalid_income"]),
        check_negative(accounts, "credit_accounts", "current_credit_limit", "invalid_credit_limit", severity_map["invalid_credit_limit"]),
        check_above_maximum(accounts, "credit_accounts", "utilization_rate", 1.0, "invalid_utilization", severity_map["invalid_utilization"]),
        check_negative(transactions, "transactions", "amount", "invalid_payment_amount", severity_map["invalid_payment_amount"]),
        check_negative(payments, "payments", "amount", "invalid_payment_amount", severity_map["invalid_payment_amount"]),
    ]
    return pd.concat(frames, ignore_index=True)
