"""Inconsistent-value and referential-integrity checks (§10): available_
credit > credit_limit, current_balance > credit_limit "when not explained
by fees/overdraft rules" (this project doesn't model overdraft, so any
occurrence is flagged), and every foreign key (transactions/credit_accounts/
payments -> customers) actually resolving."""

from __future__ import annotations

import numpy as np
import pandas as pd

ISSUE_COLUMNS = ["record_id", "dataset", "rule_id", "severity", "reason", "field"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=ISSUE_COLUMNS)


def _issues_for_mask(df: pd.DataFrame, dataset: str, mask: pd.Series, rule: str, severity: str, reason: str, field: str | None = None) -> pd.DataFrame:
    if not mask.any():
        return _empty()
    return pd.DataFrame({
        "record_id": df.loc[mask, "record_id"], "dataset": dataset,
        "rule_id": rule, "severity": severity, "reason": reason, "field": field,
    })


def check_available_credit_exceeds_limit(df: pd.DataFrame, tolerance: float, severity: str) -> pd.DataFrame:
    limit = pd.to_numeric(df["current_credit_limit"], errors="coerce")
    available = pd.to_numeric(df["available_credit"], errors="coerce")
    mask = limit.notna() & available.notna() & (available > limit + tolerance)
    return _issues_for_mask(df, "credit_accounts", mask, "inconsistent_available_credit", severity,
                             "available_credit exceeds current_credit_limit, which is impossible", "available_credit")


def check_available_credit_matches_identity(df: pd.DataFrame, tolerance: float, severity: str) -> pd.DataFrame:
    limit = pd.to_numeric(df["current_credit_limit"], errors="coerce")
    balance = pd.to_numeric(df["current_balance"], errors="coerce")
    available = pd.to_numeric(df["available_credit"], errors="coerce")
    implied = limit - balance
    mask = limit.notna() & balance.notna() & available.notna() & ((available - implied).abs() > tolerance)
    return _issues_for_mask(df, "credit_accounts", mask, "inconsistent_available_credit", severity,
                             "available_credit does not equal current_credit_limit - current_balance", "available_credit")


def check_balance_exceeds_limit(df: pd.DataFrame, tolerance: float, severity: str) -> pd.DataFrame:
    limit = pd.to_numeric(df["current_credit_limit"], errors="coerce")
    balance = pd.to_numeric(df["current_balance"], errors="coerce")
    mask = limit.notna() & balance.notna() & (balance > limit + tolerance)
    return _issues_for_mask(df, "credit_accounts", mask, "inconsistent_balance_exceeds_limit", severity,
                             "current_balance exceeds current_credit_limit with no overdraft/fee explanation "
                             "in this product model", "current_balance")


def check_referential_integrity(df: pd.DataFrame, dataset: str, col: str, valid_values: set, rule_id: str, severity: str) -> pd.DataFrame:
    if col not in df.columns:
        return _empty()
    present = df[col].notna()
    mask = present & ~df[col].isin(valid_values)
    return _issues_for_mask(df, dataset, mask, rule_id, severity, f"{col} does not exist in master data", col)


def check_consistency(datasets: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    severity_map = config["quality"]["severity"]
    tolerance = config["quality"]["tolerance"]
    customers, accounts = datasets["customers"], datasets["credit_accounts"]
    transactions, payments = datasets["transactions"], datasets["payments"]
    valid_customer_ids = set(customers["customer_id"])

    frames = [
        check_available_credit_exceeds_limit(accounts, tolerance, severity_map["inconsistent_available_credit"]),
        check_available_credit_matches_identity(accounts, tolerance, severity_map["inconsistent_available_credit"]),
        check_balance_exceeds_limit(accounts, tolerance, severity_map["inconsistent_balance_exceeds_limit"]),
        check_referential_integrity(accounts, "credit_accounts", "customer_id", valid_customer_ids,
                                     "invalid_reference_account", severity_map["invalid_reference_account"]),
        check_referential_integrity(transactions, "transactions", "customer_id", valid_customer_ids,
                                     "invalid_reference_transaction", severity_map["invalid_reference_transaction"]),
        check_referential_integrity(payments, "payments", "customer_id", valid_customer_ids,
                                     "invalid_reference_payment", severity_map["invalid_reference_payment"]),
    ]
    return pd.concat(frames, ignore_index=True)
