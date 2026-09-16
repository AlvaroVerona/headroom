"""Tests for the validation engine: each check function against small
handcrafted fixtures (precise, fast), plus an integration test of the full
engine against the real generated data (RAW = VALIDATED + QUARANTINED,
quarantine schema, score bounds, and a regression test for the by_segment
denominator-scope bug found while building this)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_limit_optimizer.data import completeness, consistency, duplicates, validity
from credit_limit_optimizer.data.ingestion import load_all_raw
from credit_limit_optimizer.data.validation import (
    QUARANTINE_SEVERITIES, build_quarantine_ledger, compute_quality_score,
    run_all_checks, run_validation, split_validated_and_quarantine,
)
from credit_limit_optimizer.utils.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_missing_annual_income_detected(config):
    df = pd.DataFrame({
        "record_id": ["CUSTOMERS_00000001", "CUSTOMERS_00000002"],
        "customer_id": ["C0000001", "C0000002"], "age": [30, 40],
        "annual_income": [None, 30000.0], "employment_status": ["employed", "employed"],
        "employment_tenure_months": [12, 24], "country": ["DE", "DE"],
    })
    issues = completeness.check_completeness({"customers": df, "credit_accounts": pd.DataFrame(columns=df.columns),
                                               "transactions": pd.DataFrame(columns=df.columns), "payments": pd.DataFrame(columns=df.columns)}, config)
    hit = issues[(issues["record_id"] == "CUSTOMERS_00000001") & (issues["rule_id"] == "missing_annual_income")]
    assert len(hit) == 1
    assert hit.iloc[0]["severity"] == config["quality"]["severity"]["missing_annual_income"]


def test_exact_duplicate_customer_detected(config):
    df = pd.DataFrame({
        "record_id": ["CUSTOMERS_00000001", "CUSTOMERS_00000002", "CUSTOMERS_00000003"],
        "customer_id": ["C1", "C1", "C2"], "age": [30, 30, 40],
    })
    issues = duplicates.check_exact_duplicates(df, "customers", "CRITICAL")
    assert set(issues["record_id"]) == {"CUSTOMERS_00000001", "CUSTOMERS_00000002"}


def test_invalid_age_detected():
    df = pd.DataFrame({"record_id": ["R1", "R2"], "age": [12, 30]})
    issues = validity.check_below_minimum(df, "customers", "age", 18, "invalid_age", "HIGH")
    assert list(issues["record_id"]) == ["R1"]


def test_invalid_utilization_detected():
    df = pd.DataFrame({"record_id": ["R1", "R2"], "utilization_rate": [1.2, 0.5]})
    issues = validity.check_above_maximum(df, "credit_accounts", "utilization_rate", 1.0, "invalid_utilization", "HIGH")
    assert list(issues["record_id"]) == ["R1"]


def test_negative_payment_amount_detected():
    df = pd.DataFrame({"record_id": ["R1", "R2"], "amount": [-50.0, 50.0]})
    issues = validity.check_negative(df, "payments", "amount", "invalid_payment_amount", "HIGH")
    assert list(issues["record_id"]) == ["R1"]


def test_available_credit_exceeds_limit_detected():
    df = pd.DataFrame({
        "record_id": ["R1", "R2"], "current_credit_limit": [1000.0, 1000.0],
        "available_credit": [1200.0, 800.0],
    })
    issues = consistency.check_available_credit_exceeds_limit(df, tolerance=0.01, severity="HIGH")
    assert list(issues["record_id"]) == ["R1"]


def test_balance_exceeds_limit_detected():
    df = pd.DataFrame({"record_id": ["R1", "R2"], "current_credit_limit": [1000.0, 1000.0], "current_balance": [1200.0, 800.0]})
    issues = consistency.check_balance_exceeds_limit(df, tolerance=0.01, severity="HIGH")
    assert list(issues["record_id"]) == ["R1"]


def test_referential_integrity_detected():
    df = pd.DataFrame({"record_id": ["R1", "R2"], "customer_id": ["C_BOGUS", "C1"]})
    issues = consistency.check_referential_integrity(df, "transactions", "customer_id", {"C1", "C2"}, "invalid_reference_transaction", "CRITICAL")
    assert list(issues["record_id"]) == ["R1"]


@pytest.fixture(scope="module")
def real_datasets():
    return load_all_raw()


def test_run_all_checks_on_real_data(real_datasets, config):
    issues = run_all_checks(real_datasets, config)
    assert len(issues) > 0
    assert set(issues["severity"]) <= {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}


def test_split_partitions_without_loss_or_duplication(real_datasets, config):
    issues = run_all_checks(real_datasets, config)
    validated, quarantined = split_validated_and_quarantine(real_datasets, issues)
    for name in real_datasets:
        assert len(validated[name]) + len(quarantined[name]) == len(real_datasets[name])
        assert set(validated[name]["record_id"]).isdisjoint(set(quarantined[name]["record_id"]))


def test_quarantine_ledger_schema(real_datasets, config):
    issues = run_all_checks(real_datasets, config)
    _, quarantined = split_validated_and_quarantine(real_datasets, issues)
    ledgers = build_quarantine_ledger(quarantined, issues)
    expected_cols = {"record_id", "rule_id", "severity", "reason", "detected_at"}
    for name, ledger in ledgers.items():
        assert expected_cols <= set(ledger.columns)
        if len(ledger) > 0:
            assert ledger["severity"].isin(QUARANTINE_SEVERITIES).all()


def test_quality_score_within_bounds(real_datasets, config):
    issues = run_all_checks(real_datasets, config)
    score = compute_quality_score(issues, real_datasets, config)
    assert 0 <= score["overall_score"] <= 100
    for v in score["by_dataset"].values():
        assert 0 <= v <= 100


def test_by_field_score_scoped_to_owning_datasets_only(real_datasets, config):
    """Regression test: an earlier version divided every field's issue
    count by the total row count across all 4 datasets combined, even for
    a field (e.g. "available_credit") that only exists in credit_accounts.csv
    -- diluting its score toward 100 with ~6.17M rows from datasets that
    don't even have that column. A field confined to one dataset must score
    close to a manual recomputation against that dataset's own row count;
    a field genuinely shared across all 4 (customer_id) is unaffected."""
    issues = run_all_checks(real_datasets, config)
    score = compute_quality_score(issues, real_datasets, config)

    field_issues = issues[issues["field"].notna()]
    single_dataset_fields = [
        f for f in field_issues["field"].dropna().unique()
        if sum(1 for df in real_datasets.values() if f in df.columns) == 1
    ]
    assert single_dataset_fields, "expected at least one field confined to a single dataset"
    for field in single_dataset_fields:
        owning_dataset = next(name for name, df in real_datasets.items() if field in df.columns)
        own_denom = len(real_datasets[owning_dataset])
        weighted_penalty = field_issues.loc[field_issues["field"] == field, "severity"].map(
            {"CRITICAL": 1.0, "HIGH": 0.6, "MEDIUM": 0.3, "LOW": 0.1, "INFO": 0.02}
        ).fillna(0.3).sum()
        expected = round(100 * (1 - min(weighted_penalty / own_denom, 1.0)), 2)
        assert abs(score["by_field"][field] - expected) < 0.01, (
            f"{field}: got {score['by_field'][field]}, expected {expected} scoped to {owning_dataset}"
        )

    # customer_id is a genuine exception: it's a column in all 4 datasets,
    # so its scoped denominator equals the old unscoped total_rows exactly.
    total_rows = sum(len(df) for df in real_datasets.values())
    assert all(field in real_datasets["customers"].columns for field in ["customer_id"])
    cid_denom = sum(len(df) for df in real_datasets.values() if "customer_id" in df.columns)
    assert cid_denom == total_rows


def test_by_segment_score_is_not_degenerate(real_datasets, config):
    """Regression test: an earlier version summed issues from ALL 4
    datasets into the by_segment numerator while dividing by a
    customer-row-only denominator, which floored every segment's score to
    0.0. by_segment must stay close to the customers dataset's own score,
    not blow through the floor."""
    issues = run_all_checks(real_datasets, config)
    score = compute_quality_score(issues, real_datasets, config)
    for seg, seg_score in score["by_segment"].items():
        assert seg_score > 50, f"{seg} score suspiciously low: {seg_score}"
        assert abs(seg_score - score["by_dataset"]["customers"]) < 15


def test_by_month_covers_full_history_and_is_not_degenerate(real_datasets, config):
    """§11 requires a quality score broken down by month -- an earlier
    version of compute_quality_score didn't compute this dimension at all.
    Scoped to transactions.csv + payments.csv (the only datasets with a
    real per-record calendar timestamp); should cover all 36 months and
    stay close to those two datasets' own scores (issues are injected at a
    uniform rate, not correlated with month, so no month should be a
    degenerate outlier)."""
    issues = run_all_checks(real_datasets, config)
    score = compute_quality_score(issues, real_datasets, config)
    assert len(score["by_month"]) == 36
    assert list(score["by_month"]) == sorted(score["by_month"])
    ts_avg = (score["by_dataset"]["transactions"] + score["by_dataset"]["payments"]) / 2
    for month, month_score in score["by_month"].items():
        assert 0 <= month_score <= 100
        assert abs(month_score - ts_avg) < 5, f"{month} score {month_score} far from transactions/payments average {ts_avg}"


def test_full_validation_run_produces_processed_and_quarantine_files():
    report = run_validation()
    assert "overall_score" in report
    from credit_limit_optimizer.data.validation import PROCESSED_DIR, QUARANTINE_DIR
    for name in ["customers", "credit_accounts", "transactions", "payments"]:
        assert (PROCESSED_DIR / f"{name}.csv").exists()
        assert (QUARANTINE_DIR / f"{name}.csv").exists()
