"""Orchestrates the full validation engine (§10-11): ingest raw -> run
every check -> split into VALIDATED (data/processed/) and QUARANTINED
(data/quarantine/) -> compute the 0-100 Data Quality Score, globally and
by dataset/field/customer_segment/month, from the *actual* check results
(never hardcoded).

Run: python -m credit_limit_optimizer.data.validation
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from credit_limit_optimizer.utils.config import load_config, PROJECT_ROOT
from credit_limit_optimizer.utils.logging import get_logger
from credit_limit_optimizer.data.ingestion import load_all_raw
from credit_limit_optimizer.data import completeness, duplicates, validity, consistency

log = get_logger("validation")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
QUARANTINE_DIR = PROJECT_ROOT / "data" / "quarantine"
REPORT_PATH = PROJECT_ROOT / "reports" / "outputs" / "quality_report.json"

QUARANTINE_SEVERITIES = {"CRITICAL", "HIGH"}
SEVERITY_WEIGHT = {"CRITICAL": 1.0, "HIGH": 0.6, "MEDIUM": 0.3, "LOW": 0.1, "INFO": 0.02}


def run_all_checks(datasets: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    frames = [
        completeness.check_completeness(datasets, config),
        duplicates.check_duplicates(datasets, config),
        validity.check_validity(datasets, config),
        consistency.check_consistency(datasets, config),
    ]
    return pd.concat(frames, ignore_index=True)


def split_validated_and_quarantine(
    datasets: dict[str, pd.DataFrame], issues: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    row_issues = issues[issues["record_id"].notna()]
    quarantine_ids = set(row_issues.loc[row_issues["severity"].isin(QUARANTINE_SEVERITIES), "record_id"])

    validated, quarantined = {}, {}
    for name, df in datasets.items():
        is_quarantined = df["record_id"].isin(quarantine_ids)
        validated[name] = df.loc[~is_quarantined].reset_index(drop=True)
        quarantined[name] = df.loc[is_quarantined].reset_index(drop=True)
    return validated, quarantined


def build_quarantine_ledger(quarantined: dict[str, pd.DataFrame], issues: pd.DataFrame) -> dict[str, pd.DataFrame]:
    detected_at = datetime.now(timezone.utc).isoformat()
    ledgers = {}
    for name, df in quarantined.items():
        if df.empty:
            ledgers[name] = pd.DataFrame(columns=["record_id", "rule_id", "severity", "reason", "detected_at"])
            continue
        relevant = issues[
            (issues["dataset"] == name) & issues["record_id"].isin(df["record_id"])
            & issues["severity"].isin(QUARANTINE_SEVERITIES)
        ]
        ledger = relevant[["record_id", "rule_id", "severity", "reason"]].copy()
        ledger["detected_at"] = detected_at
        ledgers[name] = ledger.reset_index(drop=True)
    return ledgers


# The only two datasets with a real per-record calendar timestamp --
# customers.csv and credit_accounts.csv are single-row-per-customer
# snapshots with no comparable monthly axis, so "quality by month" (§11)
# is scoped to these, the same way "quality by segment" is scoped to
# customers.csv only (see compute_quality_score).
TIMESTAMP_DATASETS = {"transactions", "payments"}


def _attach_month(issues: pd.DataFrame, datasets: dict[str, pd.DataFrame]) -> pd.Series:
    months = pd.Series(index=issues.index, dtype=object)
    for name in TIMESTAMP_DATASETS:
        mask = issues["dataset"] == name
        if not mask.any():
            continue
        df = datasets[name]
        record_to_month = df.set_index("record_id")["timestamp"].str[:7]
        months.loc[mask] = issues.loc[mask, "record_id"].map(record_to_month).to_numpy()
    return months


def _attach_customer_segment(issues: pd.DataFrame, datasets: dict[str, pd.DataFrame], customers: pd.DataFrame) -> pd.Series:
    # customer_id is not guaranteed unique in the raw data -- duplicate
    # customers are one of the injected quality issues -- so build the
    # lookup off a deduplicated frame (an exact duplicate maps to the same
    # segment anyway, so which copy survives the dedup doesn't matter).
    dedup_customers = customers.drop_duplicates(subset="customer_id")
    segment_by_customer = dedup_customers.set_index("customer_id")["customer_segment"]
    segments = pd.Series(index=issues.index, dtype=object)
    for name, df in datasets.items():
        mask = issues["dataset"] == name
        if not mask.any():
            continue
        sub_issues = issues.loc[mask]
        # record_id is always unique (position-based, assigned at
        # ingestion) regardless of what the row's own data looks like, so
        # no dedup needed here -- unlike customer_id above.
        record_to_customer = df.set_index("record_id")["customer_id"]
        customer_ids = sub_issues["record_id"].map(record_to_customer)
        segments.loc[mask] = customer_ids.map(segment_by_customer).to_numpy()
    return segments


def compute_quality_score(issues: pd.DataFrame, datasets: dict[str, pd.DataFrame], config: dict) -> dict:
    total_rows = sum(len(df) for df in datasets.values())

    def _score(subset: pd.DataFrame, denom: int) -> float:
        if denom == 0:
            return 100.0
        weighted_penalty = subset["severity"].map(SEVERITY_WEIGHT).fillna(0.3).sum()
        penalty_rate = min(weighted_penalty / denom, 1.0)
        return round(100 * (1 - penalty_rate), 2)

    overall_score = _score(issues, total_rows)
    by_dataset = {name: _score(issues[issues["dataset"] == name], len(df)) for name, df in datasets.items()}

    # Scoped per-field to the dataset(s) that actually contain that column,
    # not the full 4-dataset row total -- the same denominator-scope bug
    # class as by_segment below, just milder here. E.g. "channel" only
    # exists in transactions.csv (~4.37M rows); dividing its issue count by
    # all 4 datasets combined (~6.22M rows, including 1.8M payments rows
    # that don't even have a "channel" column) diluted every field score
    # toward 100 rather than reflecting that field's own quality. A field
    # shared by multiple datasets (e.g. "customer_id") correctly sums their
    # row counts.
    field_issues = issues[issues["field"].notna()]
    by_field = {}
    for field in field_issues["field"].dropna().unique():
        field_denom = sum(len(df) for df in datasets.values() if field in df.columns)
        by_field[field] = _score(field_issues[field_issues["field"] == field], field_denom)

    # Scoped to customers.csv's own issues only, not joined across every
    # dataset: a segment's customers can have many thousands of
    # transaction/payment rows each, so summing issues from all 4 datasets
    # while dividing by a customer-row denominator was comparing numerator
    # and denominator at completely different scales (a real bug caught
    # here -- it drove every segment's score to a floored 0.0). "Data
    # quality by segment" is a coherent question for customer-level fields;
    # it isn't a well-defined single number once you fold in a segment's
    # entire transaction history too.
    customer_issues = issues[issues["dataset"] == "customers"]
    segments = _attach_customer_segment(customer_issues, {"customers": datasets["customers"]}, datasets["customers"])
    segment_counts = datasets["customers"]["customer_segment"].value_counts()
    by_segment = {}
    for seg in segment_counts.index:
        seg_issues = customer_issues[segments == seg]
        by_segment[seg] = _score(seg_issues, int(segment_counts[seg]))

    # Scoped to transactions.csv + payments.csv only -- see TIMESTAMP_DATASETS.
    ts_issues = issues[issues["dataset"].isin(TIMESTAMP_DATASETS)]
    months_by_issue = _attach_month(ts_issues, datasets)
    month_row_counts = pd.concat(
        [datasets[name]["timestamp"].str[:7] for name in TIMESTAMP_DATASETS]
    ).value_counts()
    by_month = {
        month: _score(ts_issues[months_by_issue == month], int(month_row_counts[month]))
        for month in sorted(month_row_counts.index)
    }

    return {
        "overall_score": overall_score, "by_dataset": by_dataset,
        "by_field": by_field, "by_segment": by_segment, "by_month": by_month,
    }


def write_layers(validated: dict[str, pd.DataFrame], quarantine_ledgers: dict[str, pd.DataFrame]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in validated.items():
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        log.info("Wrote validated data/processed/%s.csv (%d rows)", name, len(df))
    for name, df in quarantine_ledgers.items():
        df.to_csv(QUARANTINE_DIR / f"{name}.csv", index=False)
        log.info("Wrote quarantine data/quarantine/%s.csv (%d rows)", name, len(df))


def run_validation() -> dict:
    config = load_config()
    datasets = load_all_raw()

    log.info("Running quality checks across %d datasets", len(datasets))
    issues = run_all_checks(datasets, config)
    log.info("Found %d issues", len(issues))
    log.info("Issues by severity: %s", issues["severity"].value_counts().to_dict())

    validated, quarantined = split_validated_and_quarantine(datasets, issues)
    quarantine_ledgers = build_quarantine_ledger(quarantined, issues)
    total_quarantined = sum(len(df) for df in quarantined.values())
    log.info("Quarantined %d records across %s", total_quarantined, list(datasets.keys()))

    score = compute_quality_score(issues, datasets, config)
    log.info("Data Quality Score: %.1f/100 -- by dataset: %s", score["overall_score"], score["by_dataset"])

    write_layers(validated, quarantine_ledgers)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_counts_raw": {name: len(df) for name, df in datasets.items()},
        "row_counts_validated": {name: len(df) for name, df in validated.items()},
        "row_counts_quarantined": {name: len(df) for name, df in quarantined.items()},
        "issue_counts_by_severity": issues["severity"].value_counts().to_dict(),
        "issue_counts_by_rule": issues["rule_id"].value_counts().to_dict(),
        **score,
    }
    with REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Wrote quality report to %s", REPORT_PATH.relative_to(PROJECT_ROOT))

    return report


def main() -> None:
    run_validation()


if __name__ == "__main__":
    main()
