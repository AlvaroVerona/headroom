"""Entry point for synthetic data generation.

Run: python -m credit_limit_optimizer.data.generator

Produces data/raw/{customers,credit_accounts,transactions,payments,
monthly_customer_behavior,labels}.csv, including intentionally injected
data-quality issues for the Phase 1 validation module to detect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_limit_optimizer.utils.config import PROJECT_ROOT, load_config
from credit_limit_optimizer.utils.logging import get_logger
from credit_limit_optimizer.data import master_data, behavior_series, monthly_table, credit_accounts as credit_accounts_mod
from credit_limit_optimizer.data import transactions_payments, labels as labels_mod, quality_injection

log = get_logger("generator")

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def generate_all(rng: np.random.Generator, cfg: dict) -> dict[str, pd.DataFrame]:
    data_cfg = cfg["data"]
    n_customers, n_months = data_cfg["n_customers"], data_cfg["n_months"]
    start_date = pd.Timestamp(data_cfg["start_date"])

    log.info("Generating %d customers (seed=%d)", n_customers, cfg["random_seed"])
    customers = master_data.generate_customers(rng, n_customers, start_date)
    log.info("Customer segments: %s", customers["customer_segment"].value_counts().to_dict())

    log.info("Assigning initial credit limits and simulating %d months of behavior", n_months)
    credit_limit = behavior_series.assign_initial_credit_limits(rng, customers)
    beh = behavior_series.simulate_behavior(rng, customers, credit_limit, n_months, start_date)

    monthly_behavior = monthly_table.build_monthly_table(customers, beh, start_date)
    log.info("Monthly behavior table: %d rows", len(monthly_behavior))

    accounts = credit_accounts_mod.build_credit_accounts(rng, customers, credit_limit, beh)
    log.info("Credit accounts: %d rows", len(accounts))

    log.info("Sampling transactions and payments from simulated behavior")
    transactions = transactions_payments.generate_transactions(rng, customers, beh, start_date)
    payments = transactions_payments.generate_payments(rng, customers, beh, start_date)
    log.info("Transactions: %d rows, Payments: %d rows", len(transactions), len(payments))

    log.info("Building default_12m / delinquent_90d / future_* labels for train/validation/test snapshots")
    label_table = labels_mod.build_all_labels(customers, beh, accounts, cfg)
    default_rate = label_table.dropna(subset=["default_12m"]).groupby("cohort")["default_12m"].mean()
    log.info("default_12m rate by cohort: %s", default_rate.round(4).to_dict())

    return {
        "customers": customers, "credit_accounts": accounts, "transactions": transactions,
        "payments": payments, "monthly_customer_behavior": monthly_behavior, "labels": label_table,
    }


def inject_quality_issues(datasets: dict[str, pd.DataFrame], config: dict, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    q = config["quality"]
    mr = q["missing_rate"]

    customers = datasets["customers"]
    customers = quality_injection.inject_missing_values(customers, rng, {
        "annual_income": mr["annual_income"], "employment_status": mr["employment_status"],
        "country": mr["optional_metadata"], "employment_tenure_months": mr["default"],
    })
    customers = quality_injection.inject_exact_duplicates(customers, rng, q["duplicate_rate"])
    customers = quality_injection.inject_impossible_values(customers, rng, q["invalid_rate"] / 3, "age", "below_min")
    customers = quality_injection.inject_impossible_values(customers, rng, q["invalid_rate"] / 3, "annual_income", "negative")

    accounts = datasets["credit_accounts"]
    accounts = quality_injection.inject_missing_values(accounts, rng, {
        "current_credit_limit": mr["default"], "apr": mr["optional_metadata"],
    })
    accounts = quality_injection.inject_impossible_values(accounts, rng, q["invalid_rate"] / 3, "current_credit_limit", "negative")
    accounts = quality_injection.inject_impossible_values(accounts, rng, q["invalid_rate"] / 3, "utilization_rate", "above_one")
    accounts = quality_injection.inject_inconsistent_credit_fields(accounts, rng, q["inconsistency_rate"])
    accounts = quality_injection.inject_invalid_references(accounts, rng, q["invalid_rate"] / 4, "customer_id")

    transactions = datasets["transactions"]
    transactions = quality_injection.inject_exact_duplicates(transactions, rng, q["duplicate_rate"])
    transactions = quality_injection.inject_missing_values(transactions, rng, {"channel": mr["optional_metadata"]})
    transactions = quality_injection.inject_impossible_values(transactions, rng, q["invalid_rate"] / 3, "amount", "negative")
    transactions = quality_injection.inject_invalid_references(transactions, rng, q["invalid_rate"] / 4, "customer_id")

    payments = datasets["payments"]
    payments = quality_injection.inject_missing_values(payments, rng, {"payment_type": mr["optional_metadata"]})
    payments = quality_injection.inject_impossible_values(payments, rng, q["invalid_rate"] / 3, "amount", "negative")
    payments = quality_injection.inject_invalid_references(payments, rng, q["invalid_rate"] / 4, "customer_id")

    return {
        "customers": customers, "credit_accounts": accounts, "transactions": transactions,
        "payments": payments, "monthly_customer_behavior": datasets["monthly_customer_behavior"],
        "labels": datasets["labels"],
    }


def write_raw(datasets: dict[str, pd.DataFrame]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in datasets.items():
        path = RAW_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        log.info("Wrote %s (%d rows)", path.relative_to(PROJECT_ROOT), len(df))


def main() -> None:
    config = load_config()
    rng = np.random.default_rng(config["random_seed"])
    datasets = generate_all(rng, config)
    datasets = inject_quality_issues(datasets, config, rng)
    write_raw(datasets)
    log.info("Data generation complete.")


if __name__ == "__main__":
    main()
