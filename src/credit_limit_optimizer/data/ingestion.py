"""Reads data/raw/*.csv and assigns the row-level lineage key (record_id)
every quality check and the quarantine layer key off. Position-based, not
derived from any business key -- a business key (customer_id, etc.) can
itself be the thing that's missing, which would make it useless as a row
identifier exactly when it's needed most."""

from __future__ import annotations

import pandas as pd

from credit_limit_optimizer.utils.config import PROJECT_ROOT
from credit_limit_optimizer.utils.logging import get_logger

log = get_logger("ingestion")

RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATASET_FILES = {
    "customers": "customers.csv",
    "credit_accounts": "credit_accounts.csv",
    "transactions": "transactions.csv",
    "payments": "payments.csv",
}


def load_raw(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in DATASET_FILES:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of {list(DATASET_FILES)}.")
    path = RAW_DIR / DATASET_FILES[dataset_name]
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found at {path}. Run `make generate-data` "
            "(or `python -m credit_limit_optimizer.data.generator`) first."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    df.insert(0, "record_id", [f"{dataset_name.upper()}_{i + 1:08d}" for i in range(len(df))])
    log.info("Ingested %s: %d rows from %s", dataset_name, len(df), path.name)
    return df


def load_all_raw() -> dict[str, pd.DataFrame]:
    return {name: load_raw(name) for name in DATASET_FILES}
