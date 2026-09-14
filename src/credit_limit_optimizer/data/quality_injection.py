"""Injects realistic, controlled data-quality problems (§10) into the
otherwise-clean generated data. Every function takes an explicit rng and
rate so injection is reproducible and each issue type is independently
testable. Applied to the RAW layer only -- validation.py (Phase 1, next)
is expected to catch what these functions break."""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_missing_values(df: pd.DataFrame, rng: np.random.Generator, rates: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    for col, rate in rates.items():
        if col not in df.columns or rate <= 0:
            continue
        n_missing = int(len(df) * rate)
        if n_missing == 0:
            continue
        idx = rng.choice(df.index, size=n_missing, replace=False)
        df.loc[idx, col] = np.nan
    return df


def inject_exact_duplicates(df: pd.DataFrame, rng: np.random.Generator, rate: float) -> pd.DataFrame:
    n_dupes = int(len(df) * rate)
    if n_dupes == 0:
        return df
    dupe_idx = rng.choice(df.index, size=n_dupes, replace=False)
    return pd.concat([df, df.loc[dupe_idx]], ignore_index=True)


def inject_impossible_values(df: pd.DataFrame, rng: np.random.Generator, rate: float, col: str, kind: str) -> pd.DataFrame:
    """kind: 'negative' flips sign, 'below_min' sets to an impossible
    below-range value, 'above_one' pushes a [0,1]-bounded field above 1."""
    df = df.copy()
    n = int(len(df) * rate)
    if n == 0 or col not in df.columns:
        return df
    idx = rng.choice(df.index, size=n, replace=False)
    if kind == "negative":
        df.loc[idx, col] = -df.loc[idx, col].abs() - 1
    elif kind == "below_min":
        # Integer columns (e.g. age) reject a float assignment outright in
        # pandas >= 2 rather than silently upcasting, so match dtype.
        values = rng.uniform(0, 17, size=n)
        if pd.api.types.is_integer_dtype(df[col]):
            values = np.round(values).astype(int)
        df.loc[idx, col] = values
    elif kind == "above_one":
        df.loc[idx, col] = 1.0 + rng.uniform(0.05, 0.5, size=n)
    return df


def inject_inconsistent_credit_fields(df: pd.DataFrame, rng: np.random.Generator, rate: float) -> pd.DataFrame:
    """Breaks available_credit = limit - balance and/or balance <= limit
    for a subset of accounts -- the "inconsistent values" example the spec
    gives explicitly (§10). Only samples rows with an intact, positive
    current_credit_limit -- if an earlier injection step already made that
    NaN or negative, deriving balance/utilization from it would cascade
    that into an unrelated, accidental NaN instead of the deliberate
    inconsistency this function is meant to create."""
    df = df.copy()
    eligible = df.index[df["current_credit_limit"].notna() & (df["current_credit_limit"] > 0)]
    n = int(len(df) * rate)
    if n == 0 or len(eligible) == 0:
        return df
    n = min(n, len(eligible))
    idx = rng.choice(eligible, size=n, replace=False)
    half = idx[: len(idx) // 2]
    other_half = idx[len(idx) // 2:]
    # available_credit inconsistent with limit - balance
    df.loc[half, "available_credit"] = df.loc[half, "current_credit_limit"] * rng.uniform(1.05, 1.3, size=len(half))
    # balance exceeds limit without an overdraft/fee explanation
    df.loc[other_half, "current_balance"] = df.loc[other_half, "current_credit_limit"] * rng.uniform(1.05, 1.25, size=len(other_half))
    df.loc[other_half, "utilization_rate"] = df.loc[other_half, "current_balance"] / df.loc[other_half, "current_credit_limit"]
    return df


def inject_invalid_references(df: pd.DataFrame, rng: np.random.Generator, rate: float, col: str) -> pd.DataFrame:
    df = df.copy()
    n = int(len(df) * rate)
    if n == 0 or col not in df.columns:
        return df
    idx = rng.choice(df.index, size=n, replace=False)
    bogus = rng.integers(9_000_000, 9_999_999, size=n)
    prefix = "C" if col == "customer_id" else ("A" if col == "account_id" else col[0].upper())
    df.loc[idx, col] = [f"{prefix}{b}" for b in bogus]
    return df
