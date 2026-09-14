"""Customer master data generation. `customer_segment` is a real, disclosed
field (per the spec's customers.csv schema) that also acts as a LATENT risk
tier driving correlated downstream behavior in behavior_series.py --
population weights and correlation strengths live here as named constants,
not in config.yaml, since they're distributional shape parameters (like
breach-point's financial_series.py), not policy thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COUNTRIES = ["DE", "FR", "ES", "IT", "NL"]
COUNTRY_WEIGHTS = [0.32, 0.24, 0.18, 0.16, 0.10]

EMPLOYMENT_STATUSES = ["employed", "self_employed", "unemployed", "retired", "student"]
EMPLOYMENT_WEIGHTS = [0.62, 0.14, 0.06, 0.12, 0.06]
EMPLOYMENT_INCOME_FACTOR = {
    "employed": 1.00, "self_employed": 1.10, "unemployed": 0.25, "retired": 0.60, "student": 0.30,
}

SEGMENTS = ["prime", "near_prime", "subprime"]
SEGMENT_WEIGHTS = [0.35, 0.40, 0.25]
SEGMENT_INCOME_FACTOR = {"prime": 1.30, "near_prime": 1.00, "subprime": 0.75}
# Baseline monthly hazard-of-distress level by segment (used by
# behavior_series.py to simulate delinquency) -- higher segment risk means
# a higher baseline probability of a missed-payment month, before any
# idiosyncratic shock. Deliberately modest gaps + substantial per-customer
# noise elsewhere so default risk isn't trivially separable by segment
# alone (spec section 8: "Do not make the target trivially predictable").
SEGMENT_BASE_DISTRESS = {"prime": 0.018, "near_prime": 0.040, "subprime": 0.080}

BASE_ANNUAL_INCOME = 28000.0
INCOME_NOISE_SIGMA = 0.35


def _age_income_factor(age: np.ndarray) -> np.ndarray:
    """Income rises with age/experience up to ~45, then flattens."""
    return 0.55 + 0.9 * np.clip(age, 18, 45) / 45 - 0.05 * np.clip(age - 45, 0, None) / 30


def generate_customers(rng: np.random.Generator, n_customers: int, start_date: pd.Timestamp) -> pd.DataFrame:
    customer_id = np.array([f"C{i:07d}" for i in range(1, n_customers + 1)])

    age = np.clip(rng.normal(38, 12, size=n_customers), 18, 80).round().astype(int)
    employment_status = rng.choice(EMPLOYMENT_STATUSES, size=n_customers, p=EMPLOYMENT_WEIGHTS)
    segment = rng.choice(SEGMENTS, size=n_customers, p=SEGMENT_WEIGHTS)
    country = rng.choice(COUNTRIES, size=n_customers, p=COUNTRY_WEIGHTS)

    max_working_months = np.clip((age - 18) * 12, 0, None)
    employment_tenure_months = np.zeros(n_customers, dtype=int)
    working = np.isin(employment_status, ["employed", "self_employed"])
    employment_tenure_months[working] = (
        rng.uniform(0, 0.7, size=working.sum()) * max_working_months[working]
    ).astype(int)
    student_mask = employment_status == "student"
    employment_tenure_months[student_mask] = rng.integers(0, 24, size=student_mask.sum())

    age_factor = _age_income_factor(age)
    employment_factor = np.array([EMPLOYMENT_INCOME_FACTOR[s] for s in employment_status])
    segment_factor = np.array([SEGMENT_INCOME_FACTOR[s] for s in segment])
    noise = rng.lognormal(mean=0, sigma=INCOME_NOISE_SIGMA, size=n_customers)
    annual_income = np.round(BASE_ANNUAL_INCOME * age_factor * employment_factor * segment_factor * noise, 2)
    annual_income = np.clip(annual_income, 4000, None)
    monthly_income = np.round(annual_income / 12, 2)

    tenure_days = rng.integers(30, 365 * 10, size=n_customers)
    customer_since = start_date - pd.to_timedelta(tenure_days, unit="D")

    return pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "employment_status": employment_status,
        "employment_tenure_months": employment_tenure_months,
        "annual_income": annual_income,
        "monthly_income": monthly_income,
        "country": country,
        "customer_since": customer_since,
        "customer_segment": segment,
    })
