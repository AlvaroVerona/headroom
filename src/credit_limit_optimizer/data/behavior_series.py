"""Simulates the 36-month ground-truth trajectory per customer: income,
spend (essential/discretionary/subscription/cash), credit line utilization,
payment behavior, and days-past-due/delinquency evolution -- all coupled
through a single latent monthly "distress" process, so the correlations
the spec asks for (higher utilization/DTI/income-volatility -> higher
delinquency -> higher default risk) are structural, not bolted on.

Days-past-due moves in 30-day cycles (0/30/60/90/...), the standard
industry delinquency-bucket convention: a missed-payment month adds a
cycle, an on-time/catch-up month removes one. default_12m and
delinquent_90d (both built in labels.py) are read directly off this
trajectory -- see CLAUDE.md for why 90+ DPD is the default definition.

Credit limit is assigned once at origination and held constant through
history (no organic mid-history limit changes) -- a deliberate
simplification: the project's entire premise is that CURRENT limits may be
suboptimal, which doesn't require them to have moved historically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_limit_optimizer.data.master_data import SEGMENT_BASE_DISTRESS

INITIAL_UTILIZATION_BY_SEGMENT = {"prime": 0.20, "near_prime": 0.40, "subprime": 0.65}
SPEND_INCOME_RATIO_BY_SEGMENT = {"prime": (0.20, 0.35), "near_prime": (0.30, 0.50), "subprime": (0.35, 0.60)}
ESSENTIAL_SHARE_BY_SEGMENT = {"prime": 0.45, "near_prime": 0.60, "subprime": 0.72}
CASH_WITHDRAWAL_SHARE_BY_SEGMENT = {"prime": 0.03, "near_prime": 0.06, "subprime": 0.11}
INCOME_VOL_SIGMA_BY_SEGMENT = {"prime": 0.03, "near_prime": 0.06, "subprime": 0.10}
FULL_PAYOFF_PROB_BY_SEGMENT = {"prime": 0.35, "near_prime": 0.18, "subprime": 0.05}
# Fraction of (prior balance + this month's spend) a REVOLVING (non-full-
# payoff) customer pays off each month. This -- not a tiny minimum-payment
# floor -- is what determines the long-run equilibrium balance: balance
# converges to roughly spend*(1-f)/f, so it needs to be a real fraction of
# the balance, not ~3% of it. A minimum-payment-sized payment can't keep
# pace with ongoing new spend, which is what originally made 34% of all
# customer-months pile up stuck at the credit ceiling instead of
# fluctuating around a segment-appropriate utilization level.
#
# Calibrated by solving that equilibrium equation for f given each
# segment's target utilization (prime ~15%, near_prime ~45%, subprime
# ~75%) at the segment's own spend_ratio/credit_limit_multiple, then
# empirically re-checked (the equilibrium formula alone under-predicts
# realized utilization once full-payoff months are mixed in, since those
# periodically reset balance to zero and pull the time-average down) --
# see CLAUDE.md.
REVOLVER_PAYOFF_FRACTION_BY_SEGMENT = {"prime": (0.35, 0.65), "near_prime": (0.25, 0.50), "subprime": (0.20, 0.42)}
LIMIT_INCOME_MULTIPLE_BY_SEGMENT = {"prime": 2.0, "near_prime": 1.3, "subprime": 1.0}

DPD_CYCLE_DAYS = 30
UTILIZATION_DISTRESS_COEF = 0.05  # extra monthly hazard per 100% utilization
# Escalating to 90+ DPD needs 3 CONSECUTIVE missed-payment months (30->60->90),
# so realistic default rates depend on genuine multi-month "bad patches" --
# weakly-persistent noise mostly just jitters an independent-ish Bernoulli
# hazard, which makes 3-in-a-row rare almost regardless of the base rate.
# High persistence + a shock big enough to swing hazard meaningfully is
# what produces those patches; calibrated empirically (see CLAUDE.md) to a
# ~3-6% default_12m rate at each snapshot -- low enough to be realistic,
# high enough for 50,000 customers to give a robust number of positive
# labels per cohort.
DISTRESS_PERSISTENCE = 0.92
DISTRESS_SHOCK_COEF = 0.14
MIN_CREDIT_LIMIT, MAX_CREDIT_LIMIT = 500.0, 12000.0


def assign_initial_credit_limits(rng: np.random.Generator, customers: pd.DataFrame) -> np.ndarray:
    segment = customers["customer_segment"].to_numpy()
    monthly_income = customers["monthly_income"].to_numpy()
    multiple = np.array([LIMIT_INCOME_MULTIPLE_BY_SEGMENT[s] for s in segment])
    noise = rng.lognormal(0, 0.25, size=len(customers))
    limit = monthly_income * multiple * noise
    limit = np.round(limit / 100) * 100  # round to nearest 100
    return np.clip(limit, MIN_CREDIT_LIMIT, MAX_CREDIT_LIMIT)


def simulate_behavior(
    rng: np.random.Generator, customers: pd.DataFrame, credit_limit: np.ndarray, n_months: int, start_date: pd.Timestamp,
) -> dict[str, np.ndarray]:
    """Returns a dict of (n_customers, n_months) arrays, one per behavioral
    field, plus the credit_limit array passed through for convenience."""
    n = len(customers)
    segment = customers["customer_segment"].to_numpy()
    base_income = customers["monthly_income"].to_numpy()

    base_distress = np.array([SEGMENT_BASE_DISTRESS[s] for s in segment])
    spend_ratio_lo = np.array([SPEND_INCOME_RATIO_BY_SEGMENT[s][0] for s in segment])
    spend_ratio_hi = np.array([SPEND_INCOME_RATIO_BY_SEGMENT[s][1] for s in segment])
    spend_ratio = rng.uniform(spend_ratio_lo, spend_ratio_hi, size=n)
    essential_share = np.array([ESSENTIAL_SHARE_BY_SEGMENT[s] for s in segment])
    cash_share = np.array([CASH_WITHDRAWAL_SHARE_BY_SEGMENT[s] for s in segment])
    income_sigma = np.array([INCOME_VOL_SIGMA_BY_SEGMENT[s] for s in segment])
    full_payoff_prob = np.array([FULL_PAYOFF_PROB_BY_SEGMENT[s] for s in segment])
    revolver_lo = np.array([REVOLVER_PAYOFF_FRACTION_BY_SEGMENT[s][0] for s in segment])
    revolver_hi = np.array([REVOLVER_PAYOFF_FRACTION_BY_SEGMENT[s][1] for s in segment])

    income = np.zeros((n, n_months))
    spend = np.zeros((n, n_months))
    essential_spend = np.zeros((n, n_months))
    discretionary_spend = np.zeros((n, n_months))
    subscription_spend = np.zeros((n, n_months))
    cash_withdrawal = np.zeros((n, n_months))
    balance = np.zeros((n, n_months))
    avg_balance = np.zeros((n, n_months))
    payment_amount = np.zeros((n, n_months))
    scheduled_amount = np.zeros((n, n_months))
    dpd = np.zeros((n, n_months))
    delinquency_count = np.zeros((n, n_months))
    missed_payment = np.zeros((n, n_months), dtype=bool)

    persistent_shock = np.zeros(n)
    balance_prev = credit_limit * np.array([INITIAL_UTILIZATION_BY_SEGMENT[s] for s in segment]) * rng.uniform(0.7, 1.3, size=n)
    balance_prev = np.clip(balance_prev, 0, credit_limit)
    dpd_prev = np.zeros(n)
    delinq_count_prev = np.zeros(n)

    growth_trend = 1.0015  # mild monthly income growth

    for t in range(n_months):
        income[:, t] = base_income * (growth_trend ** t) * rng.lognormal(0, income_sigma, size=n)

        utilization_prev = balance_prev / credit_limit
        # Proper AR(1) (rho*prev + sqrt(1-rho^2)*innovation): keeps unit
        # variance regardless of rho, so higher persistence lengthens the
        # memory of a "bad patch" without shrinking its magnitude. An
        # EWMA-style rho*prev + (1-rho)*innovation looks similar but
        # actually *shrinks* variance as rho rises (the innovation weight
        # shrinks faster than persistence extends memory) -- the original
        # bug here, which made increasing persistence barely move the
        # default rate at all.
        persistent_shock = (
            DISTRESS_PERSISTENCE * persistent_shock
            + np.sqrt(1 - DISTRESS_PERSISTENCE ** 2) * rng.normal(0, 1, size=n)
        )
        hazard = base_distress + DISTRESS_SHOCK_COEF * persistent_shock + UTILIZATION_DISTRESS_COEF * utilization_prev
        hazard = np.clip(hazard, 0.001, 0.75)
        missed = rng.random(n) < hazard
        missed_payment[:, t] = missed

        # Spend is constrained by available headroom -- a real card
        # purchase that would exceed the limit gets declined, it doesn't
        # silently inflate the balance past the ceiling. Without this, a
        # customer whose desired spend chronically exceeds headroom just
        # gets clipped to EXACTLY the limit every such month (utilization
        # pinned at 100.000%) instead of naturally fluctuating just under
        # it -- see CLAUDE.md.
        desired_spend = income[:, t] * spend_ratio * rng.lognormal(0, 0.15, size=n)
        available_headroom = np.maximum(credit_limit - balance_prev, 0)
        spend[:, t] = np.minimum(desired_spend, available_headroom)
        essential_spend[:, t] = spend[:, t] * essential_share
        discretionary_spend[:, t] = spend[:, t] * (1 - essential_share) * 0.85
        subscription_spend[:, t] = spend[:, t] - essential_spend[:, t] - discretionary_spend[:, t]
        cash_withdrawal[:, t] = spend[:, t] * cash_share * rng.lognormal(0, 0.3, size=n)

        balance_pre_payment = np.clip(balance_prev + spend[:, t], 0, None)
        scheduled_amount[:, t] = np.maximum(balance_pre_payment * 0.03, np.where(balance_pre_payment > 0, 25, 0))

        pays_in_full = (~missed) & (rng.random(n) < full_payoff_prob)
        revolver_fraction = rng.uniform(revolver_lo, revolver_hi, size=n)
        payment_amount[:, t] = np.where(
            missed, 0.0,
            np.where(pays_in_full, balance_pre_payment, balance_pre_payment * revolver_fraction),
        )
        payment_amount[:, t] = np.minimum(payment_amount[:, t], balance_pre_payment)

        balance[:, t] = np.clip(balance_pre_payment - payment_amount[:, t], 0, credit_limit)
        avg_balance[:, t] = (balance_prev + balance[:, t]) / 2

        dpd[:, t] = np.where(missed, dpd_prev + DPD_CYCLE_DAYS, np.maximum(dpd_prev - DPD_CYCLE_DAYS, 0))
        delinquency_count[:, t] = delinq_count_prev + missed.astype(float)

        balance_prev = balance[:, t]
        dpd_prev = dpd[:, t]
        delinq_count_prev = delinquency_count[:, t]

    debt_to_income = np.divide(balance, np.maximum(income, 1.0))
    savings_rate = np.divide(income - spend, np.maximum(income, 1.0))
    utilization = balance / credit_limit[:, None]
    max_utilization = np.maximum.accumulate(utilization, axis=1)
    payment_ratio = np.divide(payment_amount, np.maximum(scheduled_amount, 1e-6))
    minimum_payment_ratio = np.clip(payment_ratio, 0, None)

    return {
        "income": income, "spend": spend, "essential_spend": essential_spend,
        "discretionary_spend": discretionary_spend, "subscription_spend": subscription_spend,
        "cash_withdrawal": cash_withdrawal, "balance": balance, "avg_balance": avg_balance,
        "payment_amount": payment_amount, "scheduled_amount": scheduled_amount, "dpd": dpd,
        "delinquency_count": delinquency_count, "missed_payment": missed_payment,
        "debt_to_income": debt_to_income, "savings_rate": savings_rate, "utilization": utilization,
        "max_utilization": max_utilization, "payment_ratio": payment_ratio,
        "minimum_payment_ratio": minimum_payment_ratio,
    }
