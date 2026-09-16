"""Tests for the EDA module (§12): unit tests for the binning/stat helpers
on handcrafted fixtures, plus an integration test that runs the full EDA
against the real generated data and checks every figure/stat file lands
and the numbers are internally consistent (never fabricated)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from credit_limit_optimizer.analysis.eda import (
    FIG_DIR, REPORT_PATH, STATS_PATH,
    _default_rate_by_bin, _describe, _fmt_interval, run_eda,
)


def test_describe_matches_pandas_directly():
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    d = _describe(s)
    assert d["mean"] == pytest.approx(30.0)
    assert d["median"] == pytest.approx(30.0)
    assert d["count"] == 5


def test_describe_drops_nan():
    s = pd.Series([10.0, np.nan, 30.0])
    d = _describe(s)
    assert d["count"] == 2
    assert d["mean"] == pytest.approx(20.0)


def test_default_rate_by_bin_fixed_bins_computes_correct_rate():
    df = pd.DataFrame({
        "value": [5, 15, 25, 35, 95],
        "default_12m": [0.0, 0.0, 1.0, 1.0, 1.0],
    })
    g = _default_rate_by_bin(df, df["value"], bins=[0, 10, 30, 100], labels=["low", "mid", "high"])
    assert g.loc["low", "default_rate"] == pytest.approx(0.0)
    assert g.loc["mid", "default_rate"] == pytest.approx(0.5)  # rows 15 (0.0), 25 (1.0)
    assert g.loc["high", "default_rate"] == pytest.approx(1.0)  # rows 35, 95


def test_default_rate_by_bin_excludes_nan_default_rows():
    df = pd.DataFrame({"value": [5, 15, 25], "default_12m": [0.0, np.nan, 1.0]})
    g = _default_rate_by_bin(df, df["value"], bins=[0, 20, 30], labels=["low", "high"])
    assert g.loc["low", "count"] == 1  # the NaN-default row (value=15) is excluded
    assert g.loc["high", "count"] == 1


def test_qcut_labels_are_rounded_not_raw_float_repr():
    """Regression test: qcut's raw interval repr had absurd precision, e.g.
    '(267.28900000000004, 1569.59]' -- unreadable in a chart. round_ndigits
    controls display rounding without changing which rows fall in which bin."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "value": rng.uniform(100, 5000, size=500),
        "default_12m": rng.integers(0, 2, size=500).astype(float),
    })
    g = _default_rate_by_bin(df, df["value"], q=5, round_ndigits=0)
    for label in g.index:
        assert "." not in label or label.count(".") == 0, f"unrounded float leaked into label: {label}"
        assert len(label) < 20


def test_fmt_interval_rounds_both_edges():
    iv = pd.Interval(267.28900000000004, 1569.59)
    assert _fmt_interval(iv, 0) == "267-1570"


@pytest.fixture(scope="module")
def stats():
    return run_eda()


def test_eda_produces_all_figures(stats):
    expected_count = 16
    pngs = list(FIG_DIR.glob("*.png"))
    assert len(pngs) == expected_count, f"expected {expected_count} figures, found {[p.name for p in pngs]}"


def test_eda_writes_stats_and_report_files(stats):
    assert STATS_PATH.exists()
    assert REPORT_PATH.exists()
    with STATS_PATH.open() as f:
        reloaded = json.load(f)
    assert reloaded["n_rows"] == stats["n_rows"]


def test_default_rate_by_income_is_monotonically_decreasing(stats):
    rates = [r["default_rate"] for r in stats["default_rate_by_income"]]
    assert rates == sorted(rates, reverse=True), "higher income should default less, on average"


def test_default_rate_by_utilization_is_monotonically_increasing(stats):
    rates = [r["default_rate"] for r in stats["default_rate_by_utilization"]]
    assert rates == sorted(rates), "higher utilization should default more, on average"


def test_default_rate_by_delinquency_history_is_monotonically_increasing(stats):
    rates = [r["default_rate"] for r in stats["default_rate_by_delinquency_history"]["by_max_days_past_due"]]
    assert rates == sorted(rates)


def test_segment_default_rate_ordered_prime_lowest_subprime_highest(stats):
    by_seg = stats["default_rate_overall"]["by_segment"]
    assert by_seg["prime"] < by_seg["near_prime"] < by_seg["subprime"]


def test_no_negative_monthly_cash_flow_is_a_verified_generator_property(stats):
    """This dataset's monthly_spend is drawn as a fraction of monthly_income
    (behavior_series.py), so spend never exceeds income within a month by
    construction -- fraction_negative must be exactly 0.0, not just small."""
    assert stats["monthly_cash_flow"]["fraction_negative"] == 0.0
    assert stats["monthly_cash_flow"]["overall"]["min"] > 0


def test_correlation_matrix_is_symmetric_with_unit_diagonal(stats):
    full_matrix = stats["correlation_structure"]["full_matrix"]
    cols = list(full_matrix.keys())
    for c in cols:
        assert full_matrix[c][c] == pytest.approx(1.0, abs=1e-6)
    for a in cols:
        for b in cols:
            assert full_matrix[a][b] == pytest.approx(full_matrix[b][a], abs=1e-6)


def test_report_numbers_are_sourced_from_stats_not_hardcoded(stats):
    """Spot-check that a few numbers appearing in the markdown report match
    the stats file exactly -- catches copy-paste/stale-number drift."""
    report_text = REPORT_PATH.read_text()
    overall_rate = stats["default_rate_overall"]["overall_rate"]
    assert f"{overall_rate:.2%}" in report_text
