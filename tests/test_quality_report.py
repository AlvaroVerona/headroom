"""Tests for the formal Data Quality report (§11): integration test that
runs it against the real quality_report.json (written by validate-data)
and checks every figure/report file lands and the numbers are internally
consistent with the source JSON (never fabricated)."""

from __future__ import annotations

import pytest

from credit_limit_optimizer.data.quality_report import (
    FIG_DIR, REPORT_PATH, build_report, load_quality_report, write_markdown_report,
)


@pytest.fixture(scope="module")
def report():
    return load_quality_report()


@pytest.fixture(scope="module")
def built(report):
    return build_report(report)


def test_quality_report_json_has_all_four_score_dimensions(report):
    """§11 requires the score broken down by Dataset, Field, Customer
    segment, and Month -- by_month was missing from an earlier version."""
    for key in ["overall_score", "by_dataset", "by_field", "by_segment", "by_month"]:
        assert key in report


def test_by_month_covers_full_36_month_history(report):
    assert len(report["by_month"]) == 36


def test_quality_report_produces_all_figures(built):
    pngs = list(FIG_DIR.glob("*.png"))
    assert len(pngs) == 7, f"expected 7 figures, found {[p.name for p in pngs]}"


def test_quality_report_writes_markdown(report, built):
    write_markdown_report(report, built["month_stats"])
    assert REPORT_PATH.exists()
    text = REPORT_PATH.read_text()
    assert f"{report['overall_score']:.1f} / 100" in text


def test_month_stats_match_report_by_month(report, built):
    values = list(report["by_month"].values())
    stats = built["month_stats"]
    assert stats["min"] == pytest.approx(min(values))
    assert stats["max"] == pytest.approx(max(values))
    assert stats["mean"] == pytest.approx(sum(values) / len(values), abs=0.01)


def test_load_quality_report_raises_clear_error_when_missing(tmp_path, monkeypatch):
    import credit_limit_optimizer.data.quality_report as qr
    missing_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(qr, "QUALITY_JSON_PATH", missing_path)
    with pytest.raises(FileNotFoundError, match="make validate-data"):
        qr.load_quality_report()
