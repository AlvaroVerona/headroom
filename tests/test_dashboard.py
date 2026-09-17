"""Phase 7 tests: runs every dashboard page headlessly via Streamlit's
AppTest harness and asserts it executes without raising -- the automated
equivalent of a manual browser walkthrough, but seconds instead of minutes
and no browser/port needed under pytest."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"

PAGES = [
    APP_DIR / "app.py",
    APP_DIR / "pages" / "01_Data_Quality.py",
    APP_DIR / "pages" / "02_Risk_Models.py",
    APP_DIR / "pages" / "03_Economics.py",
    APP_DIR / "pages" / "04_Optimization.py",
    APP_DIR / "pages" / "05_Stress_Testing.py",
    APP_DIR / "pages" / "06_Monte_Carlo_Risk.py",
    APP_DIR / "pages" / "07_Monitoring.py",
]


@pytest.mark.parametrize("page_path", PAGES, ids=[p.stem for p in PAGES])
def test_page_runs_without_exception(page_path):
    at = AppTest.from_file(str(page_path), default_timeout=60)
    at.run()
    assert not at.exception, f"{page_path.name} raised: {[str(e) for e in at.exception]}"


def test_data_quality_page_renders_lineage_table():
    at = AppTest.from_file(str(APP_DIR / "pages" / "01_Data_Quality.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 1


def test_economics_page_has_example_customer_tabs():
    at = AppTest.from_file(str(APP_DIR / "pages" / "03_Economics.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 3  # one per segment's example customer


def test_monte_carlo_page_renders_var_cvar_tables():
    at = AppTest.from_file(str(APP_DIR / "pages" / "06_Monte_Carlo_Risk.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 3  # sanity-check table + one per confidence level


def test_overview_shows_headline_metrics():
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.metric) >= 7
