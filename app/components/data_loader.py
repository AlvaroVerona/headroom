"""Cached readers for the pipeline's output artifacts
(`reports/outputs/*.json`). The dashboard reads what `make all` already
produced rather than recomputing the pipeline on every page interaction --
Phase 3's model training and Phase 6's Monte Carlo simulation alone would
make the UI unusably slow if re-run on every click."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from credit_limit_optimizer.utils.config import PROJECT_ROOT, load_config

REPORTS_DIR = PROJECT_ROOT / "reports" / "outputs"


@st.cache_data
def get_config() -> dict:
    return load_config()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


@st.cache_data
def load_quality_report() -> dict:
    return _read_json(REPORTS_DIR / "quality_report.json")


@st.cache_data
def load_risk_model_metrics() -> dict:
    return _read_json(REPORTS_DIR / "risk_model_baseline_metrics.json")


@st.cache_data
def load_xgboost_metrics() -> dict:
    return _read_json(REPORTS_DIR / "xgboost_metrics.json")


@st.cache_data
def load_calibration_report() -> dict:
    return _read_json(REPORTS_DIR / "calibration_report.json")


@st.cache_data
def load_shap_report() -> dict:
    return _read_json(REPORTS_DIR / "shap_report.json")


@st.cache_data
def load_expected_loss_report() -> dict:
    return _read_json(REPORTS_DIR / "expected_loss_report.json")


@st.cache_data
def load_revenue_report() -> dict:
    return _read_json(REPORTS_DIR / "revenue_report.json")


@st.cache_data
def load_funding_cost_report() -> dict:
    return _read_json(REPORTS_DIR / "funding_cost_report.json")


@st.cache_data
def load_profitability_report() -> dict:
    return _read_json(REPORTS_DIR / "profitability_report.json")


@st.cache_data
def load_customer_optimization_report() -> dict:
    return _read_json(REPORTS_DIR / "customer_optimization_report.json")


@st.cache_data
def load_portfolio_optimization_report() -> dict:
    return _read_json(REPORTS_DIR / "portfolio_optimization_report.json")


@st.cache_data
def load_scenario_report() -> dict:
    return _read_json(REPORTS_DIR / "scenario_report.json")


@st.cache_data
def load_monte_carlo_report() -> dict:
    return _read_json(REPORTS_DIR / "monte_carlo_report.json")


@st.cache_data
def load_drift_report() -> dict:
    return _read_json(REPORTS_DIR / "drift_report.json")


def outputs_available() -> bool:
    """True once `make all` (or at least Phase 1's validate-data /
    quality-report) has run."""
    return (REPORTS_DIR / "quality_report.json").exists()


def figures_dir(name: str) -> Path:
    return PROJECT_ROOT / "reports" / "figures" / name
