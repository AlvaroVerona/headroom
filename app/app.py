"""Headroom -- Executive Overview.

Run: streamlit run app/app.py
Requires `make all` (or at least generate-data/validate-data/quality-report
through simulate/stress/monitor) to have populated reports/outputs/ first --
this dashboard reads those artifacts rather than recomputing the pipeline
live (Phase 3's model training and Phase 6's Monte Carlo simulation alone
would make the UI unusably slow otherwise).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

from app.components.data_loader import (
    load_calibration_report, load_customer_optimization_report, load_drift_report,
    load_monte_carlo_report, load_portfolio_optimization_report, load_quality_report,
    load_scenario_report, outputs_available,
)
from app.components.style import apply_layout, segment_color, status_badge

st.set_page_config(page_title="Headroom", page_icon="\U0001F4B3", layout="wide")

st.title("Headroom")
st.caption("Dynamic Credit Limit Optimization for a Digital Bank")

if not outputs_available():
    st.warning(
        "No pipeline outputs found yet. Run `make all` from the project root "
        "(takes several minutes -- trains models, runs the MIP, simulates), then reload."
    )
    st.stop()

quality = load_quality_report()
calibration = load_calibration_report()
customer_opt = load_customer_optimization_report()
portfolio_opt = load_portfolio_optimization_report()
scenario = load_scenario_report()
monte_carlo = load_monte_carlo_report()
drift = load_drift_report()

xgb_chosen = calibration.get("xgboost", {}).get("chosen_method", "isotonic")
xgb_test = calibration.get("xgboost", {}).get("test_scores", {}).get(xgb_chosen, {})
mip = portfolio_opt.get("mip_optimal_metrics", {})
base_scenario = scenario.get("summary_by_scenario", {}).get("base", {})
base_mc = monte_carlo.get("summary_by_scenario", {}).get("base", {})
var_99 = base_mc.get("var_cvar", {}).get("0.99", {})

st.caption("All figures below come from the real test-cohort pipeline run -- see each page for the full methodology.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Data Quality Score", f"{quality.get('overall_score', float('nan')):.1f} / 100")
col2.metric("Production PD model ROC-AUC", f"{xgb_test.get('roc_auc', float('nan')):.3f}")
col3.metric("Approval rate", f"{customer_opt.get('approval_rate', 0) * 100:.1f}%")
col4.metric("Funded customers", f"{mip.get('n_funded', 0):,} / {mip.get('n_pool', 0):,}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Funded portfolio profit/year", f"€{mip.get('total_expected_profit', 0):,.0f}")
col6.metric("Base scenario Expected Loss", f"€{base_scenario.get('total_expected_loss', 0):,.0f}")
col7.metric("VaR 99% (base)", f"€{var_99.get('var', 0):,.0f}")
with col8:
    st.markdown("**Model drift**")
    status = "SIGNIFICANT" if drift.get("any_significant_drift") else "STABLE"
    st.markdown(status_badge(status, status), unsafe_allow_html=True)

st.divider()

left, right = st.columns(2)

with left:
    st.markdown("#### Expected Loss by scenario (funded book)")
    summaries = scenario.get("summary_by_scenario", {})
    scenarios = list(summaries.keys())
    fig = go.Figure(go.Bar(
        x=scenarios, y=[summaries[s]["total_expected_loss"] for s in scenarios],
        marker_color=["#4C78A8" if s == "base" else "#E45756" for s in scenarios],
    ))
    apply_layout(fig, yaxis_title="EUR/year", height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("#### Funded rate by segment (Phase 5 MIP)")
    funded_rate = portfolio_opt.get("funded_rate_by_segment", {})
    segments = ["prime", "near_prime", "subprime"]
    fig = go.Figure(go.Bar(
        x=segments, y=[funded_rate.get(s, 0) * 100 for s in segments],
        marker_color=[segment_color(s) for s in segments],
    ))
    apply_layout(fig, yaxis_title="Funded %", height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Use the sidebar to open Data Quality, Risk Models, Economics, Optimization, "
    "Stress Testing, Monte Carlo Risk and Monitoring."
)
