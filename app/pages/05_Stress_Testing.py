"""Stress Testing page (Phase 6): the funded portfolio's PD/Expected
Loss/Profit re-scored under each macro scenario."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app.components.data_loader import get_config, load_scenario_report, outputs_available
from app.components.style import apply_layout, page_header

st.set_page_config(page_title="Stress Testing | Headroom", layout="wide")
page_header("Stress Testing / Scenarios", "Phase 6: the funded book (Phase 5's MIP solution) re-evaluated under each macro scenario.")

if not outputs_available():
    st.warning("No pipeline outputs found. Run `make stress` first.")
    st.stop()

report = load_scenario_report()
config = get_config()
summaries = report.get("summary_by_scenario", {})
scenarios = list(summaries.keys())

col1, col2, col3 = st.columns(3)
col1.metric("Funded customers evaluated", f"{report.get('n_funded', 0):,}")
col2.metric("Base scenario Expected Loss", f"€{summaries.get('base', {}).get('total_expected_loss', 0):,.0f}")
col3.metric("Portfolio EL limit", f"€{report.get('portfolio_expected_loss_limit', 0):,.0f}")

st.divider()
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("#### Mean PD by scenario")
    fig = go.Figure(go.Bar(
        x=scenarios, y=[summaries[s]["mean_pd"] * 100 for s in scenarios],
        marker_color=["#4C78A8" if s == "base" else "#E45756" for s in scenarios],
    ))
    apply_layout(fig, yaxis_title="Mean PD (%)", height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("#### Total Expected Profit by scenario")
    values = [summaries[s]["total_expected_profit"] for s in scenarios]
    fig = go.Figure(go.Bar(x=scenarios, y=values, marker_color=["#54A24B" if v >= 0 else "#E45756" for v in values]))
    apply_layout(fig, yaxis_title="EUR/year", height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("#### Expected Loss multiplier vs. base")
multipliers = report.get("expected_loss_multiplier_vs_base", {})
fig = go.Figure(go.Bar(x=list(multipliers.keys()), y=list(multipliers.values()), marker_color="#F58518"))
fig.add_hline(y=1.0, line_dash="dash", line_color="#333333")
apply_layout(fig, yaxis_title="EL multiplier", height=300, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("#### Scenario definitions (`config/settings.yaml`)")
scenario_config = config.get("scenarios", {})
rows = [{"scenario": name, **shocks} for name, shocks in scenario_config.items()]
st.dataframe(rows, use_container_width=True, hide_index=True)

hir, base = summaries.get("high_interest_rate", {}), summaries.get("base", {})
if hir and base and abs(hir["total_expected_profit"] - base["total_expected_profit"]) < 1.0:
    st.info(
        "**Real finding**: `high_interest_rate`'s net profit matches base almost exactly -- "
        "equal-magnitude `funding_rate_shift` and `apr_shift` (+3pp each) cancel in net profit "
        "even though gross revenue and gross funding cost both move by real, substantial "
        "amounts. See CLAUDE.md / the scenario model card for the full breakdown."
    )
