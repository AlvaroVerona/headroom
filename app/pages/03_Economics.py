"""Economics page (Phase 4): Expected Loss, Revenue, Funding Cost and
Customer Profitability, all at customers' current/observed limits."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app.components.data_loader import (
    load_expected_loss_report, load_funding_cost_report, load_profitability_report,
    load_revenue_report, outputs_available,
)
from app.components.style import apply_layout, page_header, segment_color

st.set_page_config(page_title="Economics | Headroom", layout="wide")
page_header("Economics", "Phase 4: Expected Loss = PD x LGD x EAD, Revenue, Funding Cost, Customer Profitability.")

if not outputs_available():
    st.warning("No pipeline outputs found. Run `make expected-loss economics` first.")
    st.stop()

el = load_expected_loss_report()
revenue = load_revenue_report()
funding = load_funding_cost_report()
profitability = load_profitability_report()

st.markdown("#### Portfolio summary (test cohort, current limits)")
ts = el.get("test_summary", {})
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean PD", f"{ts.get('mean_pd', 0):.2%}")
col2.metric("Mean LGD", f"{ts.get('mean_lgd', 0):.2f}")
col3.metric("Total portfolio Expected Loss", f"€{ts.get('total_portfolio_expected_loss', 0):,.0f}")
col4.metric(
    "Mean Expected Profit/year",
    f"€{profitability.get('current_state_summary', {}).get('mean_expected_profit', 0):,.2f}",
)

st.divider()
segments = ["prime", "near_prime", "subprime"]

col_left, col_right = st.columns(2)
with col_left:
    st.markdown("#### Expected Loss by segment")
    el_by_segment = el.get("el_by_segment", {})
    fig = go.Figure(go.Bar(x=segments, y=[el_by_segment.get(s, 0) for s in segments], marker_color=[segment_color(s) for s in segments]))
    apply_layout(fig, yaxis_title="Mean EL (EUR)", height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("#### Total revenue by segment (monthly)")
    rev_by_segment = revenue.get("by_segment", {})
    fig = go.Figure(go.Bar(
        x=segments, y=[rev_by_segment.get(s, {}).get("total_revenue", 0) for s in segments],
        marker_color=[segment_color(s) for s in segments],
    ))
    apply_layout(fig, yaxis_title="Mean revenue (EUR/month)", height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("#### Net Interest Margin by segment (base funding scenario)")
    nim = funding.get("nim_by_segment", {})
    fig = go.Figure(go.Bar(x=segments, y=[nim.get(s, 0) for s in segments], marker_color=[segment_color(s) for s in segments]))
    apply_layout(fig, yaxis_title="Mean NIM (EUR/month)", height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("#### Expected Customer Profit by segment (annual, current limits)")
    profit_by_segment = profitability.get("profit_by_segment", {})
    fig = go.Figure(go.Bar(x=segments, y=[profit_by_segment.get(s, 0) for s in segments], marker_color=[segment_color(s) for s in segments]))
    apply_layout(fig, yaxis_title="Mean Expected Profit (EUR/year)", height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("#### Example customer: profitability across the candidate limit grid")
examples = profitability.get("examples", [])
if examples:
    tabs = st.tabs([f"{ex['segment']} ({ex['customer_id']})" for ex in examples])
    for tab, ex in zip(tabs, examples):
        with tab:
            st.dataframe(ex["table"], use_container_width=True, hide_index=True)
