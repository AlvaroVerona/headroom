"""Optimization page (Phase 5): individual credit limit optimization
(per-customer argmax under risk/income/utilization/DSR constraints) and
portfolio optimization (OR-Tools MIP knapsack over the approved pool)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app.components.data_loader import (
    load_customer_optimization_report, load_portfolio_optimization_report, outputs_available,
)
from app.components.style import apply_layout, page_header, segment_color

st.set_page_config(page_title="Optimization | Headroom", layout="wide")
page_header("Optimization", "Phase 5: individual limit optimization, then a portfolio-level MIP over the approved pool.")

if not outputs_available():
    st.warning("No pipeline outputs found. Run `make optimize-customer optimize-portfolio` first.")
    st.stop()

customer_opt = load_customer_optimization_report()
portfolio_opt = load_portfolio_optimization_report()

st.markdown("#### Individual optimization (stage 1)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Approval rate", f"{customer_opt.get('approval_rate', 0) * 100:.1f}%")
col2.metric("Approved / total", f"{customer_opt.get('n_approved', 0):,} / {customer_opt.get('n_customers', 0):,}")
col3.metric("Mean profit uplift", f"€{customer_opt.get('mean_profit_uplift', 0):,.2f}/year")
col4.metric("Total profit uplift", f"€{customer_opt.get('total_profit_uplift', 0):,.0f}/year")

col5, col6, col7 = st.columns(3)
col5.metric("Limits increased", f"{customer_opt.get('fraction_limit_increased', 0) * 100:.1f}%")
col6.metric("Limits decreased", f"{customer_opt.get('fraction_limit_decreased', 0) * 100:.1f}%")
col7.metric("Limits unchanged", f"{customer_opt.get('fraction_limit_unchanged', 0) * 100:.1f}%")
st.caption(
    f"{customer_opt.get('fraction_decrease_from_grid_cap', 0):.0%} of decreases are explained by the "
    f"candidate grid's own maximum (€{customer_opt.get('candidate_grid_max', 0):,.0f}), not a model "
    "judgment that the customer's limit should shrink."
)

st.markdown("**Decline reasons**")
decline_reasons = customer_opt.get("decline_reason_counts", {})
if decline_reasons:
    fig = go.Figure(go.Bar(x=list(decline_reasons.keys()), y=list(decline_reasons.values()), marker_color="#E45756"))
    apply_layout(fig, yaxis_title="Declined customers", height=280, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("#### Portfolio optimization (stage 2, OR-Tools MIP)")
u, m, g, c = (
    portfolio_opt.get("unconstrained_metrics", {}), portfolio_opt.get("mip_optimal_metrics", {}),
    portfolio_opt.get("greedy_baseline_metrics", {}), portfolio_opt.get("constraints", {}),
)

col8, col9, col10 = st.columns(3)
col8.metric("Solver status", portfolio_opt.get("solver_status", "N/A"))
col9.metric("Funded customers", f"{m.get('n_funded', 0):,} / {m.get('n_pool', 0):,}")
col10.metric("MIP vs. greedy improvement", f"€{portfolio_opt.get('mip_vs_greedy_profit_improvement', 0):,.0f}/year")

st.markdown("**Constraint utilization at the MIP-optimal solution**")
constraint_rows = [
    ("Exposure", m.get("total_exposure", 0), c.get("portfolio_exposure_limit", 1)),
    ("Expected Loss", m.get("total_expected_loss", 0), c.get("portfolio_expected_loss_limit", 1)),
    ("Average PD", m.get("average_pd", 0), c.get("max_average_pd", 1)),
    ("High-risk exposure %", m.get("high_risk_exposure_pct", 0), c.get("max_high_risk_exposure_pct", 1)),
]
labels = [r[0] for r in constraint_rows]
utilization = [r[1] / r[2] if r[2] else 0 for r in constraint_rows]
colors = ["#E45756" if u >= 0.999 else "#F58518" if u >= 0.9 else "#54A24B" for u in utilization]
fig = go.Figure(go.Bar(x=utilization, y=labels, orientation="h", marker_color=colors))
fig.add_vline(x=1.0, line_dash="dash", line_color="#333333")
apply_layout(fig, xaxis_title="Fraction of portfolio limit used", height=280, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

col11, col12 = st.columns(2)
with col11:
    st.markdown("**Funded rate by segment**")
    funded_rate = portfolio_opt.get("funded_rate_by_segment", {})
    segments = ["prime", "near_prime", "subprime"]
    fig = go.Figure(go.Bar(x=segments, y=[funded_rate.get(s, 0) * 100 for s in segments], marker_color=[segment_color(s) for s in segments]))
    apply_layout(fig, yaxis_title="Funded %", height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col12:
    st.markdown("**Profit-per-euro-of-exposure by segment**")
    efficiency = portfolio_opt.get("profit_efficiency_by_segment", {})
    fig = go.Figure(go.Bar(x=segments, y=[efficiency.get(s, 0) for s in segments], marker_color=[segment_color(s) for s in segments]))
    apply_layout(fig, yaxis_title="EUR profit / EUR exposure", height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.info(
    "Funding rate is NOT \"safest first\": exposure is the binding constraint, and prime's "
    "profit-per-euro-of-exposure is the segment's worst, so prime gets funded the LEAST "
    "despite being the lowest-risk segment. See CLAUDE.md for the full explanation."
)
