"""Monte Carlo Risk page (Phase 6): simulated loss distribution, VaR/CVaR
and Economic Capital by scenario."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app.components.data_loader import load_monte_carlo_report, outputs_available
from app.components.style import apply_layout, page_header

st.set_page_config(page_title="Monte Carlo Risk | Headroom", layout="wide")
page_header(
    "Monte Carlo Loss Simulation / VaR / CVaR",
    "Phase 6: single-factor (Vasicek/ASRF) Gaussian copula default simulation over the funded book.",
)

if not outputs_available():
    st.warning("No pipeline outputs found. Run `make simulate` first.")
    st.stop()

report = load_monte_carlo_report()
summaries = report.get("summary_by_scenario", {})
scenarios = list(summaries.keys())
confidence_levels = report.get("confidence_levels", [0.95, 0.99])
last_level = str(confidence_levels[-1]) if confidence_levels else "0.99"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Simulations", f"{report.get('n_simulations', 0):,}")
col2.metric("Asset correlation", f"{report.get('asset_correlation', 0):.2f}")
col3.metric("Funded customers", f"{report.get('n_funded', 0):,}")
base = summaries.get("base", {})
base_var = base.get("var_cvar", {}).get(last_level, {})
col4.metric(f"Base VaR {float(last_level):.0%}", f"€{base_var.get('var', 0):,.0f}")

st.divider()
st.markdown("#### Analytical Expected Loss vs. Monte Carlo mean (sanity check)")
rows = [
    {
        "scenario": s, "analytical_EL": f"€{summaries[s]['expected_loss']:,.0f}",
        "MC_mean_loss": f"€{summaries[s]['mc_mean_loss']:,.0f}",
        "difference": f"{100 * (summaries[s]['mc_mean_loss'] / summaries[s]['expected_loss'] - 1):+.2f}%",
    }
    for s in scenarios
]
st.dataframe(rows, use_container_width=True, hide_index=True)
st.caption("Both describe the same expectation -- a real divergence here would flag a simulation bug, not a finding.")

st.divider()
st.markdown(f"#### Expected Loss vs. VaR {float(last_level):.0%} by scenario")
el_values = [summaries[s]["expected_loss"] for s in scenarios]
var_values = [summaries[s]["var_cvar"][last_level]["var"] for s in scenarios]
fig = go.Figure()
fig.add_trace(go.Bar(x=scenarios, y=el_values, name="Expected Loss", marker_color="#4C78A8"))
fig.add_trace(go.Bar(x=scenarios, y=var_values, name=f"VaR {float(last_level):.0%}", marker_color="#E45756"))
apply_layout(fig, yaxis_title="EUR/year", height=380, barmode="group")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("#### VaR / CVaR / Economic Capital by confidence level and scenario")
for level in confidence_levels:
    level_key = str(level)
    st.markdown(f"**{level:.0%} confidence**")
    level_rows = [
        {
            "scenario": s, "VaR": f"€{summaries[s]['var_cvar'][level_key]['var']:,.0f}",
            "CVaR": f"€{summaries[s]['var_cvar'][level_key]['cvar']:,.0f}",
            "Economic Capital": f"€{summaries[s]['economic_capital'][level_key]:,.0f}",
        }
        for s in scenarios
    ]
    st.dataframe(level_rows, use_container_width=True, hide_index=True)
