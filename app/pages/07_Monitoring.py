"""Monitoring page (Phase 6): Population Stability Index for the PD
score and top SHAP-importance features across the three real snapshot
vintages (train/validation/test)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app.components.data_loader import load_drift_report, outputs_available
from app.components.style import apply_layout, page_header, status_badge

st.set_page_config(page_title="Monitoring | Headroom", layout="wide")
page_header(
    "Drift Monitoring / PSI",
    "Phase 6: Population Stability Index across the three real, time-separated snapshot vintages.",
)

if not outputs_available():
    st.warning("No pipeline outputs found. Run `make monitor` first.")
    st.stop()

report = load_drift_report()
score_drift = report.get("score_drift", {})
feature_drift = report.get("feature_drift", {})
thresholds = report.get("thresholds", {})

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Overall verdict**")
    status = "SIGNIFICANT" if report.get("any_significant_drift") else "STABLE"
    st.markdown(status_badge(status, status), unsafe_allow_html=True)
with col2:
    st.caption(
        f"Warning threshold: PSI >= {thresholds.get('psi_warning_threshold', 0.10)}. "
        f"Critical threshold: PSI >= {thresholds.get('psi_critical_threshold', 0.25)}."
    )

st.divider()
st.markdown("#### PD score stability")
pairs = list(score_drift.keys())
values = [score_drift[p]["psi"] for p in pairs]
colors = [
    "#E45756" if v >= thresholds.get("psi_critical_threshold", 0.25)
    else "#F58518" if v >= thresholds.get("psi_warning_threshold", 0.10)
    else "#54A24B"
    for v in values
]
fig = go.Figure(go.Bar(x=pairs, y=values, marker_color=colors))
fig.add_hline(y=thresholds.get("psi_warning_threshold", 0.10), line_dash="dash", line_color="#F58518", annotation_text="Warning")
fig.add_hline(y=thresholds.get("psi_critical_threshold", 0.25), line_dash="dash", line_color="#E45756", annotation_text="Critical")
apply_layout(fig, yaxis_title="PSI", height=340, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

score_rows = [
    {"period": p, "psi": f"{row['psi']:.4f}", "severity": row["severity"], "baseline_mean_PD": f"{row['baseline_mean']:.2%}", "current_mean_PD": f"{row['current_mean']:.2%}"}
    for p, row in score_drift.items()
]
st.dataframe(score_rows, use_container_width=True, hide_index=True)

st.divider()
st.markdown(f"#### Feature stability (top {len(report.get('monitored_features', []))} SHAP-importance features)")
feature_names = list(feature_drift.keys())
period_names = list(next(iter(feature_drift.values()), {}).keys())
feature_rows = []
for feature in feature_names:
    row = {"feature": feature}
    for period in period_names:
        row[period] = f"{feature_drift[feature][period]['psi']:.4f} ({feature_drift[feature][period]['severity']})"
    feature_rows.append(row)
st.dataframe(feature_rows, use_container_width=True, hide_index=True)

delinquency = feature_drift.get("delinquency_count", {}).get("train_vs_test", {})
if delinquency.get("severity") == "SIGNIFICANT":
    st.info(
        "**Real finding, not a bug**: `delinquency_count` shows SIGNIFICANT drift because "
        "train/validation/test are the SAME customers observed at later points in their own "
        "history -- a cumulative counter mechanically rises. `days_past_due` (point-in-time) "
        "and `recent_delinquency` (recent-window) both stay stable, confirming this. See "
        "CLAUDE.md for the full explanation."
    )
