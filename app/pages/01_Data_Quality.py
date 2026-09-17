"""Data Quality page (Phase 1): overall/dataset/segment quality scores,
issue severity mix, and row-level RAW -> VALIDATED -> QUARANTINED lineage."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app.components.data_loader import load_quality_report, outputs_available
from app.components.style import apply_layout, page_header, segment_color

st.set_page_config(page_title="Data Quality | Headroom", layout="wide")
page_header("Data Quality", "Phase 1: validation engine output, 4 datasets.")

if not outputs_available():
    st.warning("No pipeline outputs found. Run `make validate-data quality-report` first.")
    st.stop()

quality = load_quality_report()

col1, col2, col3 = st.columns(3)
col1.metric("Overall Data Quality Score", f"{quality.get('overall_score', float('nan')):.1f} / 100")
raw = quality.get("row_counts_raw", {})
quarantined = quality.get("row_counts_quarantined", {})
total_raw = sum(raw.values())
total_quarantined = sum(quarantined.values())
col2.metric("Total raw rows", f"{total_raw:,}")
col3.metric("Quarantined rows", f"{total_quarantined:,} ({100 * total_quarantined / total_raw:.1f}%)")

st.divider()
left, right = st.columns(2)

with left:
    st.markdown("#### Quality score by dataset")
    by_dataset = quality.get("by_dataset", {})
    fig = go.Figure(go.Bar(x=list(by_dataset.keys()), y=list(by_dataset.values()), marker_color="#4C78A8"))
    apply_layout(fig, yaxis_title="Score / 100", height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("#### Quality score by customer segment")
    by_segment = quality.get("by_segment", {})
    segments = ["prime", "near_prime", "subprime"]
    fig = go.Figure(go.Bar(
        x=segments, y=[by_segment.get(s, 0) for s in segments],
        marker_color=[segment_color(s) for s in segments],
    ))
    apply_layout(fig, yaxis_title="Score / 100", height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("#### Issues by severity")
by_severity = quality.get("issue_counts_by_severity", {})
severity_order = [s for s in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] if s in by_severity]
fig = go.Figure(go.Bar(
    x=severity_order, y=[by_severity[s] for s in severity_order],
    marker_color=["#E45756", "#F58518", "#ECC94B", "#B0B0B8"][: len(severity_order)],
))
apply_layout(fig, yaxis_title="Issue count", height=300, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("#### RAW -> VALIDATED -> QUARANTINED lineage")
validated = quality.get("row_counts_validated", {})
rows = [
    {"dataset": name, "raw": raw.get(name, 0), "validated": validated.get(name, 0), "quarantined": quarantined.get(name, 0)}
    for name in raw
]
st.dataframe(rows, use_container_width=True, hide_index=True)
