"""Shared chart styling: a fixed categorical color per customer segment
(never reassigned when a filter changes which segments are shown) and a
status palette for severity/risk, kept distinct so a "SIGNIFICANT" badge
is never confused with a segment's own color."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

SEGMENT_COLORS = {
    "prime": "#4C78A8",
    "near_prime": "#ECC94B",
    "subprime": "#E45756",
}

STATUS_COLORS = {
    "STABLE": "#54A24B", "LOW": "#54A24B", "PASS": "#54A24B", "OPTIMAL": "#54A24B",
    "MODERATE": "#ECC94B", "FEASIBLE": "#ECC94B", "MEDIUM": "#ECC94B",
    "HIGH": "#F58518",
    "SIGNIFICANT": "#E45756", "CRITICAL": "#E45756", "FAIL": "#E45756", "INFEASIBLE": "#E45756",
    "INFO": "#B0B0B8",
}

PLOT_LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(size=13),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10),
)


def apply_layout(fig: go.Figure, **overrides) -> go.Figure:
    fig.update_layout(**{**PLOT_LAYOUT_DEFAULTS, **overrides})
    return fig


def segment_color(segment: str) -> str:
    return SEGMENT_COLORS.get(segment, "#B0B0B8")


def status_color(status: str) -> str:
    return STATUS_COLORS.get(str(status).upper(), "#B0B0B8")


def status_badge(label: str, status: str) -> str:
    color = status_color(status)
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color};'
        f'border-radius:6px;padding:2px 10px;font-weight:600;font-size:0.85em">{label}</span>'
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
