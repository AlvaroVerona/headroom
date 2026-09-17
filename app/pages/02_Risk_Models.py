"""Risk Models page (Phase 3): LR baseline vs. XGBoost, calibration
before/after, and SHAP global importance for the production PD model."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app.components.data_loader import (
    load_calibration_report, load_risk_model_metrics, load_shap_report,
    load_xgboost_metrics, outputs_available,
)
from app.components.style import apply_layout, page_header

st.set_page_config(page_title="Risk Models | Headroom", layout="wide")
page_header("Risk Models", "Phase 3: LR baseline, XGBoost (production PD source), calibration, SHAP.")

if not outputs_available():
    st.warning("No pipeline outputs found. Run `make train train-xgboost calibrate explain` first.")
    st.stop()

lr = load_risk_model_metrics()
xgb = load_xgboost_metrics()
calibration = load_calibration_report()
shap_report = load_shap_report()

st.markdown("#### LR baseline vs. XGBoost (test cohort, before calibration)")
col1, col2, col3, col4 = st.columns(4)
lr_test, xgb_test = lr["metrics"]["test"], xgb["metrics"]["test"]
col1.metric("LR ROC-AUC", f"{lr_test['roc_auc']:.3f}")
col2.metric("XGBoost ROC-AUC", f"{xgb_test['roc_auc']:.3f}", delta=f"{xgb_test['roc_auc'] - lr_test['roc_auc']:+.3f}")
col3.metric("LR PR-AUC", f"{lr_test['pr_auc']:.3f}")
col4.metric("XGBoost PR-AUC", f"{xgb_test['pr_auc']:.3f}", delta=f"{xgb_test['pr_auc'] - lr_test['pr_auc']:+.3f}")

st.caption(
    "XGBoost is the production PD source for everything downstream (Expected Loss, "
    "optimization, simulation) -- it modestly but genuinely beats the LR baseline on "
    "held-out test data. LR's role is the required, interpretable baseline."
)

st.divider()
st.markdown("#### Calibration: raw vs. calibrated (test cohort)")
xgb_cal = calibration.get("xgboost", {})
chosen = xgb_cal.get("chosen_method", "isotonic")
raw_scores = xgb_cal.get("test_scores", {}).get("raw", {})
cal_scores = xgb_cal.get("test_scores", {}).get(chosen, {})

col5, col6, col7 = st.columns(3)
col5.metric("Mean predicted probability (raw)", f"{raw_scores.get('mean_predicted_probability', 0):.1%}")
col6.metric(f"Mean predicted probability ({chosen})", f"{cal_scores.get('mean_predicted_probability', 0):.1%}")
col7.metric("Actual base rate", f"{cal_scores.get('actual_base_rate', 0):.1%}")

col8, col9 = st.columns(2)
col8.metric("Brier score (raw -> calibrated)", f"{raw_scores.get('brier_score', 0):.4f} -> {cal_scores.get('brier_score', 0):.4f}")
col9.metric("ECE (raw -> calibrated)", f"{raw_scores.get('ece', 0):.4f} -> {cal_scores.get('ece', 0):.4f}")

st.divider()
st.markdown("#### SHAP global importance (production XGBoost model)")
importance = shap_report.get("xgboost", {}).get("global_importance", {})
top_n = list(importance.items())[:15]
fig = go.Figure(go.Bar(
    x=[v for _, v in top_n][::-1], y=[k for k, _ in top_n][::-1],
    orientation="h", marker_color="#4C78A8",
))
apply_layout(fig, xaxis_title="Mean |SHAP value|", height=500, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Top risk-increasing / risk-reducing coefficients (LR baseline)")
coef_col1, coef_col2 = st.columns(2)
with coef_col1:
    st.markdown("**Risk-increasing**")
    st.dataframe(
        [{"feature": k, "coefficient": v} for k, v in list(lr["coefficients"]["most_risk_increasing"].items())[:8]],
        use_container_width=True, hide_index=True,
    )
with coef_col2:
    st.markdown("**Risk-reducing**")
    st.dataframe(
        [{"feature": k, "coefficient": v} for k, v in list(lr["coefficients"]["most_risk_reducing"].items())[:8]],
        use_container_width=True, hide_index=True,
    )
