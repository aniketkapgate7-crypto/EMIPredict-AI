"""
Model Performance & Evaluation Dashboard for EMIPredict AI Streamlit Application.
Displays empirical benchmarks, confusion matrices, residual plots, and selection rationale.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Setup path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from src.utils.artifacts import load_json_metadata

try:
    st.set_page_config(page_title="Model Performance — EMIPredict AI", page_icon="📈", layout="wide")
except Exception:
    pass

st.markdown("# 📈 Model Benchmarks & Empirical Performance")
st.caption("Detailed evaluation metrics across candidate classification and regression models trained on the EMIPredict dataset.")
st.markdown("---")


@st.cache_data
def load_comparison_tables():
    """Loads saved classification and regression comparison CSVs."""
    cls_csv = REPORTS_DIR / "classification_model_comparison.csv"
    reg_csv = REPORTS_DIR / "regression_model_comparison.csv"

    cls_df = pd.read_csv(cls_csv) if cls_csv.exists() else None
    reg_df = pd.read_csv(reg_csv) if reg_csv.exists() else None
    return cls_df, reg_df


cls_df, reg_df = load_comparison_tables()
meta = load_json_metadata(MODELS_DIR / "model_metadata.json")

tab1, tab2, tab3 = st.tabs([
    "🎯 Multiclass Classification",
    "💵 EMI Affordability Regression",
    "📑 Model Selection Report"
])

# ----------------------------------------------------
# TAB 1: CLASSIFICATION
# ----------------------------------------------------
with tab1:
    st.markdown("### 1. Classification Model Benchmark (`emi_eligibility`)")
    st.markdown(
        "**Primary Metric**: **Macro F1-Score** (ensures balanced precision and recall across all risk classes without majority-class bias)."
    )

    if cls_df is not None:
        st.dataframe(cls_df, use_container_width=True, hide_index=True)

        cls_meta = meta.get("classification_champion", {})
        if cls_meta:
            st.markdown("#### 🏆 Champion Classification Model")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(label="Model Architecture", value=cls_meta.get("model_name", "Random Forest"))
            with c2:
                st.metric(
                    label="Validation Macro F1",
                    value=f"{cls_meta.get('validation_metrics', {}).get('macro_f1', 0.0):.4f}"
                )
            with c3:
                st.metric(
                    label="Test Set Macro F1",
                    value=f"{cls_meta.get('test_metrics', {}).get('macro_f1', 0.0):.4f}"
                )
            with c4:
                st.metric(
                    label="Test Accuracy",
                    value=f"{cls_meta.get('test_metrics', {}).get('accuracy', 0.0):.4f}"
                )

        st.markdown("#### 🖼️ Diagnostic Plots")
        cm_files = list(FIGURES_DIR.glob("confusion_matrix_*.png")) if FIGURES_DIR.exists() else []
        if cm_files:
            for i in range(0, len(cm_files), 2):
                cc1, cc2 = st.columns(2)
                with cc1:
                    if i < len(cm_files):
                        st.image(str(cm_files[i]), caption=cm_files[i].stem.replace("_", " ").title(), use_container_width=True)
                with cc2:
                    if i + 1 < len(cm_files):
                        st.image(str(cm_files[i + 1]), caption=cm_files[i + 1].stem.replace("_", " ").title(), use_container_width=True)
    else:
        st.info("Classification model comparison results will appear after running training pipeline.")

# ----------------------------------------------------
# TAB 2: REGRESSION
# ----------------------------------------------------
with tab2:
    st.markdown("### 2. Regression Model Benchmark (`max_monthly_emi`)")
    st.markdown(
        "**Primary Metric**: **Mean Absolute Error (MAE)** (measures average rupee discrepancy between predicted and true monthly affordability)."
    )

    if reg_df is not None:
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        reg_meta = meta.get("regression_champion", {})
        if reg_meta:
            st.markdown("#### 🏆 Champion Regression Model")
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric(label="Model Architecture", value=reg_meta.get("model_name", "Random Forest Regressor"))
            with r2:
                val_mae = reg_meta.get('validation_metrics', {}).get('mae', 0.0)
                st.metric(label="Validation MAE", value=f"₹{val_mae:,.2f}")
            with r3:
                test_mae = reg_meta.get('test_metrics', {}).get('mae', 0.0)
                st.metric(label="Test Set MAE", value=f"₹{test_mae:,.2f}")
            with r4:
                test_r2 = reg_meta.get('test_metrics', {}).get('r2', 0.0)
                st.metric(label="Test Set R² Score", value=f"{test_r2:.4f}")

        st.markdown("#### 🖼️ Regression Diagnostics")
        reg_plots = [
            p for p in (list(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else [])
            if "actual_vs_predicted" in p.name or "residuals" in p.name
        ]
        if reg_plots:
            for i in range(0, len(reg_plots), 2):
                rc1, rc2 = st.columns(2)
                with rc1:
                    if i < len(reg_plots):
                        st.image(str(reg_plots[i]), caption=reg_plots[i].stem.replace("_", " ").title(), use_container_width=True)
                with rc2:
                    if i + 1 < len(reg_plots):
                        st.image(str(reg_plots[i + 1]), caption=reg_plots[i + 1].stem.replace("_", " ").title(), use_container_width=True)
    else:
        st.info("Regression model comparison results will appear after running training pipeline.")

# ----------------------------------------------------
# TAB 3: SELECTION REPORT
# ----------------------------------------------------
with tab3:
    sel_report_path = REPORTS_DIR / "model_selection_report.md"
    if sel_report_path.exists():
        with open(sel_report_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Model selection report will be generated after model training.")
