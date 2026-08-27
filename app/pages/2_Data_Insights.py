"""
Data Insights & Exploratory Analysis Page for EMIPredict AI Streamlit Application.
Loads the representative sample and generated EDA figures without loading the 404.8k dataset into memory.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Setup path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import DATA_SAMPLE_PATH, FIGURES_DIR, REPORTS_DIR
from src.utils.artifacts import load_json_metadata

try:
    st.set_page_config(page_title="Data Insights — EMIPredict AI", page_icon="📊", layout="wide")
except Exception:
    pass

st.markdown("# 📊 Exploratory Data Analysis & Quality Insights")
st.caption("Comprehensive data quality audit, target distributions, and feature relationships based on audited data samples.")
st.markdown("---")


@st.cache_data
def load_sample_data():
    """Loads lightweight representative dataset sample."""
    if DATA_SAMPLE_PATH.exists():
        return pd.read_csv(DATA_SAMPLE_PATH)
    return None


@st.cache_data
def load_dq_report():
    """Loads data quality audit report JSON."""
    report_file = REPORTS_DIR / "data_quality_report.json"
    if report_file.exists():
        return load_json_metadata(report_file)
    return None


df_sample = load_sample_data()
dq_report = load_dq_report()

if dq_report:
    st.markdown("### 📋 Data Quality Audit Highlights")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric(label="Total Dataset Size", value=f"{dq_report.get('total_rows', 404800):,} Rows")
        st.caption("Full production corpus")
    with q2:
        st.metric(label="Feature Space", value=f"{dq_report.get('total_columns', 27)} Columns")
        st.caption("25 inputs + 2 target variables")
    with q3:
        dup_count = dq_report.get("duplicate_rows", {}).get("count", 0)
        st.metric(label="Exact Duplicate Rows", value=f"{dup_count:,}")
        st.caption(f"{dq_report.get('duplicate_rows', {}).get('percentage', 0.0):.2f}% of dataset")
    with q4:
        missing_count = dq_report.get("missing_values_summary", {}).get("total_missing_cells", 0)
        st.metric(label="Missing Cells", value=f"{missing_count:,}")
        st.caption("Complete clean schema")

    st.markdown("---")

if df_sample is not None:
    st.markdown("### 🔍 Interactive Feature Exploration (Representative Sample)")
    st.caption("Explore relationships between income, credit rating, living expenses, and EMI affordability.")

    # Filter controls
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        scenarios = ["All"] + sorted(list(df_sample["emi_scenario"].dropna().unique()))
        selected_scenario = st.selectbox("Filter by Loan Scenario:", scenarios)
    with fcol2:
        elig_classes = ["All"] + sorted(list(df_sample["emi_eligibility"].dropna().unique()))
        selected_elig = st.selectbox("Filter by Eligibility Class:", elig_classes)

    filtered_df = df_sample.copy()
    if selected_scenario != "All":
        filtered_df = filtered_df[filtered_df["emi_scenario"] == selected_scenario]
    if selected_elig != "All":
        filtered_df = filtered_df[filtered_df["emi_eligibility"] == selected_elig]

    st.write(f"Displaying **{len(filtered_df):,}** sample records.")

    # Interactive Plots
    pcol1, pcol2 = st.columns(2)

    with pcol1:
        # Scatter: Monthly Salary vs Max Monthly EMI
        fig_scatter = px.scatter(
            filtered_df,
            x="monthly_salary",
            y="max_monthly_emi",
            color="emi_eligibility",
            color_discrete_map={
                "Eligible": "#00A896",
                "High_Risk": "#F4A261",
                "Not_Eligible": "#E63946",
            },
            title="Monthly Salary vs. Max Monthly EMI Capacity",
            labels={"monthly_salary": "Monthly Salary (₹)", "max_monthly_emi": "Max Monthly EMI (₹)"},
            opacity=0.6,
        )
        fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#FFFFFF"))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with pcol2:
        # Box plot: Credit Score by Eligibility Class
        fig_box = px.box(
            filtered_df,
            x="emi_eligibility",
            y="credit_score",
            color="emi_eligibility",
            color_discrete_map={
                "Eligible": "#00A896",
                "High_Risk": "#F4A261",
                "Not_Eligible": "#E63946",
            },
            title="Credit Score Distribution by Eligibility Tier",
            labels={"credit_score": "Credit Score", "emi_eligibility": "Eligibility Tier"},
        )
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#FFFFFF"))
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

st.markdown("### 🖼️ Saved High-Resolution Exploratory Figures")

figures = list(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
if figures:
    # Filter figures relevant to EDA
    eda_figs = [
        f for f in figures
        if any(keyword in f.name for keyword in ["distribution", "scenario", "salary", "credit", "correlation", "ratio"])
    ]
    if not eda_figs:
        eda_figs = figures

    for i in range(0, len(eda_figs), 2):
        c1, c2 = st.columns(2)
        with c1:
            if i < len(eda_figs):
                st.image(str(eda_figs[i]), caption=eda_figs[i].stem.replace("_", " ").title(), use_container_width=True)
        with c2:
            if i + 1 < len(eda_figs):
                st.image(str(eda_figs[i + 1]), caption=eda_figs[i + 1].stem.replace("_", " ").title(), use_container_width=True)
else:
    st.info("Figures will be rendered upon running the training and EDA pipelines.")
