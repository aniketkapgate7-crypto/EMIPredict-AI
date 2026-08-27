"""
Main Streamlit Application & Home Dashboard for EMIPredict AI.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Configure page layout if running standalone
try:
    st.set_page_config(
        page_title="EMIPredict AI — Intelligent Financial Risk Platform",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass


def main():
    # Header Hero Section
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0B132B 0%, #1C2541 100%);
            border: 1px solid #3A506B;
            border-radius: 14px;
            padding: 28px;
            margin-bottom: 24px;
        ">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span style="font-size: 42px;">💳</span>
                <div>
                    <h1 style="color: #00A896; margin: 0; font-size: 32px; font-weight: 800;">
                        EMIPredict AI
                    </h1>
                    <p style="color: #E0E1DD; margin: 4px 0 0 0; font-size: 16px;">
                        Next-Generation Intelligent Financial Risk Assessment & EMI Affordability Underwriting Platform
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Core Problem & Value Proposition
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 🎯 System Objectives & Core Capabilities")
        st.markdown(
            """
            EMIPredict AI addresses retail lending risk and borrower over-leveraging by delivering dual machine learning capabilities:

            1. **Multiclass EMI Eligibility Classification**:
               - Evaluates borrower solvency into three risk tiers:
                 - `Eligible`: High creditworthiness, stable cash flows, adequate debt cushion.
                 - `High_Risk`: Elevated debt-to-income or thin liquidity reserves, requiring conditional terms.
                 - `Not_Eligible`: Over-leveraged, high default probability, unviable debt commitments.
               - Optimized for **Macro F1-Score** to prevent majority-class bias.

            2. **Maximum Monthly EMI Affordability Regression**:
               - Estimates a model-recommended sustainable monthly EMI amount (`max_monthly_emi`) an applicant can sustainably afford each month.
               - Evaluated against **Mean Absolute Error (MAE)** and **RMSE**, bounded with presentation lower limit of ₹0.
            """
        )

    with col2:
        st.markdown("### 🏛️ Platform Architecture")
        st.markdown(
            """
            <div style="
                background-color: #1C2541;
                border: 1px solid #3A506B;
                border-radius: 10px;
                padding: 16px;
                font-size: 14px;
            ">
                <p>⚙️ <b>Feature Engineering</b>: 13 domain ratios (DTI, Expense Ratio, Principal Burden, Runway).</p>
                <p>⚖️ <b>Fair Lending Policy</b>: Sensitive demographics (<code>gender</code>, <code>marital_status</code>) excluded from model decisions.</p>
                <p>🧪 <b>Experiment Tracking</b>: Centralized MLflow metrics & artifact tracking.</p>
                <p>🗄️ <b>SQLAlchemy CRUD</b>: SQLite/PostgreSQL applicant record management.</p>
                <p>🚀 <b>Streamlit Cloud Ready</b>: Ultra-lightweight inference with cached joblib pipelines.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Dataset Summary KPI Cards
    st.markdown("### 📊 Dataset Overview")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(label="Total Dataset Records", value="404,800")
        st.caption("Full production corpus")
    with k2:
        st.metric(label="Total Features Observed", value="27 Columns")
        st.caption("25 inputs + 2 targets")
    with k3:
        st.metric(label="Classification Target", value="emi_eligibility")
        st.caption("3-Class Multiclass")
    with k4:
        st.metric(label="Regression Target", value="Maximum Monthly EMI")
        st.caption("`max_monthly_emi` • Rupee Affordability in ₹")

    st.markdown("---")

    # Responsible Use Notice
    st.warning(
        "⚠️ **Responsible AI & Decision-Support Notice**: "
        "EMIPredict AI is engineered exclusively as an intelligent decision-support platform for qualified credit officers. "
        "Automated predictions do not constitute legally binding loan commitments and must be reviewed alongside institutional compliance policies."
    )


main()

