"""
About & System Architecture Page for EMIPredict AI Streamlit Application.
"""

import streamlit as st

try:
    st.set_page_config(page_title="About — EMIPredict AI", page_icon="ℹ️", layout="wide")
except Exception:
    pass

st.markdown("# ℹ️ About EMIPredict AI")
st.caption("Intelligent Financial Risk Assessment & EMI Affordability Underwriting System")
st.markdown("---")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 🎯 Project Overview & Objectives")
    st.markdown(
        """
        **EMIPredict AI** is an enterprise-grade academic capstone project engineered to assist retail credit underwriting teams.
        It automates preliminary financial solvency auditing through dual machine learning tasks:

        1. **Multiclass Credit Risk Classification**: Classifies applicant eligibility into `Eligible`, `High_Risk`, or `Not_Eligible` optimizing for **Macro F1-Score**.
        2. **Affordability Regression**: Estimates sustainable monthly repayment capacity (`max_monthly_emi`) in Indian Rupees (₹) optimizing for **Mean Absolute Error (MAE)**.
        """
    )

    st.markdown("### 🛠️ Technology Stack")
    st.markdown(
        """
        - **Core Language**: Python 3.11
        - **Machine Learning & Pipeline**: Scikit-Learn, XGBoost, NumPy, Pandas, Joblib
        - **Experiment Tracking & Lineage**: MLflow (SQLite backend)
        - **Persistence Layer**: SQLAlchemy ORM (SQLite / PostgreSQL)
        - **Frontend Interface**: Streamlit Multi-Page Web App, Plotly Interactive Visuals
        - **Quality Assurance**: Pytest, Ruff Linter, GitHub Actions CI
        """
    )

    st.markdown("### ⚖️ Responsible AI & Ethical Lending Policy")
    st.markdown(
        """
        - **Exclusion of Sensitive Demographic Attributes**: Sensitive features (`gender` and `marital_status`) are preserved for fairness auditing but **strictly excluded** from the decision models to eliminate demographic bias.
        - **Non-Negativity Constraint**: Regression outputs are lower-bounded at ₹0.00 to guarantee mathematically realistic installment figures.
        - **Strict Train-Test Isolation**: All transformers and scalers are fitted exclusively on training partitions to prevent data leakage.
        """
    )

with col2:
    st.markdown("### 🏛️ Pipeline Flow")
    st.markdown(
        """
        <div style="
            background-color: #1C2541;
            border: 1px solid #3A506B;
            border-radius: 10px;
            padding: 18px;
            font-size: 14px;
        ">
            <p>1️⃣ <b>Raw Ingestion</b>: Schema audit & downcasting (404.8k rows, 27 columns).</p>
            <p>2️⃣ <b>Feature Engineering</b>: 13 domain ratios (DTI, Expense Ratio, Principal Burden, Runway).</p>
            <p>3️⃣ <b>Preprocessing</b>: Median/Mode imputation + OHE + StandardScaler.</p>
            <p>4️⃣ <b>Dual Modeling</b>: Logistic Regression, RF, XGBoost + MLflow tracking.</p>
            <p>5️⃣ <b>Inference Service</b>: Real-time validation, ratio compute & prediction.</p>
            <p>6️⃣ <b>CRUD Persistence</b>: Full applicant audit trail in database.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ⚠️ Regulatory Disclaimer")
    st.info(
        "EMIPredict AI is designed strictly as a clinical decision-support tool for qualified underwriting professionals. "
        "Outputs do not constitute automatic legal approvals or credit rejections. Final lending decisions must comply with local banking regulations."
    )

    st.markdown("### 👨‍💻 Project Developer")
    st.markdown(
        """
        - **Developer**: EMIPredict AI Engineering Team
        - **Program**: B.Tech CSE (Artificial Intelligence & Machine Learning)
        - **Repository**: [GitHub Repository](https://github.com/)
        - **License**: MIT Open Source License
        """
    )
