"""
Prediction & Risk Assessment Page for EMIPredict AI Streamlit Application.
"""

import sys
from pathlib import Path

import streamlit as st

# Setup path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.components.charts import (
    plot_financial_health_radar,
    plot_probability_distribution,
)
from app.components.input_form import render_applicant_input_form
from app.components.metrics import (
    render_eligibility_badge,
    render_financial_ratios_grid,
    render_max_emi_card,
)
from src.database.crud import create_applicant_record
from src.models.predict import predict_applicant_risk

try:
    st.set_page_config(page_title="Risk Prediction — EMIPredict AI", page_icon="🎯", layout="wide")
except Exception:
    pass

st.markdown("# 🎯 Applicant Risk & EMI Affordability Assessment")
st.caption("Interactive underwriting calculator evaluating multiclass credit eligibility and safe monthly repayment limit.")
st.markdown("---")

# Render inputs form
form_data = render_applicant_input_form()

st.markdown("---")
predict_btn = st.button("🚀 Assess Financial Risk & Predict Safe EMI", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Processing applicant profile through feature engineering and ensemble models..."):
        result = predict_applicant_risk(form_data)

    if result.get("status") == "error":
        st.error(f"❌ Prediction Assessment Error: {result.get('error_message')}")
    else:
        st.session_state["latest_prediction"] = result
        st.session_state["latest_form_data"] = form_data

if "latest_prediction" in st.session_state:
    result = st.session_state["latest_prediction"]
    form_data = st.session_state.get("latest_form_data", {})

    st.markdown("### 📊 Assessment Summary")

    # 1. Eligibility Badge
    render_eligibility_badge(
        result["predicted_eligibility"],
        result.get("probabilities", {})
    )

    # 2. Max EMI vs Requested EMI
    ratios = result.get("financial_ratios", {})
    req_monthly = ratios.get("requested_principal_per_month", 0.0)
    render_max_emi_card(result["max_monthly_emi"], req_monthly)

    st.markdown("#### 📈 Financial Ratios & Liquidity Diagnostics")
    render_financial_ratios_grid(ratios)

    st.markdown("---")

    # 3. Visualizations
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        if result.get("probabilities"):
            plot_probability_distribution(result["probabilities"])
    with vcol2:
        dti = ratios.get("current_debt_to_income_ratio", ratios.get("debt_to_income_ratio", 0.0))
        savings_ratio = ratios.get("savings_to_income_ratio", 0.0)
        emg_months = ratios.get("emergency_fund_months", 0.0)
        c_score = form_data.get("credit_score", 650.0)
        disp_ratio = (ratios.get("disposable_income", 0.0) / (form_data.get("monthly_salary", 1.0) + 1e-5))

        plot_financial_health_radar(
            dti=dti,
            savings_ratio=savings_ratio,
            emergency_months=emg_months,
            credit_score=c_score,
            disposable_ratio=disp_ratio,
        )

    # 4. Underwriting Diagnostics
    st.markdown("#### 💡 Underwriter Insights & Key Observations")
    diagnostics = result.get("diagnostics", [])
    for diag in diagnostics:
        st.info(f"• {diag}")

    st.markdown("---")

    # 5. Database Persistence Option
    st.markdown("### 🗄️ Save Assessment Record")
    save_col1, save_col2 = st.columns([3, 1])

    with save_col1:
        notes = st.text_input(
            "Underwriter Notes / Assessment Remarks",
            placeholder="Add specific comments or conditions for this applicant..."
        )

    with save_col2:
        st.write("")
        st.write("")
        save_btn = st.button("💾 Save to Applicant Database", use_container_width=True)

    if save_btn:
        save_payload = dict(form_data)
        save_payload["predicted_eligibility"] = result["predicted_eligibility"]
        save_payload["max_monthly_emi"] = result["max_monthly_emi"]
        save_payload["probabilities"] = result.get("probabilities", {})
        save_payload["financial_ratios"] = ratios
        save_payload["model_version"] = result.get("model_version", "1.0.0")
        save_payload["notes"] = notes
        save_payload["status"] = "Pending Review"

        try:
            saved_rec = create_applicant_record(save_payload)
            st.success(
                f"✅ Record successfully saved! Reference Identifier: **{saved_rec.get('applicant_identifier')}** (ID #{saved_rec.get('id')})"
            )
        except Exception as e:
            st.error(f"Failed to persist record to database: {e}")
