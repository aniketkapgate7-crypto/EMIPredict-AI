"""
Streamlit metric cards, status badges, and KPI components for EMIPredict AI.
"""

from typing import Dict

import streamlit as st


def render_eligibility_badge(eligibility_class: str, probabilities: Dict[str, float] | None = None) -> None:
    """Renders a high-contrast, accessible status card for predicted EMI eligibility."""
    class_config = {
        "Eligible": {
            "title": "ELIGIBLE FOR FINANCING",
            "icon": "✅",
            "bg_color": "rgba(0, 168, 150, 0.15)",
            "border_color": "#00A896",
            "text_color": "#48CAE4",
            "description": "Applicant financial profile demonstrates strong affordability and stable debt coverage capacity.",
        },
        "High_Risk": {
            "title": "HIGH RISK — CONDITIONAL REVIEW",
            "icon": "⚠️",
            "bg_color": "rgba(244, 162, 97, 0.15)",
            "border_color": "#F4A261",
            "text_color": "#F4A261",
            "description": "Applicant exhibits elevated debt-to-income or reduced savings cushion. Manual underwriter evaluation required.",
        },
        "Not_Eligible": {
            "title": "NOT ELIGIBLE / OVER-LEVERAGED",
            "icon": "❌",
            "bg_color": "rgba(230, 57, 70, 0.15)",
            "border_color": "#E63946",
            "text_color": "#E63946",
            "description": "Applicant current obligations exceed standard risk parameters. Requested commitment exceeds safe thresholds.",
        }
    }

    cfg = class_config.get(eligibility_class, {
        "title": eligibility_class.upper(),
        "icon": "ℹ️",
        "bg_color": "rgba(255, 255, 255, 0.1)",
        "border_color": "#FFFFFF",
        "text_color": "#FFFFFF",
        "description": "Risk assessment completed.",
    })

    st.markdown(
        f"""
        <div style="
            background-color: {cfg['bg_color']};
            border: 2px solid {cfg['border_color']};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            text-align: center;
        ">
            <h2 style="color: {cfg['text_color']}; margin: 0; font-size: 24px; font-weight: 700;">
                {cfg['icon']} {cfg['title']}
            </h2>
            <p style="color: #E0E1DD; margin-top: 8px; font-size: 15px;">
                {cfg['description']}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_max_emi_card(max_emi: float, requested_monthly: float) -> None:
    """Renders maximum recommended monthly installment alongside principal-only monthly estimate."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="
                background-color: #1C2541;
                border: 1px solid #3A506B;
                border-radius: 10px;
                padding: 16px;
                text-align: center;
            ">
                <span style="color: #A0AEC0; font-size: 13px; text-transform: uppercase;">Max Safe Monthly EMI</span>
                <h3 style="color: #00A896; font-size: 28px; margin: 6px 0; font-weight: bold;">
                    ₹{max_emi:,.2f}
                </h3>
                <span style="color: #718096; font-size: 12px;">Model Affordability Limit</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        diff = max_emi - requested_monthly
        diff_color = "#00A896" if diff >= 0 else "#E63946"
        status_text = "Affordable (Within Limit)" if diff >= 0 else "Exceeds Limit by " + f"₹{abs(diff):,.0f}"

        st.markdown(
            f"""
            <div style="
                background-color: #1C2541;
                border: 1px solid #3A506B;
                border-radius: 10px;
                padding: 16px;
                text-align: center;
            ">
                <span style="color: #A0AEC0; font-size: 13px; text-transform: uppercase;">Principal-Only Monthly Estimate</span>
                <h3 style="color: {diff_color}; font-size: 28px; margin: 6px 0; font-weight: bold;">
                    ₹{requested_monthly:,.2f}
                </h3>
                <span style="color: {diff_color}; font-size: 12px; font-weight: 600;">{status_text} (Interest Unspecified)</span>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_financial_ratios_grid(ratios: Dict[str, float]) -> None:
    """Renders a responsive 5-column financial ratios and liquidity KPI grid."""
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        dti = ratios.get("current_debt_to_income_ratio", ratios.get("debt_to_income_ratio", 0.0)) * 100
        st.metric(label="Current DTI", value=f"{dti:.1f}%")
        st.caption("Benchmark: ≤ 40% (Debt / Income)")

    with c2:
        exp_ratio = ratios.get("expense_to_income_ratio", 0.0) * 100
        st.metric(label="Expense Ratio", value=f"{exp_ratio:.1f}%")
        st.caption("Living Expenses / Salary")

    with c3:
        prop_burden = ratios.get("proposed_principal_burden_ratio", 0.0) * 100
        st.metric(label="Principal Burden", value=f"{prop_burden:.1f}%")
        st.caption("Proposed Principal / Salary")

    with c4:
        disp = ratios.get("disposable_income", 0.0)
        st.metric(label="Disposable Income", value=f"₹{disp:,.0f}")
        st.caption("Salary minus obligations")

    with c5:
        emg_months = ratios.get("emergency_fund_months", 0.0)
        st.metric(label="Emergency Runway", value=f"{emg_months:.1f} mo")
        st.caption("Benchmark: ≥ 3.0 months")
