"""
Streamlit input form component for EMIPredict AI.
Collects and validates applicant demographic, income, expense, and loan details.
"""

from typing import Any, Dict

import streamlit as st


def render_applicant_input_form() -> Dict[str, Any]:
    """
    Renders organized input controls across 6 financial categories.
    Returns a dictionary of raw applicant attributes.
    """
    st.markdown("### 📋 Applicant Financial Profile")
    st.caption("Enter the applicant's financial and employment parameters for automated risk underwriting.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👤 1. Personal & Demographics")
        age = st.number_input("Age (Years)", min_value=18, max_value=80, value=35, step=1)
        gender = st.selectbox("Gender (Audited for fairness)", ["Male", "Female", "Other"], index=0)
        marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed"], index=0)
        education = st.selectbox(
            "Education Level",
            ["Graduate", "Post Graduate", "Professional", "High School", "Doctorate", "Diploma"],
            index=0
        )

        st.markdown("#### 💼 2. Employment & Income")
        monthly_salary = st.number_input(
            "Monthly Gross Salary (₹)",
            min_value=10000.0,
            max_value=2000000.0,
            value=65000.0,
            step=5000.0,
            format="%.2f"
        )
        employment_type = st.selectbox(
            "Employment Type",
            ["Salaried", "Self-Employed", "Business", "Private", "Government", "Freelancer"],
            index=0
        )
        years_of_employment = st.number_input(
            "Total Work Experience (Years)",
            min_value=0.0,
            max_value=50.0,
            value=6.0,
            step=0.5
        )
        company_type = st.selectbox(
            "Company Type",
            ["MNC", "Corporate", "Mid-size", "Startup", "Public", "Small Business"],
            index=0
        )

        st.markdown("#### 🏠 3. Housing & Dependents")
        house_type = st.selectbox(
            "Residential House Type",
            ["Rented", "Own", "Family", "Mortgaged", "Company Provided"],
            index=0
        )
        monthly_rent = st.number_input(
            "Monthly Rent / Maintenance (₹)",
            min_value=0.0,
            max_value=300000.0,
            value=15000.0 if house_type == "Rented" else 0.0,
            step=1000.0,
            format="%.2f"
        )
        family_size = st.number_input("Family Size", min_value=1, max_value=15, value=3, step=1)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1, step=1)

    with col2:
        st.markdown("#### 🛒 4. Monthly Living Expenses")
        school_fees = st.number_input("School Fees (₹/mo)", min_value=0.0, max_value=200000.0, value=3000.0, step=500.0)
        college_fees = st.number_input("College / Higher Edu Fees (₹/mo)", min_value=0.0, max_value=200000.0, value=0.0, step=500.0)
        travel_expenses = st.number_input("Travel & Fuel (₹/mo)", min_value=0.0, max_value=100000.0, value=4500.0, step=500.0)
        groceries_utilities = st.number_input("Groceries & Utilities (₹/mo)", min_value=0.0, max_value=150000.0, value=12000.0, step=1000.0)
        other_monthly_expenses = st.number_input("Other Discretionary Expenses (₹/mo)", min_value=0.0, max_value=100000.0, value=4000.0, step=500.0)

        st.markdown("#### 💳 5. Credit & Financial Reserves")
        existing_loans = st.selectbox("Existing Active Loans?", ["No", "Yes"], index=0)
        current_emi_amount = st.number_input(
            "Current Active EMI Payments (₹/mo)",
            min_value=0.0,
            max_value=500000.0,
            value=5000.0 if existing_loans == "Yes" else 0.0,
            step=1000.0,
            format="%.2f"
        )
        credit_score = st.number_input("Credit Score (CIBIL / Experian)", min_value=300.0, max_value=900.0, value=740.0, step=10.0)
        bank_balance = st.number_input("Savings Bank Balance (₹)", min_value=0.0, max_value=10000000.0, value=150000.0, step=10000.0)
        emergency_fund = st.number_input("Liquid Emergency Fund (₹)", min_value=0.0, max_value=10000000.0, value=80000.0, step=10000.0)

        st.markdown("#### 🎯 6. Proposed Loan Request")
        emi_scenario = st.selectbox(
            "Loan Purpose / Scenario",
            [
                "Personal Loan EMI",
                "Home Loan EMI",
                "Vehicle EMI",
                "Education EMI",
                "E-commerce Shopping EMI",
                "Medical Emergency EMI",
                "Business Equipment EMI"
            ],
            index=0
        )
        requested_amount = st.number_input(
            "Requested Loan Amount (₹)",
            min_value=10000.0,
            max_value=10000000.0,
            value=300000.0,
            step=25000.0,
            format="%.2f"
        )
        requested_tenure = st.number_input(
            "Requested Tenure (Months)",
            min_value=3,
            max_value=360,
            value=24,
            step=3
        )

    return {
        "age": int(age),
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": float(monthly_salary),
        "employment_type": employment_type,
        "years_of_employment": float(years_of_employment),
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": float(monthly_rent),
        "family_size": int(family_size),
        "dependents": int(dependents),
        "school_fees": float(school_fees),
        "college_fees": float(college_fees),
        "travel_expenses": float(travel_expenses),
        "groceries_utilities": float(groceries_utilities),
        "other_monthly_expenses": float(other_monthly_expenses),
        "existing_loans": existing_loans,
        "current_emi_amount": float(current_emi_amount),
        "credit_score": float(credit_score),
        "bank_balance": float(bank_balance),
        "emergency_fund": float(emergency_fund),
        "emi_scenario": emi_scenario,
        "requested_amount": float(requested_amount),
        "requested_tenure": int(requested_tenure),
    }
