"""
Feature engineering module for EMIPredict AI.
Calculates domain-specific financial ratios and affordability metrics with zero-division safeguards.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_config import setup_logger

logger = setup_logger(__name__)

EPSILON = 1e-5


def _get_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Extracts a column as a float pd.Series with default fill, resilient to missing columns."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def calculate_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific financial ratios from raw applicant attributes.
    Protects all denominators from division by zero with EPSILON / np.maximum.
    """
    data = df.copy()

    # 1. Total monthly living expenses (Housing + Education + Travel + Groceries + Discretionary)
    rent = _get_series(data, "monthly_rent", 0.0)
    school = _get_series(data, "school_fees", 0.0)
    college = _get_series(data, "college_fees", 0.0)
    travel = _get_series(data, "travel_expenses", 0.0)
    groceries = _get_series(data, "groceries_utilities", 0.0)
    other = _get_series(data, "other_monthly_expenses", 0.0)

    data["total_monthly_expenses"] = rent + school + college + travel + groceries + other

    # 2. Total monthly obligations (Living expenses + Existing EMI debt service)
    current_emi = _get_series(data, "current_emi_amount", 0.0)
    data["total_monthly_obligations"] = data["total_monthly_expenses"] + current_emi

    # 3. Disposable Income (Gross Monthly Salary - Total Obligations)
    salary = _get_series(data, "monthly_salary", 0.0)
    data["disposable_income"] = salary - data["total_monthly_obligations"]

    # 4. Debt-to-Income (DTI) Ratio (Current Debt Payments / Gross Monthly Salary)
    # Distinct ratio: Current recurring debt service relative to gross income.
    data["debt_to_income_ratio"] = current_emi / (salary + EPSILON)
    data["current_debt_to_income_ratio"] = data["debt_to_income_ratio"]

    # 5. Expense-to-Income Ratio (Total Living Expenses / Monthly Salary)
    # Distinct ratio: Living overhead relative to gross income.
    data["expense_to_income_ratio"] = data["total_monthly_expenses"] / (salary + EPSILON)

    # 6. Total Obligation-to-Income Ratio ((Living Expenses + Debt Payments) / Monthly Salary)
    # Distinct ratio: Combined living overhead and debt servicing burden.
    data["obligation_to_income_ratio"] = data["total_monthly_obligations"] / (salary + EPSILON)

    # 7. Savings-to-Income Ratio ((Bank Balance + Emergency Fund) / Annual Salary)
    bank_bal = _get_series(data, "bank_balance", 0.0)
    emg_fund = _get_series(data, "emergency_fund", 0.0)
    annual_salary = (salary * 12.0) + EPSILON
    data["savings_to_income_ratio"] = (bank_bal + emg_fund) / annual_salary

    # 8. Emergency Fund Runway in Months (Emergency Fund / Monthly Living Expenses)
    data["emergency_fund_months"] = emg_fund / (data["total_monthly_expenses"] + EPSILON)

    # 9. Principal-Only Monthly Requested Estimate (Requested Amount / Requested Tenure)
    # Labeled as principal estimate because interest rate is unspecified in the dataset.
    req_amt = _get_series(data, "requested_amount", 0.0)
    req_tenure = _get_series(data, "requested_tenure", 1.0)
    req_tenure_safe = np.maximum(req_tenure, 1.0)
    data["requested_principal_per_month"] = req_amt / req_tenure_safe

    # 10. Proposed Principal Burden Ratio (Requested Principal / Monthly Salary)
    data["proposed_principal_burden_ratio"] = data["requested_principal_per_month"] / (salary + EPSILON)

    # 11. Requested Loan Amount to Annual Income Ratio
    data["requested_amount_to_income_ratio"] = req_amt / annual_salary

    # 12. Dependents to Family Size Ratio
    family_size = np.maximum(_get_series(data, "family_size", 1.0), 1.0)
    dependents = _get_series(data, "dependents", 0.0)
    data["dependents_ratio"] = dependents / family_size

    # 13. Employment Stability Ratio (Years of Employment / Working-Age Career Span)
    age = _get_series(data, "age", 25.0)
    working_years_potential = np.maximum(age - 18.0, 1.0)
    emp_years = _get_series(data, "years_of_employment", 0.0)
    data["employment_stability_score"] = emp_years / working_years_potential

    return data


def compute_single_applicant_ratios(applicant_dict: Dict[str, Any]) -> Dict[str, float]:
    """Calculates engineered ratios for a single applicant dictionary."""
    df_single = pd.DataFrame([applicant_dict])
    df_eng = calculate_financial_features(df_single)
    ratios = {
        "total_monthly_expenses": float(df_eng["total_monthly_expenses"].iloc[0]),
        "total_monthly_obligations": float(df_eng["total_monthly_obligations"].iloc[0]),
        "disposable_income": float(df_eng["disposable_income"].iloc[0]),
        "debt_to_income_ratio": float(df_eng["debt_to_income_ratio"].iloc[0]),
        "current_debt_to_income_ratio": float(df_eng["current_debt_to_income_ratio"].iloc[0]),
        "expense_to_income_ratio": float(df_eng["expense_to_income_ratio"].iloc[0]),
        "obligation_to_income_ratio": float(df_eng["obligation_to_income_ratio"].iloc[0]),
        "proposed_principal_burden_ratio": float(df_eng["proposed_principal_burden_ratio"].iloc[0]),
        "savings_to_income_ratio": float(df_eng["savings_to_income_ratio"].iloc[0]),
        "emergency_fund_months": float(df_eng["emergency_fund_months"].iloc[0]),
        "requested_principal_per_month": float(df_eng["requested_principal_per_month"].iloc[0]),
        "requested_amount_to_income_ratio": float(df_eng["requested_amount_to_income_ratio"].iloc[0]),
        "dependents_ratio": float(df_eng["dependents_ratio"].iloc[0]),
        "employment_stability_score": float(df_eng["employment_stability_score"].iloc[0]),
    }
    return ratios


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for engineering financial ratios.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return calculate_financial_features(X)
