"""
Tests for financial feature engineering formulas, transformers, and preprocessing pipelines.
"""

import pandas as pd
import pytest

from src.features.build_features import (
    FinancialFeatureEngineer,
    calculate_financial_features,
)
from src.features.preprocessing import (
    create_preprocessor,
    get_model_feature_lists,
)


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Sample dataframe for feature testing."""
    return pd.DataFrame({
        "age": [30],
        "gender": ["Female"],
        "marital_status": ["Single"],
        "education": ["Graduate"],
        "monthly_salary": [100000.0],
        "employment_type": ["Salaried"],
        "years_of_employment": [6.0],
        "company_type": ["MNC"],
        "house_type": ["Rented"],
        "monthly_rent": [20000.0],
        "family_size": [3],
        "dependents": [1],
        "school_fees": [5000.0],
        "college_fees": [0.0],
        "travel_expenses": [5000.0],
        "groceries_utilities": [15000.0],
        "other_monthly_expenses": [5000.0],
        "existing_loans": ["Yes"],
        "current_emi_amount": [10000.0],
        "credit_score": [750.0],
        "bank_balance": [200000.0],
        "emergency_fund": [100000.0],
        "emi_scenario": ["Personal Loan EMI"],
        "requested_amount": [240000.0],
        "requested_tenure": [24],
    })


def test_calculate_financial_features_formulas(sample_features_df):
    """Verify exact calculations of all domain features."""
    df_res = calculate_financial_features(sample_features_df)

    # 1. total_monthly_expenses = 20000 + 5000 + 0 + 5000 + 15000 + 5000 = 50000
    assert df_res["total_monthly_expenses"].iloc[0] == 50000.0

    # 2. total_monthly_obligations = 50000 + 10000 = 60000
    assert df_res["total_monthly_obligations"].iloc[0] == 60000.0

    # 3. disposable_income = 100000 - 60000 = 40000
    assert df_res["disposable_income"].iloc[0] == 40000.0

    # 4. debt_to_income_ratio (Debt only) = 10000 / 100000 = 0.10
    assert pytest.approx(df_res["debt_to_income_ratio"].iloc[0], 0.001) == 0.10
    assert pytest.approx(df_res["current_debt_to_income_ratio"].iloc[0], 0.001) == 0.10

    # 5. expense_to_income_ratio = 50000 / 100000 = 0.50
    assert pytest.approx(df_res["expense_to_income_ratio"].iloc[0], 0.001) == 0.50

    # 6. obligation_to_income_ratio = 60000 / 100000 = 0.60
    assert pytest.approx(df_res["obligation_to_income_ratio"].iloc[0], 0.001) == 0.60

    # 7. savings_to_income_ratio = (200000 + 100000) / (100000 * 12) = 300000 / 1200000 = 0.25
    assert pytest.approx(df_res["savings_to_income_ratio"].iloc[0], 0.001) == 0.25

    # 8. emergency_fund_months = 100000 / 50000 = 2.0
    assert pytest.approx(df_res["emergency_fund_months"].iloc[0], 0.001) == 2.0

    # 9. requested_principal_per_month = 240000 / 24 = 10000
    assert pytest.approx(df_res["requested_principal_per_month"].iloc[0], 0.001) == 10000.0

    # 10. proposed_principal_burden_ratio = 10000 / 100000 = 0.10
    assert pytest.approx(df_res["proposed_principal_burden_ratio"].iloc[0], 0.001) == 0.10


def test_qa_observed_case_financial_ratios():
    """
    Focused unit test for the specific QA case:
    - Monthly gross salary: ₹65,000
    - Existing active loans: No (Current EMI: ₹0)
    - Requested amount: ₹300,000 over 24 months
    - Living expenses: ₹38,500 (~59.23% expense ratio)

    Verifies:
    1. Current DTI is 0.0%
    2. Requested principal estimate is ₹12,500
    3. Proposed principal burden is ~19.23%
    4. Expense ratio remains independent from DTI (59.23% vs 0.0%)
    """
    qa_df = pd.DataFrame({
        "age": [35],
        "gender": ["Male"],
        "marital_status": ["Married"],
        "education": ["Graduate"],
        "monthly_salary": [65000.0],
        "employment_type": ["Salaried"],
        "years_of_employment": [6.0],
        "company_type": ["MNC"],
        "house_type": ["Rented"],
        "monthly_rent": [15000.0],
        "family_size": [3],
        "dependents": [1],
        "school_fees": [3000.0],
        "college_fees": [0.0],
        "travel_expenses": [4500.0],
        "groceries_utilities": [12000.0],
        "other_monthly_expenses": [4000.0],
        "existing_loans": ["No"],
        "current_emi_amount": [0.0],
        "credit_score": [740.0],
        "bank_balance": [150000.0],
        "emergency_fund": [80000.0],
        "emi_scenario": ["Personal Loan EMI"],
        "requested_amount": [300000.0],
        "requested_tenure": [24],
    })

    df_res = calculate_financial_features(qa_df)

    # 1. Salary ₹65,000 and current EMI 0 => current DTI 0%
    assert pytest.approx(df_res["debt_to_income_ratio"].iloc[0], abs=1e-4) == 0.0
    assert pytest.approx(df_res["current_debt_to_income_ratio"].iloc[0], abs=1e-4) == 0.0

    # 2. Requested 300,000 over 24 months => principal estimate 12,500
    assert pytest.approx(df_res["requested_principal_per_month"].iloc[0], 0.01) == 12500.0

    # 3. Proposed principal burden ratio approximately 19.23% (12500 / 65000 = 19.2307%)
    assert pytest.approx(df_res["proposed_principal_burden_ratio"].iloc[0] * 100.0, 0.05) == 19.23

    # 4. Expense ratio remains independent from DTI (~59.23%)
    assert pytest.approx(df_res["expense_to_income_ratio"].iloc[0] * 100.0, 0.05) == 59.23
    assert df_res["expense_to_income_ratio"].iloc[0] != df_res["debt_to_income_ratio"].iloc[0]

    # Total living expenses = 15000 + 3000 + 0 + 4500 + 12000 + 4000 = 38500
    assert df_res["total_monthly_expenses"].iloc[0] == 38500.0
    # Total monthly obligations = 38500 + 0 = 38500
    assert df_res["total_monthly_obligations"].iloc[0] == 38500.0
    # Disposable income = 65000 - 38500 = 26500
    assert df_res["disposable_income"].iloc[0] == 26500.0


def test_division_by_zero_safeguards():
    """Verify that zero salaries, zero expenses, or zero tenures do not crash or produce Inf/NaN."""
    zero_df = pd.DataFrame({
        "age": [18],
        "monthly_salary": [0.0],
        "monthly_rent": [0.0],
        "school_fees": [0.0],
        "college_fees": [0.0],
        "travel_expenses": [0.0],
        "groceries_utilities": [0.0],
        "other_monthly_expenses": [0.0],
        "current_emi_amount": [0.0],
        "bank_balance": [0.0],
        "emergency_fund": [0.0],
        "requested_amount": [0.0],
        "requested_tenure": [0],
        "family_size": [0],
        "dependents": [0],
        "years_of_employment": [0.0],
    })

    df_res = calculate_financial_features(zero_df)
    for col in df_res.columns:
        assert not df_res[col].isna().any(), f"NaN found in {col}"
        assert not df_res[col].isin([float("inf"), float("-inf")]).any(), f"Inf found in {col}"


def test_financial_feature_engineer_transformer(sample_features_df):
    """Test Scikit-Learn transformer behavior."""
    transformer = FinancialFeatureEngineer()
    res = transformer.fit_transform(sample_features_df)
    assert "disposable_income" in res.columns
    assert "debt_to_income_ratio" in res.columns
    assert "current_debt_to_income_ratio" in res.columns
    assert "expense_to_income_ratio" in res.columns
    assert "obligation_to_income_ratio" in res.columns
    assert "proposed_principal_burden_ratio" in res.columns


def test_sensitive_attributes_exclusion():
    """Verify that gender and marital_status are strictly excluded from decision models."""
    num_cols, cat_cols, sensitive = get_model_feature_lists()
    assert "gender" not in num_cols
    assert "gender" not in cat_cols
    assert "marital_status" not in num_cols
    assert "marital_status" not in cat_cols
    assert "gender" in sensitive
    assert "marital_status" in sensitive


def test_create_preprocessor_pipeline(sample_features_df):
    """Verify that preprocessing pipeline fits and transforms cleanly."""
    pipe = create_preprocessor(include_feature_engineering=True)
    transformed = pipe.fit_transform(sample_features_df)
    assert transformed.shape[0] == 1
    assert transformed.shape[1] > 10


def test_partial_dataframe_missing_columns_resilience():
    """Verify that calculate_financial_features safely handles partial DataFrames with missing columns without crashing."""
    minimal_df = pd.DataFrame({
        "monthly_salary": [65000.0],
        "current_emi_amount": [0.0],
        "requested_amount": [300000.0],
        "requested_tenure": [24],
    })
    df_res = calculate_financial_features(minimal_df)
    assert df_res["debt_to_income_ratio"].iloc[0] == 0.0
    assert df_res["current_debt_to_income_ratio"].iloc[0] == 0.0
    assert df_res["requested_principal_per_month"].iloc[0] == 12500.0
    assert pytest.approx(df_res["proposed_principal_burden_ratio"].iloc[0] * 100.0, 0.05) == 19.23
    assert df_res["total_monthly_expenses"].iloc[0] == 0.0
    assert df_res["disposable_income"].iloc[0] == 65000.0

