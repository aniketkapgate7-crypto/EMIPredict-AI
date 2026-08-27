"""
Tests for data validation, schema auditing, and loading modules.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import downcast_dtypes, load_raw_dataset
from src.data.validate_data import run_data_validation
from src.utils.validation import sanitize_input_df, validate_input_dict


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Fixture providing a minimal synthetic raw dataframe."""
    return pd.DataFrame({
        "age": [35, 45, 28],
        "gender": ["Male", "Female", "Male"],
        "marital_status": ["Married", "Single", "Married"],
        "education": ["Graduate", "Post Graduate", "Professional"],
        "monthly_salary": [50000.0, 80000.0, 35000.0],
        "employment_type": ["Salaried", "Private", "Self-Employed"],
        "years_of_employment": [5.0, 10.0, 2.0],
        "company_type": ["MNC", "Corporate", "Startup"],
        "house_type": ["Rented", "Own", "Rented"],
        "monthly_rent": [12000.0, 0.0, 8000.0],
        "family_size": [3, 2, 4],
        "dependents": [1, 0, 2],
        "school_fees": [2000.0, 0.0, 1500.0],
        "college_fees": [0.0, 0.0, 0.0],
        "travel_expenses": [3000.0, 5000.0, 2500.0],
        "groceries_utilities": [10000.0, 15000.0, 9000.0],
        "other_monthly_expenses": [2000.0, 3000.0, 1500.0],
        "existing_loans": ["No", "Yes", "No"],
        "current_emi_amount": [0.0, 8000.0, 0.0],
        "credit_score": [720.0, 680.0, 750.0],
        "bank_balance": [120000.0, 250000.0, 80000.0],
        "emergency_fund": [50000.0, 100000.0, 30000.0],
        "emi_scenario": ["Personal Loan EMI", "Vehicle EMI", "Education EMI"],
        "requested_amount": [200000.0, 500000.0, 150000.0],
        "requested_tenure": [24, 36, 12],
        "emi_eligibility": ["Eligible", "High_Risk", "Eligible"],
        "max_monthly_emi": [18500.0, 22000.0, 12000.0]
    })


def test_downcast_dtypes(sample_raw_df):
    """Test memory reduction via downcasting."""
    optimized = downcast_dtypes(sample_raw_df)
    assert optimized["age"].dtype in [np.int8, np.int16, np.int32]
    assert optimized["monthly_salary"].dtype in [np.float32]


def test_sanitize_input_df():
    """Test dataframe column standardization."""
    messy_df = pd.DataFrame({
        " Age ": [30],
        "Monthly Salary": [50000.0],
        "Gender": [" Male "]
    })
    clean_df = sanitize_input_df(messy_df)
    assert "age" in clean_df.columns
    assert "monthly_salary" in clean_df.columns
    assert clean_df["gender"].iloc[0] == "Male"


def test_run_data_validation(sample_raw_df, tmp_path):
    """Test full validation report generation."""
    report = run_data_validation(sample_raw_df, output_dir=tmp_path)
    assert report["total_rows"] == 3
    assert report["total_columns"] == 27
    assert (tmp_path / "data_quality_report.json").exists()
    assert (tmp_path / "missing_values.csv").exists()
    assert (tmp_path / "data_quality_summary.md").exists()


def test_validate_input_dict_valid():
    """Test valid applicant dictionary."""
    valid_data = {
        "age": 30,
        "monthly_salary": 50000.0,
        "credit_score": 750.0,
        "family_size": 3,
        "dependents": 1,
        "education": "Graduate",
        "gender": "Male"
    }
    is_valid, errors = validate_input_dict(valid_data)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_input_dict_invalid():
    """Test applicant dictionary with invalid range and logical conflict."""
    invalid_data = {
        "age": 10,  # Below min age
        "credit_score": 1200.0,  # Above max credit score
        "family_size": 2,
        "dependents": 3,  # Dependents > family size
        "gender": "InvalidGender"
    }
    is_valid, errors = validate_input_dict(invalid_data)
    assert is_valid is False
    assert len(errors) >= 3


def test_missing_dataset_handling(tmp_path):
    """Test graceful failure when raw file does not exist."""
    fake_path = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError):
        load_raw_dataset(file_path=fake_path)
