"""
Tests for prediction service, inference schema validation, and output structure.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.predict import predict_applicant_risk


@pytest.fixture
def valid_applicant_dict():
    """Fixture for valid applicant input."""
    return {
        "age": 35,
        "gender": "Female",
        "marital_status": "Married",
        "education": "Graduate",
        "monthly_salary": 75000.0,
        "employment_type": "Salaried",
        "years_of_employment": 7.0,
        "company_type": "MNC",
        "house_type": "Rented",
        "monthly_rent": 18000.0,
        "family_size": 3,
        "dependents": 1,
        "school_fees": 4000.0,
        "college_fees": 0.0,
        "travel_expenses": 5000.0,
        "groceries_utilities": 14000.0,
        "other_monthly_expenses": 4000.0,
        "existing_loans": "No",
        "current_emi_amount": 0.0,
        "credit_score": 750.0,
        "bank_balance": 180000.0,
        "emergency_fund": 90000.0,
        "emi_scenario": "Personal Loan EMI",
        "requested_amount": 300000.0,
        "requested_tenure": 24,
    }


def test_predict_applicant_risk_validation_failure():
    """Test response when required fields contain invalid values."""
    bad_data = {
        "age": 12,  # Invalid
        "monthly_salary": -5000.0,  # Invalid
    }
    result = predict_applicant_risk(bad_data)
    assert result["status"] == "error"
    assert "error_message" in result
    assert result["predicted_eligibility"] is None


def test_predict_applicant_risk_mocked(valid_applicant_dict):
    """Test complete inference structure using mocked pipelines."""
    mock_cls = MagicMock()
    mock_cls.predict.return_value = ["Eligible"]
    mock_cls.predict_proba.return_value = np.array([[0.85, 0.10, 0.05]])

    mock_reg = MagicMock()
    mock_reg.predict.return_value = np.array([24500.0])

    cls_meta = {
        "model_name": "Random Forest",
        "uses_label_encoding": False,
    }
    reg_meta = {
        "model_name": "Random Forest Regressor",
    }

    with patch("src.models.predict.load_prediction_pipelines") as mock_loader:
        mock_loader.return_value = (mock_cls, mock_reg, cls_meta, reg_meta)
        result = predict_applicant_risk(valid_applicant_dict)

        assert result["status"] == "success"
        assert result["predicted_eligibility"] == "Eligible"
        assert "debt_to_income_ratio" in result["financial_ratios"]
        assert "current_debt_to_income_ratio" in result["financial_ratios"]
        assert "expense_to_income_ratio" in result["financial_ratios"]
        assert "obligation_to_income_ratio" in result["financial_ratios"]
        assert "proposed_principal_burden_ratio" in result["financial_ratios"]
        assert "disposable_income" in result["financial_ratios"]
        assert "diagnostics" in result


def test_predict_applicant_risk_real_models(valid_applicant_dict):
    """Test inference against real saved model pipelines if available."""
    from src.config import MODELS_DIR
    cls_file = MODELS_DIR / "eligibility_pipeline.joblib"
    reg_file = MODELS_DIR / "max_emi_pipeline.joblib"

    if not (cls_file.exists() and reg_file.exists()):
        pytest.skip("Trained model pipelines not yet saved in models/.")

    result = predict_applicant_risk(valid_applicant_dict)
    assert result["status"] == "success"
    assert result["predicted_eligibility"] in ["Eligible", "High_Risk", "Not_Eligible"]
    assert isinstance(result["max_monthly_emi"], (int, float))
    assert result["max_monthly_emi"] >= 0.0
    assert len(result["probabilities"]) == 3
    assert "disposable_income" in result["financial_ratios"]
    assert "debt_to_income_ratio" in result["financial_ratios"]
    assert "current_debt_to_income_ratio" in result["financial_ratios"]
    assert "proposed_principal_burden_ratio" in result["financial_ratios"]


def test_predict_applicant_risk_qa_case():
    """Test inference on the audited QA case."""
    qa_input = {
        "age": 35,
        "gender": "Male",
        "marital_status": "Married",
        "education": "Graduate",
        "monthly_salary": 65000.0,
        "employment_type": "Salaried",
        "years_of_employment": 6.0,
        "company_type": "MNC",
        "house_type": "Rented",
        "monthly_rent": 15000.0,
        "family_size": 3,
        "dependents": 1,
        "school_fees": 3000.0,
        "college_fees": 0.0,
        "travel_expenses": 4500.0,
        "groceries_utilities": 12000.0,
        "other_monthly_expenses": 4000.0,
        "existing_loans": "No",
        "current_emi_amount": 0.0,
        "credit_score": 740.0,
        "bank_balance": 150000.0,
        "emergency_fund": 80000.0,
        "emi_scenario": "Personal Loan EMI",
        "requested_amount": 300000.0,
        "requested_tenure": 24,
    }
    result = predict_applicant_risk(qa_input)
    assert result["status"] == "success"
    ratios = result["financial_ratios"]
    assert ratios["current_debt_to_income_ratio"] == 0.0
    assert ratios["debt_to_income_ratio"] == 0.0
    assert pytest.approx(ratios["expense_to_income_ratio"] * 100.0, 0.05) == 59.23
    assert ratios["requested_principal_per_month"] == 12500.0
    assert pytest.approx(ratios["proposed_principal_burden_ratio"] * 100.0, 0.05) == 19.23
    assert ratios["disposable_income"] == 26500.0

