"""
Tests for SQLAlchemy database models and CRUD operations using in-memory SQLite.
"""

import pytest

from src.database.crud import (
    create_applicant_record,
    delete_applicant_record,
    get_applicant_record,
    init_db,
    list_applicant_records,
    update_applicant_record,
)


@pytest.fixture
def test_db_url():
    """In-memory SQLite database URL for fast, isolated testing."""
    return "sqlite:///:memory:"


@pytest.fixture
def sample_record_data():
    """Sample applicant record payload."""
    return {
        "age": 32,
        "gender": "Male",
        "marital_status": "Married",
        "education": "Graduate",
        "monthly_salary": 60000.0,
        "employment_type": "Salaried",
        "years_of_employment": 5.0,
        "company_type": "Corporate",
        "house_type": "Rented",
        "monthly_rent": 15000.0,
        "family_size": 3,
        "dependents": 1,
        "school_fees": 3000.0,
        "college_fees": 0.0,
        "travel_expenses": 4000.0,
        "groceries_utilities": 12000.0,
        "other_monthly_expenses": 3000.0,
        "existing_loans": "No",
        "current_emi_amount": 0.0,
        "credit_score": 730.0,
        "bank_balance": 150000.0,
        "emergency_fund": 80000.0,
        "emi_scenario": "Personal Loan EMI",
        "requested_amount": 250000.0,
        "requested_tenure": 24,
        "predicted_eligibility": "Eligible",
        "max_monthly_emi": 21000.0,
        "probabilities": {"Eligible": 0.85, "High_Risk": 0.10, "Not_Eligible": 0.05},
        "financial_ratios": {"debt_to_income_ratio": 0.56, "disposable_income": 23000.0},
        "status": "Pending Review",
        "notes": "Test applicant note",
    }


def test_crud_lifecycle(test_db_url, sample_record_data):
    """Test full Create -> Read -> List -> Update -> Delete lifecycle."""
    init_db(test_db_url)

    # 1. CREATE
    created = create_applicant_record(sample_record_data, db_url=test_db_url)
    assert created is not None
    record_id = created["id"]
    assert record_id > 0
    assert created["monthly_salary"] == 60000.0
    assert created["predicted_eligibility"] == "Eligible"
    assert created["probabilities"]["Eligible"] == 0.85

    # 2. READ
    fetched = get_applicant_record(record_id, db_url=test_db_url)
    assert fetched is not None
    assert fetched["id"] == record_id
    assert fetched["credit_score"] == 730.0

    # 3. LIST & FILTER
    records = list_applicant_records(status_filter="Pending Review", db_url=test_db_url)
    assert len(records) >= 1
    assert records[0]["id"] == record_id

    # 4. UPDATE
    updated = update_applicant_record(
        record_id,
        {"status": "Approved", "notes": "Approved by senior credit officer."},
        db_url=test_db_url
    )
    assert updated is not None
    assert updated["status"] == "Approved"
    assert updated["notes"] == "Approved by senior credit officer."

    # 5. DELETE
    deleted = delete_applicant_record(record_id, db_url=test_db_url)
    assert deleted is True

    # 6. VERIFY DELETED
    assert get_applicant_record(record_id, db_url=test_db_url) is None
