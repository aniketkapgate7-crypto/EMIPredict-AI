"""
SQLAlchemy database models for EMIPredict AI.
Defines schema for applicant financial profiles, underwriting decisions, and model outputs.
"""

import json
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ApplicantRecord(Base):
    """
    SQLAlchemy model representing an applicant's financial profile,
    model risk assessment, calculated ratios, and underwriting review state.
    """
    __tablename__ = "applicant_records"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Personal Profile
    age = Column(Integer, nullable=False)
    gender = Column(String(50), nullable=True)
    marital_status = Column(String(50), nullable=True)
    education = Column(String(50), nullable=False)

    # Employment & Income
    monthly_salary = Column(Float, nullable=False)
    employment_type = Column(String(50), nullable=False)
    years_of_employment = Column(Float, nullable=False)
    company_type = Column(String(50), nullable=False)

    # Housing & Family
    house_type = Column(String(50), nullable=False)
    monthly_rent = Column(Float, default=0.0)
    family_size = Column(Integer, default=1)
    dependents = Column(Integer, default=0)

    # Monthly Expenses
    school_fees = Column(Float, default=0.0)
    college_fees = Column(Float, default=0.0)
    travel_expenses = Column(Float, default=0.0)
    groceries_utilities = Column(Float, default=0.0)
    other_monthly_expenses = Column(Float, default=0.0)

    # Credit & Financial Position
    existing_loans = Column(String(10), default="No")
    current_emi_amount = Column(Float, default=0.0)
    credit_score = Column(Float, nullable=False)
    bank_balance = Column(Float, default=0.0)
    emergency_fund = Column(Float, default=0.0)

    # Loan Request
    emi_scenario = Column(String(100), nullable=False)
    requested_amount = Column(Float, nullable=False)
    requested_tenure = Column(Integer, nullable=False)

    # Model Predictions & Analytics
    predicted_eligibility = Column(String(50), nullable=True)
    predicted_max_emi = Column(Float, nullable=True)
    probabilities_json = Column(Text, nullable=True)
    financial_ratios_json = Column(Text, nullable=True)
    model_version = Column(String(50), default="1.0.0")

    # Workflow & Underwriting Review
    status = Column(String(50), default="Pending Review")  # Pending Review, Approved, Flagged, Rejected
    notes = Column(Text, default="")
    applicant_identifier = Column(String(100), nullable=True)  # Anonymous reference ID e.g. APP-10023

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Converts model instance to dictionary with parsed JSON fields."""
        probs = {}
        if self.probabilities_json:
            try:
                probs = json.loads(self.probabilities_json)
            except Exception:
                probs = {}

        ratios = {}
        if self.financial_ratios_json:
            try:
                ratios = json.loads(self.financial_ratios_json)
            except Exception:
                ratios = {}

        return {
            "id": self.id,
            "applicant_identifier": self.applicant_identifier or f"APP-{self.id:05d}",
            "age": self.age,
            "gender": self.gender,
            "marital_status": self.marital_status,
            "education": self.education,
            "monthly_salary": self.monthly_salary,
            "employment_type": self.employment_type,
            "years_of_employment": self.years_of_employment,
            "company_type": self.company_type,
            "house_type": self.house_type,
            "monthly_rent": self.monthly_rent,
            "family_size": self.family_size,
            "dependents": self.dependents,
            "school_fees": self.school_fees,
            "college_fees": self.college_fees,
            "travel_expenses": self.travel_expenses,
            "groceries_utilities": self.groceries_utilities,
            "other_monthly_expenses": self.other_monthly_expenses,
            "existing_loans": self.existing_loans,
            "current_emi_amount": self.current_emi_amount,
            "credit_score": self.credit_score,
            "bank_balance": self.bank_balance,
            "emergency_fund": self.emergency_fund,
            "emi_scenario": self.emi_scenario,
            "requested_amount": self.requested_amount,
            "requested_tenure": self.requested_tenure,
            "predicted_eligibility": self.predicted_eligibility,
            "predicted_max_emi": self.predicted_max_emi,
            "probabilities": probs,
            "financial_ratios": ratios,
            "model_version": self.model_version,
            "status": self.status,
            "notes": self.notes,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M:%S") if self.updated_at else None,
        }
