"""
Input validation utilities for applicant data and prediction requests.
"""

from typing import Any, Dict, List, Tuple

import pandas as pd

# Domain boundaries for reasonable loan underwriting validation
DOMAIN_RANGES = {
    "age": (18, 100),
    "monthly_salary": (0.0, 10_000_000.0),
    "years_of_employment": (0.0, 80.0),
    "monthly_rent": (0.0, 1_000_000.0),
    "family_size": (1, 30),
    "dependents": (0, 30),
    "school_fees": (0.0, 1_000_000.0),
    "college_fees": (0.0, 1_000_000.0),
    "travel_expenses": (0.0, 500_000.0),
    "groceries_utilities": (0.0, 1_000_000.0),
    "other_monthly_expenses": (0.0, 1_000_000.0),
    "current_emi_amount": (0.0, 5_000_000.0),
    "credit_score": (300.0, 900.0),
    "bank_balance": (-100_000.0, 100_000_000.0),
    "emergency_fund": (0.0, 100_000_000.0),
    "requested_amount": (1_000.0, 100_000_000.0),
    "requested_tenure": (1, 480),
}

VALID_CATEGORIES = {
    "gender": ["Male", "Female", "Other", "Non-binary"],
    "marital_status": ["Single", "Married", "Divorced", "Widowed"],
    "education": ["High School", "Graduate", "Post Graduate", "Professional", "Doctorate", "Diploma"],
    "employment_type": ["Salaried", "Self-Employed", "Business", "Private", "Government", "Freelancer"],
    "company_type": ["MNC", "Startup", "Mid-size", "Corporate", "Public", "Small Business", "Government"],
    "house_type": ["Rented", "Own", "Mortgaged", "Family", "Company Provided"],
    "existing_loans": ["Yes", "No"],
    "emi_scenario": [
        "Personal Loan EMI", "Home Loan EMI", "Vehicle EMI", "Education EMI",
        "E-commerce Shopping EMI", "Medical Emergency EMI", "Business Equipment EMI"
    ]
}


def validate_input_dict(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates a dictionary of applicant inputs against expected fields, ranges, and types.
    Returns (is_valid, error_messages).
    """
    errors: List[str] = []

    # Check numeric ranges
    for col, (min_val, max_val) in DOMAIN_RANGES.items():
        if col in data:
            val = data[col]
            try:
                numeric_val = float(val)
                if numeric_val < min_val or numeric_val > max_val:
                    errors.append(f"Field '{col}' value {numeric_val} is outside expected range [{min_val}, {max_val}].")
            except (ValueError, TypeError):
                errors.append(f"Field '{col}' must be a valid number, received: {val}")

    # Check categorical categories if present
    for col, allowed_vals in VALID_CATEGORIES.items():
        if col in data and data[col] is not None:
            val_str = str(data[col]).strip()
            # Case insensitive check
            matching = [v for v in allowed_vals if v.lower() == val_str.lower()]
            if not matching:
                errors.append(
                    f"Field '{col}' value '{val_str}' is unrecognized. Expected one of: {allowed_vals}"
                )

    # Check logical consistency
    if "family_size" in data and "dependents" in data:
        try:
            if int(data["dependents"]) >= int(data["family_size"]):
                errors.append("Dependents count cannot be greater than or equal to family size.")
        except Exception:
            pass

    return len(errors) == 0, errors


def sanitize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and sanitize dataframe column names and strings."""
    df_clean = df.copy()
    df_clean.columns = [str(col).strip().lower().replace(" ", "_") for col in df_clean.columns]
    for col in df_clean.select_dtypes(include=["object", "string"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    return df_clean
