"""
Configuration module for EMIPredict AI.
Centralizes paths, target specifications, schema lists, and runtime parameters.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Base directory of repository
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Config paths
CONFIG_PATH: Path = BASE_DIR / "configs" / "model_config.yaml"


def load_yaml_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from a YAML file with safe fallback."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


_cfg = load_yaml_config()

# Paths
DATA_RAW_PATH: Path = BASE_DIR / _cfg.get("paths", {}).get("raw_data", "data/raw/EMI_dataset.csv")
DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
DATA_SAMPLE_PATH: Path = BASE_DIR / _cfg.get("paths", {}).get("sample_data", "data/sample/emi_sample.csv")
MODELS_DIR: Path = BASE_DIR / _cfg.get("paths", {}).get("models_dir", "models")
REPORTS_DIR: Path = BASE_DIR / _cfg.get("paths", {}).get("reports_dir", "reports")
FIGURES_DIR: Path = BASE_DIR / _cfg.get("paths", {}).get("figures_dir", "reports/figures")
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    _cfg.get("paths", {}).get("database_url", f"sqlite:///{BASE_DIR / 'database' / 'applicants.db'}")
)

# Random Seed
RANDOM_STATE: int = int(_cfg.get("project", {}).get("random_state", 42))

# Targets
TARGET_CLASSIFICATION: str = _cfg.get("targets", {}).get("classification", "emi_eligibility")
TARGET_REGRESSION: str = _cfg.get("targets", {}).get("regression", "max_monthly_emi")
CLASSIFICATION_CLASSES: List[str] = _cfg.get("targets", {}).get(
    "classes", ["Eligible", "High_Risk", "Not_Eligible"]
)

# Sensitive demographic attributes to exclude from decision models for fairness
SENSITIVE_ATTRIBUTES: List[str] = _cfg.get("data_schema", {}).get(
    "sensitive_attributes", ["gender", "marital_status"]
)

# Raw Features
RAW_NUMERICAL_FEATURES: List[str] = _cfg.get("data_schema", {}).get(
    "raw_numerical_features",
    [
        "age", "monthly_salary", "years_of_employment", "monthly_rent",
        "family_size", "dependents", "school_fees", "college_fees",
        "travel_expenses", "groceries_utilities", "other_monthly_expenses",
        "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
        "requested_amount", "requested_tenure"
    ]
)

RAW_CATEGORICAL_FEATURES: List[str] = _cfg.get("data_schema", {}).get(
    "raw_categorical_features",
    [
        "education", "employment_type", "company_type", "house_type",
        "existing_loans", "emi_scenario"
    ]
)

# Engineered Features
ENGINEERED_NUMERICAL_FEATURES: List[str] = _cfg.get("data_schema", {}).get(
    "engineered_numerical_features",
    [
        "total_monthly_expenses", "total_monthly_obligations", "disposable_income",
        "debt_to_income_ratio", "expense_to_income_ratio", "obligation_to_income_ratio",
        "proposed_principal_burden_ratio", "savings_to_income_ratio",
        "emergency_fund_months", "requested_principal_per_month",
        "requested_amount_to_income_ratio", "dependents_ratio",
        "employment_stability_score"
    ]
)

# Split Settings
TRAIN_SIZE: float = float(_cfg.get("split", {}).get("train_size", 0.70))
VAL_SIZE: float = float(_cfg.get("split", {}).get("val_size", 0.15))
TEST_SIZE: float = float(_cfg.get("split", {}).get("test_size", 0.15))

# Compute & MLflow
N_JOBS: int = int(_cfg.get("compute", {}).get("n_jobs", 2))
SAMPLE_FOR_TUNING: int = int(_cfg.get("compute", {}).get("sample_for_tuning", 50000))
CV_FOLDS: int = int(_cfg.get("compute", {}).get("cv_folds", 3))

MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI",
    _cfg.get("mlflow", {}).get("tracking_uri", f"sqlite:///{BASE_DIR / 'mlflow.db'}")
)
MLFLOW_EXP_CLASSIFICATION: str = _cfg.get("mlflow", {}).get(
    "experiment_classification", "EMIPredict_Classification"
)
MLFLOW_EXP_REGRESSION: str = _cfg.get("mlflow", {}).get(
    "experiment_regression", "EMIPredict_Regression"
)
