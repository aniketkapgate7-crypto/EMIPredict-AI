"""
Prediction and inference service for EMIPredict AI.
Loads saved production pipelines, validates incoming data, computes financial ratios,
and returns structured predictions, probabilities, and financial health diagnostics.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from src.config import (
    CLASSIFICATION_CLASSES,
    MODELS_DIR,
)
from src.features.build_features import compute_single_applicant_ratios
from src.logging_config import setup_logger
from src.utils.artifacts import load_joblib_artifact, load_json_metadata
from src.utils.validation import sanitize_input_df, validate_input_dict

logger = setup_logger(__name__)

# Cache for loaded models
_CACHED_MODELS: Dict[str, Any] = {}


def load_prediction_pipelines(
    models_dir: Path | str | None = None
) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any]]:
    """
    Loads saved classification and regression pipelines and their metadata.
    Uses in-memory cache to ensure high-performance inference.
    """
    global _CACHED_MODELS

    path = Path(models_dir) if models_dir else MODELS_DIR

    if "cls_pipeline" not in _CACHED_MODELS:
        cls_file = path / "eligibility_pipeline.joblib"
        if not cls_file.exists():
            raise FileNotFoundError(
                f"Classification model not found at {cls_file}. "
                "Please run 'python -m src.models.train_classification' first."
            )
        _CACHED_MODELS["cls_pipeline"] = load_joblib_artifact(cls_file)
        _CACHED_MODELS["cls_meta"] = load_json_metadata(path / "eligibility_metadata.json")

    if "reg_pipeline" not in _CACHED_MODELS:
        reg_file = path / "max_emi_pipeline.joblib"
        if not reg_file.exists():
            raise FileNotFoundError(
                f"Regression model not found at {reg_file}. "
                "Please run 'python -m src.models.train_regression' first."
            )
        _CACHED_MODELS["reg_pipeline"] = load_joblib_artifact(reg_file)
        _CACHED_MODELS["reg_meta"] = load_json_metadata(path / "max_emi_metadata.json")

    return (
        _CACHED_MODELS["cls_pipeline"],
        _CACHED_MODELS["reg_pipeline"],
        _CACHED_MODELS["cls_meta"],
        _CACHED_MODELS["reg_meta"],
    )


def predict_applicant_risk(
    applicant_data: Dict[str, Any],
    models_dir: Path | str | None = None
) -> Dict[str, Any]:
    """
    Performs comprehensive risk assessment and max EMI prediction for a single applicant.

    Returns structured results:
    - predicted_eligibility (Eligible | High_Risk | Not_Eligible)
    - probabilities (dict of class: probability)
    - max_monthly_emi (₹ estimate, bounded >= 0)
    - financial_ratios (DTI, disposable income, savings ratio, etc.)
    - status (success | error)
    - error_message (if validation fails)
    """
    # 1. Validate inputs
    is_valid, validation_errors = validate_input_dict(applicant_data)
    if not is_valid:
        return {
            "status": "error",
            "error_message": "; ".join(validation_errors),
            "predicted_eligibility": None,
            "probabilities": {},
            "max_monthly_emi": None,
            "financial_ratios": {},
        }

    # 2. Compute financial health ratios
    ratios = compute_single_applicant_ratios(applicant_data)

    # 3. Load ML models
    try:
        cls_pipeline, reg_pipeline, cls_meta, reg_meta = load_prediction_pipelines(models_dir)
    except FileNotFoundError as e:
        return {
            "status": "error",
            "error_message": str(e),
            "predicted_eligibility": None,
            "probabilities": {},
            "max_monthly_emi": None,
            "financial_ratios": ratios,
        }

    # 4. Format Input DataFrame
    df_input = pd.DataFrame([applicant_data])
    df_input = sanitize_input_df(df_input)

    # 5. Multiclass Classification Inference
    try:
        # Predict class probabilities
        probas = cls_pipeline.predict_proba(df_input)[0]

        # Check if pipeline was trained with encoded classes or raw strings
        if cls_meta.get("uses_label_encoding") and "class_mapping" in cls_meta:
            inv_map = {v: k for k, v in cls_meta["class_mapping"].items()}
            raw_pred = cls_pipeline.predict(df_input)[0]
            pred_class = inv_map.get(raw_pred, str(raw_pred))
            prob_dict = {
                inv_map.get(i, CLASSIFICATION_CLASSES[i]): round(float(p), 4)
                for i, p in enumerate(probas)
            }
        else:
            pred_class = str(cls_pipeline.predict(df_input)[0])
            # Match probabilities with model classes
            if hasattr(cls_pipeline.named_steps.get("classifier", None), "classes_"):
                model_classes = list(cls_pipeline.named_steps["classifier"].classes_)
                prob_dict = {
                    str(c): round(float(probas[i]), 4)
                    for i, c in enumerate(model_classes)
                }
            else:
                prob_dict = {
                    CLASSIFICATION_CLASSES[i]: round(float(probas[i]), 4)
                    for i in range(len(probas))
                }
    except Exception as e:
        logger.error(f"Classification inference failed: {e}")
        return {
            "status": "error",
            "error_message": f"Classification inference error: {str(e)}",
            "predicted_eligibility": None,
            "probabilities": {},
            "max_monthly_emi": None,
            "financial_ratios": ratios,
        }

    # 6. Maximum Monthly EMI Regression Inference
    try:
        raw_emi_pred = float(reg_pipeline.predict(df_input)[0])
        # Apply documented presentation-level lower bound of ₹0
        max_emi_pred = max(0.0, round(raw_emi_pred, 2))
    except Exception as e:
        logger.error(f"Regression inference failed: {e}")
        return {
            "status": "error",
            "error_message": f"Regression inference error: {str(e)}",
            "predicted_eligibility": pred_class,
            "probabilities": prob_dict,
            "max_monthly_emi": None,
            "financial_ratios": ratios,
        }

    # 7. Affordability & Recommendation Diagnostics
    dti = ratios.get("current_debt_to_income_ratio", ratios.get("debt_to_income_ratio", 0.0))
    req_monthly = ratios.get("requested_principal_per_month", 0.0)
    proposed_burden = ratios.get("proposed_principal_burden_ratio", 0.0)
    obligation_ratio = ratios.get("obligation_to_income_ratio", 0.0)
    disp_income = ratios.get("disposable_income", 0.0)

    insights = []
    if dti > 0.40:
        insights.append(
            f"Current Debt-to-Income (DTI) ratio ({dti * 100:.1f}%) exceeds standard academic benchmark (≤ 40%)."
        )
    if proposed_burden > 0.40:
        insights.append(
            f"Proposed principal burden ratio ({proposed_burden * 100:.1f}%) exceeds academic prudence benchmark (≤ 40% of gross salary)."
        )
    if req_monthly > max_emi_pred and max_emi_pred > 0:
        insights.append(
            f"Principal-only monthly estimate (₹{req_monthly:,.0f}) exceeds model-recommended maximum safe EMI (₹{max_emi_pred:,.0f})."
        )
    if obligation_ratio > 0.70:
        insights.append(
            f"Total monthly obligation-to-income ratio ({obligation_ratio * 100:.1f}% including living expenses and active debt) indicates high cash-flow utilization against academic benchmark (≤ 70%)."
        )
    if disp_income < max_emi_pred:
        insights.append(
            f"Current disposable surplus (₹{disp_income:,.0f}) is lower than recommended installment capacity."
        )
    if not insights:
        insights.append("Applicant profile meets standard financial affordability criteria against academic benchmarks.")

    return {
        "status": "success",
        "predicted_eligibility": pred_class,
        "probabilities": prob_dict,
        "max_monthly_emi": max_emi_pred,
        "financial_ratios": ratios,
        "diagnostics": insights,
        "model_version": "1.0.0",
        "disclaimer": "EMIPredict AI provides intelligent underwriting decision support. Final loan approval requires institutional underwriter verification.",
    }
