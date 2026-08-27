from pathlib import Path
from typing import Any, Dict

import mlflow
import pandas as pd

from src.config import (
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERICAL_FEATURES,
    REPORTS_DIR,
    SENSITIVE_ATTRIBUTES,
)
from src.logging_config import setup_logger
from src.utils.artifacts import load_json_metadata, save_json_metadata
from src.utils.validation import DOMAIN_RANGES, VALID_CATEGORIES

logger = setup_logger(__name__)


def export_mlflow_run_summary(output_path: Path | None = None) -> pd.DataFrame:
    """Exports all logged MLflow runs across classification and regression to CSV."""
    if output_path is None:
        output_path = REPORTS_DIR / "mlflow_run_summary.csv"

    records = []
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            for run in runs:
                row = {
                    "experiment_name": exp.name,
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", "Unnamed"),
                    "status": run.info.status,
                    "start_time": pd.to_datetime(run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
                    if run.info.start_time
                    else "",
                }
                for k, v in run.data.params.items():
                    row[f"param_{k}"] = v
                for k, v in run.data.metrics.items():
                    row[f"metric_{k}"] = round(v, 4) if isinstance(v, float) else v
                records.append(row)
    except Exception as e:
        logger.warning(f"Could not export MLflow runs directly: {e}")

    df_summary = pd.DataFrame(records) if records else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(output_path, index=False)
    logger.info(f"Exported {len(records)} MLflow runs to {output_path}")
    return df_summary


def generate_input_schema(output_path: Path | None = None) -> Dict[str, Any]:
    """Generates a complete JSON input schema for inference and validation."""
    if output_path is None:
        output_path = MODELS_DIR / "input_schema.json"

    schema = {
        "title": "EMIPredict AI Applicant Input Schema",
        "description": "Input schema specification for EMI eligibility classification and max EMI regression.",
        "type": "object",
        "required_features": RAW_NUMERICAL_FEATURES + [
            col for col in RAW_CATEGORICAL_FEATURES if col not in SENSITIVE_ATTRIBUTES
        ],
        "sensitive_attributes_audited": SENSITIVE_ATTRIBUTES,
        "properties": {},
    }

    for col in RAW_NUMERICAL_FEATURES:
        min_v, max_v = DOMAIN_RANGES.get(col, (0, 1000000))
        schema["properties"][col] = {
            "type": "number",
            "domain_min": min_v,
            "domain_max": max_v,
            "description": f"Numerical attribute: {col.replace('_', ' ').title()}",
        }

    for col in RAW_CATEGORICAL_FEATURES:
        opts = VALID_CATEGORIES.get(col, [])
        schema["properties"][col] = {
            "type": "string",
            "allowed_values": opts,
            "description": f"Categorical attribute: {col.replace('_', ' ').title()}",
        }

    save_json_metadata(schema, output_path)
    return schema


def generate_model_reports_and_metadata(
    cls_metadata_path: Path | None = None,
    reg_metadata_path: Path | None = None
) -> None:
    """
    Synthesizes model comparison metrics into model_selection_report.md,
    model_card.md, and consolidated model_metadata.json.
    """
    if cls_metadata_path is None:
        cls_metadata_path = MODELS_DIR / "eligibility_metadata.json"
    if reg_metadata_path is None:
        reg_metadata_path = MODELS_DIR / "max_emi_metadata.json"

    cls_meta = load_json_metadata(cls_metadata_path)
    reg_meta = load_json_metadata(reg_metadata_path)

    # 1. Consolidated Model Metadata JSON
    consolidated_metadata = {
        "project": "EMIPredict AI",
        "version": "1.0.0",
        "classification_champion": cls_meta,
        "regression_champion": reg_meta,
        "excluded_features_for_fairness": SENSITIVE_ATTRIBUTES,
    }
    save_json_metadata(consolidated_metadata, MODELS_DIR / "model_metadata.json")

    # 2. Model Selection Report Markdown
    sel_report_path = REPORTS_DIR / "model_selection_report.md"
    with open(sel_report_path, "w", encoding="utf-8") as f:
        f.write("# EMIPredict AI — Model Selection & Architecture Report\n\n")
        f.write("## Executive Summary\n")
        f.write(
            "This report documents the empirical evaluation, trade-off analysis, and selection "
            "of production machine learning models for the EMIPredict AI platform.\n\n"
        )

        f.write("## 1. Classification Model Selection (`emi_eligibility`)\n")
        f.write(
            "- **Primary Selection Metric**: **Macro F1-Score** (to ensure robust, balanced performance across all risk classes).\n"
        )
        if cls_meta:
            f.write(f"- **Selected Champion**: `{cls_meta.get('model_name')}`\n")
            f.write(f"- **Validation Macro F1**: {cls_meta.get('validation_metrics', {}).get('macro_f1', 'N/A')}\n")
            f.write(f"- **Validation Accuracy**: {cls_meta.get('validation_metrics', {}).get('accuracy', 'N/A')}\n")
            f.write(f"- **Test Set Macro F1**: {cls_meta.get('test_metrics', {}).get('macro_f1', 'N/A')}\n\n")

        f.write("## 2. Regression Model Selection (`max_monthly_emi`)\n")
        f.write(
            "- **Primary Selection Metric**: **Mean Absolute Error (MAE)** (measures expected rupee error directly in INR).\n"
        )
        if reg_meta:
            f.write(f"- **Selected Champion**: `{reg_meta.get('model_name')}`\n")
            f.write(f"- **Validation MAE**: INR {reg_meta.get('validation_metrics', {}).get('mae', 'N/A')}\n")
            f.write(f"- **Validation RMSE**: INR {reg_meta.get('validation_metrics', {}).get('rmse', 'N/A')}\n")
            f.write(f"- **Validation R2**: {reg_meta.get('validation_metrics', {}).get('r2', 'N/A')}\n")
            f.write(f"- **Test Set MAE**: INR {reg_meta.get('test_metrics', {}).get('mae', 'N/A')}\n\n")

        f.write("## 3. Responsible AI & Fairness Decisions\n")
        f.write(
            "- Demographic attributes (`gender`, `marital_status`) were strictly excluded from model feature inputs "
            "to prevent discriminatory lending bias and comply with ethical AI principles.\n"
            "- Preprocessing pipelines were strictly fitted only on training partitions to prevent data leakage.\n"
        )

    # 3. Model Card Markdown
    card_path = REPORTS_DIR / "model_card.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write("# EMIPredict AI — Model Card\n\n")
        f.write("## Model Details\n")
        f.write("- **Organization**: EMIPredict AI Open Source Project\n")
        f.write("- **Model Date**: August 2026\n")
        f.write("- **Model Version**: 1.0.0\n")
        f.write("- **Model Type**: Scikit-Learn Pipeline combining Feature Engineering + ColumnTransformer + Ensemble Classifiers/Regressors\n")
        f.write("- **License**: MIT\n\n")

        f.write("## Intended Use\n")
        f.write("- **Primary Use Case**: Assisting financial institutions and loan officers with preliminary applicant risk profiling and monthly repayment affordability estimation.\n")
        f.write("- **Out of Scope**: Automatic, unsupervised loan rejection or granting without human underwriter review.\n\n")

        f.write("## Training & Evaluation Data\n")
        f.write("- **Source**: `data/raw/EMI_dataset.csv` (404,800 records across 25 input variables and 2 target variables).\n")
        f.write("- **Partitioning**: 70% Training, 15% Validation, 15% Test (Stratified Split).\n\n")

        f.write("## Ethical Considerations\n")
        f.write("- **Fairness**: Sensitive demographic features (`gender`, `marital_status`) are omitted from input features.\n")
        f.write("- **Non-Negativity Constraint**: Regression predictions are bounded at INR 0 to avoid impossible negative repayment commitments.\n")

    # Generate schema
    generate_input_schema()

    # Export MLflow runs summary
    export_mlflow_run_summary()

    logger.info("Generated model reports, model card, input schema, and MLflow run summary.")


if __name__ == "__main__":
    generate_model_reports_and_metadata()
