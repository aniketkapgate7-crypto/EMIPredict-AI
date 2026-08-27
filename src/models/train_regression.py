"""
Regression training pipeline for EMIPredict AI.
Memory-optimized for 8 GB RAM Windows systems.
Trains and compares Linear Regression, Ridge, Random Forest, and XGBoost regressors
for predicting maximum safe monthly installment (max_monthly_emi).
Logs experiments in MLflow and selects the winning model based on MAE.
"""

import time
from typing import Any, Dict, List, Tuple

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config import (
    DATA_PROCESSED_DIR,
    MLFLOW_EXP_REGRESSION,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_REGRESSION,
)
from src.features.preprocessing import create_preprocessor
from src.logging_config import setup_logger
from src.models.evaluate import evaluate_regression_model
from src.utils.artifacts import save_joblib_artifact, save_json_metadata

logger = setup_logger(__name__)


def setup_mlflow():
    """Configures MLflow tracking URI and experiment for regression."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXP_REGRESSION)
    logger.info(f"MLflow configured: URI='{MLFLOW_TRACKING_URI}', Exp='{MLFLOW_EXP_REGRESSION}'")


def load_partitions() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads train, validation, and test datasets from processed parquet files."""
    train_path = DATA_PROCESSED_DIR / "train.parquet"
    val_path = DATA_PROCESSED_DIR / "val.parquet"
    test_path = DATA_PROCESSED_DIR / "test.parquet"

    if not train_path.exists():
        from src.data.prepare_data import prepare_sample_and_splits
        logger.info("Processed partitions missing. Generating now...")
        return prepare_sample_and_splits()[:3]

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    return train_df, val_df, test_df


def train_regression_models() -> Tuple[Pipeline, pd.DataFrame, Dict[str, Any]]:
    """
    Executes the full regression training pipeline.
    Trains Linear Regression, Ridge, Random Forest, XGBoost, and Decision Tree.
    Logs each run to MLflow, compares validation MAE, and saves the best model.
    """
    setup_mlflow()
    train_df, val_df, test_df = load_partitions()

    # Separate features and target (remove emi_eligibility to prevent target leakage)
    X_train = train_df.drop(columns=[TARGET_REGRESSION, "emi_eligibility"], errors="ignore")
    y_train = train_df[TARGET_REGRESSION]

    X_val = val_df.drop(columns=[TARGET_REGRESSION, "emi_eligibility"], errors="ignore")
    y_val = val_df[TARGET_REGRESSION]

    X_test = test_df.drop(columns=[TARGET_REGRESSION, "emi_eligibility"], errors="ignore")
    y_test = test_df[TARGET_REGRESSION]

    logger.info(f"Train size: {len(X_train):,}, Val size: {len(X_val):,}, Test size: {len(X_test):,}")

    models_config = {
        "Linear Regression": {
            "model": LinearRegression(),
        },
        "Ridge Regression": {
            "model": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(
                max_depth=12,
                random_state=RANDOM_STATE
            ),
        },
        "Random Forest": {
            "model": RandomForestRegressor(
                n_estimators=50,
                max_depth=12,
                max_samples=0.5,
                random_state=RANDOM_STATE,
                n_jobs=1
            ),
        },
        "XGBoost": {
            "model": XGBRegressor(
                n_estimators=80,
                max_depth=6,
                learning_rate=0.1,
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=1
            ),
        }
    }

    comparison_results: List[Dict[str, Any]] = []
    trained_pipelines: Dict[str, Pipeline] = {}
    validation_metrics_all: Dict[str, Dict[str, Any]] = {}

    for name, cfg in models_config.items():
        logger.info(f"--- Training {name} ---")
        start_time = time.time()

        # Build end-to-end pipeline
        pipe = Pipeline(steps=[
            ("preprocessing", create_preprocessor(include_feature_engineering=True)),
            ("regressor", cfg["model"])
        ])

        with mlflow.start_run(run_name=name):
            # Fit pipeline
            pipe.fit(X_train, y_train)
            train_duration = round(time.time() - start_time, 2)

            # Validation predictions
            preds = pipe.predict(X_val)

            # Evaluate
            val_metrics = evaluate_regression_model(
                y_true=y_val,
                y_pred=preds,
                model_name=name,
                apply_lower_bound=True,
                save_plots=True
            )

            # Log to MLflow
            mlflow.log_param("model_type", name)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))
            mlflow.log_param("training_duration_seconds", train_duration)

            mlflow.log_metric("val_mae", val_metrics["mae"])
            mlflow.log_metric("val_rmse", val_metrics["rmse"])
            mlflow.log_metric("val_r2", val_metrics["r2"])
            mlflow.log_metric("val_mape", val_metrics["mape"])

            if "actual_vs_predicted_plot" in val_metrics:
                mlflow.log_artifact(val_metrics["actual_vs_predicted_plot"])
            if "residual_plot" in val_metrics:
                mlflow.log_artifact(val_metrics["residual_plot"])

            comparison_results.append({
                "Model": name,
                "MAE (INR)": val_metrics["mae"],
                "RMSE (INR)": val_metrics["rmse"],
                "R2 Score": val_metrics["r2"],
                "MAPE": val_metrics["mape"],
                "Training Time (s)": train_duration,
            })

            trained_pipelines[name] = pipe
            validation_metrics_all[name] = val_metrics
            logger.info(
                f"{name} Results -> MAE: INR {val_metrics['mae']:,.2f} | "
                f"RMSE: INR {val_metrics['rmse']:,.2f} | R2: {val_metrics['r2']:.4f} | Time: {train_duration}s"
            )

    # Save Comparison CSV (Sorted by MAE ascending)
    comp_df = pd.DataFrame(comparison_results).sort_values(by="MAE (INR)", ascending=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comp_csv_path = REPORTS_DIR / "regression_model_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    logger.info(f"Saved regression comparison table to {comp_csv_path}")

    # Select Best Model based on lowest MAE
    best_model_name = comp_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    best_val_metrics = validation_metrics_all[best_model_name]
    logger.info(f"Selected Champion Regression Model: {best_model_name} (MAE: INR {best_val_metrics['mae']:,.2f})")

    # Evaluate Best Pipeline on untouched Test set
    test_preds = best_pipeline.predict(X_test)
    test_metrics = evaluate_regression_model(
        y_true=y_test,
        y_pred=test_preds,
        model_name=f"{best_model_name} (Test Set)",
        apply_lower_bound=True,
        save_plots=True
    )
    logger.info(f"Test Set MAE for {best_model_name}: INR {test_metrics['mae']:,.2f} | RMSE: INR {test_metrics['rmse']:,.2f}")

    # Save Selected Model Pipeline for Production & Streamlit
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_joblib_artifact(best_pipeline, MODELS_DIR / "max_emi_pipeline.joblib")

    # Save model metadata
    model_metadata = {
        "model_name": best_model_name,
        "task": "regression",
        "target_column": TARGET_REGRESSION,
        "primary_metric": "mae",
        "validation_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "presentation_lower_bound": 0.0,
        "saved_path": str(MODELS_DIR / "max_emi_pipeline.joblib"),
    }
    save_json_metadata(model_metadata, MODELS_DIR / "max_emi_metadata.json")

    return best_pipeline, comp_df, model_metadata


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    best_pipe, comparison, meta = train_regression_models()
    print("\n--- Regression Training Complete ---")
    print(comparison.to_string(index=False))
