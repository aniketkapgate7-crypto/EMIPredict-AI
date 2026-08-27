"""
Classification training pipeline for EMIPredict AI.
Memory-optimized for 8 GB RAM Windows systems.
Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost classifiers.
Tracks experiments in MLflow and selects the winning model based on Macro F1-score.
"""

import time
from typing import Any, Dict, List, Tuple

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    CLASSIFICATION_CLASSES,
    DATA_PROCESSED_DIR,
    MLFLOW_EXP_CLASSIFICATION,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_CLASSIFICATION,
)
from src.features.preprocessing import create_preprocessor
from src.logging_config import setup_logger
from src.models.evaluate import evaluate_classification_model
from src.utils.artifacts import save_joblib_artifact, save_json_metadata

logger = setup_logger(__name__)


def setup_mlflow():
    """Configures MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXP_CLASSIFICATION)
    logger.info(f"MLflow configured: URI='{MLFLOW_TRACKING_URI}', Exp='{MLFLOW_EXP_CLASSIFICATION}'")


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


def train_classification_models() -> Tuple[Pipeline, pd.DataFrame, Dict[str, Any]]:
    """
    Executes the full multiclass classification training pipeline.
    Trains Logistic Regression, Random Forest, XGBoost, and Decision Tree.
    Logs each run to MLflow, compares validation Macro F1, and saves the best model.
    """
    setup_mlflow()
    train_df, val_df, test_df = load_partitions()

    # Separate features and target
    X_train = train_df.drop(columns=[TARGET_CLASSIFICATION, "max_monthly_emi"], errors="ignore")
    y_train = train_df[TARGET_CLASSIFICATION]

    X_val = val_df.drop(columns=[TARGET_CLASSIFICATION, "max_monthly_emi"], errors="ignore")
    y_val = val_df[TARGET_CLASSIFICATION]

    X_test = test_df.drop(columns=[TARGET_CLASSIFICATION, "max_monthly_emi"], errors="ignore")
    y_test = test_df[TARGET_CLASSIFICATION]

    # Label Encoder for XGBoost compatibility
    le = LabelEncoder()
    le.fit(CLASSIFICATION_CLASSES)
    y_train_enc = le.transform(y_train)

    # Class mappings
    label_map = {cls: idx for idx, cls in enumerate(CLASSIFICATION_CLASSES)}
    inv_label_map = {idx: cls for cls, idx in label_map.items()}

    logger.info(f"Target classes: {CLASSIFICATION_CLASSES}")
    logger.info(f"Train size: {len(X_train):,}, Val size: {len(X_val):,}, Test size: {len(X_test):,}")

    models_config = {
        "Logistic Regression": {
            "model": LogisticRegression(
                class_weight="balanced",
                max_iter=500,
                random_state=RANDOM_STATE
            ),
            "use_encoded_y": False,
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=12,
                random_state=RANDOM_STATE
            ),
            "use_encoded_y": False,
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=50,
                max_depth=12,
                max_samples=0.5,  # Subsample to keep memory usage under 200MB
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1          # Single process to prevent joblib multiprocessing IPC memory explosion
            ),
            "use_encoded_y": False,
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=80,
                max_depth=6,
                learning_rate=0.1,
                tree_method="hist",
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=1
            ),
            "use_encoded_y": True,
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
            ("classifier", cfg["model"])
        ])

        target_train = y_train_enc if cfg["use_encoded_y"] else y_train

        with mlflow.start_run(run_name=name):
            # Fit pipeline
            pipe.fit(X_train, target_train)
            train_duration = round(time.time() - start_time, 2)

            # Validation predictions
            preds_raw = pipe.predict(X_val)
            probas = pipe.predict_proba(X_val)

            if cfg["use_encoded_y"]:
                preds = [inv_label_map[p] for p in preds_raw]
            else:
                preds = list(preds_raw)

            # Evaluate
            val_metrics = evaluate_classification_model(
                y_true=y_val,
                y_pred=preds,
                y_proba=probas,
                classes=CLASSIFICATION_CLASSES,
                model_name=name,
                save_plots=True,
            )

            # Log to MLflow
            mlflow.log_param("model_type", name)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))
            mlflow.log_param("training_duration_seconds", train_duration)

            mlflow.log_metric("val_accuracy", val_metrics["accuracy"])
            mlflow.log_metric("val_macro_precision", val_metrics["macro_precision"])
            mlflow.log_metric("val_macro_recall", val_metrics["macro_recall"])
            mlflow.log_metric("val_macro_f1", val_metrics["macro_f1"])
            mlflow.log_metric("val_weighted_f1", val_metrics["weighted_f1"])
            if val_metrics["roc_auc_ovr"] is not None:
                mlflow.log_metric("val_roc_auc_ovr", val_metrics["roc_auc_ovr"])

            if "confusion_matrix_plot" in val_metrics:
                mlflow.log_artifact(val_metrics["confusion_matrix_plot"])

            comparison_results.append({
                "Model": name,
                "Macro F1": val_metrics["macro_f1"],
                "Weighted F1": val_metrics["weighted_f1"],
                "Accuracy": val_metrics["accuracy"],
                "Macro Precision": val_metrics["macro_precision"],
                "Macro Recall": val_metrics["macro_recall"],
                "ROC-AUC (OvR)": val_metrics["roc_auc_ovr"],
                "Training Time (s)": train_duration,
            })

            trained_pipelines[name] = pipe
            validation_metrics_all[name] = val_metrics
            logger.info(
                f"{name} Results -> Macro F1: {val_metrics['macro_f1']:.4f} | "
                f"Accuracy: {val_metrics['accuracy']:.4f} | Time: {train_duration}s"
            )

    # Save Comparison CSV
    comp_df = pd.DataFrame(comparison_results).sort_values(by="Macro F1", ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comp_csv_path = REPORTS_DIR / "classification_model_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    logger.info(f"Saved classification comparison table to {comp_csv_path}")

    # Select Best Model based on Macro F1
    best_model_name = comp_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    best_val_metrics = validation_metrics_all[best_model_name]
    logger.info(
        f"Selected Champion Classification Model: {best_model_name} (Macro F1: {best_val_metrics['macro_f1']:.4f})"
    )

    # Evaluate Best Pipeline on untouched Test set
    test_preds_raw = best_pipeline.predict(X_test)
    test_probas = best_pipeline.predict_proba(X_test)

    if models_config[best_model_name]["use_encoded_y"]:
        test_preds = [inv_label_map[p] for p in test_preds_raw]
    else:
        test_preds = list(test_preds_raw)

    test_metrics = evaluate_classification_model(
        y_true=y_test,
        y_pred=test_preds,
        y_proba=test_probas,
        classes=CLASSIFICATION_CLASSES,
        model_name=f"{best_model_name} (Test Set)",
        save_plots=True
    )
    logger.info(f"Test Set Macro F1 for {best_model_name}: {test_metrics['macro_f1']:.4f}")

    # Save Selected Model Pipeline for Production & Streamlit
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_joblib_artifact(best_pipeline, MODELS_DIR / "eligibility_pipeline.joblib")

    # Save model metadata
    model_metadata = {
        "model_name": best_model_name,
        "task": "multiclass_classification",
        "target_column": TARGET_CLASSIFICATION,
        "classes": CLASSIFICATION_CLASSES,
        "primary_metric": "macro_f1",
        "validation_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "uses_label_encoding": models_config[best_model_name]["use_encoded_y"],
        "class_mapping": label_map if models_config[best_model_name]["use_encoded_y"] else None,
        "saved_path": str(MODELS_DIR / "eligibility_pipeline.joblib"),
    }
    save_json_metadata(model_metadata, MODELS_DIR / "eligibility_metadata.json")

    return best_pipeline, comp_df, model_metadata


if __name__ == "__main__":
    best_pipe, comparison, meta = train_classification_models()
    print("\n--- Classification Training Complete ---")
    print(comparison.to_string(index=False))
