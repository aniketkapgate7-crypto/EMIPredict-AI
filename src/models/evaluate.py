"""
Model evaluation module for EMIPredict AI.
Calculates rigorous metrics for multiclass classification and regression,
and generates publication-ready diagnostic plots.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server/CLI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from src.config import FIGURES_DIR
from src.logging_config import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------
# CLASSIFICATION EVALUATION
# ---------------------------------------------------------

def evaluate_classification_model(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    model_name: str = "Classifier",
    save_plots: bool = True,
    output_dir: Path = FIGURES_DIR
) -> Dict[str, Any]:
    """
    Evaluates classification predictions against ground truth.
    Primary selection metric: Macro F1-Score.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if classes is None:
        classes = sorted(list(set(y_true)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    roc_auc_ovr = None
    if y_proba is not None:
        try:
            roc_auc_ovr = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC for {model_name}: {e}")

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    report_dict = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )

    metrics = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),  # Primary Metric
        "weighted_f1": round(weighted_f1, 4),
        "roc_auc_ovr": round(roc_auc_ovr, 4) if roc_auc_ovr is not None else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    if save_plots:
        # Plot Confusion Matrix
        clean_name = model_name.lower().replace(" ", "_")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax
        )
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("True Class", fontsize=10)
        plt.tight_layout()
        cm_path = output_dir / f"confusion_matrix_{clean_name}.png"
        plt.savefig(cm_path, dpi=300)
        plt.close(fig)
        metrics["confusion_matrix_plot"] = str(cm_path)

    return metrics


# ---------------------------------------------------------
# REGRESSION EVALUATION
# ---------------------------------------------------------

def evaluate_regression_model(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str = "Regressor",
    apply_lower_bound: bool = True,
    save_plots: bool = True,
    output_dir: Path = FIGURES_DIR
) -> Dict[str, Any]:
    """
    Evaluates regression predictions against ground truth.
    Primary selection metric: Mean Absolute Error (MAE).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluation is done on the model predictions; presentation lower bound is applied where specified
    y_pred_eval = np.array(y_pred)
    if apply_lower_bound:
        y_pred_eval = np.maximum(y_pred_eval, 0.0)

    y_true_arr = np.array(y_true)

    mae = float(mean_absolute_error(y_true_arr, y_pred_eval))
    rmse = float(root_mean_squared_error(y_true_arr, y_pred_eval))
    r2 = float(r2_score(y_true_arr, y_pred_eval))

    # Protect MAPE from division by zero in true values
    non_zero_mask = y_true_arr > 0
    if np.sum(non_zero_mask) > 0:
        mape = float(
            mean_absolute_percentage_error(
                y_true_arr[non_zero_mask], y_pred_eval[non_zero_mask]
            )
        )
    else:
        mape = 0.0

    residuals = y_true_arr - y_pred_eval

    metrics = {
        "model_name": model_name,
        "mae": round(mae, 2),    # Primary Metric
        "rmse": round(rmse, 2),  # Main Secondary Metric
        "r2": round(r2, 4),
        "mape": round(mape, 4),
        "residual_mean": round(float(np.mean(residuals)), 2),
        "residual_std": round(float(np.std(residuals)), 2),
    }

    if save_plots:
        clean_name = model_name.lower().replace(" ", "_")

        # 1. Actual vs Predicted Plot (Sample 2000 points if large)
        sample_indices = np.random.choice(
            len(y_true_arr), size=min(2000, len(y_true_arr)), replace=False
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            y_true_arr[sample_indices],
            y_pred_eval[sample_indices],
            alpha=0.3,
            color="#00A896",
            edgecolors="none"
        )
        max_val = max(np.max(y_true_arr[sample_indices]), np.max(y_pred_eval[sample_indices]))
        ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Ideal 1:1 Line")
        ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual Maximum EMI (₹)", fontsize=10)
        ax.set_ylabel("Predicted Maximum EMI (₹)", fontsize=10)
        ax.legend()
        plt.tight_layout()
        actual_pred_path = output_dir / f"actual_vs_predicted_{clean_name}.png"
        plt.savefig(actual_pred_path, dpi=300)
        plt.close(fig)
        metrics["actual_vs_predicted_plot"] = str(actual_pred_path)

        # 2. Residual Distribution Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.histplot(residuals[sample_indices], kde=True, color="#028090", ax=ax)
        ax.axvline(0, color="red", linestyle="--", lw=1.5)
        ax.set_title(f"Residual Distribution — {model_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Residual (Actual - Predicted) (₹)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        plt.tight_layout()
        residual_path = output_dir / f"residuals_{clean_name}.png"
        plt.savefig(residual_path, dpi=300)
        plt.close(fig)
        metrics["residual_plot"] = str(residual_path)

    return metrics
