"""
Data loading module for EMIPredict AI.
Loads dataset with memory optimizations, schema normalization, and string anomaly corrections.
"""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DATA_RAW_PATH, RAW_NUMERICAL_FEATURES, TARGET_REGRESSION
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def clean_numeric_string(val: str) -> str:
    """Fixes formatting artifacts like '64300.0.0' or '23400.0.0.0' into standard decimal strings."""
    if not isinstance(val, str):
        return val
    val = val.strip()
    # Match patterns like 123.0.0 or 123.4.0.0
    val = re.sub(r"(\.\d+)(\.0)+$", r"\1", val)
    return val


def sanitize_numerical_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Identifies and sanitizes numerical columns that contain string formatting anomalies.
    Returns cleaned dataframe and a dictionary of corrections made.
    """
    df_clean = df.copy()
    corrections = {}
    target_num_cols = list(RAW_NUMERICAL_FEATURES) + [TARGET_REGRESSION]

    for col in target_num_cols:
        if col in df_clean.columns and df_clean[col].dtype == "object":
            s = df_clean[col].astype(str)
            # Detect multi-dot artifacts
            mask_artifact = s.str.contains(r"\.\d+\.", regex=True)
            num_artifacts = int(mask_artifact.sum())
            if num_artifacts > 0:
                df_clean[col] = s.apply(clean_numeric_string)
                corrections[col] = num_artifacts
                logger.info(f"Corrected {num_artifacts} formatting artifacts in numerical column '{col}'")

            # Convert to numeric float/int
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    return df_clean, corrections


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcasts float64 and int64 columns to float32/int32 to reduce memory footprint."""
    df_opt = df.copy()
    start_mem = df_opt.memory_usage(deep=True).sum() / (1024 * 1024)

    for col in df_opt.columns:
        col_type = df_opt[col].dtype
        if np.issubdtype(col_type, np.floating):
            df_opt[col] = pd.to_numeric(df_opt[col], downcast="float")
        elif np.issubdtype(col_type, np.integer):
            df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")
        elif col_type == "object":
            num_unique = df_opt[col].nunique()
            num_total = len(df_opt[col])
            if num_unique / num_total < 0.2:
                df_opt[col] = df_opt[col].astype("category")

    end_mem = df_opt.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(
        f"Memory optimization: {start_mem:.2f} MB down to {end_mem:.2f} MB "
        f"({((start_mem - end_mem) / start_mem) * 100:.1f}% reduction)"
    )
    return df_opt


def load_raw_dataset(
    file_path: Optional[Path | str] = None,
    downcast: bool = True,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Loads the raw EMI dataset from CSV, validates existence, normalizes column names,
    fixes known numerical string artifacts, and optionally downcasts dtypes.
    """
    path = Path(file_path) if file_path else DATA_RAW_PATH
    if not path.exists():
        msg = f"Dataset not found at '{path}'. Please ensure EMI_dataset.csv is placed in data/raw/."
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Loading raw dataset from {path}...")
    df = pd.read_csv(path, nrows=sample_size, low_memory=False)

    # Normalize column names: strip whitespace, convert to lowercase snake_case
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Sanitize known numerical columns containing string formatting artifacts
    df, corrections = sanitize_numerical_columns(df)

    logger.info(f"Loaded dataset successfully with shape: {df.shape}")
    if downcast:
        df = downcast_dtypes(df)

    return df


if __name__ == "__main__":
    df = load_raw_dataset()
    print("Dataset Loaded Successfully:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")
