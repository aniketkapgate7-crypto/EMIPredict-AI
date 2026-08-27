"""
Data preparation and partitioning module for EMIPredict AI.
Generates an anonymized, representative sample for Streamlit and creates
stratified 70/15/15 train/val/test partitions.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_SAMPLE_PATH,
    RANDOM_STATE,
    TARGET_CLASSIFICATION,
    TEST_SIZE,
    VAL_SIZE,
)
from src.data.load_data import load_raw_dataset
from src.logging_config import setup_logger
from src.utils.artifacts import save_json_metadata

logger = setup_logger(__name__)


def prepare_sample_and_splits(
    df: pd.DataFrame | None = None,
    sample_size: int = 5000,
    save_parquet: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into 70% Train, 15% Validation, 15% Test (stratified by target),
    and creates a compact sample for rapid EDA in Streamlit.
    """
    if df is None:
        df = load_raw_dataset(downcast=True)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DATA_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Total dataset records: {len(df):,}")

    # 1. Generate stratified sample for Streamlit EDA
    stratify_col = df[TARGET_CLASSIFICATION] if TARGET_CLASSIFICATION in df.columns else None
    if stratify_col is not None and len(df) > sample_size:
        sample_df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=stratify_col,
            random_state=RANDOM_STATE
        )
    else:
        sample_df = df.sample(n=min(len(df), sample_size), random_state=RANDOM_STATE)

    sample_df.to_csv(DATA_SAMPLE_PATH, index=False)
    logger.info(f"Saved representative sample ({len(sample_df)} rows) to {DATA_SAMPLE_PATH}")

    # 2. Train / Temp split (70% train, 30% temp)
    temp_size = VAL_SIZE + TEST_SIZE
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        stratify=stratify_col,
        random_state=RANDOM_STATE
    )

    # 3. Validation / Test split (15% val, 15% test from total)
    temp_stratify = temp_df[TARGET_CLASSIFICATION] if TARGET_CLASSIFICATION in temp_df.columns else None
    val_ratio_of_temp = VAL_SIZE / temp_size  # 0.15 / 0.30 = 0.50

    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - val_ratio_of_temp,
        stratify=temp_stratify,
        random_state=RANDOM_STATE
    )

    logger.info(f"Split completed: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    if save_parquet:
        train_path = DATA_PROCESSED_DIR / "train.parquet"
        val_path = DATA_PROCESSED_DIR / "val.parquet"
        test_path = DATA_PROCESSED_DIR / "test.parquet"

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)
        logger.info(f"Saved processed partitions to {DATA_PROCESSED_DIR}")

    split_metadata = {
        "total_records": len(df),
        "train_records": len(train_df),
        "train_percentage": round(len(train_df) / len(df) * 100, 2),
        "val_records": len(val_df),
        "val_percentage": round(len(val_df) / len(df) * 100, 2),
        "test_records": len(test_df),
        "test_percentage": round(len(test_df) / len(df) * 100, 2),
        "stratified_by": TARGET_CLASSIFICATION,
        "random_state": RANDOM_STATE,
        "sample_records": len(sample_df),
    }
    save_json_metadata(split_metadata, DATA_PROCESSED_DIR / "split_metadata.json")

    return train_df, val_df, test_df, sample_df


if __name__ == "__main__":
    prepare_sample_and_splits()
