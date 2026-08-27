"""
Preprocessing pipeline construction for EMIPredict AI.
Constructs scikit-learn ColumnTransformers with imputation, scaling, and one-hot encoding.
Enforces Responsible AI policy by excluding sensitive demographic variables.
"""

from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ENGINEERED_NUMERICAL_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERICAL_FEATURES,
    SENSITIVE_ATTRIBUTES,
)
from src.features.build_features import FinancialFeatureEngineer
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def get_model_feature_lists() -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (numerical_features, categorical_features, excluded_sensitive_features).
    Guarantees sensitive attributes (gender, marital_status) are excluded from decision pipelines.
    """
    num_features = list(RAW_NUMERICAL_FEATURES) + list(ENGINEERED_NUMERICAL_FEATURES)

    # Filter out sensitive attributes if present
    cat_features = [
        col for col in RAW_CATEGORICAL_FEATURES
        if col not in SENSITIVE_ATTRIBUTES
    ]

    logger.info(
        f"Feature configuration: {len(num_features)} numerical features, "
        f"{len(cat_features)} categorical features. "
        f"Excluded sensitive attributes for fairness: {SENSITIVE_ATTRIBUTES}"
    )
    return num_features, cat_features, SENSITIVE_ATTRIBUTES


def create_preprocessor(include_feature_engineering: bool = True) -> Pipeline | ColumnTransformer:
    """
    Builds the complete scikit-learn preprocessing ColumnTransformer or Pipeline.
    """
    num_cols, cat_cols, _ = get_model_feature_lists()

    # Numerical Transformer: Median Imputation + Standard Scaling
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical Transformer: Most Frequent Imputation + One-Hot Encoding
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",  # Drops unmodeled columns and sensitive demographic variables
        verbose_feature_names_out=False
    )

    if include_feature_engineering:
        full_pipeline = Pipeline(steps=[
            ("feature_engineer", FinancialFeatureEngineer()),
            ("preprocessor", col_transformer),
        ])
        return full_pipeline

    return col_transformer


def get_feature_names(transformer: ColumnTransformer) -> List[str]:
    """Extracts output feature names from a fitted ColumnTransformer."""
    try:
        return list(transformer.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction
        names = []
        for name, pipe, cols in transformer.transformers_:
            if name == "remainder" or name == "drop":
                continue
            if hasattr(pipe, "named_steps") and "ohe" in pipe.named_steps:
                ohe = pipe.named_steps["ohe"]
                if hasattr(ohe, "get_feature_names_out"):
                    names.extend(ohe.get_feature_names_out(cols))
                else:
                    names.extend(cols)
            else:
                names.extend(cols)
        return names
