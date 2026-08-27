
"""
Artifact management utilities for EMIPredict AI.
Handles serializing/deserializing pipelines with joblib, saving schemas and metrics.
"""

import json
from pathlib import Path
from typing import Any, Dict

import joblib

from src.logging_config import setup_logger

logger = setup_logger(__name__)


def save_joblib_artifact(obj: Any, file_path: Path | str, compress: int = 3) -> None:
    """Save an object using compressed joblib."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=compress)
    logger.info(f"Saved artifact to {path}")


def load_joblib_artifact(file_path: Path | str) -> Any:
    """Load an artifact saved with joblib."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {path}")
    return joblib.load(path)


def save_json_metadata(data: Dict[str, Any], file_path: Path | str) -> None:
    """Save dictionary metadata to formatted JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved metadata JSON to {path}")


def load_json_metadata(file_path: Path | str) -> Dict[str, Any]:
    """Load JSON metadata file."""
    path = Path(file_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
