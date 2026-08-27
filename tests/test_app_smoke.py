"""
Smoke tests for Streamlit application components, pages, and module imports.
"""

import importlib


def test_app_imports():
    """Verify that root application and all core modules import cleanly."""
    modules = [
        "src.config",
        "src.logging_config",
        "src.data.load_data",
        "src.data.validate_data",
        "src.data.prepare_data",
        "src.features.build_features",
        "src.features.preprocessing",
        "src.models.evaluate",
        "src.models.predict",
        "src.models.select_model",
        "src.database.models",
        "src.database.crud",
        "app.components.input_form",
        "app.components.metrics",
        "app.components.charts",
    ]

    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"Failed to import {mod_name}"


def test_streamlit_pages_exist():
    """Verify that all Streamlit page scripts exist on disk."""
    from pathlib import Path

    page_files = [
        "streamlit_app.py",
        "app/streamlit_app.py",
        "app/pages/1_Prediction.py",
        "app/pages/2_Data_Insights.py",
        "app/pages/3_Model_Performance.py",
        "app/pages/4_Experiment_Tracking.py",
        "app/pages/5_Applicant_Records.py",
        "app/pages/6_About.py",
    ]

    for pf in page_files:
        p = Path(pf)
        assert p.exists(), f"Missing page file: {pf}"

