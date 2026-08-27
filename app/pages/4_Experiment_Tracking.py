"""
MLflow Experiment Tracking Page for EMIPredict AI Streamlit Application.
Connects to MLflow tracking server or falls back to static run logs.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Setup path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import MLFLOW_TRACKING_URI, REPORTS_DIR

try:
    st.set_page_config(page_title="Experiment Tracking — EMIPredict AI", page_icon="🧪", layout="wide")
except Exception:
    pass

st.markdown("# 🧪 MLflow Experiment Tracking & MLOps Audit")
st.caption("Centralized experiment runs, hyperparameter logs, and model lineage recorded during model training.")
st.markdown("---")


def get_mlflow_runs():
    """Attempts to load runs directly from MLflow tracking backend."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        experiments = client.search_experiments()

        all_runs = []
        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            for r in runs:
                run_dict = {
                    "Experiment": exp.name,
                    "Run Name": r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
                    "Status": r.info.status,
                    "Start Time": pd.to_datetime(r.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S") if r.info.start_time else "N/A",
                }
                # Add metrics
                for k, v in r.data.metrics.items():
                    run_dict[f"Metric: {k}"] = round(v, 4) if isinstance(v, float) else v
                # Add params
                for k, v in r.data.params.items():
                    run_dict[f"Param: {k}"] = v

                all_runs.append(run_dict)

        if all_runs:
            return pd.DataFrame(all_runs), True, None
        return None, True, "No runs found in active tracking URI."
    except Exception as e:
        return None, False, str(e)


# Status Header
st.markdown("### 📡 MLflow Backend Status")
st.code(f"MLFLOW_TRACKING_URI = {MLFLOW_TRACKING_URI}", language="bash")

runs_df, is_connected, err_msg = get_mlflow_runs()

if is_connected and runs_df is not None and len(runs_df) > 0:
    st.success("✅ Connected to MLflow Tracking Database. Real-time run logs loaded.")
    st.dataframe(runs_df, use_container_width=True, hide_index=True)
else:
    if err_msg:
        st.warning(f"ℹ️ MLflow Server Offline / Standalone Mode: {err_msg}")

    # Fallback to static summary
    static_summary_path = REPORTS_DIR / "mlflow_run_summary.csv"
    if static_summary_path.exists():
        st.markdown("#### 📄 Static Exported Experiment Summary")
        df_static = pd.read_csv(static_summary_path)
        st.dataframe(df_static, use_container_width=True, hide_index=True)
    else:
        st.info("No static experiment summary found. Train models to populate MLflow runs.")

st.markdown("---")

st.markdown("### 💻 How to Launch Local MLflow UI")
st.markdown(
    """
    To launch the full interactive MLflow UI on your local Windows workstation, execute the following command in PowerShell:
    """
)
st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000", language="powershell")
st.caption("Then navigate to http://127.0.0.1:5000 in your browser to inspect model artifacts, parameters, and comparison charts.")
