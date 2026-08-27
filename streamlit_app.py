"""
EMIPredict AI - Streamlit Entry Point & Multi-Page Navigation Router.
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root directory to python path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Configure root page layout and metadata
st.set_page_config(
    page_title="EMIPredict AI — Intelligent Financial Risk Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Persistent sidebar branding header
with st.sidebar:
    st.markdown("## 💳 EMIPredict AI")
    st.caption("Intelligent Financial Risk & EMI Affordability Platform")
    st.markdown("---")

# Define all pages for native Streamlit navigation
pages = [
    st.Page("app/streamlit_app.py", title="Home Dashboard", icon="🏠", default=True),
    st.Page("app/pages/1_Prediction.py", title="1_Prediction: Assess Risk & Max EMI", icon="🎯"),
    st.Page("app/pages/2_Data_Insights.py", title="2_Data_Insights: Dataset EDA & Visuals", icon="📊"),
    st.Page("app/pages/3_Model_Performance.py", title="3_Model_Performance: Benchmarks & Eval", icon="📈"),
    st.Page("app/pages/4_Experiment_Tracking.py", title="4_Experiment_Tracking: MLflow Runs", icon="🧪"),
    st.Page("app/pages/5_Applicant_Records.py", title="5_Applicant_Records: Database CRUD", icon="🗄️"),
    st.Page("app/pages/6_About.py", title="6_About: Architecture & Ethics", icon="ℹ️"),
]

pg = st.navigation(pages)

# Persistent sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("🔒 *Responsible AI Underwriting System*")
    st.caption("Version 1.0.0 • Production Ready")

pg.run()

