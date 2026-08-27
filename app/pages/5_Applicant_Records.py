"""
Applicant Records Database Management (CRUD) Page for EMIPredict AI Streamlit Application.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Setup path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.database.crud import (
    delete_applicant_record,
    get_applicant_record,
    list_applicant_records,
    update_applicant_record,
)

try:
    st.set_page_config(page_title="Applicant Records — EMIPredict AI", page_icon="🗄️", layout="wide")
except Exception:
    pass

st.markdown("# 🗄️ Applicant Underwriting Records & Database CRUD")
st.caption("Centralized repository of applicant profiles, model predictions, underwriting statuses, and notes.")
st.markdown("---")

st.warning(
    "⚠️ **Storage Notice**: In local development, records are saved in SQLite (`database/applicants.db`). "
    "When deployed to Streamlit Cloud, SQLite persistence is ephemeral across container restarts. "
    "Set `DATABASE_URL` in environment secrets to point to a persistent PostgreSQL instance for enterprise production."
)

# Filter Controls
fcol1, fcol2, fcol3 = st.columns([2, 2, 3])
with fcol1:
    status_filter = st.selectbox(
        "Filter by Review Status:",
        ["All", "Pending Review", "Approved", "Flagged for Review", "Rejected"],
        index=0
    )
with fcol2:
    elig_filter = st.selectbox(
        "Filter by Model Eligibility:",
        ["All", "Eligible", "High_Risk", "Not_Eligible"],
        index=0
    )
with fcol3:
    search_q = st.text_input("🔍 Search Identifier / Loan Purpose / Notes:", "")

records = list_applicant_records(
    limit=200,
    status_filter=status_filter if status_filter != "All" else None,
    eligibility_filter=elig_filter if elig_filter != "All" else None,
    search_query=search_q if search_q.strip() else None
)

st.markdown(f"### 📋 Stored Applications ({len(records)} found)")

if records:
    # Summary Table View
    summary_list = []
    for r in records:
        summary_list.append({
            "ID": r["id"],
            "Identifier": r["applicant_identifier"],
            "Loan Purpose": r["emi_scenario"],
            "Salary (₹)": f"₹{r['monthly_salary']:,.0f}",
            "Requested (₹)": f"₹{r['requested_amount']:,.0f}",
            "Eligibility": r["predicted_eligibility"],
            "Max EMI (₹)": f"₹{r['predicted_max_emi']:,.0f}" if r["predicted_max_emi"] else "N/A",
            "Status": r["status"],
            "Created": r["created_at"],
        })
    df_recs = pd.DataFrame(summary_list)
    st.dataframe(df_recs, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Record Inspector & CRUD actions
    st.markdown("### ✏️ Manage Selected Application")
    record_ids = [r["id"] for r in records]
    selected_id = st.selectbox(
        "Select Application ID to Review / Update / Delete:",
        record_ids,
        format_func=lambda x: f"Record #{x} — {[r['applicant_identifier'] for r in records if r['id'] == x][0]}"
    )

    if selected_id:
        target_rec = get_applicant_record(selected_id)
        if target_rec:
            dcol1, dcol2 = st.columns([3, 2])

            with dcol1:
                st.markdown(f"#### 👤 Financial Profile — **{target_rec['applicant_identifier']}**")
                p1, p2, p3 = st.columns(3)
                p1.write(f"**Age**: {target_rec['age']} yrs")
                p1.write(f"**Education**: {target_rec['education']}")
                p1.write(f"**Employment**: {target_rec['employment_type']}")

                p2.write(f"**Salary**: ₹{target_rec['monthly_salary']:,.2f}")
                p2.write(f"**Credit Score**: {target_rec['credit_score']}")
                p2.write(f"**Company**: {target_rec['company_type']}")

                p3.write(f"**House**: {target_rec['house_type']}")
                p3.write(f"**Bank Balance**: ₹{target_rec['bank_balance']:,.2f}")
                p3.write(f"**Emergency Fund**: ₹{target_rec['emergency_fund']:,.2f}")

                st.markdown("##### 💳 Loan Details & Model Output")
                l1, l2 = st.columns(2)
                l1.write(f"**Loan Scenario**: {target_rec['emi_scenario']}")
                l1.write(f"**Requested Amount**: ₹{target_rec['requested_amount']:,.2f}")
                l1.write(f"**Requested Tenure**: {target_rec['requested_tenure']} mo")

                l2.write(f"**Predicted Eligibility**: `{target_rec['predicted_eligibility']}`")
                l2.write(f"**Max Monthly EMI**: ₹{target_rec['predicted_max_emi']:,.2f}" if target_rec['predicted_max_emi'] else "N/A")
                l2.write(f"**Model Version**: {target_rec['model_version']}")

            with dcol2:
                st.markdown("#### ⚙️ Underwriter Action Panel")

                new_status = st.selectbox(
                    "Update Underwriting Status:",
                    ["Pending Review", "Approved", "Flagged for Review", "Rejected"],
                    index=["Pending Review", "Approved", "Flagged for Review", "Rejected"].index(
                        target_rec["status"]
                    ) if target_rec["status"] in ["Pending Review", "Approved", "Flagged for Review", "Rejected"] else 0
                )
                new_notes = st.text_area("Underwriter Notes:", value=target_rec["notes"] or "", height=100)

                ucol1, ucol2 = st.columns(2)
                with ucol1:
                    if st.button("💾 Save Changes", type="primary", use_container_width=True):
                        updated = update_applicant_record(
                            selected_id,
                            {"status": new_status, "notes": new_notes}
                        )
                        st.success("✅ Application record updated!")
                        st.rerun()

                with ucol2:
                    confirm_delete = st.checkbox("Confirm Deletion", key=f"del_confirm_{selected_id}")
                    if confirm_delete:
                        if st.button("🗑️ Delete Record", type="secondary", use_container_width=True):
                            delete_applicant_record(selected_id)
                            st.warning(f"Record #{selected_id} deleted.")
                            st.rerun()
else:
    st.info("No applicant records found in database. Use the Prediction page to create and save assessments.")
