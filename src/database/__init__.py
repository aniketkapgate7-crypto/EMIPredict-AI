"""
Database and persistence layer for EMIPredict AI.
"""

from src.database.crud import (
    create_applicant_record,
    delete_applicant_record,
    get_applicant_record,
    get_db_session,
    init_db,
    list_applicant_records,
    update_applicant_record,
)
from src.database.models import ApplicantRecord, Base

__all__ = [
    "ApplicantRecord",
    "Base",
    "init_db",
    "get_db_session",
    "create_applicant_record",
    "get_applicant_record",
    "list_applicant_records",
    "update_applicant_record",
    "delete_applicant_record",
]
