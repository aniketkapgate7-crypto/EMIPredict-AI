"""
CRUD operations and database session management for EMIPredict AI.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from sqlalchemy import create_engine, desc
from sqlalchemy.orm import Session, sessionmaker

from src.config import DATABASE_URL
from src.database.models import ApplicantRecord, Base
from src.logging_config import setup_logger

logger = setup_logger(__name__)

# Engine and session factory
_ENGINE = None
_SESSION_FACTORY = None


def get_engine(db_url: str | None = None):
    """Returns or initializes SQLAlchemy engine."""
    global _ENGINE
    if db_url is None:
        db_url = DATABASE_URL

    if _ENGINE is None or str(_ENGINE.url) != db_url:
        # If SQLite path, ensure directory exists
        if db_url.startswith("sqlite:///"):
            raw_path = db_url.replace("sqlite:///", "")
            if raw_path != ":memory:":
                Path(raw_path).parent.mkdir(parents=True, exist_ok=True)

        _ENGINE = create_engine(
            db_url,
            echo=False,
            connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {}
        )
    return _ENGINE


def init_db(db_url: str | None = None) -> None:
    """Creates database tables if they do not exist."""
    engine = get_engine(db_url)
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialized at {engine.url}")


@contextmanager
def get_db_session(db_url: str | None = None) -> Generator[Session, None, None]:
    """Context manager providing a transactional database session."""
    engine = get_engine(db_url)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session rolled back due to error: {e}")
        raise
    finally:
        session.close()


def create_applicant_record(data: Dict[str, Any], db_url: str | None = None) -> Dict[str, Any]:
    """Inserts a new applicant record and prediction into database."""
    init_db(db_url)

    # Format JSON fields if passed as dicts
    probs_raw = data.get("probabilities", data.get("probabilities_json", {}))
    probs_str = json.dumps(probs_raw) if isinstance(probs_raw, (dict, list)) else str(probs_raw or "")

    ratios_raw = data.get("financial_ratios", data.get("financial_ratios_json", {}))
    ratios_str = json.dumps(ratios_raw) if isinstance(ratios_raw, (dict, list)) else str(ratios_raw or "")

    with get_db_session(db_url) as session:
        record = ApplicantRecord(
            age=int(data.get("age", 25)),
            gender=str(data.get("gender", "Unspecified")),
            marital_status=str(data.get("marital_status", "Unspecified")),
            education=str(data.get("education", "Graduate")),
            monthly_salary=float(data.get("monthly_salary", 0.0)),
            employment_type=str(data.get("employment_type", "Salaried")),
            years_of_employment=float(data.get("years_of_employment", 0.0)),
            company_type=str(data.get("company_type", "Corporate")),
            house_type=str(data.get("house_type", "Rented")),
            monthly_rent=float(data.get("monthly_rent", 0.0)),
            family_size=int(data.get("family_size", 1)),
            dependents=int(data.get("dependents", 0)),
            school_fees=float(data.get("school_fees", 0.0)),
            college_fees=float(data.get("college_fees", 0.0)),
            travel_expenses=float(data.get("travel_expenses", 0.0)),
            groceries_utilities=float(data.get("groceries_utilities", 0.0)),
            other_monthly_expenses=float(data.get("other_monthly_expenses", 0.0)),
            existing_loans=str(data.get("existing_loans", "No")),
            current_emi_amount=float(data.get("current_emi_amount", 0.0)),
            credit_score=float(data.get("credit_score", 650.0)),
            bank_balance=float(data.get("bank_balance", 0.0)),
            emergency_fund=float(data.get("emergency_fund", 0.0)),
            emi_scenario=str(data.get("emi_scenario", "Personal Loan EMI")),
            requested_amount=float(data.get("requested_amount", 100000.0)),
            requested_tenure=int(data.get("requested_tenure", 12)),
            predicted_eligibility=str(data.get("predicted_eligibility", "")),
            predicted_max_emi=float(data.get("max_monthly_emi", data.get("predicted_max_emi", 0.0))),
            probabilities_json=probs_str,
            financial_ratios_json=ratios_str,
            model_version=str(data.get("model_version", "1.0.0")),
            status=str(data.get("status", "Pending Review")),
            notes=str(data.get("notes", "")),
            applicant_identifier=data.get("applicant_identifier"),
        )
        session.add(record)
        session.flush()
        if not record.applicant_identifier:
            record.applicant_identifier = f"APP-{record.id:05d}"
        session.commit()
        session.refresh(record)
        return record.to_dict()


def get_applicant_record(record_id: int, db_url: str | None = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single applicant record by primary key."""
    init_db(db_url)
    with get_db_session(db_url) as session:
        record = session.query(ApplicantRecord).filter(ApplicantRecord.id == record_id).first()
        return record.to_dict() if record else None


def list_applicant_records(
    limit: int = 100,
    offset: int = 0,
    status_filter: Optional[str] = None,
    eligibility_filter: Optional[str] = None,
    search_query: Optional[str] = None,
    db_url: str | None = None
) -> List[Dict[str, Any]]:
    """Lists applicant records with optional filtering and pagination."""
    init_db(db_url)
    with get_db_session(db_url) as session:
        query = session.query(ApplicantRecord)

        if status_filter and status_filter != "All":
            query = query.filter(ApplicantRecord.status == status_filter)
        if eligibility_filter and eligibility_filter != "All":
            query = query.filter(ApplicantRecord.predicted_eligibility == eligibility_filter)
        if search_query:
            query = query.filter(
                (ApplicantRecord.applicant_identifier.ilike(f"%{search_query}%")) |
                (ApplicantRecord.emi_scenario.ilike(f"%{search_query}%")) |
                (ApplicantRecord.notes.ilike(f"%{search_query}%"))
            )

        records = query.order_by(desc(ApplicantRecord.created_at)).offset(offset).limit(limit).all()
        return [r.to_dict() for r in records]


def update_applicant_record(
    record_id: int,
    update_data: Dict[str, Any],
    db_url: str | None = None
) -> Optional[Dict[str, Any]]:
    """Updates status, notes, or fields for an existing applicant record."""
    init_db(db_url)
    with get_db_session(db_url) as session:
        record = session.query(ApplicantRecord).filter(ApplicantRecord.id == record_id).first()
        if not record:
            return None

        for key, value in update_data.items():
            if hasattr(record, key) and key != "id":
                setattr(record, key, value)

        session.commit()
        session.refresh(record)
        return record.to_dict()


def delete_applicant_record(record_id: int, db_url: str | None = None) -> bool:
    """Deletes an applicant record by ID."""
    init_db(db_url)
    with get_db_session(db_url) as session:
        record = session.query(ApplicantRecord).filter(ApplicantRecord.id == record_id).first()
        if not record:
            return False
        session.delete(record)
        session.commit()
        return True
