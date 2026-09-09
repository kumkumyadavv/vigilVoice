import hashlib
import json

from sqlalchemy.orm import Session

from models import AuditLog


def create_audit_hash(
    data: dict,
    previous_hash: str = ""
) -> str:

    payload = {
        "data": data,
        "previous_hash": previous_hash
    }

    serialized = json.dumps(
        payload,
        sort_keys=True
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def save_audit_log(
    db: Session,
    analysis_id: int,
    data: dict
):

    previous_log = (
        db.query(AuditLog)
        .order_by(AuditLog.id.desc())
        .first()
    )

    previous_hash = (
        previous_log.data_hash
        if previous_log
        else ""
    )

    current_hash = create_audit_hash(
        data,
        previous_hash
    )

    audit = AuditLog(
        analysis_id=analysis_id,
        event_type="CALL_ANALYSIS",
        data_hash=current_hash,
        previous_hash=previous_hash
    )

    db.add(audit)
    db.commit()

    return audit