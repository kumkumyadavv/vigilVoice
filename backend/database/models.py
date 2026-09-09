from datetime import datetime
import hashlib
import json
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.db import Base


class CallAnalysis(Base):

    __tablename__ = "call_analyses"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    filename: Mapped[str] = mapped_column(
        String(255)
    )

    duration: Mapped[float] = mapped_column(
        Float
    )

    sample_rate: Mapped[int] = mapped_column(
        Integer
    )

    spoof_probability: Mapped[float] = mapped_column(
        Float
    )

    real_probability: Mapped[float] = mapped_column(
        Float
    )

    voice_risk: Mapped[float] = mapped_column(
        Float
    )

    fraud_probability: Mapped[float] = mapped_column(
        Float
    )

    fraud_risk: Mapped[float] = mapped_column(
        Float
    )

    fraud_type: Mapped[str] = mapped_column(
        String(100)
    )

    transcription: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    language: Mapped[str] = mapped_column(
        String(20),
        default=""
    )

    combined_risk: Mapped[float] = mapped_column(
        Float
    )

    risk_level: Mapped[str] = mapped_column(
        String(20)
    )

    action: Mapped[str] = mapped_column(
        String(100)
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )

    suspicious_segments = relationship(
        "SuspiciousSegment",
        back_populates="analysis",
        cascade="all, delete-orphan"
    )


class SuspiciousSegment(Base):

    __tablename__ = "suspicious_segments"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("call_analyses.id"),
        nullable=False,
        index=True
    )

    start: Mapped[float] = mapped_column(
        Float
    )

    end: Mapped[float] = mapped_column(
        Float
    )

    spoof_probability: Mapped[float] = mapped_column(
        Float
    )

    voice_risk: Mapped[float] = mapped_column(
        Float
    )

    analysis = relationship(
        "CallAnalysis",
        back_populates="suspicious_segments"
    )


class AuditLog(Base):

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("call_analyses.id"),
        nullable=False,
        index=True
    )

    event_type: Mapped[str] = mapped_column(
        String(100)
    )

    data_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False
    )

    previous_hash: Mapped[str] = mapped_column(
        String(64),
        default=""
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )