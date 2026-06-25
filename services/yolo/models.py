from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import declarative_base, relationship, mapped_column, Mapped

Base = declarative_base()


class PredictionSession(Base):
    __tablename__ = "prediction_sessions"

    uid: Mapped[str] = mapped_column(String, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    original_image: Mapped[str] = mapped_column(String)
    predicted_image: Mapped[str] = mapped_column(String)

    objects: Mapped[list["DetectionObject"]] = relationship(
        "DetectionObject",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class DetectionObject(Base):
    __tablename__ = "detection_objects"
    __table_args__ = (
        Index("idx_prediction_uid", "prediction_uid"),
        Index("idx_label", "label"),
        Index("idx_score", "score"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_uid: Mapped[str] = mapped_column(
        String,
        ForeignKey("prediction_sessions.uid"),
    )
    label: Mapped[str] = mapped_column(String)
    score: Mapped[float] = mapped_column(Float)
    box: Mapped[str] = mapped_column(String)

    session: Mapped[PredictionSession] = relationship(
        "PredictionSession",
        back_populates="objects",
    )
