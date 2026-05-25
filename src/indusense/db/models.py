"""SQLAlchemy models for Bronze, Silver and Gold layers."""

from __future__ import annotations

import enum
import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Identity,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from indusense.db.base import Base, TimestampMixin


INCIDENT_TYPE_COLS: list[str] = [
    "type_surchauffe",
    "type_baisse_pression",
    "type_vibration",
    "type_bruit_mecanique",
    "type_surconsommation",
    "type_blocage_mecanique",
    "type_alarme_capteur",
    "type_arret_urgence",
    "type_defaut_qualite",
]


def _pg_enum(enum_cls: type, *, name: str) -> Enum:
    # SQLAlchemy defaults to member names (RUNNING, TRAIN…) for native enums;
    # values_callable forces it to use the Python values (running, train…)
    # which match what the Alembic migration stored in PostgreSQL.
    return Enum(enum_cls, name=name, values_callable=lambda x: [e.value for e in x])


class IngestionStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class DataQualitySeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class SensorType(str, enum.Enum):
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"


class SplitSet(str, enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    HOLDOUT = "holdout"


class IngestionBatch(TimestampMixin, Base):
    __tablename__ = "ingestion_batch"

    ingestion_batch_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    source_name: Mapped[str] = mapped_column(String(64), nullable=False)
    source_file: Mapped[str] = mapped_column(String(255), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rows_read: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_loaded: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_rejected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[IngestionStatus] = mapped_column(
        _pg_enum(IngestionStatus, name="ingestion_status"),
        nullable=False,
        default=IngestionStatus.PENDING,
    )


class Machine(TimestampMixin, Base):
    __tablename__ = "machine"
    __table_args__ = (
        CheckConstraint("machine_code ~ '^MACH-[0-9]{2}$'", name="machine_code_format"),
        CheckConstraint("max_daily_capacity > 0", name="machine_capacity_positive"),
        CheckConstraint(
            "criticality IN ('LOW', 'MEDIUM', 'HIGH')",
            name="machine_criticality_allowed",
        ),
    )

    machine_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        primary_key=True,
    )
    machine_code: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    commissioning_date: Mapped[date | None] = mapped_column(Date)
    max_daily_capacity: Mapped[int | None] = mapped_column(Integer)
    model: Mapped[str | None] = mapped_column(String(32))
    production_line: Mapped[str | None] = mapped_column(String(16), index=True)
    location: Mapped[str | None] = mapped_column(String(16), index=True)
    criticality: Mapped[str | None] = mapped_column(String(8))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    sensor_readings: Mapped[list["SilverSensorReading"]] = relationship(back_populates="machine")
    incidents: Mapped[list["SilverIncident"]] = relationship(back_populates="machine")
    hourly_features: Mapped[list["GoldMachineHourlyFeature"]] = relationship(back_populates="machine")


class Operator(TimestampMixin, Base):
    __tablename__ = "operator"

    operator_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        primary_key=True,
    )
    operator_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    badge_hash: Mapped[str | None] = mapped_column(String(128))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    incidents: Mapped[list["SilverIncident"]] = relationship(back_populates="operator")


class BronzeTemperatureRaw(TimestampMixin, Base):
    __tablename__ = "bronze_temperature_raw"

    temperature_raw_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        primary_key=True,
    )
    ingestion_batch_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("ingestion_batch.ingestion_batch_id", ondelete="CASCADE"),
        nullable=False,
    )
    row_number: Mapped[int] = mapped_column(Integer, nullable=False)
    machine_id_raw: Mapped[str | None] = mapped_column(String(64))
    timestamp_raw: Mapped[str | None] = mapped_column(String(128))
    temperature_raw: Mapped[str | None] = mapped_column(String(64))
    parse_ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rejected_reason: Mapped[str | None] = mapped_column(Text)


class BronzePressureRaw(TimestampMixin, Base):
    __tablename__ = "bronze_pressure_raw"

    pressure_raw_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        primary_key=True,
    )
    ingestion_batch_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("ingestion_batch.ingestion_batch_id", ondelete="CASCADE"),
        nullable=False,
    )
    row_number: Mapped[int] = mapped_column(Integer, nullable=False)
    machine_id_raw: Mapped[str | None] = mapped_column(String(64))
    timestamp_raw: Mapped[str | None] = mapped_column(String(128))
    pressure_raw: Mapped[str | None] = mapped_column(String(64))
    parse_ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rejected_reason: Mapped[str | None] = mapped_column(Text)


class BronzeIncidentRaw(TimestampMixin, Base):
    __tablename__ = "bronze_incident_raw"

    incident_raw_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        primary_key=True,
    )
    ingestion_batch_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("ingestion_batch.ingestion_batch_id", ondelete="CASCADE"),
        nullable=False,
    )
    row_number: Mapped[int] = mapped_column(Integer, nullable=False)
    incident_code_raw: Mapped[str | None] = mapped_column(String(64))
    machine_id_raw: Mapped[str | None] = mapped_column(String(64))
    operator_name_raw: Mapped[str | None] = mapped_column(String(128))
    operator_badge_raw: Mapped[str | None] = mapped_column(String(64))
    occurred_at_raw: Mapped[str | None] = mapped_column(String(128))
    severity_raw: Mapped[str | None] = mapped_column(String(16))
    shift_raw: Mapped[str | None] = mapped_column(String(32))
    comment_raw: Mapped[str | None] = mapped_column(Text)
    type_surchauffe: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_baisse_pression: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_vibration: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_bruit_mecanique: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_surconsommation: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_blocage_mecanique: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_alarme_capteur: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_arret_urgence: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_defaut_qualite: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    parse_ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rejected_reason: Mapped[str | None] = mapped_column(Text)


class BronzeWeatherRaw(TimestampMixin, Base):
    __tablename__ = "bronze_weather_raw"

    weather_raw_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        primary_key=True,
    )
    ingestion_batch_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("ingestion_batch.ingestion_batch_id", ondelete="CASCADE"),
        nullable=False,
    )
    row_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_raw: Mapped[str | None] = mapped_column(String(128))
    temp_raw: Mapped[str | None] = mapped_column(String(32))
    humidity_raw: Mapped[str | None] = mapped_column(String(32))
    pressure_raw: Mapped[str | None] = mapped_column(String(32))
    wind_speed_raw: Mapped[str | None] = mapped_column(String(32))
    is_imputed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    parse_ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rejected_reason: Mapped[str | None] = mapped_column(Text)


class DataQualityIssue(TimestampMixin, Base):
    __tablename__ = "data_quality_issue"

    dq_issue_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        primary_key=True,
    )
    ingestion_batch_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("ingestion_batch.ingestion_batch_id", ondelete="CASCADE"),
        nullable=False,
    )
    dataset_name: Mapped[str] = mapped_column(String(64), nullable=False)
    rule_code: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[DataQualitySeverity] = mapped_column(
        _pg_enum(DataQualitySeverity, name="data_quality_severity"),
        nullable=False,
    )
    entity_key: Mapped[str | None] = mapped_column(String(128))
    details: Mapped[str | None] = mapped_column(Text)


class SilverSensorReading(TimestampMixin, Base):
    __tablename__ = "silver_sensor_reading"
    __table_args__ = (
        UniqueConstraint(
            "machine_id",
            "observed_at",
            "sensor_type",
            name="silver_sensor_observation",
        ),
    )

    sensor_reading_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        primary_key=True,
    )
    machine_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("machine.machine_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    sensor_type: Mapped[SensorType] = mapped_column(
        _pg_enum(SensorType, name="sensor_type"),
        nullable=False,
    )
    sensor_value: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    unit: Mapped[str] = mapped_column(String(16), nullable=False)
    is_missing: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_duplicate: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_outlier: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    ingestion_batch_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("ingestion_batch.ingestion_batch_id", ondelete="RESTRICT"),
        nullable=False,
    )

    machine: Mapped["Machine"] = relationship(back_populates="sensor_readings")


class SilverIncident(TimestampMixin, Base):
    __tablename__ = "silver_incident"
    __table_args__ = (
        CheckConstraint("severity BETWEEN 1 AND 5", name="silver_incident_severity_range"),
    )

    incident_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        primary_key=True,
    )
    incident_code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    machine_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("machine.machine_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    operator_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("operator.operator_id", ondelete="SET NULL"),
    )
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    severity: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    shift: Mapped[str | None] = mapped_column(String(32))
    comment: Mapped[str | None] = mapped_column(Text)
    is_label_event: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    type_surchauffe: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_baisse_pression: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_vibration: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_bruit_mecanique: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_surconsommation: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_blocage_mecanique: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_alarme_capteur: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_arret_urgence: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    type_defaut_qualite: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    ingestion_batch_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("ingestion_batch.ingestion_batch_id", ondelete="RESTRICT"),
        nullable=False,
    )

    machine: Mapped["Machine"] = relationship(back_populates="incidents")
    operator: Mapped["Operator"] = relationship(back_populates="incidents")


class SilverWeatherReading(TimestampMixin, Base):
    __tablename__ = "silver_weather_reading"
    __table_args__ = (
        UniqueConstraint("observed_at", name="silver_weather_observed_at_unique"),
    )

    weather_reading_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        primary_key=True,
    )
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    temp_celsius: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    humidity_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    pressure_hpa: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    wind_speed_ms: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    is_imputed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    ingestion_batch_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("ingestion_batch.ingestion_batch_id", ondelete="RESTRICT"),
        nullable=False,
    )


class GoldMachineHourlyFeature(TimestampMixin, Base):
    __tablename__ = "gold_machine_hourly_feature"
    __table_args__ = (
        UniqueConstraint("machine_id", "window_start", name="gold_machine_window_start"),
        CheckConstraint("window_end > window_start", name="gold_window_order"),
    )

    feature_row_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        primary_key=True,
    )
    machine_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("machine.machine_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # 6-hour rolling features (short-term spikes)
    temp_mean_6h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    temp_max_6h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    temp_std_6h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    pressure_mean_6h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    pressure_max_6h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    pressure_std_6h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    # 12-hour rolling features (medium-term drift)
    temp_mean_12h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    temp_max_12h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    temp_std_12h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    pressure_mean_12h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    pressure_max_12h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    pressure_std_12h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    # 24-hour rolling features (daily pattern)
    temp_mean_24h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    temp_max_24h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    temp_std_24h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    pressure_mean_24h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    pressure_max_24h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    pressure_std_24h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    # Trend and anomaly features
    temp_trend_6h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    pressure_trend_6h: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    temp_zscore_24h: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    # Incident lookback features
    incident_count_prev_24h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    incident_max_severity_prev_24h: Mapped[int | None] = mapped_column(SmallInteger)
    incident_count_prev_7d: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    hours_since_last_incident: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    # Incident type rolling counts (24h lookback per type)
    type_surchauffe_count_prev_24h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    type_baisse_pression_count_prev_24h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    type_vibration_count_prev_24h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    type_bruit_mecanique_count_prev_24h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    type_surconsommation_count_prev_24h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    type_blocage_mecanique_count_prev_24h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    type_alarme_capteur_count_prev_24h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    type_arret_urgence_count_prev_24h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    type_defaut_qualite_count_prev_24h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Ambient weather features (joined from SilverWeatherReading by hour)
    ambient_temp_c: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    ambient_humidity_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    ambient_pressure_hpa: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    # Multi-horizon failure labels
    label_failure_next_6h: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    label_failure_next_12h: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    label_failure_next_24h: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    label_failure_next_48h: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    split_set: Mapped[SplitSet] = mapped_column(
        _pg_enum(SplitSet, name="split_set"),
        nullable=False,
        default=SplitSet.TRAIN,
    )

    machine: Mapped["Machine"] = relationship(back_populates="hourly_features")
