"""Bronze layer: CSV/TSV files → BronzeRaw tables with IngestionBatch tracking."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa
from loguru import logger
from pydantic import ValidationError
from sqlalchemy.orm import Session

from indusense.db.models import (
    BronzeIncidentRaw,
    BronzePressureRaw,
    BronzeTemperatureRaw,
    BronzeWeatherRaw,
    INCIDENT_TYPE_COLS,
    IngestionBatch,
    IngestionStatus,
)
from indusense.schemas.ingestion import IncidentInput, PressureInput, TemperatureInput


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _create_batch(session: Session, source_name: str, source_file: str) -> IngestionBatch:
    batch = IngestionBatch(
        ingestion_batch_id=uuid.uuid4(),
        source_name=source_name,
        source_file=source_file,
        started_at=_now(),
        status=IngestionStatus.RUNNING,
    )
    session.add(batch)
    session.flush()
    return batch


def _finish_batch(
    session: Session,
    batch: IngestionBatch,
    rows_read: int,
    rows_loaded: int,
    rows_rejected: int,
) -> None:
    batch.finished_at = _now()
    batch.rows_read = rows_read
    batch.rows_loaded = rows_loaded
    batch.rows_rejected = rows_rejected
    batch.status = IngestionStatus.COMPLETED if rows_rejected == 0 else IngestionStatus.PARTIAL
    session.flush()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)


def _bulk_insert(session: Session, model: type, rows: list[dict[str, Any]]) -> None:
    if rows:
        session.execute(sa.insert(model), rows)


def load_temperature_file(session: Session, path: Path) -> IngestionBatch:
    """Parse temperature CSV, validate each row, insert into BronzeTemperatureRaw."""
    batch = _create_batch(session, "temperature", path.name)
    df = _read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    rows: list[dict[str, Any]] = []
    rows_loaded = rows_rejected = 0

    for row_num, row in enumerate(df.itertuples(index=False), start=1):
        machine_id_raw = str(getattr(row, "machine_id", "")).strip()
        timestamp_raw = str(getattr(row, "timestamp", "")).strip()
        temperature_raw = str(getattr(row, "temperature", "")).strip()
        raw: dict[str, Any] = {
            "ingestion_batch_id": batch.ingestion_batch_id,
            "row_number": row_num,
            "machine_id_raw": machine_id_raw or None,
            "timestamp_raw": timestamp_raw or None,
            "temperature_raw": temperature_raw or None,
            "parse_ok": False,
            "rejected_reason": None,
        }
        try:
            TemperatureInput(
                machine_id_raw=machine_id_raw,
                timestamp_raw=timestamp_raw,
                temperature_raw=temperature_raw or None,
            )
            raw["parse_ok"] = True
            rows_loaded += 1
        except (ValidationError, ValueError, Exception) as exc:
            raw["rejected_reason"] = str(exc)[:512]
            rows_rejected += 1
        rows.append(raw)

    _bulk_insert(session, BronzeTemperatureRaw, rows)
    _finish_batch(session, batch, len(df), rows_loaded, rows_rejected)
    logger.info(
        "Bronze temperature loaded: {} ok / {} rejected (batch {})",
        rows_loaded, rows_rejected, batch.ingestion_batch_id,
    )
    return batch


def load_pressure_file(session: Session, path: Path) -> IngestionBatch:
    """Parse pressure TSV/CSV, validate each row, insert into BronzePressureRaw."""
    batch = _create_batch(session, "pressure", path.name)
    df = _read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    pressure_col = "pressure_bar" if "pressure_bar" in df.columns else "pressure"

    rows: list[dict[str, Any]] = []
    rows_loaded = rows_rejected = 0

    for row_num, row in enumerate(df.itertuples(index=False), start=1):
        machine_id_raw = str(getattr(row, "machine_id", "")).strip()
        timestamp_raw = str(getattr(row, "timestamp", "")).strip()
        pressure_raw = str(getattr(row, pressure_col, "")).strip()
        raw: dict[str, Any] = {
            "ingestion_batch_id": batch.ingestion_batch_id,
            "row_number": row_num,
            "machine_id_raw": machine_id_raw or None,
            "timestamp_raw": timestamp_raw or None,
            "pressure_raw": pressure_raw or None,
            "parse_ok": False,
            "rejected_reason": None,
        }
        try:
            PressureInput(
                machine_id_raw=machine_id_raw,
                timestamp_raw=timestamp_raw,
                pressure_raw=pressure_raw or None,
            )
            raw["parse_ok"] = True
            rows_loaded += 1
        except (ValidationError, ValueError, Exception) as exc:
            raw["rejected_reason"] = str(exc)[:512]
            rows_rejected += 1
        rows.append(raw)

    _bulk_insert(session, BronzePressureRaw, rows)
    _finish_batch(session, batch, len(df), rows_loaded, rows_rejected)
    logger.info(
        "Bronze pressure loaded: {} ok / {} rejected (batch {})",
        rows_loaded, rows_rejected, batch.ingestion_batch_id,
    )
    return batch


def load_incidents_file(session: Session, path: Path) -> IngestionBatch:
    """Parse incidents CSV (with type columns), validate each row, insert into BronzeIncidentRaw."""
    batch = _create_batch(session, "incidents", path.name)
    df = _read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    rows: list[dict[str, Any]] = []
    rows_loaded = rows_rejected = 0

    for row_num, row in enumerate(df.itertuples(index=False), start=1):
        incident_code_raw = str(getattr(row, "incident_id", "")).strip()
        machine_id_raw = str(getattr(row, "machine_id", "")).strip()
        date_raw = str(getattr(row, "date", "")).strip()
        time_raw = str(getattr(row, "time", "")).strip()
        severity_raw = str(getattr(row, "severity", "")).strip()
        operator_name_raw = str(getattr(row, "operator_name", "")).strip() or None
        operator_badge_raw = str(getattr(row, "operator_badge", "")).strip() or None
        shift_raw = str(getattr(row, "shift", "")).strip() or None
        comment_raw = str(getattr(row, "comment", "")).strip() or None

        type_values = {
            col: int(str(getattr(row, col, "0")).strip() or "0")
            for col in INCIDENT_TYPE_COLS
        }

        raw: dict[str, Any] = {
            "ingestion_batch_id": batch.ingestion_batch_id,
            "row_number": row_num,
            "incident_code_raw": incident_code_raw or None,
            "machine_id_raw": machine_id_raw or None,
            "operator_name_raw": operator_name_raw,
            "operator_badge_raw": operator_badge_raw,
            "occurred_at_raw": f"{date_raw} {time_raw}" if date_raw else None,
            "severity_raw": severity_raw or None,
            "shift_raw": shift_raw,
            "comment_raw": comment_raw,
            **type_values,
            "parse_ok": False,
            "rejected_reason": None,
        }
        try:
            IncidentInput(
                incident_code_raw=incident_code_raw,
                machine_id_raw=machine_id_raw,
                date_raw=date_raw,
                time_raw=time_raw,
                severity_raw=severity_raw,
                operator_name_raw=operator_name_raw,
                operator_badge_raw=operator_badge_raw,
                shift_raw=shift_raw,
                comment_raw=comment_raw,
                **type_values,
            )
            raw["parse_ok"] = True
            rows_loaded += 1
        except (ValidationError, ValueError, Exception) as exc:
            raw["rejected_reason"] = str(exc)[:512]
            rows_rejected += 1
        rows.append(raw)

    _bulk_insert(session, BronzeIncidentRaw, rows)
    _finish_batch(session, batch, len(df), rows_loaded, rows_rejected)
    logger.info(
        "Bronze incidents loaded: {} ok / {} rejected (batch {})",
        rows_loaded, rows_rejected, batch.ingestion_batch_id,
    )
    return batch


def load_weather_data(
    session: Session,
    start: date,
    end: date,
    api_key: str | None = None,
) -> IngestionBatch:
    """Fetch (or load from cache) hourly weather for [start, end] → BronzeWeatherRaw."""
    from indusense.weather.fetcher import fetch_or_load_weather

    batch = _create_batch(session, "weather", f"openweather_{start}_{end}")
    weather_df = fetch_or_load_weather(start, end, api_key=api_key)

    rows: list[dict[str, Any]] = []
    rows_loaded = rows_rejected = 0

    for row_num, (_, r) in enumerate(weather_df.iterrows(), start=1):
        ts = r.get("observed_at")
        raw: dict[str, Any] = {
            "ingestion_batch_id": batch.ingestion_batch_id,
            "row_number": row_num,
            "timestamp_raw": str(ts) if ts is not None else None,
            "temp_raw": str(r.get("temp_celsius")) if r.get("temp_celsius") is not None else None,
            "humidity_raw": str(r.get("humidity_pct")) if r.get("humidity_pct") is not None else None,
            "pressure_raw": str(r.get("pressure_hpa")) if r.get("pressure_hpa") is not None else None,
            "wind_speed_raw": str(r.get("wind_speed_ms")) if r.get("wind_speed_ms") is not None else None,
            "is_imputed": bool(r.get("is_imputed", False)),
            "parse_ok": True,
            "rejected_reason": None,
        }
        rows.append(raw)
        rows_loaded += 1

    _bulk_insert(session, BronzeWeatherRaw, rows)
    _finish_batch(session, batch, len(weather_df), rows_loaded, rows_rejected)
    logger.info(
        "Bronze weather loaded: {} rows ({} imputed) for {} → {}",
        rows_loaded,
        weather_df["is_imputed"].astype(bool).sum(),
        start, end,
    )
    return batch
