"""Full ingestion pipeline orchestration: CSV → Bronze → Silver → Gold."""

from __future__ import annotations

import os
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa
from loguru import logger
from sqlalchemy.orm import Session

from indusense.db.models import (
    BronzeIncidentRaw,
    BronzePressureRaw,
    BronzeTemperatureRaw,
    GoldMachineHourlyFeature,
    IngestionBatch,
    SilverIncident,
    SilverSensorReading,
    SilverWeatherReading,
)
from indusense.db.session import SessionLocal
from indusense.pipeline.bronze import (
    load_incidents_file,
    load_pressure_file,
    load_temperature_file,
    load_weather_data,
)
from indusense.pipeline.gold import build_gold
from indusense.pipeline.silver import (
    build_incident_silver,
    build_pressure_silver,
    build_temperature_silver,
    build_weather_silver,
)

_SOURCE_STEMS = {
    "capteurs_temperature": "temperature",
    "capteurs_pression": "pressure",
    "releves_incidents": "incidents",
}


def _find_source_files(data_dir: Path) -> dict[str, Path]:
    """Return {source_type: path} for known source files in data_dir."""
    found: dict[str, Path] = {}
    for path in sorted(data_dir.iterdir()):
        if path.is_file() and not path.name.startswith("."):
            source_type = _SOURCE_STEMS.get(path.stem)
            if source_type:
                found[source_type] = path
    return found


def _detect_weather_date_range(temp_path: Path) -> tuple[date, date]:
    """Read temperature CSV to determine the sensor date range."""
    df = pd.read_csv(temp_path, sep=None, engine="python", dtype=str, keep_default_na=False)
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if ts.empty:
        today = date.today()
        return today, today
    return ts.min().date(), ts.max().date()


def _truncate_facts(session: Session) -> None:
    """Clear all fact tables in FK-safe order (keep Machine and Operator dimensions)."""
    session.execute(sa.delete(GoldMachineHourlyFeature))
    session.execute(sa.delete(SilverSensorReading))
    session.execute(sa.delete(SilverIncident))
    session.execute(sa.delete(SilverWeatherReading))
    # Deleting IngestionBatch cascades to all Bronze tables
    session.execute(sa.delete(IngestionBatch))
    session.flush()
    logger.info("Fact tables truncated (Bronze, Silver, Gold)")


def _create_silver_batch(session: Session) -> uuid.UUID:
    """Create an IngestionBatch record to anchor Silver records' FK constraint."""
    from datetime import datetime, timezone

    from indusense.db.models import IngestionStatus

    batch_id = uuid.uuid4()
    session.add(
        IngestionBatch(
            ingestion_batch_id=batch_id,
            source_name="silver_build",
            source_file="silver_build",
            started_at=datetime.now(tz=timezone.utc),
            finished_at=datetime.now(tz=timezone.utc),
            rows_read=0,
            rows_loaded=0,
            rows_rejected=0,
            status=IngestionStatus.COMPLETED,
        )
    )
    session.flush()
    return batch_id


def run_pipeline(
    data_dir: Path = Path("datas"),
    session_factory: Any = None,
    openweather_api_key: str | None = None,
) -> dict[str, Any]:
    """
    Execute the full Bronze → Silver → Gold pipeline.

    Truncates all fact tables before loading to ensure idempotent full reloads.
    Machine and Operator dimension records are preserved via upsert.

    Weather data is fetched from OpenWeather API (or served from cache at
    datas/weather_cache.csv). Set OPENWEATHER_API_KEY env var or pass
    openweather_api_key to enable live fetching for uncached historical dates.
    Future dates (beyond today) are automatically imputed with monthly means.

    Returns a summary dict with row counts for each stage.
    """
    if session_factory is None:
        session_factory = SessionLocal

    api_key = openweather_api_key or os.environ.get("OPENWEATHER_API_KEY")

    source_files = _find_source_files(data_dir)
    missing = {"temperature", "pressure", "incidents"} - source_files.keys()
    if missing:
        raise FileNotFoundError(
            f"Missing source files for: {sorted(missing)} in {data_dir}"
        )

    weather_start, weather_end = _detect_weather_date_range(source_files["temperature"])

    summary: dict[str, Any] = {}

    with session_factory() as session:
        # ── Bronze ────────────────────────────────────────────────────────────
        logger.info("Step 1/3 — Bronze: loading source files")
        _truncate_facts(session)

        temp_batch = load_temperature_file(session, source_files["temperature"])
        press_batch = load_pressure_file(session, source_files["pressure"])
        inc_batch = load_incidents_file(session, source_files["incidents"])
        weather_batch = load_weather_data(session, weather_start, weather_end, api_key=api_key)
        session.commit()

        summary["bronze_temperature_loaded"] = temp_batch.rows_loaded
        summary["bronze_temperature_rejected"] = temp_batch.rows_rejected
        summary["bronze_pressure_loaded"] = press_batch.rows_loaded
        summary["bronze_pressure_rejected"] = press_batch.rows_rejected
        summary["bronze_incidents_loaded"] = inc_batch.rows_loaded
        summary["bronze_incidents_rejected"] = inc_batch.rows_rejected
        summary["bronze_weather_loaded"] = weather_batch.rows_loaded

        # ── Silver ────────────────────────────────────────────────────────────
        logger.info("Step 2/3 — Silver: normalizing, deduplicating, imputing")
        silver_batch_id = _create_silver_batch(session)

        silver_temp = build_temperature_silver(session, silver_batch_id)
        silver_press = build_pressure_silver(session, silver_batch_id)
        silver_inc = build_incident_silver(session, silver_batch_id)
        silver_weather = build_weather_silver(session, silver_batch_id)
        session.commit()

        summary["silver_temperature_inserted"] = silver_temp
        summary["silver_pressure_inserted"] = silver_press
        summary["silver_incidents_inserted"] = silver_inc
        summary["silver_weather_inserted"] = silver_weather

        # ── Gold ──────────────────────────────────────────────────────────────
        logger.info("Step 3/3 — Gold: building multi-horizon rolling window features")
        gold_rows = build_gold(session)
        session.commit()

        summary["gold_rows_inserted"] = gold_rows

    logger.info("Pipeline complete: {}", summary)
    return summary
