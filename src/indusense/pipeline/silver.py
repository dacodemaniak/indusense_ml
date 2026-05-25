"""Silver layer: Bronze tables → normalized SilverSensorReading, SilverIncident, SilverWeatherReading."""

from __future__ import annotations

import uuid
from datetime import timezone
from typing import Any

import pandas as pd
import sqlalchemy as sa
from loguru import logger
from sqlalchemy.orm import Session

from indusense.db.models import (
    BronzeIncidentRaw,
    BronzePressureRaw,
    BronzeTemperatureRaw,
    BronzeWeatherRaw,
    INCIDENT_TYPE_COLS,
    Machine,
    Operator,
    SensorType,
    SilverIncident,
    SilverSensorReading,
    SilverWeatherReading,
)
from indusense.processing import (
    ImputationContext,
    build_incident_silver_candidate,
    build_sensor_silver_candidate,
    deduplicate_incidents,
    deduplicate_sensor_records,
    evaluate_imputation_strategies,
    normalize_temperature_units,
    summarize_imputation_decisions,
)
from indusense.schemas.ingestion import normalize_machine_code


def _normalize_machine_code_safe(value: str | None) -> str | None:
    """Return normalized machine code or None on any parse failure."""
    if not value:
        return None
    try:
        return normalize_machine_code(str(value).strip())
    except (ValueError, AttributeError):
        return None


def _upsert_machines(session: Session, machine_codes: set[str]) -> dict[str, int]:
    """Ensure all machine_codes exist in Machine dimension; return code→id map."""
    existing = {
        row.machine_code: row.machine_id
        for row in session.execute(sa.select(Machine.machine_code, Machine.machine_id)).all()
    }
    for code in sorted(machine_codes - existing.keys()):
        session.add(Machine(machine_code=code, is_active=True))
    session.flush()
    return {
        row.machine_code: row.machine_id
        for row in session.execute(
            sa.select(Machine.machine_code, Machine.machine_id).where(
                Machine.machine_code.in_(machine_codes)
            )
        ).all()
    }


def _upsert_operators(session: Session, operator_keys: set[str]) -> dict[str, int]:
    """Ensure all operator_keys exist in Operator dimension; return key→id map."""
    existing = {
        row.operator_key: row.operator_id
        for row in session.execute(sa.select(Operator.operator_key, Operator.operator_id)).all()
    }
    for key in sorted(operator_keys - existing.keys()):
        session.add(Operator(operator_key=key, is_active=True))
    session.flush()
    return {
        row.operator_key: row.operator_id
        for row in session.execute(
            sa.select(Operator.operator_key, Operator.operator_id).where(
                Operator.operator_key.in_(operator_keys)
            )
        ).all()
    }


def _bulk_insert(session: Session, model: type, rows: list[dict[str, Any]], batch_size: int = 500) -> int:
    for i in range(0, len(rows), batch_size):
        session.execute(sa.insert(model), rows[i : i + batch_size])
    return len(rows)


def build_temperature_silver(session: Session, batch_id: uuid.UUID) -> int:
    """Build Silver temperature readings from Bronze (normalize → dedup → impute → insert)."""
    bronze_rows = session.execute(
        sa.select(
            BronzeTemperatureRaw.machine_id_raw,
            BronzeTemperatureRaw.timestamp_raw,
            BronzeTemperatureRaw.temperature_raw,
        ).where(BronzeTemperatureRaw.parse_ok.is_(True))
    ).all()

    if not bronze_rows:
        logger.warning("No valid Bronze temperature rows — skipping Silver temperature build")
        return 0

    df = pd.DataFrame(bronze_rows, columns=["machine_id_raw", "timestamp_raw", "temperature_raw"])
    df["machine_id_std"] = df["machine_id_raw"].apply(_normalize_machine_code_safe)
    df["event_ts"] = pd.to_datetime(
        df["timestamp_raw"].str.strip().str.replace("T", " ", regex=False),
        errors="coerce",
    )
    df["temperature"] = pd.to_numeric(df["temperature_raw"].str.strip(), errors="coerce")
    df["is_missing"] = df["temperature"].isna()
    df = df.dropna(subset=["machine_id_std", "event_ts"]).reset_index(drop=True)

    df_normalized, _ = normalize_temperature_units(df, value_column="temperature")
    df_clean, _ = deduplicate_sensor_records(
        df_normalized, dataset_name="capteurs_temperature", value_column="temperature_normalized"
    )

    context = ImputationContext(
        dataset_name="capteurs_temperature",
        target_column="temperature_normalized",
        group_column="machine_id_std",
        time_column="event_ts",
        plausible_min=0.0,
        plausible_max=120.0,
    )
    metrics, imputed_cols = evaluate_imputation_strategies(df_clean, context)
    decisions = summarize_imputation_decisions(metrics)
    method = decisions.loc[
        decisions["dataset_name"] == "capteurs_temperature", "recommended_method"
    ].iloc[0]

    silver = build_sensor_silver_candidate(
        df_clean,
        value_column="temperature_normalized",
        sensor_type="temperature",
        unit="C",
        chosen_method=method,
        imputed_columns=imputed_cols,
    )

    machine_map = _upsert_machines(session, set(silver["machine_id_std"].dropna().unique()))

    rows: list[dict[str, Any]] = []
    for _, r in silver.iterrows():
        mid = machine_map.get(r["machine_id_std"])
        if mid is None:
            continue
        val = r.get("sensor_value")
        ts = pd.Timestamp(r["event_ts"]).to_pydatetime().replace(tzinfo=timezone.utc)
        rows.append({
            "machine_id": mid,
            "observed_at": ts,
            "sensor_type": SensorType.TEMPERATURE,
            "sensor_value": float(val) if pd.notna(val) else None,
            "unit": "C",
            "is_missing": bool(r.get("is_missing", False)),
            "is_duplicate": bool(r.get("is_duplicate", False)),
            "is_outlier": False,
            "ingestion_batch_id": batch_id,
        })

    inserted = _bulk_insert(session, SilverSensorReading, rows)
    logger.info("Silver temperature: {} rows inserted (method={})", inserted, method)
    return inserted


def build_pressure_silver(session: Session, batch_id: uuid.UUID) -> int:
    """Build Silver pressure readings from Bronze (dedup → impute → insert)."""
    bronze_rows = session.execute(
        sa.select(
            BronzePressureRaw.machine_id_raw,
            BronzePressureRaw.timestamp_raw,
            BronzePressureRaw.pressure_raw,
        ).where(BronzePressureRaw.parse_ok.is_(True))
    ).all()

    if not bronze_rows:
        logger.warning("No valid Bronze pressure rows — skipping Silver pressure build")
        return 0

    df = pd.DataFrame(bronze_rows, columns=["machine_id_raw", "timestamp_raw", "pressure_raw"])
    df["machine_id_std"] = df["machine_id_raw"].apply(_normalize_machine_code_safe)
    df["event_ts"] = pd.to_datetime(
        df["timestamp_raw"].str.strip()
        .str.replace("T", " ", regex=False)
        .str.replace(r"(?:Z|[+-]\d{2}:\d{2})$", "", regex=True),
        errors="coerce",
    )
    df["pressure_bar"] = pd.to_numeric(df["pressure_raw"].str.strip(), errors="coerce")
    df["is_missing"] = df["pressure_bar"].isna()
    df = df.dropna(subset=["machine_id_std", "event_ts"]).reset_index(drop=True)

    df_clean, _ = deduplicate_sensor_records(
        df, dataset_name="capteurs_pression", value_column="pressure_bar"
    )

    context = ImputationContext(
        dataset_name="capteurs_pression",
        target_column="pressure_bar",
        group_column="machine_id_std",
        time_column="event_ts",
        plausible_min=100.0,
        plausible_max=250.0,
    )
    metrics, imputed_cols = evaluate_imputation_strategies(df_clean, context)
    decisions = summarize_imputation_decisions(metrics)
    method = decisions.loc[
        decisions["dataset_name"] == "capteurs_pression", "recommended_method"
    ].iloc[0]

    silver = build_sensor_silver_candidate(
        df_clean,
        value_column="pressure_bar",
        sensor_type="pressure",
        unit="bar",
        chosen_method=method,
        imputed_columns=imputed_cols,
    )

    machine_map = _upsert_machines(session, set(silver["machine_id_std"].dropna().unique()))

    rows: list[dict[str, Any]] = []
    for _, r in silver.iterrows():
        mid = machine_map.get(r["machine_id_std"])
        if mid is None:
            continue
        val = r.get("sensor_value")
        ts = pd.Timestamp(r["event_ts"]).to_pydatetime().replace(tzinfo=timezone.utc)
        rows.append({
            "machine_id": mid,
            "observed_at": ts,
            "sensor_type": SensorType.PRESSURE,
            "sensor_value": float(val) if pd.notna(val) else None,
            "unit": "bar",
            "is_missing": bool(r.get("is_missing", False)),
            "is_duplicate": bool(r.get("is_duplicate", False)),
            "is_outlier": False,
            "ingestion_batch_id": batch_id,
        })

    inserted = _bulk_insert(session, SilverSensorReading, rows)
    logger.info("Silver pressure: {} rows inserted (method={})", inserted, method)
    return inserted


def build_incident_silver(session: Session, batch_id: uuid.UUID) -> int:
    """Build Silver incidents from Bronze (dedup → candidate → insert), propagating type columns."""
    # Build dynamic SELECT to include all type columns
    type_col_attrs = [getattr(BronzeIncidentRaw, col) for col in INCIDENT_TYPE_COLS]
    bronze_rows = session.execute(
        sa.select(
            BronzeIncidentRaw.incident_code_raw,
            BronzeIncidentRaw.machine_id_raw,
            BronzeIncidentRaw.operator_name_raw,
            BronzeIncidentRaw.operator_badge_raw,
            BronzeIncidentRaw.occurred_at_raw,
            BronzeIncidentRaw.severity_raw,
            BronzeIncidentRaw.shift_raw,
            BronzeIncidentRaw.comment_raw,
            *type_col_attrs,
        ).where(BronzeIncidentRaw.parse_ok.is_(True))
    ).all()

    if not bronze_rows:
        logger.warning("No valid Bronze incident rows — skipping Silver incident build")
        return 0

    columns = [
        "incident_id", "machine_id_raw", "operator_name", "operator_badge",
        "occurred_at_raw", "severity_raw", "shift", "comment",
        *INCIDENT_TYPE_COLS,
    ]
    df = pd.DataFrame(bronze_rows, columns=columns)
    df["incident_id"] = df["incident_id"].str.strip().str.upper()
    df["machine_id_std"] = df["machine_id_raw"].apply(_normalize_machine_code_safe)
    df["event_ts"] = pd.to_datetime(df["occurred_at_raw"].str.strip(), errors="coerce")
    df["severity"] = pd.to_numeric(df["severity_raw"].str.strip(), errors="coerce")
    df = df.dropna(subset=["machine_id_std", "event_ts", "incident_id"]).reset_index(drop=True)

    df_clean, _ = deduplicate_incidents(df, key_column="incident_id")
    silver = build_incident_silver_candidate(df_clean)

    machine_map = _upsert_machines(session, set(silver["machine_id_std"].dropna().unique()))
    operator_keys = set(silver["operator_name_clean"].fillna("UNKNOWN").unique())
    operator_map = _upsert_operators(session, operator_keys)

    rows: list[dict[str, Any]] = []
    for _, r in silver.iterrows():
        mid = machine_map.get(r["machine_id_std"])
        if mid is None:
            continue
        op_key = r.get("operator_name_clean") or "UNKNOWN"
        ts = pd.Timestamp(r["event_ts"]).to_pydatetime().replace(tzinfo=timezone.utc)
        type_vals = {col: int(r.get(col, 0) or 0) for col in INCIDENT_TYPE_COLS}
        rows.append({
            "incident_code": r["incident_id"],
            "machine_id": mid,
            "operator_id": operator_map.get(op_key),
            "occurred_at": ts,
            "severity": int(r["severity"]),
            "shift": r.get("shift") or None,
            "comment": r.get("comment_clean") or None,
            "is_label_event": True,
            **type_vals,
            "ingestion_batch_id": batch_id,
        })

    inserted = _bulk_insert(session, SilverIncident, rows)
    logger.info("Silver incidents: {} rows inserted", inserted)
    return inserted


def build_weather_silver(session: Session, batch_id: uuid.UUID) -> int:
    """Normalize BronzeWeatherRaw → SilverWeatherReading (one row per UTC hour)."""
    bronze_rows = session.execute(
        sa.select(
            BronzeWeatherRaw.timestamp_raw,
            BronzeWeatherRaw.temp_raw,
            BronzeWeatherRaw.humidity_raw,
            BronzeWeatherRaw.pressure_raw,
            BronzeWeatherRaw.wind_speed_raw,
            BronzeWeatherRaw.is_imputed,
        ).where(BronzeWeatherRaw.parse_ok.is_(True))
    ).all()

    if not bronze_rows:
        logger.warning("No valid Bronze weather rows — skipping Silver weather build")
        return 0

    df = pd.DataFrame(
        bronze_rows,
        columns=["timestamp_raw", "temp_raw", "humidity_raw", "pressure_raw", "wind_speed_raw", "is_imputed"],
    )
    df["observed_at"] = pd.to_datetime(df["timestamp_raw"], utc=True, errors="coerce")
    df = df.dropna(subset=["observed_at"])
    # Floor to hour and deduplicate (keep last in case of duplicates)
    df["observed_at"] = df["observed_at"].dt.floor("h")
    df = df.sort_values("observed_at").drop_duplicates(subset=["observed_at"], keep="last")

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        ts = r["observed_at"].to_pydatetime()

        def _f(val: Any) -> float | None:
            try:
                return float(val) if val is not None and str(val) not in ("nan", "None", "") else None
            except (ValueError, TypeError):
                return None

        rows.append({
            "observed_at": ts,
            "temp_celsius": _f(r["temp_raw"]),
            "humidity_pct": _f(r["humidity_raw"]),
            "pressure_hpa": _f(r["pressure_raw"]),
            "wind_speed_ms": _f(r["wind_speed_raw"]),
            "is_imputed": bool(r["is_imputed"]),
            "ingestion_batch_id": batch_id,
        })

    inserted = _bulk_insert(session, SilverWeatherReading, rows)
    imputed_count = sum(1 for r in rows if r["is_imputed"])
    logger.info(
        "Silver weather: {} rows inserted ({} imputed by monthly mean)",
        inserted, imputed_count,
    )
    return inserted
