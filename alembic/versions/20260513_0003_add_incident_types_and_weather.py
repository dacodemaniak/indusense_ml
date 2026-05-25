"""Add incident type labels (Bronze/Silver), weather tables (Bronze/Silver), Gold type counts and weather columns."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260513_0003"
down_revision = "20260512_0002"
branch_labels = None
depends_on = None

_INCIDENT_TYPE_COLS = [
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


def upgrade() -> None:
    # ── bronze_incident_raw: 9 type columns ───────────────────────────────────
    for col in _INCIDENT_TYPE_COLS:
        op.add_column(
            "bronze_incident_raw",
            sa.Column(col, sa.SmallInteger(), nullable=False, server_default="0"),
        )

    # ── bronze_weather_raw: new table ─────────────────────────────────────────
    op.create_table(
        "bronze_weather_raw",
        sa.Column("weather_raw_id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column("ingestion_batch_id", sa.Uuid(), sa.ForeignKey("ingestion_batch.ingestion_batch_id", ondelete="CASCADE"), nullable=False),
        sa.Column("row_number", sa.Integer(), nullable=False),
        sa.Column("timestamp_raw", sa.String(128), nullable=True),
        sa.Column("temp_raw", sa.String(32), nullable=True),
        sa.Column("humidity_raw", sa.String(32), nullable=True),
        sa.Column("pressure_raw", sa.String(32), nullable=True),
        sa.Column("wind_speed_raw", sa.String(32), nullable=True),
        sa.Column("is_imputed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("parse_ok", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("rejected_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # ── silver_incident: 9 type columns ───────────────────────────────────────
    for col in _INCIDENT_TYPE_COLS:
        op.add_column(
            "silver_incident",
            sa.Column(col, sa.SmallInteger(), nullable=False, server_default="0"),
        )

    # ── silver_weather_reading: new table ─────────────────────────────────────
    op.create_table(
        "silver_weather_reading",
        sa.Column("weather_reading_id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("temp_celsius", sa.Numeric(8, 2), nullable=True),
        sa.Column("humidity_pct", sa.Numeric(5, 2), nullable=True),
        sa.Column("pressure_hpa", sa.Numeric(8, 2), nullable=True),
        sa.Column("wind_speed_ms", sa.Numeric(8, 2), nullable=True),
        sa.Column("is_imputed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("ingestion_batch_id", sa.Uuid(), sa.ForeignKey("ingestion_batch.ingestion_batch_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.UniqueConstraint("observed_at", name="silver_weather_observed_at_unique"),
    )

    # ── gold_machine_hourly_feature: 9 type count columns ─────────────────────
    for col in _INCIDENT_TYPE_COLS:
        op.add_column(
            "gold_machine_hourly_feature",
            sa.Column(f"{col}_count_prev_24h", sa.Integer(), nullable=False, server_default="0"),
        )

    # ── gold_machine_hourly_feature: 3 ambient weather columns ────────────────
    op.add_column("gold_machine_hourly_feature", sa.Column("ambient_temp_c", sa.Numeric(8, 2), nullable=True))
    op.add_column("gold_machine_hourly_feature", sa.Column("ambient_humidity_pct", sa.Numeric(5, 2), nullable=True))
    op.add_column("gold_machine_hourly_feature", sa.Column("ambient_pressure_hpa", sa.Numeric(8, 2), nullable=True))


def downgrade() -> None:
    # Gold weather columns
    op.drop_column("gold_machine_hourly_feature", "ambient_pressure_hpa")
    op.drop_column("gold_machine_hourly_feature", "ambient_humidity_pct")
    op.drop_column("gold_machine_hourly_feature", "ambient_temp_c")

    # Gold type count columns
    for col in reversed(_INCIDENT_TYPE_COLS):
        op.drop_column("gold_machine_hourly_feature", f"{col}_count_prev_24h")

    # silver_weather_reading (drop_table supprime l'index observé_at automatiquement)
    op.drop_table("silver_weather_reading")

    # silver_incident type columns
    for col in reversed(_INCIDENT_TYPE_COLS):
        op.drop_column("silver_incident", col)

    # bronze_weather_raw
    op.drop_table("bronze_weather_raw")

    # bronze_incident_raw type columns
    for col in reversed(_INCIDENT_TYPE_COLS):
        op.drop_column("bronze_incident_raw", col)
