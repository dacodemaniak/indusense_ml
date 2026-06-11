"""Add BronzeTelemetryRaw, SilverTelemetryReading, Maintenance, max_hourly_capacity_pieces, new Gold columns."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260611_0004"
down_revision = "20260513_0003"
branch_labels = None
depends_on = None

_VOLTAGE_ROTATION_WINDOWS = [6, 12, 24]


def upgrade() -> None:
    # ── machine: add hourly capacity column ───────────────────────────────────
    op.add_column(
        "machine",
        sa.Column("max_hourly_capacity_pieces", sa.Integer(), nullable=True),
    )

    # ── maintenance: new reference table ─────────────────────────────────────
    op.create_table(
        "maintenance",
        sa.Column(
            "maintenance_db_id",
            sa.BigInteger(),
            sa.Identity(always=False),
            primary_key=True,
        ),
        sa.Column("maintenance_id", sa.Integer(), nullable=False, unique=True),
        sa.Column(
            "machine_id",
            sa.BigInteger(),
            sa.ForeignKey("machine.machine_id", ondelete="RESTRICT"),
            nullable=False,
            index=True,
        ),
        sa.Column("maintenance_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("maintenance_type", sa.String(16), nullable=False),
        sa.Column("action_type", sa.String(32), nullable=False),
        sa.Column("component", sa.String(64), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("related_incident_code", sa.String(16), nullable=True),
        sa.Column("duration_hours", sa.Numeric(6, 2), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("idx_maintenance_machine_at", "maintenance", ["machine_id", "maintenance_at"])

    # ── bronze_telemetry_raw: unified telemetry ingestion ─────────────────────
    op.create_table(
        "bronze_telemetry_raw",
        sa.Column(
            "telemetry_raw_id",
            sa.BigInteger(),
            sa.Identity(always=False),
            primary_key=True,
        ),
        sa.Column(
            "ingestion_batch_id",
            sa.Uuid(),
            sa.ForeignKey("ingestion_batch.ingestion_batch_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("row_number", sa.Integer(), nullable=False),
        sa.Column("machine_id_raw", sa.String(64), nullable=True),
        sa.Column("timestamp_raw", sa.String(128), nullable=True),
        sa.Column("temperature_raw", sa.String(64), nullable=True),
        sa.Column("pressure_raw", sa.String(64), nullable=True),
        sa.Column("voltage_raw", sa.String(64), nullable=True),
        sa.Column("rotation_raw", sa.String(64), nullable=True),
        sa.Column("pieces_raw", sa.String(64), nullable=True),
        sa.Column("parse_ok", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("rejected_reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # ── silver_telemetry_reading: unified hourly telemetry per machine ─────────
    op.create_table(
        "silver_telemetry_reading",
        sa.Column(
            "telemetry_reading_id",
            sa.BigInteger(),
            sa.Identity(always=False),
            primary_key=True,
        ),
        sa.Column(
            "machine_id",
            sa.BigInteger(),
            sa.ForeignKey("machine.machine_id", ondelete="RESTRICT"),
            nullable=False,
            index=True,
        ),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("temperature_c", sa.Numeric(8, 3), nullable=True),
        sa.Column("pressure_bar", sa.Numeric(10, 3), nullable=True),
        sa.Column("voltage_mean_v", sa.Numeric(10, 3), nullable=True),
        sa.Column("rotation_mean_rpm", sa.Numeric(10, 3), nullable=True),
        sa.Column("pieces_produced", sa.Integer(), nullable=True),
        sa.Column("is_missing_temp", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("is_missing_pressure", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("is_duplicate", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column(
            "ingestion_batch_id",
            sa.Uuid(),
            sa.ForeignKey("ingestion_batch.ingestion_batch_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("machine_id", "observed_at", name="silver_telemetry_machine_ts"),
    )

    # ── gold: voltage rolling features (6h / 12h / 24h) ─────────────────────
    for w in _VOLTAGE_ROTATION_WINDOWS:
        op.add_column(
            "gold_machine_hourly_feature",
            sa.Column(f"voltage_mean_{w}h", sa.Numeric(10, 3), nullable=True),
        )
        op.add_column(
            "gold_machine_hourly_feature",
            sa.Column(f"voltage_std_{w}h", sa.Numeric(10, 3), nullable=True),
        )

    # ── gold: rotation rolling features (6h / 12h / 24h) ─────────────────────
    for w in _VOLTAGE_ROTATION_WINDOWS:
        op.add_column(
            "gold_machine_hourly_feature",
            sa.Column(f"rotation_mean_{w}h", sa.Numeric(10, 3), nullable=True),
        )
        op.add_column(
            "gold_machine_hourly_feature",
            sa.Column(f"rotation_std_{w}h", sa.Numeric(10, 3), nullable=True),
        )

    # ── gold: production & maintenance features ───────────────────────────────
    op.add_column(
        "gold_machine_hourly_feature",
        sa.Column("pieces_produced_sum_24h", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "gold_machine_hourly_feature",
        sa.Column("capacity_utilization_pct", sa.Numeric(6, 3), nullable=True),
    )
    op.add_column(
        "gold_machine_hourly_feature",
        sa.Column("days_since_last_maintenance", sa.Numeric(8, 2), nullable=True),
    )
    op.add_column(
        "gold_machine_hourly_feature",
        sa.Column("maintenance_count_prev_30d", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("gold_machine_hourly_feature", "maintenance_count_prev_30d")
    op.drop_column("gold_machine_hourly_feature", "days_since_last_maintenance")
    op.drop_column("gold_machine_hourly_feature", "capacity_utilization_pct")
    op.drop_column("gold_machine_hourly_feature", "pieces_produced_sum_24h")

    for w in reversed(_VOLTAGE_ROTATION_WINDOWS):
        op.drop_column("gold_machine_hourly_feature", f"rotation_std_{w}h")
        op.drop_column("gold_machine_hourly_feature", f"rotation_mean_{w}h")

    for w in reversed(_VOLTAGE_ROTATION_WINDOWS):
        op.drop_column("gold_machine_hourly_feature", f"voltage_std_{w}h")
        op.drop_column("gold_machine_hourly_feature", f"voltage_mean_{w}h")

    op.drop_table("silver_telemetry_reading")
    op.drop_table("bronze_telemetry_raw")
    op.drop_index("idx_maintenance_machine_at", table_name="maintenance")
    op.drop_table("maintenance")
    op.drop_column("machine", "max_hourly_capacity_pieces")
