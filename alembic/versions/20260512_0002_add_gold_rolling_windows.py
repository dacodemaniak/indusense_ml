"""Add multi-horizon rolling window features to gold_machine_hourly_feature."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260512_0002"
down_revision = "20260507_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("gold_machine_hourly_feature", sa.Column("temp_mean_6h", sa.Numeric(12, 3), nullable=True))
    op.add_column("gold_machine_hourly_feature", sa.Column("temp_max_6h", sa.Numeric(12, 3), nullable=True))
    op.add_column("gold_machine_hourly_feature", sa.Column("temp_std_6h", sa.Numeric(12, 3), nullable=True))
    op.add_column("gold_machine_hourly_feature", sa.Column("pressure_mean_6h", sa.Numeric(12, 3), nullable=True))
    op.add_column("gold_machine_hourly_feature", sa.Column("pressure_max_6h", sa.Numeric(12, 3), nullable=True))
    op.add_column("gold_machine_hourly_feature", sa.Column("pressure_std_6h", sa.Numeric(12, 3), nullable=True))

    op.add_column("gold_machine_hourly_feature", sa.Column("temp_mean_12h", sa.Numeric(12, 3), nullable=True))
    op.add_column("gold_machine_hourly_feature", sa.Column("temp_max_12h", sa.Numeric(12, 3), nullable=True))
    op.add_column("gold_machine_hourly_feature", sa.Column("temp_std_12h", sa.Numeric(12, 3), nullable=True))
    op.add_column("gold_machine_hourly_feature", sa.Column("pressure_mean_12h", sa.Numeric(12, 3), nullable=True))
    op.add_column("gold_machine_hourly_feature", sa.Column("pressure_max_12h", sa.Numeric(12, 3), nullable=True))
    op.add_column("gold_machine_hourly_feature", sa.Column("pressure_std_12h", sa.Numeric(12, 3), nullable=True))

    op.add_column("gold_machine_hourly_feature", sa.Column("pressure_max_24h", sa.Numeric(12, 3), nullable=True))

    op.add_column("gold_machine_hourly_feature", sa.Column("temp_trend_6h", sa.Numeric(12, 3), nullable=True))
    op.add_column("gold_machine_hourly_feature", sa.Column("pressure_trend_6h", sa.Numeric(12, 3), nullable=True))
    op.add_column("gold_machine_hourly_feature", sa.Column("temp_zscore_24h", sa.Numeric(8, 4), nullable=True))

    op.add_column("gold_machine_hourly_feature", sa.Column("incident_count_prev_7d", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("gold_machine_hourly_feature", sa.Column("hours_since_last_incident", sa.Numeric(10, 2), nullable=True))

    op.add_column("gold_machine_hourly_feature", sa.Column("label_failure_next_6h", sa.Boolean(), nullable=False, server_default="false"))
    op.add_column("gold_machine_hourly_feature", sa.Column("label_failure_next_12h", sa.Boolean(), nullable=False, server_default="false"))
    op.add_column("gold_machine_hourly_feature", sa.Column("label_failure_next_48h", sa.Boolean(), nullable=False, server_default="false"))


def downgrade() -> None:
    for column in [
        "label_failure_next_48h",
        "label_failure_next_12h",
        "label_failure_next_6h",
        "hours_since_last_incident",
        "incident_count_prev_7d",
        "temp_zscore_24h",
        "pressure_trend_6h",
        "temp_trend_6h",
        "pressure_max_24h",
        "pressure_std_12h",
        "pressure_max_12h",
        "pressure_mean_12h",
        "temp_std_12h",
        "temp_max_12h",
        "temp_mean_12h",
        "pressure_std_6h",
        "pressure_max_6h",
        "pressure_mean_6h",
        "temp_std_6h",
        "temp_max_6h",
        "temp_mean_6h",
    ]:
        op.drop_column("gold_machine_hourly_feature", column)
