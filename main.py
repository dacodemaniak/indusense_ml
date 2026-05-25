"""CLI entry point for the InduSense ingestion pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="InduSense — ingestion pipeline CSV → Bronze → Silver → Gold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datas"),
        metavar="DIR",
        help="Directory containing capteurs_temperature.csv, capteurs_pression.tsv, releves_incidents.csv",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        metavar="DIR",
        help="Directory for log files",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Run pending Alembic migrations before ingestion",
    )
    return parser.parse_args()


def _run_migrations() -> None:
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")


def main() -> int:
    args = _parse_args()

    from indusense.core.logging import configure_logging

    configure_logging(log_dir=args.log_dir, filename="pipeline.log")

    from loguru import logger

    if args.migrate:
        logger.info("Running Alembic migrations")
        _run_migrations()
        logger.info("Migrations applied")

    if not args.data_dir.is_dir():
        logger.error("Data directory not found: {}", args.data_dir)
        return 1

    from indusense.pipeline import run_pipeline

    try:
        summary = run_pipeline(data_dir=args.data_dir)
    except FileNotFoundError as exc:
        logger.error("{}", exc)
        return 1
    except Exception as exc:
        logger.exception("Pipeline failed: {}", exc)
        return 1

    print("\nPipeline terminé avec succès :")
    col_width = max(len(k) for k in summary) + 2
    for key, value in summary.items():
        print(f"  {key:<{col_width}} {value:>8,}".replace(",", " "))

    return 0


if __name__ == "__main__":
    sys.exit(main())
