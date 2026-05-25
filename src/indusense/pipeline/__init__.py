"""Ingestion pipeline: CSV files → Bronze → Silver → Gold."""

from indusense.pipeline.runner import run_pipeline

__all__ = ["run_pipeline"]
