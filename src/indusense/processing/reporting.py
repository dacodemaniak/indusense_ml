"""Markdown artifact reporting for data ingestion runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from mdutils.mdutils import MdUtils


def _append_table(md: MdUtils, frame: pd.DataFrame, max_rows: int = 20) -> None:
    if frame.empty:
        md.new_paragraph("Aucune ligne a afficher.")
        return

    display_frame = frame.head(max_rows).copy().fillna("")
    headers = list(display_frame.columns)
    rows = [str(item) for row in display_frame.astype(str).itertuples(index=False) for item in row]
    md.new_table(
        columns=len(headers),
        rows=len(display_frame) + 1,
        text=headers + rows,
        text_align="left",
    )


def write_data_ingestion_report(
    artifact_dir: Path,
    *,
    overview: pd.DataFrame,
    normalization_windows: pd.DataFrame,
    deduplication_summary: pd.DataFrame,
    imputation_metrics: pd.DataFrame,
    imputation_decisions: pd.DataFrame,
    gold_preview: pd.DataFrame,
) -> Path:
    """Write a Markdown report and companion CSV artifacts for a run."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    overview.to_csv(artifact_dir / "overview.csv", index=False)
    normalization_windows.to_csv(artifact_dir / "temperature_normalization_windows.csv", index=False)
    deduplication_summary.to_csv(artifact_dir / "deduplication_summary.csv", index=False)
    imputation_metrics.to_csv(artifact_dir / "imputation_metrics.csv", index=False)
    imputation_decisions.to_csv(artifact_dir / "imputation_decisions.csv", index=False)
    gold_preview.to_csv(artifact_dir / "gold_dataset_preview.csv", index=False)

    md = MdUtils(file_name=str(artifact_dir / "report"), title="Rapport de preparation des donnees")
    md.new_header(level=1, title="Rapport de preparation des donnees")
    md.new_paragraph(
        "Ce rapport synthétise les normalisations, dedoublonnages, evaluations d'imputation "
        "et premieres decisions de preparation vers les couches Silver et Gold."
    )

    md.new_header(level=2, title="Vue d'ensemble")
    _append_table(md, overview)

    md.new_header(level=2, title="Normalisation temperature")
    if normalization_windows.empty:
        md.new_paragraph("Aucune fenetre Fahrenheit n'a ete detectee.")
    else:
        _append_table(md, normalization_windows)

    md.new_header(level=2, title="Dedupllication")
    if deduplication_summary.empty:
        md.new_paragraph("Aucun doublon n'a ete resolu.")
    else:
        _append_table(md, deduplication_summary)

    md.new_header(level=2, title="Benchmarks d'imputation")
    _append_table(md, imputation_metrics)

    md.new_header(level=2, title="Decisions recommandees")
    _append_table(md, imputation_decisions)

    md.new_header(level=2, title="Apercu du Gold Dataset")
    _append_table(md, gold_preview)

    report_path = Path(md.file_name + ".md")
    md.create_md_file()
    logger.info("Markdown ingestion report written to {}", report_path)
    return report_path
