#!/usr/bin/env python3
r"""Compute patient-cluster bootstrap intervals for VIPER Top-k metrics.

The patient is the resampling unit. In each replicate, all causal variants from
each sampled patient are retained, and variant-level denominators are
recalculated. Unless overridden, the script uses 10,000 percentile-bootstrap
replicates and random seed 42. Paired model differences use identical patient
draws for both methods.

Required inputs are the IEI S2 sheet and/or the IRD summary workbook. To
reproduce the IEI de-overlap analysis, also provide the final training JSONL
and the IEI metadata workbook as shown below.

Example:
    python scripts/bootstrap_topk_ci.py \
        --iei-s2 /path/to/Supplementary\ Tables-20261023.xlsx \
        --ird-summary /path/to/IRD/summary-62S.xlsx \
        --iei-training-jsonl /path/to/Trainset-20250819.jsonl \
        --iei-metadata-xlsx Eval/example_metadata.xlsx \
        --output-dir results/ci

The output directory contains TSV tables corresponding to S3a/S3b/S3c,
S5a/S5b, S15a/S15b/S15c, and the VIPER positive-call intervals.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from openpyxl import load_workbook


@dataclass(frozen=True)
class EstimateCI:
    estimate: float
    low: float
    high: float


MODEL_ORDER = (
    "Base",
    "VIPER",
    "GeneT",
    "Xrare observed",
    "Xrare rank-augmented",
    "Exomiser",
    "PHIVE",
    "PhenIX",
)


def build_patient_bootstrap_indices(
    patient_ids: Iterable[object],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Return patient-level bootstrap positions in sorted patient order."""
    patient_ids = sorted(set(patient_ids))
    if not patient_ids:
        raise ValueError("At least one patient is required.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")
    return np.random.default_rng(seed).integers(
        0, len(patient_ids), size=(n_resamples, len(patient_ids))
    )


def _canonical_model_and_field(group: str, field: str) -> tuple[str, str] | None:
    group_key = str(group).strip()
    field_key = str(field).strip()
    if field_key == "rank":
        if group_key == "Baseline(zero-shot)":
            return "Base", "rank"
        if group_key in {"VIPER", "GeneT", "Exomiser", "PHIVE", "PhenIX"}:
            return group_key, "rank"
    if group_key == "Xrare":
        if field_key in {"rank_obs", "rank_ori"}:
            return "Xrare observed", field_key
        if field_key == "rank_aug":
            return "Xrare rank-augmented", field_key
    return None


def _read_grouped_sheet(path: str, sheet_name: str) -> list[list[object]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    if sheet_name not in workbook.sheetnames:
        workbook.close()
        raise KeyError(f"Sheet {sheet_name!r} not found in {path}.")
    try:
        return [
            list(row) for row in workbook[sheet_name].iter_rows(values_only=True)
        ]
    finally:
        workbook.close()


def _find_field_header_row(
    rows: list[list[object]], required_fields: set[str]
) -> int:
    for row_index, row in enumerate(rows):
        normalized = {
            str(value).strip()
            for value in row
            if value not in (None, "")
        }
        if required_fields.issubset(normalized):
            if row_index == 0:
                raise ValueError("A group header row must precede the field header.")
            return row_index
    missing = ", ".join(sorted(required_fields))
    raise KeyError(f"Input must contain {missing} fields.")


def _load_grouped_ranks(
    path: str,
    *,
    sheet_name: str,
    include_gene: bool,
    include_overlap: bool,
) -> pd.DataFrame:
    rows = _read_grouped_sheet(path, sheet_name)
    field_header_index = _find_field_header_row(rows, {"case_ID", "cpra"})
    group_row = rows[field_header_index - 1]
    field_row = rows[field_header_index]
    width = max(len(group_row), len(field_row))
    groups: list[str] = []
    current_group = ""
    for column in range(width):
        group = group_row[column] if column < len(group_row) else None
        if group not in (None, ""):
            current_group = str(group).strip()
        groups.append(current_group)

    fields: dict[str, int] = {}
    for column, field in enumerate(field_row):
        if field not in (None, ""):
            fields[str(field).strip()] = column

    model_columns: dict[str, int] = {}
    positive_columns: dict[str, int] = {}
    for column, (group, field) in enumerate(zip(groups, field_row)):
        mapped = _canonical_model_and_field(group, field or "")
        if mapped:
            model_columns[mapped[0]] = column
        if field == "label" and group in {"VIPER"}:
            positive_columns["VIPER"] = column
    if not model_columns:
        raise KeyError("No model rank columns could be identified.")

    overlap_column = None
    if include_overlap and "training_overlap_any" in fields:
        overlap_column = fields["training_overlap_any"]

    records: list[dict[str, object]] = []
    for row in rows[field_header_index + 1 :]:
        if not row or row[fields["case_ID"]] in (None, ""):
            continue
        record: dict[str, object] = {
            "patient_id": row[fields["case_ID"]],
            "cpra": row[fields["cpra"]],
        }
        if include_gene and "Gene" in fields:
            record["gene"] = row[fields["Gene"]]
        for model in MODEL_ORDER:
            column = model_columns.get(model)
            if column is None:
                continue
            value = row[column] if column < len(row) else None
            record[model] = pd.to_numeric(value, errors="coerce")
        positive_column = positive_columns.get("VIPER")
        if positive_column is not None:
            value = row[positive_column] if positive_column < len(row) else None
            record["VIPER_positive"] = str(value).strip().lower() == "positive"
        if overlap_column is not None:
            value = row[overlap_column] if overlap_column < len(row) else None
            if isinstance(value, str):
                value = value.strip().lower() in {"true", "1", "yes"}
            record["training_overlap_any"] = bool(value)
        records.append(record)
    frame = pd.DataFrame.from_records(records)
    ordered = ["patient_id", "cpra"]
    if include_gene:
        ordered.append("gene")
    for model in MODEL_ORDER:
        if model in frame.columns:
            ordered.append(model)
        if model == "VIPER" and "VIPER_positive" in frame.columns:
            ordered.append("VIPER_positive")
    if "training_overlap_any" in frame.columns:
        ordered.append("training_overlap_any")
    return frame[ordered]


def load_iei_s2(path: str) -> pd.DataFrame:
    return _load_grouped_ranks(
        path, sheet_name="S2", include_gene=True, include_overlap=False
    )


def load_ird_summary(path: str) -> pd.DataFrame:
    return _load_grouped_ranks(
        path, sheet_name="Sheet1", include_gene=False, include_overlap=True
    )


def _prepare_ranks(data: pd.DataFrame, rank_column: str) -> tuple[pd.DataFrame, list]:
    if rank_column not in data.columns:
        raise KeyError(f"Missing rank column: {rank_column}")
    if "patient_id" not in data.columns:
        raise KeyError("Missing patient_id column.")
    working = data[["patient_id", rank_column]].copy()
    working["_rank"] = pd.to_numeric(working[rank_column], errors="coerce")
    patient_ids = sorted(working["patient_id"].unique())
    return working, patient_ids


def _bootstrap_ci(
    estimate: float,
    replicates: np.ndarray,
) -> EstimateCI:
    low, high = np.percentile(replicates, [2.5, 97.5])
    return EstimateCI(float(estimate), float(low), float(high))


def variant_topk_ci(
    data: pd.DataFrame,
    *,
    rank_column: str,
    k: int,
    indices: np.ndarray,
) -> EstimateCI:
    """Calculate variant-level Top-k recall using patient-cluster resampling."""
    working, patient_ids = _prepare_ranks(data, rank_column)
    if indices.shape[1] != len(patient_ids):
        raise ValueError("Bootstrap indices do not match the number of patients.")

    hits = []
    denominators = []
    for patient_id in patient_ids:
        ranks = working.loc[working["patient_id"] == patient_id, "_rank"]
        hits.append(int((ranks <= k).sum()))
        denominators.append(int(len(ranks)))
    hits = np.asarray(hits)
    denominators = np.asarray(denominators)

    estimate = hits.sum() / denominators.sum()
    replicates = hits[indices].sum(axis=1) / denominators[indices].sum(axis=1)
    return _bootstrap_ci(estimate, replicates)


def patient_topk_ci(
    data: pd.DataFrame,
    *,
    rank_column: str,
    k: int,
    indices: np.ndarray,
    mode: Literal["any", "complete"],
) -> EstimateCI:
    """Calculate patient-level any-variant or complete Top-k recovery."""
    if mode not in {"any", "complete"}:
        raise ValueError("mode must be 'any' or 'complete'.")
    working, patient_ids = _prepare_ranks(data, rank_column)
    if indices.shape[1] != len(patient_ids):
        raise ValueError("Bootstrap indices do not match the number of patients.")

    event = []
    for patient_id in patient_ids:
        ranks = working.loc[working["patient_id"] == patient_id, "_rank"]
        if mode == "any":
            event.append(bool((ranks <= k).any()))
        else:
            event.append(bool(len(ranks) and ranks.notna().all() and (ranks <= k).all()))
    event = np.asarray(event, dtype=float)

    estimate = event.mean()
    replicates = event[indices].mean(axis=1)
    return _bootstrap_ci(estimate, replicates)


def paired_variant_topk_difference(
    data: pd.DataFrame,
    *,
    reference_column: str,
    comparator_column: str,
    k: int,
    indices: np.ndarray,
) -> EstimateCI:
    """Calculate a paired variant-level Top-k difference on identical draws."""
    reference, patient_ids = _prepare_ranks(data, reference_column)
    comparator, comparator_ids = _prepare_ranks(data, comparator_column)
    if patient_ids != comparator_ids:
        raise ValueError("Reference and comparator contain different patients.")
    if indices.shape[1] != len(patient_ids):
        raise ValueError("Bootstrap indices do not match the number of patients.")

    reference_hits = []
    comparator_hits = []
    denominators = []
    for patient_id in patient_ids:
        ref_ranks = reference.loc[reference["patient_id"] == patient_id, "_rank"]
        cmp_ranks = comparator.loc[comparator["patient_id"] == patient_id, "_rank"]
        if len(ref_ranks) != len(cmp_ranks):
            raise ValueError("Reference and comparator have different row counts.")
        reference_hits.append(int((ref_ranks <= k).sum()))
        comparator_hits.append(int((cmp_ranks <= k).sum()))
        denominators.append(int(len(ref_ranks)))

    reference_hits = np.asarray(reference_hits)
    comparator_hits = np.asarray(comparator_hits)
    denominators = np.asarray(denominators)
    reference_estimate = reference_hits.sum() / denominators.sum()
    comparator_estimate = comparator_hits.sum() / denominators.sum()
    estimate = reference_estimate - comparator_estimate

    sampled_denominators = denominators[indices].sum(axis=1)
    replicates = (
        reference_hits[indices].sum(axis=1) / sampled_denominators
        - comparator_hits[indices].sum(axis=1) / sampled_denominators
    )
    return _bootstrap_ci(estimate, replicates)


def filter_non_overlapping(data: pd.DataFrame) -> pd.DataFrame:
    """Return records without an exact training-set match."""
    if "training_overlap_any" not in data.columns:
        raise KeyError("Missing training_overlap_any column.")
    overlap = data["training_overlap_any"]
    if overlap.dtype == object:
        overlap = overlap.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    return data.loc[~overlap.astype(bool)].reset_index(drop=True)


def _normalize_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace(" ", "").upper()


def _training_variant_sets(
    training_jsonl: str,
) -> tuple[set[str], set[tuple[str, str]], set[tuple[str, str]]]:
    pattern = re.compile(
        r"The basic information of variant\s+(chr\S+)\s+is as follows:\s*"
        r"Gene symbol:\s*([^\n]*)\s*Transcript:\s*([^\n]*)\s*"
        r"DNA change:\s*([^\n]*)\s*Protein change:\s*([^\n]*)"
    )
    cpras: set[str] = set()
    gene_hgvsc: set[tuple[str, str]] = set()
    gene_hgvsp: set[tuple[str, str]] = set()
    with open(training_jsonl, encoding="utf-8") as handle:
        for line in handle:
            conversation = json.loads(line)
            text = "\n".join(
                message.get("content", "")
                for message in conversation.get("messages", [])
            ).replace("\\n", "\n")
            match = pattern.search(text)
            if not match:
                continue
            cpra, gene, _transcript, hgvsc, hgvsp = [
                value.strip() for value in match.groups()
            ]
            cpra = _normalize_identifier(cpra)
            gene = _normalize_identifier(gene)
            hgvsc = _normalize_identifier(hgvsc)
            hgvsp = _normalize_identifier(hgvsp)
            cpras.add(cpra)
            if gene and hgvsc:
                gene_hgvsc.add((gene, hgvsc))
            if gene and hgvsp:
                gene_hgvsp.add((gene, hgvsp))
    return cpras, gene_hgvsc, gene_hgvsp


def compute_iei_overlap_mask(
    data: pd.DataFrame,
    *,
    training_jsonl: str,
    metadata_xlsx: str,
) -> pd.Series:
    """Match evaluation rows by CPRA, gene+HGVSc or gene+HGVSp."""
    if not {"cpra", "gene"}.issubset(data.columns):
        raise KeyError("IEI overlap detection requires cpra and gene columns.")
    training_cpras, training_hgvsc, training_hgvsp = _training_variant_sets(
        training_jsonl
    )
    metadata = pd.read_excel(
        metadata_xlsx,
        usecols=["cpra", "gene symbol", "dna change", "protein change"],
    ).drop_duplicates(["cpra", "gene symbol"])
    annotation_map = {
        (
            _normalize_identifier(row["cpra"]),
            _normalize_identifier(row["gene symbol"]),
        ): (
            _normalize_identifier(row["dna change"]),
            _normalize_identifier(row["protein change"]),
        )
        for _, row in metadata.iterrows()
    }

    hits = []
    for row in data.itertuples(index=False):
        cpra = _normalize_identifier(row.cpra)
        gene = _normalize_identifier(row.gene)
        hgvsc, hgvsp = annotation_map.get((cpra, gene), ("", ""))
        hits.append(
            cpra in training_cpras
            or (gene, hgvsc) in training_hgvsc
            or bool(hgvsp and (gene, hgvsp) in training_hgvsp)
        )
    return pd.Series(hits, index=data.index, dtype=bool)


def _binary_cluster_ci(
    data: pd.DataFrame,
    *,
    outcome_column: str,
    indices: np.ndarray,
    patient_level: bool,
) -> EstimateCI:
    if outcome_column not in data.columns:
        raise KeyError(f"Missing outcome column: {outcome_column}")
    patient_ids = sorted(data["patient_id"].unique())
    if indices.shape[1] != len(patient_ids):
        raise ValueError("Bootstrap indices do not match the number of patients.")
    outcome = data[outcome_column].astype(bool)

    if patient_level:
        event = np.asarray(
            [
                bool(outcome.loc[data["patient_id"] == patient_id].any())
                for patient_id in patient_ids
            ],
            dtype=float,
        )
        estimate = event.mean()
        replicates = event[indices].mean(axis=1)
    else:
        hits = np.asarray(
            [
                int(outcome.loc[data["patient_id"] == patient_id].sum())
                for patient_id in patient_ids
            ]
        )
        denominators = np.asarray(
            [int((data["patient_id"] == patient_id).sum()) for patient_id in patient_ids]
        )
        estimate = hits.sum() / denominators.sum()
        replicates = hits[indices].sum(axis=1) / denominators[indices].sum(axis=1)
    return _bootstrap_ci(estimate, replicates)


def compute_cohort_tables(
    data: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    models: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute model, paired, patient-level and label CI tables."""
    requested_models = tuple(models) if models is not None else MODEL_ORDER
    models = [model for model in requested_models if model in data.columns]
    if "VIPER" not in models:
        raise KeyError("VIPER rank column is required.")
    patient_ids = sorted(data["patient_id"].unique())
    indices = build_patient_bootstrap_indices(
        patient_ids, n_resamples=n_resamples, seed=seed
    )

    summary_rows = []
    for model in models:
        for k in (1, 5, 10):
            result = variant_topk_ci(
                data, rank_column=model, k=k, indices=indices
            )
            summary_rows.append(
                {
                    "model": model,
                    "k": k,
                    "estimate": result.estimate,
                    "ci_low": result.low,
                    "ci_high": result.high,
                }
            )

    paired_rows = []
    for comparator in models:
        if comparator == "VIPER":
            continue
        for k in (1, 5, 10):
            result = paired_variant_topk_difference(
                data,
                reference_column="VIPER",
                comparator_column=comparator,
                k=k,
                indices=indices,
            )
            paired_rows.append(
                {
                    "comparison": f"VIPER minus {comparator}",
                    "k": k,
                    "estimate": result.estimate,
                    "ci_low": result.low,
                    "ci_high": result.high,
                }
            )

    complete_rows = []
    for model in models:
        result = patient_topk_ci(
            data,
            rank_column=model,
            k=10,
            indices=indices,
            mode="complete",
        )
        complete_rows.append(
            {
                "model": model,
                "estimate": result.estimate,
                "ci_low": result.low,
                "ci_high": result.high,
            }
        )

    label_rows = []
    if "VIPER_positive" in data.columns:
        for metric, patient_level in (
            ("patient_any_positive_call", True),
            ("variant_positive_call", False),
        ):
            result = _binary_cluster_ci(
                data,
                outcome_column="VIPER_positive",
                indices=indices,
                patient_level=patient_level,
            )
            label_rows.append(
                {
                    "metric": metric,
                    "estimate": result.estimate,
                    "ci_low": result.low,
                    "ci_high": result.high,
                }
            )

    paired = pd.DataFrame.from_records(paired_rows)
    return {
        "model_summary": pd.DataFrame.from_records(summary_rows),
        "paired_topk": paired,
        "paired_top10": paired.loc[paired["k"] == 10].copy(),
        "patient_complete_top10": pd.DataFrame.from_records(complete_rows),
        "label_metrics": pd.DataFrame.from_records(label_rows),
    }


def write_cohort_tables(
    tables: dict[str, pd.DataFrame],
    *,
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Path]:
    """Write the standard CI outputs as machine-readable TSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        "model_summary": output_dir / f"{prefix}a_model_ci.tsv",
        "paired_topk": output_dir / f"{prefix}b_paired_ci.tsv",
        "patient_complete_top10": output_dir
        / f"{prefix}c_patient_complete_ci.tsv",
        "label_metrics": output_dir / f"{prefix}_label_metrics_ci.tsv",
    }
    for key, path in destinations.items():
        tables[key].to_csv(path, sep="\t", index=False)
    return destinations


def _combine_deoverlap_outputs(
    cohort_tables: dict[str, dict[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_frames = []
    paired_frames = []
    for cohort, tables in cohort_tables.items():
        model_frame = tables["model_summary"].copy()
        model_frame.insert(0, "cohort", cohort)
        model_frames.append(model_frame)
        paired_frame = tables["paired_top10"].copy()
        paired_frame.insert(0, "cohort", cohort)
        paired_frames.append(paired_frame)
    return (
        pd.concat(model_frames, ignore_index=True),
        pd.concat(paired_frames, ignore_index=True),
    )


def run_analysis(
    *,
    iei_s2: str | None,
    ird_summary: str | None,
    output_dir: str | Path,
    n_resamples: int = 10_000,
    seed: int = 42,
    iei_training_jsonl: str | None = None,
    iei_metadata_xlsx: str | None = None,
) -> None:
    """Generate full-cohort and de-overlap CI source tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlap_rows = []
    deoverlap_tables: dict[str, dict[str, pd.DataFrame]] = {}

    if iei_s2:
        iei = load_iei_s2(iei_s2)
        iei_tables = compute_cohort_tables(
            iei, n_resamples=n_resamples, seed=seed
        )
        write_cohort_tables(iei_tables, output_dir=output_dir, prefix="S3")

        if bool(iei_training_jsonl) != bool(iei_metadata_xlsx):
            raise ValueError(
                "iei_training_jsonl and iei_metadata_xlsx must be provided together."
            )
        if iei_training_jsonl and iei_metadata_xlsx:
            overlap = compute_iei_overlap_mask(
                iei,
                training_jsonl=iei_training_jsonl,
                metadata_xlsx=iei_metadata_xlsx,
            )
            iei = iei.assign(training_overlap_any=overlap)
            iei_clean = filter_non_overlapping(iei)
            deoverlap_tables["IEI"] = compute_cohort_tables(
                iei_clean, n_resamples=n_resamples, seed=seed
            )
            overlap_rows.append(
                {
                    "cohort": "IEI",
                    "evaluated_variants": len(iei),
                    "evaluated_patients": iei["patient_id"].nunique(),
                    "overlap_variants": int(overlap.sum()),
                    "overlap_patients": iei.loc[overlap, "patient_id"].nunique(),
                    "deoverlap_variants": len(iei_clean),
                    "deoverlap_patients": iei_clean["patient_id"].nunique(),
                }
            )

    if ird_summary:
        ird = load_ird_summary(ird_summary)
        ird_models = [
            "Base",
            "VIPER",
            "GeneT",
            "Xrare observed",
            "Exomiser",
            "PHIVE",
            "PhenIX",
        ]
        ird_tables = compute_cohort_tables(
            ird,
            n_resamples=n_resamples,
            seed=seed,
            models=ird_models,
        )
        write_cohort_tables(ird_tables, output_dir=output_dir, prefix="S5")
        ird_clean = filter_non_overlapping(ird)
        deoverlap_tables["IRD"] = compute_cohort_tables(
            ird_clean,
            n_resamples=n_resamples,
            seed=seed,
            models=ird_models,
        )
        overlap_rows.append(
            {
                "cohort": "IRD",
                "evaluated_variants": len(ird),
                "evaluated_patients": ird["patient_id"].nunique(),
                "overlap_variants": int(ird["training_overlap_any"].sum()),
                "overlap_patients": ird.loc[
                    ird["training_overlap_any"], "patient_id"
                ].nunique(),
                "deoverlap_variants": len(ird_clean),
                "deoverlap_patients": ird_clean["patient_id"].nunique(),
            }
        )

    if not deoverlap_tables:
        raise ValueError("No cohort was supplied for de-overlap analysis.")
    model_deoverlap, paired_deoverlap = _combine_deoverlap_outputs(
        deoverlap_tables
    )
    model_deoverlap.to_csv(
        output_dir / "S15b_deoverlap_model_ci.tsv", sep="\t", index=False
    )
    paired_deoverlap.to_csv(
        output_dir / "S15c_deoverlap_paired_ci.tsv", sep="\t", index=False
    )
    pd.DataFrame.from_records(overlap_rows).to_csv(
        output_dir / "S15a_overlap_audit.tsv", sep="\t", index=False
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute VIPER Top-k confidence intervals using patient-cluster "
            "percentile bootstrap resampling."
        )
    )
    parser.add_argument("--iei-s2", help="Supplementary table XLSX containing S2.")
    parser.add_argument(
        "--ird-summary", help="IRD summary XLSX containing the Sheet1 evaluation table."
    )
    parser.add_argument(
        "--iei-training-jsonl",
        help="Final IEI training JSONL, required for de-overlap CI.",
    )
    parser.add_argument(
        "--iei-metadata-xlsx",
        help="IEI metadata XLSX with cpra, gene, DNA and protein columns.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if not args.iei_s2 and not args.ird_summary:
        raise SystemExit("Provide at least one of --iei-s2 or --ird-summary.")
    run_analysis(
        iei_s2=args.iei_s2,
        ird_summary=args.ird_summary,
        output_dir=args.output_dir,
        n_resamples=args.n_resamples,
        seed=args.seed,
        iei_training_jsonl=args.iei_training_jsonl,
        iei_metadata_xlsx=args.iei_metadata_xlsx,
    )


if __name__ == "__main__":
    main()
