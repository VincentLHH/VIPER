#!/usr/bin/env python3
"""Audit IRD test-to-training overlap after CPRA-to-HGVSc reconciliation.

The hgvsc column in summary-62S.xlsx is a case-level set, repeated on every
row for a case. This script therefore consumes a separately generated VEP
candidate table and joins it to the row-level CPRA values. It never assigns
HGVSc values by row order.

Expected annotation table columns:
    case_id, cpra, gene, vep_candidates

vep_candidates is a semicolon-separated, normalized list of HGVSc candidates
for the CPRA and expected gene. An optional hgvsc value records the
CPRA-specific VEP result selected for the primary transcript. It is reported
as cpra_hgvsc and sanity-checked against the strict assignment, but is not
used to construct that assignment.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


SUMMARY = Path("data/IRD/summary-62S.xlsx")
TRAINING = Path("data/IRD/Trainset-IRD.jsonl")
ANNOTATION = Path("audit/ird_cpra_vep_candidates_20261020.tsv")
OUTPUT = Path("audit/ird_cpra_hgvsc_row_mapping_20261020.tsv")

TRAINING_PATTERN = re.compile(
    r"Gene symbol:\s*([^\s]+).*?DNA change:\s*([^\n]+)", re.ASCII | re.DOTALL
)


def normalize_hgvsc(value: str) -> str:
    """Normalize an HGVSc token for matching, preserving the c. coordinates."""
    token = value.strip().rstrip("#¤").strip()
    if " p." in token:
        token = token.split(" p.", 1)[0]
    return token.replace(" ", "").upper()


def case_hgvsc_set(value: object) -> list[str]:
    """Parse the newline-delimited case-level HGVSc set."""
    return [
        normalize_hgvsc(line)
        for line in str(value).splitlines()
        if line.strip()
    ]


def parse_training(path: Path) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Return all and positive gene+HGVSc pairs in the IRD training corpus."""
    all_pairs: set[tuple[str, str]] = set()
    positive_pairs: set[tuple[str, str]] = set()

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = "\n".join(
                message.get("content", "") for message in record.get("messages", [])
            )
            match = TRAINING_PATTERN.search(text)
            if not match:
                continue

            gene, hgvsc = match.groups()
            pair = (gene.strip(), normalize_hgvsc(hgvsc))
            all_pairs.add(pair)

            final_label = record["messages"][-1].get("content", "").strip().lower()
            if final_label == "positive":
                positive_pairs.add(pair)

    return all_pairs, positive_pairs


def maximum_partial_assignments(
    edges: dict[int, set[str]],
) -> tuple[int, list[dict[int, str]]]:
    """Enumerate maximum-cardinality injective CPRA-to-HGVSc assignments."""
    row_ids = sorted(edges)
    best: list[dict[int, str]] = []
    best_size = -1

    def visit(
        position: int, used: set[str], assignment: dict[int, str]
    ) -> None:
        nonlocal best, best_size
        if position == len(row_ids):
            size = len(assignment)
            if size > best_size:
                best_size = size
                best = [dict(assignment)]
            elif size == best_size:
                best.append(dict(assignment))
            return

        row_id = row_ids[position]
        # Try assignment first so that the maximum number of rows is retained.
        for hgvsc in sorted(edges[row_id]):
            if hgvsc in used:
                continue
            used.add(hgvsc)
            assignment[row_id] = hgvsc
            visit(position + 1, used, assignment)
            assignment.pop(row_id)
            used.remove(hgvsc)

        # A row without a VEP/set edge remains unresolved.
        visit(position + 1, used, assignment)

    visit(0, set(), {})
    return best_size, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--training", type=Path, default=TRAINING)
    parser.add_argument("--annotation", type=Path, default=ANNOTATION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    summary = pd.read_excel(args.summary, header=[0, 1])
    annotation = pd.read_csv(args.annotation, sep="\t", dtype=str).fillna("")
    all_training, positive_training = parse_training(args.training)

    annotation_key = ["case_id", "cpra"]
    if annotation.duplicated(annotation_key).any():
        duplicates = annotation.loc[
            annotation.duplicated(annotation_key, keep=False), annotation_key
        ]
        raise ValueError(f"Duplicate annotation keys:\n{duplicates.to_string(index=False)}")

    annotation = annotation.set_index(annotation_key)

    case_set_column = (
        ("Mapping_audit", "hgvsc_case_set_original")
        if ("Mapping_audit", "hgvsc_case_set_original") in summary.columns
        else ("Sample_Info", "hgvsc")
    )
    row_records: dict[int, dict[str, object]] = {}
    case_rows: dict[int, list[int]] = defaultdict(list)
    for index, row in summary.iterrows():
        case_id = int(row[("Sample_Info", "case_ID")])
        cpra = str(row[("Sample_Info", "cpra")]).strip()
        gene = str(row[("Sample_Info", "Gene")]).strip()
        sample_set = case_hgvsc_set(row[case_set_column])
        key = (str(case_id), cpra)

        candidates: set[str] = set()
        preassigned = ""
        annotation_status = "available"
        if key in annotation.index:
            annotation_record = annotation.loc[key]
            candidates = {
                normalize_hgvsc(value)
                for value in str(annotation_record.get("vep_candidates", "")).split(";")
                if value.strip()
            }
            preassigned = normalize_hgvsc(
                str(annotation_record.get("hgvsc", ""))
            )
        else:
            annotation_status = "annotation_missing"

        row_records[index] = {
            "case_id": case_id,
            "cpra": cpra,
            "gene": gene,
            "sample_hgvsc_set": sample_set,
            "vep_candidates": candidates,
            "annotation_status": annotation_status,
            "preassigned": preassigned,
            "mapped_hgvsc": "",
            "mapping_status": "",
            "unassigned_case_hgvsc": "",
            "viper_rank": float(row[("VIPER", "rank")]),
        }
        case_rows[case_id].append(index)

    # Assign within each case and require an unambiguous maximum matching.
    for case_id, indices in case_rows.items():
        case_sets = {tuple(row_records[i]["sample_hgvsc_set"]) for i in indices}
        if len(case_sets) != 1:
            raise ValueError(f"Case {case_id} has non-constant case-level HGVSc sets")
        sample_set = set(next(iter(case_sets)))

        edges = {
            i: set(row_records[i]["vep_candidates"]) & sample_set  # type: ignore[arg-type]
            for i in indices
        }
        best_size, assignments = maximum_partial_assignments(edges)
        distinct = {
            tuple(sorted(assignment.items()))
            for assignment in assignments
            if len(assignment) == best_size
        }

        if len(distinct) == 1:
            resolved = dict(assignments[0])
            for i in indices:
                if i in resolved:
                    row_records[i]["mapped_hgvsc"] = resolved[i]
                    row_records[i]["mapping_status"] = "vep+sample-set-unique"
                elif row_records[i]["annotation_status"] == "annotation_missing":
                    row_records[i]["mapping_status"] = "annotation_missing"
                else:
                    row_records[i]["mapping_status"] = "unresolved_no_vep_set_edge"
            assigned_values = set(resolved.values())
            leftover = sample_set - assigned_values
            for i in indices:
                row_records[i]["unassigned_case_hgvsc"] = ";".join(sorted(leftover))
        else:
            for i in indices:
                row_records[i]["mapping_status"] = "unresolved_ambiguous_matching"
                row_records[i]["unassigned_case_hgvsc"] = ";".join(sorted(sample_set))

    # The same CPRA must receive the same HGVSc wherever it occurs.
    by_cpra: dict[str, list[int]] = defaultdict(list)
    for i, record in row_records.items():
        if record["mapped_hgvsc"]:
            by_cpra[str(record["cpra"])].append(i)
    for cpra, indices in by_cpra.items():
        assigned = {str(row_records[i]["mapped_hgvsc"]) for i in indices}
        if len(assigned) > 1:
            for i in indices:
                row_records[i]["mapped_hgvsc"] = ""
                row_records[i]["mapping_status"] = "unresolved_cross_case_inconsistent"

    output_rows: list[dict[str, object]] = []
    for i in summary.index:
        record = row_records[i]
        mapped_hgvsc = str(record["mapped_hgvsc"])
        pair = (
            (str(record["gene"]), mapped_hgvsc)
            if mapped_hgvsc
            else None
        )

        preassigned = str(record["preassigned"])
        if preassigned and mapped_hgvsc and preassigned != mapped_hgvsc:
            raise ValueError(
                f"Preassigned HGVSc differs from derived assignment for "
                f"case {record['case_id']} / {record['cpra']}: "
                f"{preassigned} versus {mapped_hgvsc}"
            )

        output_rows.append(
            {
                "case_id": record["case_id"],
                "cpra": record["cpra"],
                "gene": record["gene"],
                "sample_hgvsc_set": ";".join(record["sample_hgvsc_set"]),
                "cpra_hgvsc": preassigned,
                "mapped_hgvsc": mapped_hgvsc,
                "mapping_status": record["mapping_status"],
                "unassigned_case_hgvsc": record["unassigned_case_hgvsc"],
                "vep_candidates": ";".join(sorted(record["vep_candidates"])),
                "matches_any_training": bool(pair and pair in all_training),
                "matches_positive_training": bool(pair and pair in positive_training),
                "viper_rank": record["viper_rank"],
            }
        )

    output = pd.DataFrame(output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False)

    mapped = output["mapped_hgvsc"].ne("")
    overlaps = output["matches_any_training"]
    positive_overlaps = output["matches_positive_training"]
    deoverlap = output.loc[~overlaps]

    print(f"Rows: {len(output)}")
    print(f"Cases: {output['case_id'].nunique()}")
    print(f"Strictly mapped rows: {mapped.sum()}/{len(output)}")
    print(
        "Any-label training overlap: "
        f"{overlaps.sum()}/{len(output)} variants; "
        f"{output.loc[overlaps, 'cpra'].nunique()} unique CPRAs; "
        f"{output.loc[overlaps, 'case_id'].nunique()} patients"
    )
    print(
        "Positive training overlap: "
        f"{positive_overlaps.sum()}/{len(output)} variants; "
        f"{output.loc[positive_overlaps, 'cpra'].nunique()} unique CPRAs; "
        f"{output.loc[positive_overlaps, 'case_id'].nunique()} patients"
    )
    print(
        "De-overlap subset: "
        f"{len(deoverlap)} variants; {deoverlap['case_id'].nunique()} patients; "
        f"VIPER Top-10 {(deoverlap['viper_rank'] <= 10).mean():.2%}"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
