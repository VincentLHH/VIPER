#!/usr/bin/env python3
"""Reproduce the DeepSeek-R1 prompt-format pilot from Prompteng_info_1v1.xlsx."""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


INPUT = Path("data/PromptEng/Prompteng_info_1v1.xlsx")
OUTPUT = Path("audit/prompteng_category_summary_20261020.tsv")
CATEGORY_ORDER = [
    "zeroshot",
    "fewshot",
    "fewshot+CoT",
    "fewshot+CoT+RAG",
    "fewshot+CoT+RAG+Align",
]


def load_backslash_zip_excel(path: Path) -> pd.DataFrame:
    """Read an xlsx whose internal paths incorrectly use backslashes."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx") as handle:
        with ZipFile(path) as source, ZipFile(handle.name, "w") as target:
            for info in source.infolist():
                target.writestr(info.filename.replace("\\", "/"), source.read(info))
        return pd.read_excel(handle.name)


def normalize_binary(value: object) -> str:
    token = str(value).strip().lower().split()[0]
    if token not in {"positive", "negative"}:
        raise ValueError(f"Unexpected classification: {value!r}")
    return token


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - half_width, center + half_width


def exact_mcnemar(new_correct: np.ndarray, old_correct: np.ndarray) -> tuple[int, int, float]:
    only_new = int(((old_correct == 0) & (new_correct == 1)).sum())
    only_old = int(((old_correct == 1) & (new_correct == 0)).sum())
    discordant = only_new + only_old
    if discordant == 0:
        return only_new, only_old, 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(0, min(only_new, only_old) + 1)
    ) / (2**discordant)
    return only_new, only_old, min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = load_backslash_zip_excel(args.input)
    frame["correct"] = (
        frame["results"].map(normalize_binary)
        == frame["expect"].map(normalize_binary)
    ).astype(int)

    if set(frame["category"]) != set(CATEGORY_ORDER):
        raise ValueError("Unexpected prompt categories")
    counts = frame["category"].value_counts()
    if counts.nunique() != 1:
        raise ValueError(f"Unbalanced prompt categories: {counts.to_dict()}")

    reference = frame.loc[frame["category"] == CATEGORY_ORDER[0]].reset_index(drop=True)
    if not reference["expect"].map(normalize_binary).isin(
        {"positive", "negative"}
    ).all():
        raise ValueError("Unexpected reference labels")
    if reference["expect"].map(normalize_binary).value_counts().nunique() != 1:
        raise ValueError("The reference prompt is not label-balanced")

    rng = np.random.default_rng(args.seed)
    total = len(reference)
    bootstrap_indices = [
        rng.integers(0, total, total) for _ in range(args.bootstrap)
    ]

    rows: list[dict[str, object]] = []
    for category in CATEGORY_ORDER:
        current = frame.loc[frame["category"] == category].reset_index(drop=True)
        if not current["cpra"].equals(reference["cpra"]):
            raise ValueError(f"CPRA order differs for {category}")
        if not current["expect"].map(normalize_binary).equals(
            reference["expect"].map(normalize_binary)
        ):
            raise ValueError(f"Expected labels differ for {category}")

        current_ok = current["correct"].to_numpy()
        reference_ok = reference["correct"].to_numpy()
        successes = int(current_ok.sum())
        low, high = wilson_interval(successes, total)
        only_new, only_old, p_value = exact_mcnemar(current_ok, reference_ok)
        difference = current_ok - reference_ok
        bootstrapped = np.array(
            [difference[index].mean() for index in bootstrap_indices]
        )
        diff_low, diff_high = np.percentile(bootstrapped, [2.5, 97.5])

        rows.append(
            {
                "category": category,
                "correct": successes,
                "total": total,
                "accuracy": successes / total,
                "wilson_95_low": low,
                "wilson_95_high": high,
                "paired_gain_vs_zeroshot": only_new,
                "paired_loss_vs_zeroshot": only_old,
                "paired_difference": difference.mean(),
                "paired_95_low": diff_low,
                "paired_95_high": diff_high,
                "exact_mcnemar_p": p_value,
            }
        )

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False)
    print(output.to_string(index=False))
    print(f"Unique CPRAs: {frame['cpra'].nunique()}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
