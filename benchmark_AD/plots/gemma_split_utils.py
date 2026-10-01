"""Helpers for Gemma split prediction files.

Gemma zero-shot predictions are stored by domain and tool mode instead of in
the main all-plus CSVs. These helpers load the four split files for one seed
and pool them into one DataFrame with only the columns requested by a plot.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


GEMMA_ZERO_SHOT_DATA_DIR = Path(__file__).parent / "data" / "gemma_zeroshot_full_data"
GEMMA_FEW_SHOT_DATA_DIR = (
    Path(__file__).parent / "data" / "full_data_all_plus_semi_sup_with_gemma"
)

GEMMA_SPLITS = [
    ("EscapeRoom_v3_gsm_hard", "no_tool"),
    ("EscapeRoom_v3_gsm_hard", "tool"),
    ("EscapeRoom_v3_live_code_bench", "no_tool"),
    ("EscapeRoom_v3_live_code_bench", "tool"),
]


def zero_shot_methods(llm_csv_suffix: dict[str, str]) -> dict[str, str]:
    return {
        key: suffix
        for key, suffix in llm_csv_suffix.items()
        if key.endswith("_zs")
    }


def few_shot_methods(llm_csv_suffix: dict[str, str]) -> dict[str, str]:
    return {
        key: suffix
        for key, suffix in llm_csv_suffix.items()
        if key.endswith("_fs")
    }


def load_gemma_few_shot_predictions(
    *,
    seed: int,
    method_suffixes: dict[str, str],
    prefix: str,
    base_cols: list[str],
) -> pd.DataFrame:
    """Load Gemma few-shot predictions merged into the with_gemma full data."""
    path = GEMMA_FEW_SHOT_DATA_DIR / f"full_test_data_with_preds_seed{seed}.csv"
    wanted = [f"{prefix}{suffix}" for suffix in method_suffixes.values()]
    if not path.exists():
        return pd.DataFrame(columns=base_cols + wanted)

    cols = list(pd.read_csv(path, nrows=0).columns)
    usecols = [col for col in base_cols if col in cols]
    usecols += [col for col in wanted if col in cols]
    if len(usecols) <= len([col for col in base_cols if col in cols]):
        return pd.DataFrame(columns=base_cols + wanted)
    return pd.read_csv(path, usecols=usecols)


def load_gemma_zero_shot_predictions(
    *,
    seed: int,
    method_suffixes: dict[str, str],
    prefix: str,
    base_cols: list[str],
) -> pd.DataFrame:
    """Load pooled Gemma zero-shot predictions for one seed.

    ``prefix`` should be ``"pred_score_"`` for AUC plots or
    ``"pred_label_"`` for recall plots. Column names are preserved exactly as
    in the CSVs so existing ``csv_score_col`` / ``csv_label_col`` helpers keep
    working.
    """
    wanted = [f"{prefix}{suffix}" for suffix in method_suffixes.values()]
    parts: list[pd.DataFrame] = []

    for dataset, trace_type in GEMMA_SPLITS:
        path = (
            GEMMA_ZERO_SHOT_DATA_DIR
            / dataset
            / trace_type
            / f"full_test_data_with_preds_seed{seed}.csv"
        )
        if not path.exists():
            continue
        cols = list(pd.read_csv(path, nrows=0).columns)
        usecols = [col for col in base_cols if col in cols]
        usecols += [col for col in wanted if col in cols]
        if len(usecols) <= len([col for col in base_cols if col in cols]):
            continue
        parts.append(pd.read_csv(path, usecols=usecols))

    if not parts:
        return pd.DataFrame(columns=base_cols + wanted)
    return pd.concat(parts, ignore_index=True)
