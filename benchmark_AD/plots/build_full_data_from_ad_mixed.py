"""Build wide per-seed prediction CSVs from AD_mixed_csv_download.

`method_overall_aucroc.py` expects one CSV per seed with a shared label column
and one `pred_score_<method>` column per detector. The downloaded files are
stored one method per directory, so this script validates row alignment and
merges them into the expected layout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR / "AD_mixed_csv_download"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "full_data_all_plus_semi_sup"
SEEDS = [1, 2, 3, 4, 5]
DATASET = "EscapeRoom_v3_all_mixed"


def method_from_dir(path: Path, seed: int) -> str:
    suffix = f"_{DATASET}_seed{seed}"
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected directory name: {name}")
    return name[: -len(suffix)]


def build_seed(source_dir: Path, output_dir: Path, seed: int) -> pd.DataFrame:
    files = sorted(source_dir.glob(f"*_{DATASET}_seed{seed}/predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction files found for seed {seed} in {source_dir}")

    merged: pd.DataFrame | None = None
    key_cols = ["test_id", "gt_label", "trace_type", "domain"]
    methods: list[str] = []

    for path in files:
        method = method_from_dir(path.parent, seed)
        df = pd.read_csv(path)
        missing = [col for col in [*key_cols, "pred_label", "pred_score"] if col not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        current_keys = df[key_cols].reset_index(drop=True)
        if merged is None:
            merged = current_keys.rename(columns={"gt_label": "step_label"})
        else:
            expected_keys = merged[["test_id", "step_label", "trace_type", "domain"]].rename(
                columns={"step_label": "gt_label"}
            )
            if not current_keys.equals(expected_keys.reset_index(drop=True)):
                raise ValueError(f"Rows are not aligned for {path}")

        merged[f"pred_label_{method}"] = df["pred_label"].to_numpy()
        merged[f"pred_score_{method}"] = df["pred_score"].to_numpy()
        methods.append(method)

    assert merged is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"full_test_data_with_preds_seed{seed}.csv"
    merged.to_csv(out_path, index=False)
    print(f"seed {seed}: {len(merged)} rows, {len(methods)} methods -> {out_path}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = parser.parse_args()

    for seed in args.seeds:
        build_seed(args.source_dir, args.output_dir, seed)


if __name__ == "__main__":
    main()
