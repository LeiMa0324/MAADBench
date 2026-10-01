"""
fetch_number_gt.py — Extract LiveCodeBench-Execution questions whose ground
truth (the executed `output`) is a plain number.

The LiveCodeBench "Execution" task gives a program + a call and asks for the
program's output. Here we keep only the samples whose expected output parses to
an int or a float (bools excluded), and write them to a JSONL file.

Usage
-----
    python benchmark_core/domains/livecodebench/scripts/fetch_number_gt.py

The script writes normalized JSONL data into the sibling ``data`` directory.
"""

from __future__ import annotations

import argparse
import ast
import json
from datetime import date, datetime
from pathlib import Path

from datasets import load_dataset

DATASET = "livecodebench/execution-v2"
SPLIT = "test"


def as_number(s):
    """Return the numeric value of an output string, or None if not a number."""
    s = (s or "").strip()
    try:
        v = ast.literal_eval(s)
    except Exception:
        try:
            v = float(s)
        except Exception:
            return None
    if isinstance(v, bool):          # bool is a subclass of int — exclude it
        return None
    if isinstance(v, (int, float)):
        return v
    return None


def _jsonable(v):
    """Make a value JSON-serialisable (datetimes -> ISO strings)."""
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return v


def main():
    ap = argparse.ArgumentParser(description="Extract number-gt LiveCodeBench-Execution questions.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output JSONL path (default: execution_number_gt_<difficulty>.jsonl next to this script).")
    ap.add_argument("--dataset", type=str, default=DATASET)
    ap.add_argument("--split", type=str, default=SPLIT)
    ap.add_argument("--difficulty", nargs="+", default=["medium", "hard"],
                    help="Keep these difficulties (easy/medium/hard); use 'all' for no filter. "
                         "Default: medium hard.")
    args = ap.parse_args()

    keep_all = "all" in args.difficulty
    keep_diffs = set(args.difficulty)

    if args.out:
        out_path = Path(args.out)
    else:
        order = ["easy", "medium", "hard"]
        tag = "all" if keep_all else "_".join(d for d in order if d in keep_diffs)
        out_path = Path(__file__).resolve().parent.parent / "data" / f"execution_number_gt_{tag}.jsonl"

    print(f"[fetch] loading {args.dataset} [{args.split}] ...")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"[fetch] {len(ds)} total samples; filtering numeric ground truth "
          f"(difficulty={'all' if keep_all else sorted(keep_diffs)}) ...")

    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            if not keep_all and ex.get("difficulty") not in keep_diffs:
                continue
            gt = as_number(ex.get("output"))
            if gt is None:
                continue
            record = {k: _jsonable(v) for k, v in ex.items()}
            record["gt_number"] = gt
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[fetch] wrote {kept}/{len(ds)} number-gt questions -> {out_path}")


if __name__ == "__main__":
    main()
