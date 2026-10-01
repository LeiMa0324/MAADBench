"""Plot failure-mode distribution by model for the new output/traces layout."""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from benchmark_core.MAST_annotator.FM_annotate import (
    FM_COL,
    classify_unit_tool_call,
)


RUN_RE = re.compile(
    r"^run_(?P<timestamp>\d{8}_\d{6})_"
    r"(?P<domain>gsm-hard|livecodebench)_"
    r"(?P<tool_mode>tool|no_tool)_"
    r"(?P<model>.+)_"
    r"(?P<temperature>-?\d+(?:\.\d+)?)$"
)

MODEL_ORDER = [
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "deepseek-reasoner",
    "gpt-4.1",
    "gpt-5.4",
]

MODEL_DISPLAY = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "deepseek-reasoner": "DeepSeek-R1",
    "gpt-4.1": "GPT-4.1",
    "gpt-5.4": "GPT-5.4",
}

FM_CODES = [
    "FM-1.1",
    "FM-2.3",
    "FM-2.5",
    "FM-2.6",
    "FM-3.2",
    "FM-3.3",
    "FM-4.1",
    "FM-4.2",
    "FM-4.3",
]
BAR_NUMS = [f"({idx})" for idx in range(1, len(FM_CODES) + 1)]

FM_LEGEND = {
    "FM-1.1": "Disobey task spec",
    "FM-2.3": "Task derailment",
    "FM-2.5": "Ignored input",
    "FM-2.6": "Reason-action mism.",
    "FM-3.2": "Incomplete verif.",
    "FM-3.3": "Incorrect verif.",
    "FM-4.1": "Deception suscept.",
    "FM-4.2": "Incorrect reasoning",
    "FM-4.3": "Error propagation",
}

FM_DISPLAY = {
    "FM-4.3": "FM-5",
}

MAST_COLOR = "#DBC3D6"
OURS_COLOR = "#B887AD"
MAST_FMS = {"FM-1.1", "FM-2.3", "FM-2.5", "FM-2.6", "FM-3.2", "FM-3.3"}
FM_COLORS = {code: MAST_COLOR if code in MAST_FMS else OURS_COLOR for code in FM_CODES}

BASE_FONT = 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw failure-mode distribution by model from output/traces."
    )
    parser.add_argument("--traces-root", type=Path, default=REPO_ROOT / "output/traces")
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "FM_by_model.pdf",
        help="Output figure path. Supported suffixes: .pdf, .png.",
    )
    parser.add_argument("--models", nargs="*", default=MODEL_ORDER)
    parser.add_argument(
        "--annotate-missing",
        action="store_true",
        help="Run the deterministic MAST annotator for summaries missing taxonomy columns.",
    )
    return parser.parse_args()


def discover_summary_paths(traces_root: Path) -> list[tuple[str, Path]]:
    summaries: list[tuple[str, Path]] = []
    for path in sorted(traces_root.rglob("run_*")):
        if not path.is_dir():
            continue
        match = RUN_RE.match(path.name)
        if not match:
            continue
        summary = path / "summary.csv"
        if not summary.exists():
            continue
        run_log = path / "run.log"
        if run_log.exists() and "KeyboardInterrupt" in run_log.read_text(errors="ignore"):
            continue
        summaries.append((match.group("model"), summary))
    return summaries


def annotate_missing_summaries(traces_root: Path) -> None:
    """Populate post-hoc FM taxonomy columns before plotting when absent."""
    from benchmark_core.MAST_annotator.FM_annotate import annotate

    run_dirs: list[Path] = []
    for _model, summary_path in discover_summary_paths(traces_root):
        with summary_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            if FM_COL not in fieldnames:
                run_dirs.append(summary_path.parent)

    if not run_dirs:
        print("All summaries already contain Failure_mode(MAST).")
        return

    print(f"Annotating {len(run_dirs)} summaries with post-hoc MAST taxonomy...")
    for run_dir in run_dirs:
        print(f"FM Annotation: {run_dir.relative_to(REPO_ROOT)}")
        annotate(run_dir)


def mapped_fm(row: dict[str, str]) -> str:
    fm = (row.get(FM_COL) or row.get("fm_id") or "").strip()
    if fm:
        return fm
    if (
        row.get("action_name") == "UNIT_TOOL_CALL"
        and row.get("action_status") == "wrong"
    ):
        return classify_unit_tool_call(row.get("error_type", ""), row.get("desc", ""))
    return ""


def collect_fm_percentages(
    traces_root: Path,
    models: list[str],
) -> dict[str, dict[str, float]]:
    selected_models = set(models)
    counters = {model: Counter() for model in models}
    for model, summary_path in discover_summary_paths(traces_root):
        if model not in selected_models:
            continue
        with summary_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                fm = mapped_fm(row)
                if fm in FM_CODES:
                    counters[model][fm] += 1

    percentages: dict[str, dict[str, float]] = {}
    for model in models:
        total = sum(counters[model].values())
        if total == 0:
            continue
        percentages[model] = {
            code: counters[model].get(code, 0) / total * 100
            for code in FM_CODES
        }
    return percentages


def write_csv(model_fm_pcts: dict[str, dict[str, float]], output_path: Path) -> None:
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "llm", *FM_CODES])
        writer.writeheader()
        for model, pcts in model_fm_pcts.items():
            row = {"model": model, "llm": MODEL_DISPLAY.get(model, model)}
            row.update({code: pcts.get(code, 0) for code in FM_CODES})
            writer.writerow(row)


def fm_label(idx: int) -> str:
    code = FM_CODES[idx]
    return f"{BAR_NUMS[idx]} {FM_DISPLAY.get(code, code)}: {FM_LEGEND[code]}"


def plot(model_fm_pcts: dict[str, dict[str, float]], models: list[str], output: Path) -> None:
    mpl.rcParams["font.family"] = "Trebuchet MS"
    mpl.rcParams["font.weight"] = "normal"

    present_models = [model for model in models if model in model_fm_pcts]
    n_models = len(present_models)
    n_fm = len(FM_CODES)
    x = np.arange(n_fm)
    bar_width = 0.65

    fig, axes = plt.subplots(1, n_models, figsize=(3.8 * n_models, 4.0), sharey=True)
    if n_models == 1:
        axes = [axes]

    max_value = max(
        [value for pcts in model_fm_pcts.values() for value in pcts.values()] or [0]
    )
    y_max = max(48, min(100, np.ceil((max_value + 5) / 5) * 5))

    for idx, model in enumerate(present_models):
        ax = axes[idx]
        vals = [model_fm_pcts[model].get(code, 0) for code in FM_CODES]
        colors = [FM_COLORS[code] for code in FM_CODES]
        ax.bar(x, vals, width=bar_width, color=colors, edgecolor="black", linewidth=0.6)

        for bar_idx, value in enumerate(vals):
            if value > 0.5:
                ax.text(
                    bar_idx,
                    value + 0.8,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=BASE_FONT - 5,
                )
            else:
                ax.text(
                    bar_idx,
                    0.8,
                    "0",
                    ha="center",
                    va="bottom",
                    fontsize=BASE_FONT - 5,
                    color="gray",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(BAR_NUMS[:n_fm], fontsize=BASE_FONT - 4, ha="center")
        ax.set_ylim(0, y_max)
        ax.set_title(
            MODEL_DISPLAY.get(model, model),
            fontsize=BASE_FONT + 2,
            fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=BASE_FONT - 3)

        sep_x = len([code for code in FM_CODES if code in MAST_FMS]) - 0.5
        ax.axvline(x=sep_x, color="lightgray", linewidth=1.5, linestyle="-")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.set_ylabel("% of Failures", fontsize=BASE_FONT)

    mast_idx = [idx for idx, code in enumerate(FM_CODES) if code in MAST_FMS]
    ours_idx = [idx for idx, code in enumerate(FM_CODES) if code not in MAST_FMS]
    mast_col1 = "\n".join(fm_label(idx) for idx in mast_idx[:3])
    mast_col2 = "\n".join(fm_label(idx) for idx in mast_idx[3:])
    ours_col3 = "\n".join(fm_label(idx) for idx in ours_idx)
    fs_leg = BASE_FONT

    fig.text(
        0.38,
        0.05,
        "MAST Failure Modes",
        ha="center",
        va="top",
        fontsize=fs_leg + 1,
        fontweight="bold",
        color="0.3",
    )
    fig.text(0.10, -0.07, mast_col1, ha="left", va="top", fontsize=fs_leg, color="0.3", linespacing=1.7)
    fig.text(0.35, -0.07, mast_col2, ha="left", va="top", fontsize=fs_leg, color="0.3", linespacing=1.7)

    fig.text(
        0.76,
        0.05,
        "Our Failure Modes",
        ha="center",
        va="top",
        fontsize=fs_leg + 1,
        fontweight="bold",
        color="0.3",
    )
    fig.text(0.66, -0.07, ours_col3, ha="left", va="top", fontsize=fs_leg, color="0.3", linespacing=1.7)

    legend_elements = [
        Patch(facecolor=MAST_COLOR, edgecolor="black", linewidth=0.6, label="MAST"),
        Patch(facecolor=OURS_COLOR, edgecolor="black", linewidth=0.6, label="Ours"),
    ]
    axes[0].legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=BASE_FONT - 3,
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgray",
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_csv(model_fm_pcts, output)
    print(f"Saved to {output}")
    print(f"Saved to {output.with_suffix('.csv')}")


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() not in {".pdf", ".png"}:
        raise SystemExit("Only .pdf and .png outputs are supported.")
    if args.annotate_missing:
        annotate_missing_summaries(args.traces_root)
    pcts = collect_fm_percentages(args.traces_root, args.models)
    plot(pcts, args.models, args.output)


if __name__ == "__main__":
    main()
