"""Per-family recall at three propagation roles — Origin, Middle,
Downstream — drawn as four side-by-side bar subplots (one per family).
Each role has two bars: in-pattern recall (solid) and the apples-to-apples
global reference recall (hatched).

Roles:
  - Origin     = the most upstream failed action in that pattern
  - Downstream = the most downstream failed action in that pattern
  - Middle     = anything in between (patterns of ≥3 failed actions only)

Bars are computed by:
  - In-pattern   = mean recall per (pattern, position, method) in this
                   (family, role), weighted by the pattern's puzzle count.
  - Global ref   = mean global recall on the SAME (action_pos, method)
                   cells, weighted the same way.
"""
import sys
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".mplconfig"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["font.family"] = "Trebuchet MS"
mpl.rcParams["font.weight"] = "normal"
BASE_FONT = 16
mpl.rcParams["font.size"] = BASE_FONT

CSV_PATH = Path(__file__).parent / "method_recall_by_propagation_pattern.csv"
DATA_DIR = Path(__file__).parent / "data" / "full_data_all_plus_semi_sup_with_gemma"
SEEDS = [1, 2, 3, 4, 5]
PLOT_PATH = Path(__file__).with_suffix(".pdf")

CHAIN = ["SELECT_PUZZLE", "OBSERVE_CLUE", "SOLVE_CLUE",
         "OBSERVE_ITEM", "UNIT_TOOL_CALL", "APPLY_DELTA", "OBSERVE_PUZZLE"]
CHAIN_SHORT = ["SP", "OC", "SC", "OI", "TC", "AD", "OP"]
CHAIN_POS = {a: i for i, a in enumerate(CHAIN)}
ACTION_MERGE = {"OBSERVE_PLANNED_CLUE": "OBSERVE_CLUE"}

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from methods_meta import (
    LLM_CSV_SUFFIX, METHOD_SHORT, METHOD_CATEGORY, CATEGORY_ORDER,
    csv_label_col, ordered_methods,
)
from gemma_split_utils import load_gemma_zero_shot_predictions, zero_shot_methods

ROLES = ["Origin", "Middle", "Downstream"]
ROLE_DISPLAY = {"Origin": "Origin", "Middle": "Mid", "Downstream": "Down"}

# Single teal color for both lines; the two are distinguished by line
# style (solid vs dashed) and marker shape (circle vs square).
LINE_COLOR = "#1D7069"         # dark teal
IN_PATTERN_MARKER = "o"        # filled circle — in-propagation
GLOBAL_REF_MARKER = "s"        # filled square — in-isolation


def _action_type(aid: str) -> str:
    text = str(aid)
    if "APPLY_DELTA_TOOL_CALL" in text:
        return "UNIT_TOOL_CALL"
    for action in (
        "OBSERVE_PLANNED_CLUE",
        "UNIT_TOOL_CALL",
        "SELECT_PUZZLE",
        "OBSERVE_CLUE",
        "SOLVE_CLUE",
        "OBSERVE_ITEM",
        "APPLY_DELTA",
        "OBSERVE_PUZZLE",
    ):
        if action in text:
            return ACTION_MERGE.get(action, action)
    return ACTION_MERGE.get(text, text)


def methods_in_label_cols(cols: list[str]) -> set[str]:
    return {c[len("pred_label_"):] for c in cols if c.startswith("pred_label_")}


def discover_methods(seed_headers: dict[int, list[str]]) -> list[str]:
    label_sets = [methods_in_label_cols(cols) for cols in seed_headers.values()]
    common_label = set.intersection(*label_sets)
    any_label = set().union(*label_sets)
    raw_llm_cols = set(LLM_CSV_SUFFIX.values())
    csv_methods = sorted(
        m for m in common_label
        if not m.endswith("_polluted")
        and m != "random"
        and m not in raw_llm_cols
    )
    llm_methods = sorted(
        k for k, suf in LLM_CSV_SUFFIX.items() if suf in any_label
    )
    return ordered_methods(csv_methods + llm_methods)


def load_seed_frames(methods: list[str],
                     seed_headers: dict[int, list[str]]) -> list[pd.DataFrame]:
    base_cols = [
        "model_name",
        "trace_id",
        "room_id",
        "puzzle_id",
        "action_id",
        "step_label",
    ]
    gemma_zs_methods = {
        k: v for k, v in zero_shot_methods(LLM_CSV_SUFFIX).items()
        if k in methods
    }
    seed_dfs: list[pd.DataFrame] = []
    for s in SEEDS:
        cols = set(seed_headers[s])
        label_cols = [
            csv_label_col(m) for m in methods
            if csv_label_col(m) in cols
        ]
        usecols = [c for c in base_cols if c in cols] + label_cols
        df = pd.read_csv(DATA_DIR / f"full_test_data_with_preds_seed{s}.csv",
                         usecols=usecols)
        df = df.copy()
        df["seed"] = s
        df["action_type"] = df["action_id"].map(_action_type)
        df = df[df["action_type"].isin(CHAIN_POS)].copy()
        df["action_pos"] = df["action_type"].map(CHAIN_POS)
        df["step_label"] = df["step_label"].astype(int)

        gemma_df = load_gemma_zero_shot_predictions(
            seed=s,
            method_suffixes=gemma_zs_methods,
            prefix="pred_label_",
            base_cols=base_cols,
        )
        if not gemma_df.empty:
            gemma_df = gemma_df.copy()
            gemma_df["seed"] = s
            gemma_df["action_type"] = gemma_df["action_id"].map(_action_type)
            gemma_df = gemma_df[gemma_df["action_type"].isin(CHAIN_POS)].copy()
            gemma_df["action_pos"] = gemma_df["action_type"].map(CHAIN_POS)
            gemma_df["step_label"] = gemma_df["step_label"].astype(int)
            df = pd.concat([df, gemma_df], ignore_index=True)
        seed_dfs.append(df)
    return seed_dfs


def build_recall_by_pattern(methods: list[str],
                            seed_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    big = pd.concat(seed_dfs, ignore_index=True)

    grp_cols = ["seed", "model_name", "trace_id", "room_id", "puzzle_id"]
    anom = big[big["step_label"] == 1]
    pattern_per_puzzle = (
        anom.groupby(grp_cols)["action_pos"]
            .apply(lambda s: tuple(sorted(set(s))))
    )
    multi = pattern_per_puzzle[pattern_per_puzzle.map(len) >= 2]
    if multi.empty:
        return pd.DataFrame()

    pattern_groups: dict[tuple[int, ...], list[tuple]] = defaultdict(list)
    for key, pat in multi.items():
        pattern_groups[pat].append(key)

    sorted_patterns = sorted(
        pattern_groups.items(),
        key=lambda kv: (-len(kv[1]), len(kv[0]), kv[0]),
    )
    big_idx = big.set_index(grp_cols).sort_index()
    records: list[dict[str, object]] = []

    for pattern, keys in sorted_patterns:
        sub_all = big_idx[big_idx.index.isin(set(keys))]
        n_puzzles = len(keys)
        pattern_str = "{" + ",".join(CHAIN_SHORT[p] for p in pattern) + "}"
        for pos in pattern:
            sub_pos = sub_all[
                (sub_all["action_pos"] == pos) &
                (sub_all["step_label"] == 1)
            ]
            if sub_pos.empty:
                continue
            for m in methods:
                col = csv_label_col(m)
                if col not in sub_pos.columns:
                    continue
                col_vals = pd.to_numeric(sub_pos[col], errors="coerce")
                valid = col_vals.notna()
                n = int(valid.sum())
                if n == 0:
                    continue
                tp = int((col_vals[valid] == 1).sum())
                records.append({
                    "pattern": pattern_str,
                    "n_puzzles": n_puzzles,
                    "position": pos,
                    "action": CHAIN_SHORT[pos],
                    "method": m,
                    "n_anomalies": n,
                    "tp": tp,
                    "recall": tp / n,
                })
    return pd.DataFrame(records)


def compute_global_method_recall(methods: list[str]
                                 ) -> dict[str, dict[int, float]]:
    """Per-(method, action_pos) recall over ALL anomalous steps."""
    seed_headers = {
        s: list(pd.read_csv(DATA_DIR / f"full_test_data_with_preds_seed{s}.csv",
                            nrows=0).columns)
        for s in SEEDS
    }
    seed_dfs = load_seed_frames(methods, seed_headers)
    big = pd.concat(seed_dfs, ignore_index=True)
    anom = big[big["step_label"] == 1]

    out: dict[str, dict[int, float]] = {m: {} for m in methods}
    for m in methods:
        col = csv_label_col(m)
        if col not in anom.columns:
            continue
        for pos in range(len(CHAIN)):
            sub = anom[anom["action_pos"] == pos]
            if sub.empty:
                continue
            col_vals = pd.to_numeric(sub[col], errors="coerce")
            valid = col_vals.notna()
            n = int(valid.sum())
            if n == 0:
                continue
            tp = int((col_vals[valid] == 1).sum())
            out[m][pos] = tp / n
    return out


def assign_role(positions_in_pattern: list[int], pos: int) -> str:
    sorted_pos = sorted(positions_in_pattern)
    if pos == sorted_pos[0]:
        return "Origin"
    if pos == sorted_pos[-1]:
        return "Downstream"
    return "Middle"


def main() -> None:
    seed_headers = {
        s: list(pd.read_csv(DATA_DIR / f"full_test_data_with_preds_seed{s}.csv",
                            nrows=0).columns)
        for s in SEEDS
    }
    methods = discover_methods(seed_headers)
    seed_dfs = load_seed_frames(methods, seed_headers)
    df = build_recall_by_pattern(methods, seed_dfs)
    df.to_csv(CSV_PATH, index=False)
    if df.empty:
        print("No multi-anomaly puzzles found.")
        return

    methods = ordered_methods([m for m in methods if m in set(df["method"])])
    family_methods: dict[str, list[str]] = {
        cat: [m for m in methods if METHOD_CATEGORY.get(m) == cat]
        for cat in CATEGORY_ORDER
    }

    global_recall = compute_global_method_recall(methods)

    # Pattern → sorted list of failed positions.
    pattern_positions: dict[str, list[int]] = {}
    for patt, sub in df.groupby("pattern"):
        pattern_positions[patt] = sorted(sub["position"].unique())

    df = df.copy()
    df["role"] = df.apply(
        lambda r: assign_role(pattern_positions[r["pattern"]], r["position"]),
        axis=1,
    )
    df["global_recall"] = df.apply(
        lambda r: global_recall.get(r["method"], {}).get(r["position"], np.nan),
        axis=1,
    )

    # Aggregate: weighted mean recall and global-ref recall per (family, role).
    in_pat = pd.DataFrame(index=CATEGORY_ORDER, columns=ROLES, dtype=float)
    glob = pd.DataFrame(index=CATEGORY_ORDER, columns=ROLES, dtype=float)
    counts = pd.DataFrame(index=CATEGORY_ORDER, columns=ROLES, dtype=int)
    for cat in CATEGORY_ORDER:
        ms = family_methods[cat]
        if not ms:
            continue
        cat_df = df[df["method"].isin(ms)]
        for role in ROLES:
            sub = cat_df[cat_df["role"] == role].dropna(
                subset=["recall", "global_recall"])
            counts.at[cat, role] = len(sub)
            if sub.empty:
                continue
            w = sub["n_puzzles"].astype(float)
            in_pat.at[cat, role] = float(np.average(sub["recall"], weights=w))
            glob.at[cat, role] = float(np.average(sub["global_recall"],
                                                   weights=w))

    # ── Plot: 2×2 grid of subplots, one per family. ────────────────────
    families_present = [c for c in CATEGORY_ORDER if family_methods[c]]
    n_fam = len(families_present)
    n_cols = 2
    n_rows = (n_fam + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 2.2 * n_rows),
                             sharey='row', sharex=True,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.2})
    axes_list = list(np.atleast_1d(axes).flatten())

    x = np.arange(len(ROLES), dtype=float)

    for ax, cat in zip(axes_list, families_present):
        in_y   = [in_pat.at[cat, role] for role in ROLES]
        glob_y = [glob.at[cat, role]   for role in ROLES]
        ax.plot(x, in_y,
                color=LINE_COLOR,
                marker=IN_PATTERN_MARKER, markersize=10,
                markeredgecolor="black", markeredgewidth=0.6,
                linewidth=2.4, label="In-propagation", zorder=3)
        ax.plot(x, glob_y,
                color=LINE_COLOR,
                marker=GLOBAL_REF_MARKER, markersize=10,
                markeredgecolor="black", markeredgewidth=0.6,
                linewidth=2.4, linestyle="--", label="In-isolation",
                zorder=3)

        # On each in-propagation marker, label the delta vs in-isolation
        # with an explicit sign. Positive deltas sit above the marker,
        # negative deltas below — so the label never overlaps the other line.
        for xi, v_in, v_g in zip(x, in_y, glob_y):
            if np.isnan(v_in) or np.isnan(v_g):
                continue
            delta = v_in - v_g
            sign = "+" if delta >= 0 else "−"
            label = f"{sign}{abs(delta):.2f}"
            if delta >= 0:
                ax.text(xi, v_in + 0.06, label,
                        ha="center", va="bottom",
                        fontsize=BASE_FONT - 3, color=LINE_COLOR,
                        fontweight=600)
            else:
                ax.text(xi, v_in - 0.06, label,
                        ha="center", va="top",
                        fontsize=BASE_FONT - 3, color=LINE_COLOR,
                        fontweight=600)

        ax.set_xticks(x)
        ax.set_xticklabels([ROLE_DISPLAY.get(r, r) for r in ROLES],
                           fontsize=BASE_FONT - 1)
        ax.set_xlim(-0.6, len(ROLES) - 0.4)
        ax.set_ylim(0.08, 0.88)
        ax.set_title(cat, fontsize=BASE_FONT + 1, fontweight="bold",
                     color='black')
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

    # Y-label on every left-column subplot.
    for row in range(n_rows):
        axes_list[row * n_cols].set_ylabel("Recall", fontsize=BASE_FONT)

    # Single shared legend at the top.
    style_handles = [
        plt.Line2D([0], [0], color=LINE_COLOR,
                   marker=IN_PATTERN_MARKER, markersize=10,
                   markeredgecolor="black", markeredgewidth=0.6,
                   linewidth=2.4, label="In-propagation"),
        plt.Line2D([0], [0], color=LINE_COLOR,
                   marker=GLOBAL_REF_MARKER, markersize=10,
                   markeredgecolor="black", markeredgewidth=0.6,
                   linewidth=2.4, linestyle="--", label="In-isolation"),
    ]
    fig.legend(handles=style_handles,
               loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=2, fontsize=BASE_FONT,
               frameon=False, columnspacing=2.0, handletextpad=0.6)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.10,
                        wspace=0.08, hspace=0.28)
    fig.savefig(PLOT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {PLOT_PATH}")
    print("\nIn-pattern mean recall (weighted by n_puzzles):")
    print(in_pat.round(3).to_string())
    print("\nGlobal-ref mean recall (weighted to match):")
    print(glob.round(3).to_string())
    print("\nN of (pattern × position × method) cases per (family, role):")
    print(counts.to_string())


if __name__ == "__main__":
    main()
