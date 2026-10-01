"""Inventory the prediction methods across seed1-seed5 CSVs and compute
per-(action x method) ROC-AUC for each seed.

Stage 1: scans columns of the form pred_label_<method> and
pred_score_<method> in each seed CSV; reports methods per seed and
verifies the method sets are consistent across seeds.

Stage 2: derives `action` from `action_id`, merges OBSERVE_PLANNED_CLUE into
OBSERVE_CLUE and APPLY_DELTA_TOOL_CALL into UNIT_TOOL_CALL, then computes
roc_auc_score(step_label, pred_score_<method>) per (action x method) for each
seed and prints per-seed and mean tables.
"""
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".mplconfig"),
)

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# --- Style ---
mpl.rcParams['font.family'] = 'Trebuchet MS'
mpl.rcParams['font.weight'] = 'normal'

# Global font size; all text sizes are derived as BASE_FONT +/- delta.
BASE_FONT = 21
mpl.rcParams['font.size'] = BASE_FONT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - fallback for local sklearn/numpy ABI issues.
    def roc_auc_score(y_true, y_score):
        y = pd.Series(y_true).astype(int).to_numpy()
        scores = pd.Series(y_score).astype(float)
        if scores.isna().any():
            raise ValueError("Input contains NaN.")
        ranks = scores.rank(method="average").to_numpy()
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Only one class present in y_true.")
        return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

DATA_DIR = Path(__file__).parent / "data" / "full_data_all_plus_semi_sup_with_gemma"
PLOT_PATH = Path(__file__).with_suffix(".pdf")
SEEDS = [1, 2, 3, 4, 5]

ACTION_MERGE = {"OBSERVE_PLANNED_CLUE": "OBSERVE_CLUE"}

ACTION_ORDER = [
    "SELECT_PUZZLE",
    "OBSERVE_CLUE",
    "SOLVE_CLUE",
    "OBSERVE_ITEM",
    "UNIT_TOOL_CALL",
    "APPLY_DELTA",
    "OBSERVE_PUZZLE",
]
ACTION_ABBR = {
    "SELECT_PUZZLE":  "SP",
    "OBSERVE_CLUE":   "OC",
    "SOLVE_CLUE":     "SC",
    "OBSERVE_ITEM":   "OI",
    "UNIT_TOOL_CALL":  "TC",
    "APPLY_DELTA":    "AD",
    "OBSERVE_PUZZLE": "OP",
}
ACTION_LEGEND_NAME = {
    "OBSERVE_ITEM":   "OBSERVE_INSTRU",
    "UNIT_TOOL_CALL": "TOOL_CALL",
}
ACTION_SUCCESS_RATE = {
    "SELECT_PUZZLE":  79.7,
    "OBSERVE_CLUE":   96.7,
    "SOLVE_CLUE":     53.5,
    "OBSERVE_ITEM":   80.5,
    "UNIT_TOOL_CALL":  np.nan,
    "APPLY_DELTA":    35.8,
    "OBSERVE_PUZZLE": 98.2,
}
KNOWN_ACTIONS = sorted(
    ACTION_ORDER + list(ACTION_MERGE),
    key=len,
    reverse=True,
)

# Method/family metadata is shared across all exp_analysis scripts.
from methods_meta import (
    LLM_CSV_SUFFIX, METHOD_SHORT, METHOD_SHORTEST, METHOD_CATEGORY,
    CATEGORY_ORDER, csv_score_col, ordered_methods,
)
from gemma_split_utils import load_gemma_zero_shot_predictions, zero_shot_methods


def methods_in_cols(cols: list[str]) -> tuple[set[str], set[str]]:
    label = {c[len("pred_label_"):] for c in cols if c.startswith("pred_label_")}
    score = {c[len("pred_score_"):] for c in cols if c.startswith("pred_score_")}
    return label, score


def extract_action(action_id: str) -> str:
    text = str(action_id)
    if "APPLY_DELTA_TOOL_CALL" in text:
        return "UNIT_TOOL_CALL"
    for action in KNOWN_ACTIONS:
        if action in text:
            return ACTION_MERGE.get(action, action)
    return ACTION_MERGE.get(text, text)


def auc_per_action_method(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["action"] = df["action_id"].map(extract_action)
    actions = sorted(df["action"].unique())

    result = pd.DataFrame(index=actions, columns=methods, dtype=float)
    for action in actions:
        sub = df[df["action"] == action]
        y = sub["step_label"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        for m in methods:
            col = csv_score_col(m)
            if col not in sub.columns:
                continue
            score = sub[col].to_numpy()
            try:
                result.at[action, m] = roc_auc_score(y, score)
            except ValueError:
                result.at[action, m] = np.nan
    return result


def nanmean_2d(values: np.ndarray, axis: int) -> np.ndarray:
    valid = np.isfinite(values)
    counts = valid.sum(axis=axis)
    sums = np.where(valid, values, 0.0).sum(axis=axis)
    out = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def nanstd_2d(values: np.ndarray, axis: int, ddof: int = 1) -> np.ndarray:
    mean = np.expand_dims(nanmean_2d(values, axis=axis), axis=axis)
    valid = np.isfinite(values)
    counts = valid.sum(axis=axis)
    sq = np.where(valid, (values - mean) ** 2, 0.0).sum(axis=axis)
    denom = counts - ddof
    out = np.full(sq.shape, np.nan, dtype=float)
    np.divide(sq, denom, out=out, where=denom > 0)
    return np.sqrt(out)


def report_inventory(per_seed_methods: dict[int, tuple[set[str], set[str]]]
                     ) -> list[str]:
    all_methods = sorted(set().union(*(l | sc for l, sc in per_seed_methods.values())))

    header = f"{'method':<25} " + " ".join(f"s{s}L/s{s}S" for s in SEEDS)
    print(header)
    print("-" * len(header))
    for m in all_methods:
        cells = []
        for s in SEEDS:
            label, score = per_seed_methods[s]
            cells.append(f"  {'Y' if m in label else '-'}/{'Y' if m in score else '-'} ")
        print(f"{m:<25}" + " ".join(cells))

    print()
    label_sets = [l for l, _ in per_seed_methods.values()]
    score_sets = [sc for _, sc in per_seed_methods.values()]
    labels_consistent = all(s == label_sets[0] for s in label_sets)
    scores_consistent = all(s == score_sets[0] for s in score_sets)

    print(f"pred_label methods consistent across seeds: {labels_consistent}")
    print(f"pred_score methods consistent across seeds: {scores_consistent}")

    if not labels_consistent or not scores_consistent:
        ref_label, ref_score = per_seed_methods[SEEDS[0]]
        for s in SEEDS[1:]:
            label, score = per_seed_methods[s]
            ldiff = (label ^ ref_label)
            sdiff = (score ^ ref_score)
            if ldiff:
                print(f"  seed{s} label diff vs seed{SEEDS[0]}: {sorted(ldiff)}")
            if sdiff:
                print(f"  seed{s} score diff vs seed{SEEDS[0]}: {sorted(sdiff)}")
    else:
        print(f"All {len(all_methods)} methods present with both label & score in every seed.")

    return all_methods


def main() -> None:
    # ── Stage 1: column inventory ────────────────────────────────────────
    per_seed_methods: dict[int, tuple[set[str], set[str]]] = {}
    seed_headers: dict[int, list[str]] = {}
    for s in SEEDS:
        csv = DATA_DIR / f"full_test_data_with_preds_seed{s}.csv"
        cols = list(pd.read_csv(csv, nrows=0).columns)
        seed_headers[s] = cols
        per_seed_methods[s] = methods_in_cols(cols)

    print("=" * 70)
    print("Method inventory")
    print("=" * 70)
    report_inventory(per_seed_methods)

    # Restrict to methods that have score columns in every seed (AUC needs
    # score), drop polluted variants / the random baseline, and map raw LLM
    # column suffixes back to internal keys (gemma-4-E2B_zero_shot ->
    # gemma4_e2b_zs).
    raw_llm_cols = set(LLM_CSV_SUFFIX.values())
    common_score = set.intersection(*(sc for _, sc in per_seed_methods.values()))
    any_score = set().union(*(sc for _, sc in per_seed_methods.values()))
    csv_methods = sorted(
        m for m in common_score
        if not m.endswith("_polluted")
        and m != "random"
        and m not in raw_llm_cols
    )
    # Include LLM keys whose column exists in *any* seed (not just all):
    # auc_per_action_method skips missing columns and we average with
    # nanmean downstream, so partially-present LLMs still get plotted.
    llm_methods = sorted(
        k for k, suf in LLM_CSV_SUFFIX.items() if suf in any_score
    )
    methods_with_score = csv_methods + llm_methods
    gemma_zs_methods = {
        k: v for k, v in zero_shot_methods(LLM_CSV_SUFFIX).items()
        if k in llm_methods
    }
    read_cols_by_seed: dict[int, list[str]] = {}
    for s in SEEDS:
        cols = set(seed_headers[s])
        score_cols = [
            csv_score_col(m) for m in methods_with_score
            if csv_score_col(m) in cols
        ]
        read_cols_by_seed[s] = ["action_id", "step_label"] + score_cols

    # ── Stage 2: per-seed AUC per (action x method) ──────────────────────
    print("\n" + "=" * 70)
    print("Per-seed AUC-ROC (rows=action, cols=method)")
    print("=" * 70)

    per_seed_auc: dict[int, pd.DataFrame] = {}
    action_sr_parts: list[pd.DataFrame] = []
    for s in SEEDS:
        csv = DATA_DIR / f"full_test_data_with_preds_seed{s}.csv"
        df = pd.read_csv(csv, usecols=read_cols_by_seed[s])
        df["action"] = df["action_id"].map(extract_action)
        action_sr_parts.append(df[["action", "step_label"]])

        auc = auc_per_action_method(df, methods_with_score)
        if gemma_zs_methods:
            gemma_df = load_gemma_zero_shot_predictions(
                seed=s,
                method_suffixes=gemma_zs_methods,
                prefix="pred_score_",
                base_cols=["action_id", "step_label"],
            )
            if not gemma_df.empty:
                gemma_auc = auc_per_action_method(gemma_df, list(gemma_zs_methods))
                for action in gemma_auc.index:
                    if action not in auc.index:
                        auc.loc[action] = np.nan
                    for method in gemma_auc.columns:
                        auc.at[action, method] = gemma_auc.at[action, method]
        per_seed_auc[s] = auc

        print(f"\n--- seed {s} (n={len(df)}) ---")
        with pd.option_context("display.float_format", "{:.4f}".format,
                               "display.width", 200,
                               "display.max_columns", None):
            print(auc.round(4))

    # Mean / std across seeds
    stacked = pd.concat([per_seed_auc[s] for s in SEEDS],
                        keys=SEEDS, names=["seed"])
    mean_auc = stacked.groupby(level=1).mean()
    std_auc = stacked.groupby(level=1).std()

    print("\n" + "=" * 70)
    print("Mean AUC across seeds")
    print("=" * 70)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 200,
                           "display.max_columns", None):
        print(mean_auc.round(4))

    action_df = pd.concat(action_sr_parts, ignore_index=True)
    action_success_rate = (
        (1.0 - action_df.groupby("action")["step_label"].mean()) * 100.0
    ).to_dict()

    plot_heatmap(mean_auc, std_auc, PLOT_PATH, action_success_rate)
    print(f"\nSaved heatmap to: {PLOT_PATH}")


def plot_heatmap(mean_auc: pd.DataFrame, std_auc: pd.DataFrame,
                 out_path: Path,
                 action_success_rate: dict[str, float] | None = None) -> None:
    """Flipped heatmap: rows = actions, cols = methods. Family brackets on top
    of method columns; an extra rightmost column shows the across-method mean
    AUC per action."""
    actions = [a for a in ACTION_ORDER if a in mean_auc.index]
    extra = [a for a in mean_auc.index if a not in ACTION_ORDER]
    actions = actions + extra

    # Use the global method ordering from methods_meta.py.
    methods = ordered_methods(list(mean_auc.columns))

    matrix = mean_auc.loc[actions, methods].to_numpy()
    std_matrix = std_auc.loc[actions, methods].to_numpy()

    # Append an extra column: across-method mean AUC per action.
    # std for that column = std across the per-method mean AUCs (method spread).
    n_methods = matrix.shape[1]
    n_actions_only = matrix.shape[0]
    avg_col = nanmean_2d(matrix, axis=1).reshape(-1, 1)
    avg_col_std = nanstd_2d(matrix, axis=1, ddof=1).reshape(-1, 1)
    matrix = np.hstack([matrix, avg_col])
    std_matrix = np.hstack([std_matrix, avg_col_std])

    # Append an extra row: across-action mean AUC per method (and overall mean
    # in the corner cell). std for that row = std across actions.
    avg_row = nanmean_2d(matrix[:n_actions_only, :], axis=0).reshape(1, -1)
    avg_row_std = nanstd_2d(matrix[:n_actions_only, :], axis=0, ddof=1).reshape(1, -1)
    matrix = np.vstack([matrix, avg_row])
    std_matrix = np.vstack([std_matrix, avg_row_std])

    n_rows, _ = matrix.shape  # n_actions + 1 (Avg row)

    cmap = LinearSegmentedColormap.from_list(
        "teal_fm",
        ["#E9F8F6", "#CCE5E2", "#AFD1CE", "#92BEBA", "#75AAA6",
         "#579791", "#3A837D", "#1D7069", "#005C55"],
    )
    cmap.set_bad("white")

    # Split the matrix into the main (per-method cols) and the narrow Avg
    # column so the latter can be drawn at a smaller width via GridSpec.
    main_data = matrix[:, :n_methods]
    main_std  = std_matrix[:, :n_methods]
    avg_data  = matrix[:, n_methods:n_methods + 1].copy()
    avg_data[-1, 0] = np.nan  # blank the corner cell (Avg row × Avg col)
    avg_std   = std_matrix[:, n_methods:n_methods + 1]

    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    AVG_W = 0.8  # Avg column width as fraction of one main column.
    fig = plt.figure(figsize=(1.0 * (n_methods + AVG_W) + 2.5,
                              0.55 * n_rows + 1.2))
    gs = GridSpec(
        2, 2,
        height_ratios=[n_rows, 0.6],
        width_ratios=[n_methods, AVG_W],
        hspace=0.10, wspace=0.005,
        left=0.06, right=0.99, top=0.92, bottom=0.04,
        figure=fig,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_avg  = fig.add_subplot(gs[0, 1], sharey=ax_main)
    # Sub-grid for the colorbar so we can shrink it horizontally (centered
    # in the middle ~40% of the bottom row).
    sub_cbar = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :],
                                       width_ratios=[3, 4, 3])
    ax_cbar = fig.add_subplot(sub_cbar[0, 1])

    im_main = ax_main.imshow(np.ma.masked_invalid(main_data), cmap=cmap, aspect="auto",
                             vmin=0.0, vmax=1.0)
    ax_avg.imshow(np.ma.masked_invalid(avg_data), cmap=cmap, aspect="auto",
                  vmin=0.0, vmax=1.0)

    mean_row_idx = n_rows - 1

    # Annotate main cells: AUCROC value only, centered.
    CELL_FONT = BASE_FONT + 4
    for i in range(n_rows):
        for j in range(n_methods):
            val = main_data[i, j]
            if np.isnan(val):
                continue
            color = "white" if round(val, 2) >= 0.64 else "black"
            weight = 600 if i == mean_row_idx else 500
            ax_main.text(j, i, f"{val:.2f}",
                         ha="center", va="center",
                         fontsize=CELL_FONT, color=color, fontweight=weight)

    # Annotate Avg-column cells (skip corner blank).
    for i in range(n_rows):
        if i == mean_row_idx:
            continue  # corner intentionally blank
        val = avg_data[i, 0]
        if np.isnan(val):
            continue
        color = "white" if round(val, 2) >= 0.64 else "black"
        ax_avg.text(0, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=CELL_FONT, color=color, fontweight=600)

    # Main x ticks: METHOD_SHORT (full label) above the heatmap, rotated
    # 45 degrees so longer names fit. Convention matches
    # method_overall_aucroc.py.
    LABEL_FONT = BASE_FONT + 3
    LOWER_Y = 1.015
    ax_main.set_xticks(range(n_methods))
    ax_main.set_xticklabels(
        [METHOD_SHORT.get(m, m) for m in methods],
        fontsize=LABEL_FONT,
        rotation=45, ha="left", rotation_mode="anchor",
    )
    ax_main.xaxis.set_label_position("top")
    ax_main.xaxis.tick_top()
    ax_main.tick_params(axis="x", length=0, pad=4)

    # Main y ticks: action abbreviation (larger) over the success rate
    # (smaller), plus a single bold "Avg" label for the bottom row.
    ax_main.set_yticks(range(n_rows))
    ax_main.set_yticklabels([""] * n_rows)
    ax_main.tick_params(axis="y", length=0)
    ABBR_FONT = BASE_FONT + 2
    SR_FONT = BASE_FONT -2
    LABEL_X = -0.005  # axes-fraction, just left of the heatmap edge
    for i, a in enumerate(actions):
        ax_main.text(LABEL_X, i - 0.18, ACTION_ABBR.get(a, a),
                     transform=ax_main.get_yaxis_transform(),
                     ha="right", va="center", fontsize=ABBR_FONT)
        sr = (action_success_rate or {}).get(a, ACTION_SUCCESS_RATE.get(a, np.nan))
        sr_label = f"({sr:.1f}%)" if np.isfinite(sr) else ""
        ax_main.text(LABEL_X, i + 0.22,
                     sr_label,
                     transform=ax_main.get_yaxis_transform(),
                     ha="right", va="center", fontsize=SR_FONT)
    ax_main.text(LABEL_X, n_rows - 1, "Avg",
                 transform=ax_main.get_yaxis_transform(),
                 ha="right", va="center",
                 fontsize=ABBR_FONT, fontweight=700)

    # Avg-column axis: only the "Avg" header on top; hide its y labels.
    # Draw the header manually at the lower-row position so it aligns with
    # the staggered main labels.
    ax_avg.set_xticks([0])
    ax_avg.set_xticklabels([""])
    ax_avg.xaxis.set_label_position("top")
    ax_avg.xaxis.tick_top()
    ax_avg.tick_params(axis="x", length=0)
    ax_avg.text(0, LOWER_Y, "Avg",
                transform=ax_avg.get_xaxis_transform(),
                ha="center", va="bottom",
                fontsize=LABEL_FONT)
    plt.setp(ax_avg.get_yticklabels(), visible=False)
    ax_avg.tick_params(left=False)

    # Inter-family separators on ax_main: thick black lines so the
    # supervision groupings are clearly delimited inside the heatmap.
    cat_seq = [METHOD_CATEGORY.get(m) for m in methods]
    for j in range(1, len(methods)):
        if cat_seq[j] != cat_seq[j - 1]:
            ax_main.axvline(j - 0.5, color="black", linewidth=2.5,
                            linestyle="-", alpha=1.0, zorder=5)
    # Strong horizontal separator between data rows and the Avg row, on both axes.
    for ax in (ax_main, ax_avg):
        ax.axhline(len(actions) - 0.5, color="white", linewidth=4.0,
                   linestyle="-", alpha=1.0)

    # Family bracket labels above ax_main columns.
    family_groups: list[tuple[int, int, str]] = []
    n_method_cols = len(methods)
    j = 0
    while j < n_method_cols:
        cat = METHOD_CATEGORY.get(methods[j])
        if cat is None:
            j += 1
            continue
        k = j
        while k + 1 < n_method_cols and METHOD_CATEGORY.get(methods[k + 1]) == cat:
            k += 1
        family_groups.append((j, k, cat))
        j = k + 1

    # Brackets pushed up to clear the rotated METHOD_SHORT labels at the top.
    brack_y = 1.44
    tick_h = 0.012
    label_y = 1.47
    bracket_color = "black"
    for start, end, cat in family_groups:
        x_left = start - 0.4
        x_right = end + 0.4
        mid_x = (start + end) / 2
        ax_main.annotate('', xy=(x_left, brack_y),
                         xycoords=('data', 'axes fraction'),
                         xytext=(x_right, brack_y),
                         textcoords=('data', 'axes fraction'),
                         arrowprops=dict(arrowstyle='-', color=bracket_color, lw=1.2),
                         annotation_clip=False)
        for xx in [x_left, x_right]:
            ax_main.annotate('', xy=(xx, brack_y - tick_h),
                             xycoords=('data', 'axes fraction'),
                             xytext=(xx, brack_y),
                             textcoords=('data', 'axes fraction'),
                             arrowprops=dict(arrowstyle='-', color=bracket_color, lw=1.2),
                             annotation_clip=False)
        ax_main.annotate(cat, xy=(mid_x, label_y),
                         xycoords=('data', 'axes fraction'),
                         fontsize=BASE_FONT + 14, fontweight=700, color=bracket_color,
                         ha='center', va='bottom',
                         annotation_clip=False)

    # Horizontal colorbar in the bottom GridSpec slot, between heatmap and legend.
    cbar = fig.colorbar(im_main, cax=ax_cbar, orientation="horizontal")
    cbar.set_label("AUCROC (mean over available seeds)", fontsize=BASE_FONT +5 ,
                   labelpad=4)
    cbar.ax.tick_params(labelsize=BASE_FONT )

    # Action legend below the colorbar. The family/method-name mapping
    # legend is dropped here -- the rotated METHOD_SHORT labels at the
    # top already double as that key.
    action_body = ", ".join(
        f"{ACTION_ABBR.get(a, a)} ({ACTION_LEGEND_NAME.get(a, a)})" for a in actions
    )
    fig.text(0.05, -0.18, r"$\bf{Action}$ — " + action_body,
             ha="left", va="top",
             fontsize=BASE_FONT + 6)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
