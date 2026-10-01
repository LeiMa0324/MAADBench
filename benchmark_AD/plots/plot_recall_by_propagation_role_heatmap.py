"""Heatmap of per-method recall across propagation roles —
Origin / Middle / Downstream.

Layout mirrors method_aucroc_by_action.py exactly: rows are the three
propagation roles, columns are methods grouped by supervision family
with bracket labels above, plus an extra Avg column (across-method mean
per role) on the right and an Avg row (across-role mean per method) at
the bottom. Colormap, family separators, colorbar, and legend layout
all match the AUCROC heatmap.

Recall per (method, role) is the weighted mean (by n_puzzles) over all
(pattern, position) cells in that role for that method, taken from
method_recall_by_propagation_pattern.csv.
"""
import sys
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".mplconfig"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

mpl.rcParams["font.family"] = "Trebuchet MS"
mpl.rcParams["font.weight"] = "normal"

BASE_FONT = 21
mpl.rcParams["font.size"] = BASE_FONT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).parent / "method_recall_by_propagation_pattern.csv"
PLOT_PATH = Path(__file__).with_suffix(".pdf")

ROLES = ["Origin", "Middle", "Downstream"]
ROLE_DISPLAY = {"Origin": "Origin", "Middle": "Mid", "Downstream": "Down"}

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from methods_meta import (
    METHOD_SHORT, METHOD_SHORTEST, METHOD_CATEGORY,
    CATEGORY_ORDER, ordered_methods,
)


def _nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    valid = np.isfinite(values)
    counts = valid.sum(axis=axis)
    sums = np.where(valid, values, 0.0).sum(axis=axis)
    out = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def ensure_recall_csv_current() -> None:
    """Rebuild the propagation-pattern recall CSV when missing or stale.

    The builder script owns the raw-data plumbing, including the current
    ``full_data_all_plus_semi_sup_with_gemma`` source and Gemma zero-shot
    split directory. Keeping the rebuild here prevents this heatmap from
    accidentally using an old intermediate CSV.
    """
    import plot_recall_delta_by_role_2X2 as recall_builder

    deps = [
        recall_builder.DATA_DIR / f"full_test_data_with_preds_seed{s}.csv"
        for s in recall_builder.SEEDS
    ]
    deps.append(Path(__file__).parent / "gemma_split_utils.py")
    deps.append(Path(__file__).parent / "plot_recall_delta_by_role_2X2.py")
    deps.extend([
        Path(__file__).parent / "data" / "gemma_zeroshot_full_data" / "manifest.json",
        Path(__file__).parent / "data" / "gemma_zeroshot_full_data" / "index.csv",
    ])
    deps = [p for p in deps if p.exists()]

    if CSV_PATH.exists() and deps:
        csv_mtime = CSV_PATH.stat().st_mtime
        if csv_mtime >= max(p.stat().st_mtime for p in deps):
            return

    seed_headers = {
        s: list(pd.read_csv(
            recall_builder.DATA_DIR / f"full_test_data_with_preds_seed{s}.csv",
            nrows=0,
        ).columns)
        for s in recall_builder.SEEDS
    }
    methods = recall_builder.discover_methods(seed_headers)
    seed_dfs = recall_builder.load_seed_frames(methods, seed_headers)
    df = recall_builder.build_recall_by_pattern(methods, seed_dfs)
    df.to_csv(CSV_PATH, index=False)
    print(f"Rebuilt propagation-pattern recall CSV: {CSV_PATH}")


def assign_role(positions_in_pattern: list[int], pos: int) -> str:
    sorted_pos = sorted(positions_in_pattern)
    if pos == sorted_pos[0]:
        return "Origin"
    if pos == sorted_pos[-1]:
        return "Downstream"
    return "Middle"


def compute_recall_by_role(df: pd.DataFrame,
                           methods: list[str]) -> pd.DataFrame:
    """Weighted-by-n_puzzles mean recall per (role, method)."""
    pattern_positions: dict[str, list[int]] = {}
    for patt, sub in df.groupby("pattern"):
        pattern_positions[patt] = sorted(sub["position"].unique())

    df = df.copy()
    df["role"] = df.apply(
        lambda r: assign_role(pattern_positions[r["pattern"]], r["position"]),
        axis=1,
    )

    out = pd.DataFrame(index=ROLES, columns=methods, dtype=float)
    for m in methods:
        m_df = df[df["method"] == m]
        for role in ROLES:
            sub = m_df[m_df["role"] == role].dropna(subset=["recall"])
            if sub.empty:
                continue
            w = sub["n_puzzles"].astype(float)
            out.at[role, m] = float(np.average(sub["recall"], weights=w))
    return out


def plot_heatmap(mean_recall: pd.DataFrame, out_path: Path) -> None:
    """Same layout as method_aucroc_by_action.plot_heatmap, with rows =
    propagation roles instead of actions."""
    roles = [r for r in ROLES if r in mean_recall.index]

    methods = ordered_methods(list(mean_recall.columns))
    matrix = mean_recall.loc[roles, methods].to_numpy()

    n_methods = matrix.shape[1]
    n_roles_only = matrix.shape[0]
    avg_col = _nanmean(matrix, axis=1).reshape(-1, 1)
    matrix = np.hstack([matrix, avg_col])

    avg_row = _nanmean(matrix[:n_roles_only, :], axis=0).reshape(1, -1)
    matrix = np.vstack([matrix, avg_row])

    n_rows, _ = matrix.shape  # n_roles + 1 (Avg row)

    cmap = LinearSegmentedColormap.from_list(
        "teal_fm",
        ["#E9F8F6", "#CCE5E2", "#AFD1CE", "#92BEBA", "#75AAA6",
         "#579791", "#3A837D", "#1D7069", "#005C55"],
    )
    cmap.set_bad("white")

    main_data = matrix[:, :n_methods]
    avg_data = matrix[:, n_methods:n_methods + 1].copy()
    avg_data[-1, 0] = np.nan  # blank corner cell

    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    AVG_W = 0.8
    fig = plt.figure(figsize=(1.0 * (n_methods + AVG_W) + 2.5,
                              0.8 * n_rows + 1.0))
    gs = GridSpec(
        2, 2,
        height_ratios=[n_rows, 0.25],
        width_ratios=[n_methods, AVG_W],
        hspace=0.10, wspace=0.005,
        left=0.06, right=0.99, top=0.92, bottom=0.04,
        figure=fig,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_avg = fig.add_subplot(gs[0, 1], sharey=ax_main)
    sub_cbar = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :],
                                       width_ratios=[3, 4, 3])
    ax_cbar = fig.add_subplot(sub_cbar[0, 1])

    im_main = ax_main.imshow(np.ma.masked_invalid(main_data), cmap=cmap, aspect="auto",
                             vmin=0.0, vmax=1.0)
    ax_avg.imshow(np.ma.masked_invalid(avg_data), cmap=cmap, aspect="auto",
                  vmin=0.0, vmax=1.0)

    mean_row_idx = n_rows - 1

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

    for i in range(n_rows):
        if i == mean_row_idx:
            continue
        val = avg_data[i, 0]
        if np.isnan(val):
            continue
        color = "white" if round(val, 2) >= 0.64 else "black"
        ax_avg.text(0, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=CELL_FONT, color=color, fontweight=600)

    # Main x ticks: method short labels, staggered onto two rows so the
    # larger font fits without overlap. 1st/3rd/5th... (1-indexed odd) sit
    # on the lower row close to the heatmap; 2nd/4th/6th... on the upper.
    # UPPER_Y is larger than in method_aucroc_by_action.py because this
    # heatmap is shorter (4 rows vs 7), so axes-fraction maps to less
    # vertical space and one text line spans more axes-fraction.
    LABEL_FONT = BASE_FONT + 4
    LOWER_Y = 1.015
    UPPER_Y = 1.12
    ax_main.set_xticks(range(n_methods))
    ax_main.set_xticklabels([""] * n_methods)
    ax_main.xaxis.set_label_position("top")
    ax_main.xaxis.tick_top()
    ax_main.tick_params(axis="x", length=0)
    LINE_GAP = 0.004
    for j in range(n_methods):
        y_top = (LOWER_Y if (j % 2 == 0) else UPPER_Y) - LINE_GAP
        ax_main.plot([j, j], [1.0, y_top],
                     transform=ax_main.get_xaxis_transform(),
                     color="gray", linestyle=":", linewidth=0.8,
                     clip_on=False, zorder=1)
    for j, m in enumerate(methods):
        y = LOWER_Y if (j % 2 == 0) else UPPER_Y
        ax_main.text(j, y, METHOD_SHORTEST.get(m, m),
                     transform=ax_main.get_xaxis_transform(),
                     ha="center", va="bottom",
                     fontsize=LABEL_FONT)

    ax_main.set_yticks(range(n_rows))
    ylabels = [ROLE_DISPLAY.get(r, r) for r in roles] + ["Avg"]
    ax_main.set_yticklabels(ylabels, fontsize=BASE_FONT + 3, linespacing=1.0)

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

    cat_seq = [METHOD_CATEGORY.get(m) for m in methods]
    for j in range(1, len(methods)):
        if cat_seq[j] != cat_seq[j - 1]:
            ax_main.axvline(j - 0.5, color="gray", linewidth=0.8,
                            linestyle="-", alpha=0.5)
    for ax in (ax_main, ax_avg):
        ax.axhline(len(roles) - 0.5, color="white", linewidth=4.0,
                   linestyle="-", alpha=1.0)

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

    # Bracket offsets are in axes fraction; this heatmap is only 4 rows tall
    # (vs 7 in the AUCROC heatmap), so the offsets are scaled up so brackets
    # don't crowd the staggered method tick labels above the heatmap.
    brack_y = 1.30
    tick_h = 0.020
    label_y = 1.36
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

    cbar = fig.colorbar(im_main, cax=ax_cbar, orientation="horizontal")
    cbar.set_label("Recall", fontsize=BASE_FONT + 5,
                   labelpad=4)
    cbar.ax.tick_params(labelsize=BASE_FONT)

    role_body = ", ".join(
        f"{ROLE_DISPLAY.get(r, r)} ({r})" for r in roles
    )
    LEGEND_FAMILY_SUFFIX: dict[str, str] = {}

    # The legend uses fig-fraction Y coords. Scale spacings so they match
    # the visual cadence of method_aucroc_by_action.py, whose figure is
    # 6.6 in tall. This figure is shorter, so each fig-fraction step would
    # otherwise be smaller in inches than the (unchanged) text height.
    fig_h_ref = 6.6
    scale = max(1.0, fig_h_ref / fig.get_figheight())
    LEGEND_X = 0.05
    LEGEND_X_RIGHT = 0.95
    LEGEND_OFFSET = -0.06 * scale
    ROLE_Y = -0.04 * scale + LEGEND_OFFSET
    SEP_Y = -0.10 * scale + LEGEND_OFFSET
    FAMILY_START_Y = -0.16 * scale + LEGEND_OFFSET
    fig.text(LEGEND_X, ROLE_Y, r"$\bf{Role}$ — " + role_body,
             ha="left", va="top",
             fontsize=BASE_FONT + 5)

    from matplotlib.lines import Line2D
    fig.add_artist(Line2D([LEGEND_X, LEGEND_X_RIGHT], [SEP_Y, SEP_Y],
                          color="lightgray", linewidth=0.8,
                          transform=fig.transFigure))

    line_y = FAMILY_START_Y
    line_step = 0.06 * scale
    WRAP_FAMILIES = {"Semi-supervised", "Semi", "Semi supervised"}
    for cat in CATEGORY_ORDER:
        fam_methods = [m for m in methods if METHOD_CATEGORY.get(m) == cat]
        if not fam_methods:
            continue
        items = [
            f"{METHOD_SHORTEST.get(m, m)} ({METHOD_SHORT.get(m, m)})"
            for m in fam_methods
        ]
        cat_disp = cat + LEGEND_FAMILY_SUFFIX.get(cat, "")
        bold_cat = (cat_disp
                    .replace(" ", r"\ ")
                    .replace("(", r"\mathrm{(}")
                    .replace(")", r"\mathrm{)}"))
        if cat in WRAP_FAMILIES and len(items) > 1:
            half = (len(items) + 1) // 2
            indent = " " * (len(cat_disp) + 3)
            body = (", ".join(items[:half]) + ",\n"
                    + indent + ", ".join(items[half:]))
            n_lines = 2
        else:
            body = ", ".join(items)
            n_lines = 1
        fig.text(LEGEND_X, line_y, r"$\bf{" + bold_cat + r"}$ — " + body,
                 ha="left", va="top",
                 fontsize=BASE_FONT + 5)
        line_y -= line_step * n_lines

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_recall_csv_current()
    if not CSV_PATH.exists():
        raise SystemExit(
            f"{CSV_PATH} not found — run "
            "method_recall_by_propagation_pattern.py first.")
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("Empty CSV; nothing to plot.")
        return

    methods = ordered_methods(list(df["method"].unique()))
    mean_recall = compute_recall_by_role(df, methods)

    print("Per-(role, method) weighted mean recall:")
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 200,
                           "display.max_columns", None):
        print(mean_recall.round(4))

    plot_heatmap(mean_recall, PLOT_PATH)
    print(f"\nSaved heatmap to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
