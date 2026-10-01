"""Combined slope plot (horizontal layout): per-method AUC-ROC across
two side-by-side subplots.

Left  — split by LLM backbone (5 dots per method, one per backbone),
        reusing the logic from method_auc_by_LLM_backbones.py.
Right — split by failure type (2 dots per method: silent vs loud),
        reusing the logic from method_auc_silent_loud.py.

Both subplots share the y axis; method labels appear under each. Each
subplot has its own legend inside the chart and a title above. Family
bracket labels (Unsupervised / OCC / Semi-sup / Supervised) sit above
each subplot.
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

mpl.rcParams["font.family"] = "Trebuchet MS"
mpl.rcParams["font.weight"] = "normal"
BASE_FONT = 20
markersize = 14
mpl.rcParams["font.size"] = BASE_FONT

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmark_analyze.silent_loud import classify_dataframe, is_format_error, SILENT, LOUD  # noqa: E402

from methods_meta import (
    LLM_CSV_SUFFIX, METHOD_SHORT, METHOD_CATEGORY,
    BACKBONE_ORDER, BACKBONE_DISPLAY,
    csv_score_col, ordered_methods,
)
from gemma_split_utils import (
    load_gemma_few_shot_predictions,
    load_gemma_zero_shot_predictions,
    few_shot_methods,
    zero_shot_methods,
)

DATA_DIR = Path(__file__).parent / "data" / "full_data_all_plus_semi_sup_with_gemma"
SUMMARY_DIR = ROOT / "output" / "traces"
PLOT_PATH = Path(__file__).with_suffix(".pdf")
SEEDS = [1, 2, 3, 4, 5]
MODELS = BACKBONE_ORDER

# Backbone colors (cool→warm gradient).
BACKBONE_COLOR = {
    "gpt-5.4":                  "#172869",
    "gpt-4.1":                  "#088BBE",
    "deepseek-reasoner":        "#EA7580",
    "claude-sonnet-4-6":        "#1BB6AF",
    "claude-opus-4-8":          "#F8CD9C",
}
SILENT_COLOR = "#5773CC"
LOUD_COLOR   = "#FFB900"

# Override the family bracket label only for display (key in METHOD_CATEGORY
# stays "Semi-supervised").
CATEGORY_DISPLAY = {
    "Semi-supervised": "Semi-sup",
}

# Override the backbone legend label only for display.
BACKBONE_LEGEND_DISPLAY = {
    "claude-opus-4-8":           "Opus-4.8",
    "claude-sonnet-4-6":         "Sonnet-4.6",
    "deepseek-reasoner":         "DS-R1",
    "gpt-4.1":                   "GPT-4.1",
    "gpt-5.4":                   "GPT-5.4",
}


# ── Data plumbing ──────────────────────────────────────────────────────

def methods_in_cols(cols: list[str]) -> set[str]:
    return {c[len("pred_score_"):] for c in cols if c.startswith("pred_score_")}


def auc_per_backbone_method(df, methods, backbones):
    result = pd.DataFrame(index=backbones, columns=methods, dtype=float)
    for bb in backbones:
        sub = df[df["model_name"] == bb]
        y = sub["step_label"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        for m in methods:
            col = csv_score_col(m)
            if col not in sub.columns:
                continue
            score = sub[col].to_numpy()
            try:
                result.at[bb, m] = roc_auc_score(y, score)
            except ValueError:
                result.at[bb, m] = np.nan
    return result


def auc_per_method(df, methods, positive_mask, negative_mask):
    keep = positive_mask | negative_mask
    y = positive_mask[keep].astype(int).to_numpy()
    out: dict[str, float] = {}
    for m in methods:
        col = csv_score_col(m)
        if col not in df.columns or len(np.unique(y)) < 2:
            out[m] = float("nan")
            continue
        score = df.loc[keep, col].to_numpy()
        try:
            out[m] = float(roc_auc_score(y, score))
        except ValueError:
            out[m] = float("nan")
    return out


def build_format_error_lookup() -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for path in sorted(SUMMARY_DIR.rglob("summary.csv")):
        sm = pd.read_csv(path)
        for _, r in sm.iterrows():
            if "action_status" in sm.columns and r.get("action_status") != "wrong":
                continue
            model = r.get("llm_model", r.get("model_name"))
            if model not in MODELS:
                continue
            fa = r.get("action_name", r.get("failed_action"))
            if pd.isna(fa) or not str(fa).strip():
                continue
            if is_format_error(r.get("answer")):
                trace_filename = r.get("trace_filename")
                action_id = r.get("action_id")
                if pd.isna(trace_filename) or pd.isna(action_id):
                    continue
                keys.add((str(model), str(trace_filename), str(action_id)))
    return keys


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
            return "OBSERVE_CLUE" if action == "OBSERVE_PLANNED_CLUE" else action
    return text


def is_fmt_silent_row(row, fmt_keys) -> bool:
    # Exact archive-compatible matching requires trace_filename. The AD full
    # CSVs currently do not carry it, so return False instead of using a coarse
    # (model, room, puzzle, action) key that collides across domain/tool/temp
    # runs and over-removes silent anomalies.
    if "trace_filename" not in row.index:
        return False
    trace_filename = row.get("trace_filename")
    if pd.isna(trace_filename):
        return False
    return (
        str(row["model_name"]),
        str(trace_filename),
        str(row["action_id"]),
    ) in fmt_keys


# ── Plot helpers ───────────────────────────────────────────────────────

def draw_dots(ax, xs, values_per_series, colors_per_series):
    """Draw connected dot columns. `values_per_series` is a list (one entry
    per series) of length-n iterables of y-values."""
    n = len(xs)
    # Vertical connector through all the dots at each x.
    for i in range(n):
        ys = [series[i] for series in values_per_series
              if np.isfinite(series[i])]
        if len(ys) >= 2:
            ax.plot([xs[i]] * len(ys), ys, "-", color="#888888",
                    linewidth=0.8, alpha=0.7, zorder=1)
    # Scatter for each series.
    for series, color in zip(values_per_series, colors_per_series):
        ax.scatter(xs, series, s=markersize ** 2, color=color,
                   edgecolor="black", linewidth=0.6, zorder=2)


def draw_family_brackets(ax, methods, brack_y=1.04, label_y=1.07,
                         tick_h=0.025):
    family_groups: list[tuple[int, int, str]] = []
    n = len(methods)
    j = 0
    while j < n:
        cat = METHOD_CATEGORY.get(methods[j])
        if cat is None:
            j += 1
            continue
        k = j
        while k + 1 < n and METHOD_CATEGORY.get(methods[k + 1]) == cat:
            k += 1
        family_groups.append((j, k, cat))
        j = k + 1
    for j_idx in range(1, len(family_groups)):
        sep_x = family_groups[j_idx][0] - 0.5
        ax.axvline(sep_x, color="lightgray", linewidth=0.8,
                   linestyle="-", alpha=0.6)
    for start, end, cat in family_groups:
        x_left = start - 0.4
        x_right = end + 0.4
        mid = (start + end) / 2
        ax.annotate("", xy=(x_left, brack_y),
                    xycoords=("data", "axes fraction"),
                    xytext=(x_right, brack_y),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
                    annotation_clip=False)
        for xx in (x_left, x_right):
            ax.annotate("", xy=(xx, brack_y - tick_h),
                        xycoords=("data", "axes fraction"),
                        xytext=(xx, brack_y),
                        textcoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
                        annotation_clip=False)
        ax.annotate(CATEGORY_DISPLAY.get(cat, cat), xy=(mid, label_y),
                    xycoords=("data", "axes fraction"),
                    fontsize=BASE_FONT+2 , fontweight=500, color="black",
                    ha="center", va="bottom", annotation_clip=False)


def draw_family_separators(ax, methods):
    """Lighter version: only the inter-family vertical lines (no bracket)."""
    n = len(methods)
    cats = [METHOD_CATEGORY.get(m) for m in methods]
    for i in range(1, n):
        if cats[i] != cats[i - 1]:
            ax.axvline(i - 0.5, color="lightgray", linewidth=0.8,
                       linestyle="-", alpha=0.6)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    seed_headers: dict[int, list[str]] = {}
    per_seed_score: dict[int, set[str]] = {}
    for s in SEEDS:
        csv = DATA_DIR / f"full_test_data_with_preds_seed{s}.csv"
        cols = list(pd.read_csv(csv, nrows=0).columns)
        seed_headers[s] = cols
        per_seed_score[s] = methods_in_cols(cols)

    common_score = set.intersection(*per_seed_score.values())
    any_score = set().union(*per_seed_score.values())
    raw_llm_cols = set(LLM_CSV_SUFFIX.values())
    csv_methods = sorted(
        m for m in common_score
        if not m.endswith("_polluted")
        and m != "random"
        and m not in raw_llm_cols
    )
    # LLM keys: use union, not intersection — the per-seed AUC functions
    # already NaN-out missing columns and we nanmean across seeds.
    llm_methods = sorted(
        k for k, suf in LLM_CSV_SUFFIX.items() if suf in any_score
    )
    methods = ordered_methods(csv_methods + llm_methods)
    gemma_zs_methods = {
        k: v for k, v in zero_shot_methods(LLM_CSV_SUFFIX).items()
        if k in llm_methods
    }
    gemma_fs_methods = {
        k: v for k, v in few_shot_methods(LLM_CSV_SUFFIX).items()
        if k in llm_methods
    }

    base_cols = [
        "model_name",
        "room_id",
        "puzzle_id",
        "action_id",
        "trace_filename",
        "step_label",
        "trace_label",
    ]
    seed_dfs: list[pd.DataFrame] = []
    gemma_zs_seed_dfs: list[pd.DataFrame] = []
    gemma_fs_seed_dfs: list[pd.DataFrame] = []
    for s in SEEDS:
        cols = set(seed_headers[s])
        score_cols = [
            csv_score_col(m) for m in methods
            if csv_score_col(m) in cols
        ]
        usecols = [c for c in base_cols if c in cols] + score_cols
        seed_dfs.append(pd.read_csv(
            DATA_DIR / f"full_test_data_with_preds_seed{s}.csv",
            usecols=usecols,
        ))
        gemma_zs_seed_dfs.append(load_gemma_zero_shot_predictions(
            seed=s,
            method_suffixes=gemma_zs_methods,
            prefix="pred_score_",
            base_cols=base_cols,
        ))
        gemma_fs_seed_dfs.append(load_gemma_few_shot_predictions(
            seed=s,
            method_suffixes=gemma_fs_methods,
            prefix="pred_score_",
            base_cols=base_cols,
        ))

    backbones = [bb for bb in BACKBONE_ORDER
                 if any((df["model_name"] == bb).any() for df in seed_dfs)]

    # ── Top row data: AUC per (backbone, method), averaged over seeds. ──
    bb_auc_by_seed = []
    for df, gemma_zs_df, gemma_fs_df in zip(
        seed_dfs, gemma_zs_seed_dfs, gemma_fs_seed_dfs
    ):
        auc = auc_per_backbone_method(df, methods, backbones)
        for gemma_df, gemma_methods in (
            (gemma_zs_df, gemma_zs_methods),
            (gemma_fs_df, gemma_fs_methods),
        ):
            if gemma_df.empty or not gemma_methods:
                continue
            gemma_auc = auc_per_backbone_method(
                gemma_df, list(gemma_methods), backbones
            )
            for bb in gemma_auc.index:
                for method in gemma_auc.columns:
                    auc.at[bb, method] = gemma_auc.at[bb, method]
        bb_auc_by_seed.append(auc)
    bb_mean = pd.concat(bb_auc_by_seed, keys=range(len(seed_dfs))).groupby(
        level=1).mean().loc[backbones]

    # ── Bottom row data: silent vs loud AUC, averaged over seeds. ──
    fmt_keys = build_format_error_lookup()
    silent_act_aucs = {m: [] for m in methods}
    loud_aucs       = {m: [] for m in methods}
    for df, gemma_zs_df, gemma_fs_df in zip(
        seed_dfs, gemma_zs_seed_dfs, gemma_fs_seed_dfs
    ):
        cls = classify_dataframe(df, failed_col="step_label",
                                  trace_label_col="trace_label")
        is_silent = cls == SILENT
        is_loud   = cls == LOUD
        is_normal = df["step_label"].astype(int) == 0
        fmt_mask = pd.Series(False, index=df.index)
        for idx in df.index[is_silent]:
            fmt_mask.at[idx] = is_fmt_silent_row(df.loc[idx], fmt_keys)
        is_silent_actions = is_silent & (~fmt_mask)
        sa = auc_per_method(df, methods, is_silent_actions, is_normal)
        la = auc_per_method(df, methods, is_loud,           is_normal)

        for gemma_df, gemma_methods in (
            (gemma_zs_df, gemma_zs_methods),
            (gemma_fs_df, gemma_fs_methods),
        ):
            if gemma_df.empty or not gemma_methods:
                continue
            gemma_cls = classify_dataframe(
                gemma_df, failed_col="step_label", trace_label_col="trace_label"
            )
            gemma_silent = gemma_cls == SILENT
            gemma_loud = gemma_cls == LOUD
            gemma_normal = gemma_df["step_label"].astype(int) == 0
            gemma_fmt_mask = pd.Series(False, index=gemma_df.index)
            for idx in gemma_df.index[gemma_silent]:
                gemma_fmt_mask.at[idx] = is_fmt_silent_row(gemma_df.loc[idx], fmt_keys)
            gemma_silent_actions = gemma_silent & (~gemma_fmt_mask)
            sa.update(auc_per_method(
                gemma_df, list(gemma_methods), gemma_silent_actions, gemma_normal
            ))
            la.update(auc_per_method(
                gemma_df, list(gemma_methods), gemma_loud, gemma_normal
            ))

        for m in methods:
            silent_act_aucs[m].append(sa[m])
            loud_aucs[m].append(la[m])
    def mean_or_nan(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        return float(finite.mean()) if len(finite) else np.nan

    silent_act_mean = np.array([mean_or_nan(silent_act_aucs[m]) for m in methods])
    loud_mean       = np.array([mean_or_nan(loud_aucs[m])       for m in methods])

    # ── Plot: 2 side-by-side subplots, shared y axis. ──────────────────
    n = len(methods)
    fig_w = 0.42 * n * 2 + 2.0
    fig_h = fig_w / 5.2
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h), sharey=True,
        gridspec_kw={"wspace": 0.02},
    )

    x_pos = np.arange(n)

    # ── Left subplot: per-backbone AUC dots. ───────────────────────────
    series_left = [[bb_mean.at[bb, m] for m in methods] for bb in backbones]
    colors_left = [BACKBONE_COLOR.get(bb, "#888888") for bb in backbones]
    draw_dots(ax_left, x_pos, series_left, colors_left)
    ax_left.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_left.set_xlim(-0.6, n - 0.4)
    ax_left.set_ylim(0.2, 1.0)
    ax_left.set_ylabel("AUCROC", fontsize=BASE_FONT - 1)
    ax_left.tick_params(axis="y", labelsize=BASE_FONT - 3)
    ax_left.grid(axis="y", linestyle=":", alpha=0.5)
    draw_family_separators(ax_left, methods)
    draw_family_brackets(ax_left, methods)
    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels([METHOD_SHORT.get(m, m) for m in methods],
                            fontsize=BASE_FONT - 3, color="black",
                            rotation=45, ha="right", rotation_mode="anchor")

    def _bb_label(bb: str) -> str:
        if bb in BACKBONE_LEGEND_DISPLAY:
            return BACKBONE_LEGEND_DISPLAY[bb]
        return BACKBONE_DISPLAY.get(bb, bb).replace("\n", " ")

    bb_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none",
                   color=BACKBONE_COLOR[bb], markeredgecolor="black",
                   markersize=markersize, label=_bb_label(bb))
        for bb in backbones
    ]
    # Split into two: first half upper-left, rest lower-right.
    split = (len(bb_handles) + 1) // 2
    leg_tl = ax_left.legend(handles=bb_handles[:split], loc="upper left",
                            ncol=split,
                            fontsize=BASE_FONT - 3,
                            frameon=True, edgecolor="lightgray", fancybox=True,
                            handletextpad=0.4, columnspacing=0.1,
                            borderaxespad=0.6)
    ax_left.add_artist(leg_tl)
    ax_left.legend(handles=bb_handles[split:], loc="lower right",
                   ncol=len(bb_handles) - split,
                   fontsize=BASE_FONT - 3,
                   frameon=True, edgecolor="lightgray", fancybox=True,
                   handletextpad=0.4, columnspacing=0.1,
                   borderaxespad=0.6)
    ax_left.set_title("AUCROC on test-set splits by LLM backbone",
                      fontsize=BASE_FONT+4, fontweight="bold", y=1.25)

    # ── Right subplot: silent vs loud AUC dots. ───────────────────────
    series_right = [silent_act_mean, loud_mean]
    colors_right = [SILENT_COLOR, LOUD_COLOR]
    draw_dots(ax_right, x_pos, series_right, colors_right)
    ax_right.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_right.set_xlim(-0.6, n - 0.4)
    ax_right.tick_params(axis="y", labelsize=BASE_FONT - 3)
    ax_right.grid(axis="y", linestyle=":", alpha=0.5)
    draw_family_separators(ax_right, methods)
    draw_family_brackets(ax_right, methods)
    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels([METHOD_SHORT.get(m, m) for m in methods],
                             fontsize=BASE_FONT - 2, color="black",
                             rotation=45, ha="right", rotation_mode="anchor")

    sl_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none",
                   color=SILENT_COLOR, markeredgecolor="black",
                   markersize=markersize, label="Silent"),
        plt.Line2D([0], [0], marker="o", linestyle="none",
                   color=LOUD_COLOR,   markeredgecolor="black",
                   markersize=markersize, label="Loud"),
    ]
    ax_right.legend(handles=sl_handles, loc="upper left",
                    ncol=2,
                    fontsize=BASE_FONT - 3,
                    frameon=True, edgecolor="lightgray", fancybox=True,
                    handletextpad=0.4, columnspacing=0.6,
                    borderaxespad=0.6)
    ax_right.set_title("AUCROC on test-set splits by silent/loud anomalies only",
                       fontsize=BASE_FONT +4, fontweight="bold", y=1.25)

    fig.subplots_adjust(left=0.045, right=0.995, top=0.76, bottom=0.24,
                        wspace=0.02)
    fig.savefig(PLOT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined chart to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
