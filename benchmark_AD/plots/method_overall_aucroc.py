"""Bar chart of overall ROC-AUC per method across the 5 seeds.

For each seed CSV in data/full_data_all_plus_semi_sup_with_gemma/, compute one
AUC per method using all rows (no per-action split). Plot one bar per method
with seed-to-seed std error bars, methods grouped by family with bracket
annotations on top.
"""
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".mplconfig"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgba

mpl.rcParams["font.family"] = "Trebuchet MS"
mpl.rcParams["font.weight"] = "normal"
BASE_FONT = 20
mpl.rcParams["font.size"] = BASE_FONT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - fallback for local sklearn/numpy ABI issues.
    def roc_auc_score(y_true, y_score):
        y = pd.Series(y_true).astype(int).to_numpy()
        ranks = pd.Series(y_score).astype(float).rank(method="average").to_numpy()
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Only one class present in y_true.")
        return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

DATA_DIR = Path(__file__).parent / "data" / "full_data_all_plus_semi_sup_with_gemma"
LLM_DATA_DIR = DATA_DIR  # LLM lives in the same dir now.
GEMMA_ZERO_SHOT_DATA_DIR = Path(__file__).parent / "data" / "gemma_zeroshot_full_data"
PLOT_PATH = Path(__file__).with_suffix(".pdf")
SEEDS = [1, 2, 3, 4, 5]

# Method/family/modality metadata is shared across all exp_analysis scripts.
# This script ignores polluted variants and the random baseline entirely.
from methods_meta import (
    LLM_CSV_SUFFIX, METHOD_SHORT, METHOD_CATEGORY,
    CATEGORY_ORDER, METHOD_ORDER_IN_CATEGORY,
    METHOD_MODALITY, MODALITY_ORDER,
    ordered_methods,
)

# Extra (mean, std) values for methods that aren't in the CSV. Currently
# empty — polluted variants and the random baseline are intentionally
# excluded. Add LaTeX-table-only methods here if you need to overlay
# external numbers.
EXTERNAL_AUC: dict[str, tuple[float, float]] = {}

# Bar gradient palette (kept for reference / fallback).
BAR_PALETTE = ["#D7433B", "#F06A63", "#FF8E5E", "#FFCC3D", "#95CAA6", "#008D98"]
BAR_CMAP = LinearSegmentedColormap.from_list("bar_palette", BAR_PALETTE)

# Bars are colored by data modality (METHOD_MODALITY/MODALITY_ORDER imported
# from methods_meta — single source of truth across all exp_analysis plots).
# Palette: #D7433B, #F06A63, #FF8E5E, #FFCC3D, #95CAA6, #008D98.
# Picked 4 evenly-spaced colors (#F06A63, #FFCC3D unused).
MODALITY_COLOR = {
    "Tabular":      "#D7433B",  # red
    "Graph":        "#FF8E5E",  # orange
    "MAS-specific": "#95CAA6",  # sage
    "LLM":          "#008D98",  # teal
}

GEMMA_SPLITS = [
    ("EscapeRoom_v3_gsm_hard", "no_tool"),
    ("EscapeRoom_v3_gsm_hard", "tool"),
    ("EscapeRoom_v3_live_code_bench", "no_tool"),
    ("EscapeRoom_v3_live_code_bench", "tool"),
]

def methods_in(cols: list[str]) -> set[str]:
    return {c[len("pred_score_"):] for c in cols if c.startswith("pred_score_")}


def overall_auc_per_method(df: pd.DataFrame, methods: list[str]) -> dict[str, float]:
    y = df["step_label"].to_numpy()
    out: dict[str, float] = {}
    for m in methods:
        score = df[f"pred_score_{m}"].to_numpy()
        try:
            out[m] = roc_auc_score(y, score)
        except ValueError:
            out[m] = float("nan")
    return out


def auc_from_columns(df: pd.DataFrame, score_col: str) -> float:
    sub = df[["step_label", score_col]].dropna(subset=[score_col])
    if sub.empty:
        return float("nan")
    y = sub["step_label"].astype(int).to_numpy()
    score = sub[score_col].to_numpy()
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return float("nan")


def gemma_zero_shot_aucs_for_seed(seed: int, methods: dict[str, str]) -> dict[str, float]:
    """Pool math/code × tool/no-tool split files once, then compute Gemma AUCs."""
    score_cols = {
        internal_key: f"pred_score_{csv_suffix}"
        for internal_key, csv_suffix in methods.items()
    }
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
        available_score_cols = [col for col in score_cols.values() if col in cols]
        if not available_score_cols:
            continue
        parts.append(pd.read_csv(path, usecols=["step_label"] + available_score_cols))

    if not parts:
        return {internal_key: float("nan") for internal_key in methods}
    pooled = pd.concat(parts, ignore_index=True)
    return {
        internal_key: (
            auc_from_columns(pooled, score_col)
            if score_col in pooled.columns else float("nan")
        )
        for internal_key, score_col in score_cols.items()
    }


def main() -> None:
    # ── Collect overall AUC per (seed × method) ─────────────────────────
    seed_headers: dict[int, list[str]] = {}
    method_score_set: set[str] | None = None
    for s in SEEDS:
        cols = list(pd.read_csv(DATA_DIR / f"full_test_data_with_preds_seed{s}.csv",
                                nrows=0).columns)
        seed_headers[s] = cols
        scores = methods_in(cols)
        method_score_set = scores if method_score_set is None else method_score_set & scores

    assert method_score_set is not None
    # Drop the random baseline, polluted variants, and raw LLM column names.
    # Keep all-empty non-polluted methods in place so missing configurations
    # render as blank bars.
    # (those go through LLM_CSV_SUFFIX with internal keys instead).
    raw_llm_cols = set(LLM_CSV_SUFFIX.values())
    csv_keep = sorted(
        m for m in method_score_set
        if m != "random"
        and not m.endswith("_polluted")
        and m not in raw_llm_cols
    )

    # ── LLM AUCs computed from full_data_preds_with_llm/ ────────────────
    llm_seed_aucs: dict[str, list[float]] = {k: [] for k in LLM_CSV_SUFFIX}
    zero_shot_methods = {
        internal_key: csv_suffix
        for internal_key, csv_suffix in LLM_CSV_SUFFIX.items()
        if internal_key.endswith("_zs")
    }
    few_shot_methods = {
        internal_key: csv_suffix
        for internal_key, csv_suffix in LLM_CSV_SUFFIX.items()
        if not internal_key.endswith("_zs")
    }
    for s in SEEDS:
        zs_aucs = gemma_zero_shot_aucs_for_seed(s, zero_shot_methods)
        for internal_key in zero_shot_methods:
            llm_seed_aucs[internal_key].append(zs_aucs.get(internal_key, float("nan")))

        few_shot_cols = [
            f"pred_score_{csv_suffix}"
            for csv_suffix in few_shot_methods.values()
            if f"pred_score_{csv_suffix}" in seed_headers[s]
        ]
        df_llm = pd.read_csv(
            LLM_DATA_DIR / f"full_test_data_with_preds_seed{s}.csv",
            usecols=["step_label"] + few_shot_cols,
        )
        for internal_key, csv_suffix in few_shot_methods.items():
            col = f"pred_score_{csv_suffix}"
            if col not in df_llm.columns:
                llm_seed_aucs[internal_key].append(float("nan"))
                continue
            llm_seed_aucs[internal_key].append(auc_from_columns(df_llm, col))

    print("\nOverall ROC-AUC per LLM method (rows = seed, cols = method):")
    llm_table = pd.DataFrame(llm_seed_aucs, index=[f"seed{s}" for s in SEEDS])
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 200,
                           "display.max_columns", None):
        print(llm_table)
        print("\nMean / std (LLM, ddof=1):")
        print(pd.DataFrame({"mean": llm_table.mean(),
                            "std":  llm_table.std(ddof=1)}).round(4))

    # ── Build the full plotting list using the global ordering from
    # methods_meta.py (so this plot stays in sync with every other figure).
    csv_methods = csv_keep
    csv_set = set(csv_methods)
    llm_set = set(LLM_CSV_SUFFIX)
    available = csv_set | llm_set | set(EXTERNAL_AUC)
    methods = ordered_methods(available)

    # ── Per-method (mean, std). CSV methods recompute from raw data;
    # external methods take fixed (mean, std) from the LaTeX table.
    seed_aucs: dict[str, list[float]] = {m: [] for m in csv_methods}
    for s in SEEDS:
        score_cols = [f"pred_score_{m}" for m in csv_methods]
        df = pd.read_csv(
            DATA_DIR / f"full_test_data_with_preds_seed{s}.csv",
            usecols=["step_label"] + score_cols,
        )
        aucs = overall_auc_per_method(df, csv_methods)
        for m, v in aucs.items():
            seed_aucs[m].append(v)

    print("\nOverall ROC-AUC per CSV method (rows = seed, cols = method):")
    table = pd.DataFrame(seed_aucs, index=[f"seed{s}" for s in SEEDS])
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 200,
                           "display.max_columns", None):
        print(table)
        print("\nMean / std (CSV, ddof=1):")
        print(pd.DataFrame({"mean": table.mean(), "std": table.std(ddof=1)}).round(4))

    means_dict: dict[str, float] = {}
    stds_dict:  dict[str, float] = {}
    for m in csv_methods:
        vals = np.asarray(seed_aucs[m], dtype=float)
        means_dict[m] = float(np.mean(vals))
        stds_dict[m]  = float(np.std(vals, ddof=1))
    for m, vals_list in llm_seed_aucs.items():
        vals = np.asarray(vals_list, dtype=float)
        if np.isfinite(vals).any():
            means_dict[m] = float(np.nanmean(vals))
            stds_dict[m] = float(np.nanstd(vals, ddof=1))
        else:
            means_dict[m] = float("nan")
            stds_dict[m] = float("nan")
    for m, (mu, sd) in EXTERNAL_AUC.items():
        means_dict.setdefault(m, mu)
        stds_dict.setdefault(m, sd)

    # ── Plot ─────────────────────────────────────────────────────────────
    n = len(methods)
    fig, ax = plt.subplots(figsize=(0.7 * n + 1.0, 6.8))

    positions = np.arange(n) + 1
    means = np.array([means_dict[m] for m in methods])
    stds  = np.array([stds_dict[m]  for m in methods])
    # Bar colors: encode the method's data-modality (Tabular / Graph /
    # MAS-specific / LLM). Falls back to a neutral gray if a method is
    # missing from the modality map.
    bar_colors = [
        MODALITY_COLOR.get(METHOD_MODALITY.get(m, ""), "#888888")
        for m in methods
    ]

    ax.bar(positions, means, width=0.7,
           color=[to_rgba(c, alpha=0.85) for c in bar_colors],
           edgecolor="black", linewidth=1.4,
           yerr=stds, capsize=9,
           error_kw=dict(elinewidth=2.2, ecolor="black", capthick=2.2),
           zorder=2)

    # Per-method significance vs Random — sourced from the AUC column of
    # the main LaTeX results table (tab:main_results). Values:
    #   "**"  = AUC cell colored blue!50  (p < 0.01)
    #   "*"   = AUC cell colored blue!25  (p < 0.05)
    #   "ns"  = AUC cell uncolored        (p >= 0.05)
    SIG_LEVELS: dict[str, str] = {
        # Unsupervised
        "isolation_forest":              "*",
        "knn":                           "ns",
        "lof":                           "ns",
        "gemma4_e2b_zs":                 "**",
        "gemma4_e4b_zs":                 "**",
        "gemma4_31b_zs":                 "**",
        "gemma4_26b_a4b_zs":             "**",
        # One-Class
        "blindguard_clean":              "ns",
        "ocsvm_clean":                   "**",
        "autoencoder_clean":             "ns",
        "deepsvdd_clean":                "*",
        "tam_clean":                     "ns",
        "dominant_clean":                "**",
        # Semi-Supervised
        "deepsad":                       "**",
        "devnet":                        "**",
        "ggad":                          "ns",
        "g_safeguard_semi_supervised":   "**",
        "gemma4_e2b_fs":                 "*",
        "gemma4_e4b_fs":                 "*",
        "gemma4_31b_fs":                 "*",
        "gemma4_26b_a4b_fs":             "**",
        # Supervised
        "g_safeguard":                   "**",
        "svm":                           "**",
        "random_forest":                 "**",
        "xgboost":                       "**",
    }

    # Prefer p-values written by make_main_results_tables.py so the markers
    # stay in sync with the table; the dict above is the fallback.
    pv_path = Path(__file__).parent / "main_results_pvalues.csv"
    if pv_path.exists():
        for _, r in pd.read_csv(pv_path).iterrows():
            SIG_LEVELS[r["method"]] = ("**" if r["AUC"] < 0.01
                                       else "*" if r["AUC"] < 0.05 else "ns")

    def _sig_marker(method: str) -> str:
        return SIG_LEVELS.get(method, "ns")

    # Numeric labels above each bar, with a significance marker appended.
    for x, mu, sd, m in zip(positions, means, stds, methods):
        if not np.isfinite(mu) or not np.isfinite(sd):
            continue
        num = f"{mu:.2f}".lstrip("0") if mu < 1 else f"{mu:.2f}"
        marker = _sig_marker(m)
        ax.text(x, mu + sd + 0.012, num,
                ha="right", va="bottom",
                fontsize=BASE_FONT, fontweight=500,
                color="black", zorder=4)
        if marker in ("**", "*"):
            ax.text(x, mu + sd + 0.012, marker,
                    ha="left", va="bottom",
                    fontsize=BASE_FONT, fontweight=700,
                    color="#C0392B", zorder=4)
        else:
            ax.text(x, mu + sd + 0.012, " ns",
                    ha="left", va="bottom",
                    fontsize=BASE_FONT, fontweight=400,
                    color="#999999", style="italic", zorder=4)

    ax.axhline(0.5, color="#444444", linestyle="--", linewidth=2.0, alpha=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_SHORT.get(m, m) for m in methods],
                       fontsize=BASE_FONT - 2,
                       rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("AUCROC", fontsize=BASE_FONT)
    ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
    ax.set_ylim(0.3, 1.0)
    ax.margins(x=0)
    ax.tick_params(axis="y", labelsize=BASE_FONT - 2)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Family separators between adjacent groups.
    cat_seq = [METHOD_CATEGORY.get(m) for m in methods]
    for j in range(1, n):
        if cat_seq[j] != cat_seq[j - 1]:
            ax.axvline(positions[j] - 0.5, color="gray",
                       linewidth=0.8, alpha=0.5)

    # Family bracket labels above each method group.
    family_groups: list[tuple[int, int, str]] = []
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

    brack_y = 1.02
    tick_h = 0.018
    label_y = 1.06
    for start, end, cat in family_groups:
        x_left = positions[start] - 0.4
        x_right = positions[end] + 0.4
        mid_x = (positions[start] + positions[end]) / 2
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
        ax.annotate(cat, xy=(mid_x, label_y),
                    xycoords=("data", "axes fraction"),
                    fontsize=BASE_FONT +2, fontweight=700, color="black",
                    ha="center", va="bottom", annotation_clip=False)

    # Modality color legend — single row with all 4 modalities.
    from matplotlib.patches import Patch
    modality_handles = [
        Patch(facecolor=MODALITY_COLOR[mod], edgecolor="black",
              linewidth=1.0, label=mod)
        for mod in MODALITY_ORDER
    ]
    ax.legend(handles=modality_handles, loc="lower center",
              bbox_to_anchor=(0.5, 1.22),
              ncol=len(MODALITY_ORDER),
              title="Method families",
              title_fontsize=BASE_FONT + 2,
              frameon=False, fontsize=BASE_FONT + 2,
              handlelength=1.4, handleheight=1.0,
              columnspacing=1.6, borderaxespad=0.0)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved bar chart to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
