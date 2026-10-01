"""Generate the main-results LaTeX tables (metrics + p-values vs Random).

Sources:
  * data/full_data_all_plus_semi_sup_with_gemma/ (all methods; Gemma few-shot
    only filled for seeds 1-3)
  * data/gemma_zeroshot_full_data/ (Gemma zero-shot, pooled over the 4
    dataset x tool splits per seed)
Per seed we compute F1 / Acc / AUC / Bal.Acc on pooled rows, then mean +- std
over seeds. p-values: one-tailed two-sample (Welch) t-test, Method > Random.
Also prints sanity diagnostics. Output: main_results_tables.tex
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from methods_meta import LLM_CSV_SUFFIX, METHOD_ORDER_IN_CATEGORY, METHOD_CATEGORY, CATEGORY_ORDER, METHOD_MODALITY
from gemma_split_utils import GEMMA_SPLITS

HERE = Path(__file__).parent
MAIN = HERE / "data" / "full_data_all_plus_semi_sup_with_gemma"
ZS = HERE / "data" / "gemma_zeroshot_full_data"
SEEDS = [1, 2, 3, 4, 5]
METRICS = ["F1", "Acc", "AUC", "Bal. Acc"]

DISPLAY = {
    "isolation_forest": "Isolation Forest", "knn": "KNN", "lof": "LOF",
    "autoencoder_clean": "Autoencoder", "deepsvdd_clean": "DeepSVDD",
    "ocsvm_clean": "OC-SVM", "dominant_clean": "DOMINANT", "tam_clean": "TAM",
    "blindguard_clean": "BlindGuard", "deepsad": "DeepSAD", "devnet": "DevNet",
    "ggad": "GGAD", "g_safeguard_semi_supervised": "G-Safe (semi-sup)",
    "svm": "SVM", "random_forest": "Random Forest", "xgboost": "XGBoost",
    "g_safeguard": "G-Safeguard",
}
for k, v in {"e2b": "E2B", "e4b": "E4B", "31b": "31B", "26b_a4b": "26B-A4B"}.items():
    DISPLAY[f"gemma4_{k}_zs"] = f"Gemma4-{v} (zero-shot)"
    DISPLAY[f"gemma4_{k}_fs"] = f"Gemma4-{v} (few-shot)"
FAMILY = {"Tabular": "Tabular", "Graph": "Graph", "MAS-specific": "MAS-specific", "LLM": "LLM"}
SUP_NAME = {"Unsupervised": "Unsupervised", "OCC": "One-Class",
            "Semi-supervised": "Semi-Supervised", "Supervised": "Supervised"}


def auc(y, s):
    y = np.asarray(y, int); r = pd.Series(s).rank(method="average").to_numpy()
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def metrics(y, pred, score):
    y = np.asarray(y, int); p = np.asarray(pred, int)
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum()
    fn = ((p == 0) & (y == 1)).sum(); tn = ((p == 0) & (y == 0)).sum()
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    tpr = tp / (tp + fn); tnr = tn / (tn + fp)
    return dict(F1=f1, Acc=(tp + tn) / len(y), AUC=auc(y, score), **{"Bal. Acc": (tpr + tnr) / 2})


def col_pair(key):
    suf = LLM_CSV_SUFFIX.get(key, key)
    return f"pred_label_{suf}", f"pred_score_{suf}"


def load_seed(seed, key):
    lc, sc = col_pair(key)
    if key.endswith("_zs"):
        parts = []
        for ds, tt in GEMMA_SPLITS:
            p = ZS / ds / tt / f"full_test_data_with_preds_seed{seed}.csv"
            if p.exists():
                parts.append(pd.read_csv(p, usecols=["step_label", lc, sc]))
        df = pd.concat(parts, ignore_index=True) if parts else None
    else:
        df = pd.read_csv(MAIN / f"full_test_data_with_preds_seed{seed}.csv", usecols=["step_label", lc, sc])
    if df is None:
        return None
    df = df.dropna(subset=[lc, sc])
    return df if len(df) else None


def per_seed(key):
    lc, sc = col_pair(key)
    out = {}
    for s in SEEDS:
        df = load_seed(s, key)
        if df is None:
            continue
        m = metrics(df["step_label"], df[lc], df[sc]); m["n"] = len(df); m["pos"] = df["step_label"].mean()
        out[s] = m
    return out


def fmt(x):
    return f"{x:.3f}".lstrip("0") if x < 1 else f"{x:.3f}"


def fmt_p(p):
    return "$<.001$" if p < 0.001 else f"${fmt(p)}$"


def main():
    order = [m for c in CATEGORY_ORDER for m in METHOD_ORDER_IN_CATEGORY[c]]
    res = {m: per_seed(m) for m in order}
    res["random"] = per_seed("random")
    vals = {m: {k: np.array([res[m][s][k] for s in res[m]]) for k in METRICS} for m in res}

    # ---- diagnostics ----
    print("== per-method seeds used / n rows / pos rate ==")
    for m in ["random"] + order:
        r = res[m]
        print(f"{m:30s} seeds={list(r)} n={[r[s]['n'] for s in r]} pos={[round(r[s]['pos'],3) for s in r]}")

    # ---- best (bold) per metric ----
    best = {k: max(order, key=lambda m: vals[m][k].mean()) for k in METRICS}
    ref = vals["random"]

    def pv(m, k):
        return stats.ttest_ind(vals[m][k], ref[k], equal_var=False, alternative="greater").pvalue

    def cell(m, k):
        mu, sd = vals[m][k].mean(), vals[m][k].std(ddof=1) if len(vals[m][k]) > 1 else float("nan")
        body = f"{fmt(mu)}_{{ \\pm {fmt(sd)} }}"
        if best[k] == m:
            body = f"\\mathbf{{{body}}}"
        p = pv(m, k)
        color = "\\cellcolor{blue!50} " if p < .01 else "\\cellcolor{blue!25} " if p < .05 else ""
        return f"{color}${body}$"

    hdr = ("\\toprule\n\\textbf{Supervision} & \\textbf{Method} & \\textbf{Method Family} & "
           "\\textbf{F1} & \\textbf{Acc} & \\textbf{AUC} & \\textbf{Bal. Acc} \\\\\n\\midrule\n")
    rnd = "\\multirow{1}{*}{-} & Random & Random & " + " & ".join(
        f"${fmt(ref[k].mean())}_{{ \\pm {fmt(ref[k].std(ddof=1))} }}$" for k in METRICS) + "  \\\\\n\\midrule\n"
    rnd_p = "\\multirow{1}{*}{-} & Random & Random & - & - & - & -  \\\\\n\\midrule\n"

    def body(cellfn, rnd_row):
        s = rnd_row
        for ci, c in enumerate(CATEGORY_ORDER):
            ms = METHOD_ORDER_IN_CATEGORY[c]
            s += f"\\multirow{{{len(ms)}}}{{*}}{{{SUP_NAME[c]}}} \n"
            for m in ms:
                s += f"& {DISPLAY[m]} & {FAMILY[METHOD_MODALITY[m]]} & " + " & ".join(cellfn(m, k) for k in METRICS) + "  \\\\\n"
            s += "\\midrule\n" if ci < len(CATEGORY_ORDER) - 1 else "\\bottomrule\n"
        return s

    best_p = {k: pv(best[k], k) for k in METRICS}
    t1 = ("% ----------------- results table ------------------\n\\begin{table*}[ht]\n    \\centering\n    \\small\n"
          "    \\setlength{\\tabcolsep}{3pt}\n    \\caption{\n        Overall anomaly detection performance across four supervision settings. "
          "Dark blue cells indicate $p < 0.01$ and light blue cells indicate $p < 0.05$ relative to \\textit{Random} performance.\n    }\n"
          "    \\label{tab:main_results}\n    \\resizebox{\\linewidth}{!}{\n    \\begin{tabular}{lllcccc}\n"
          + hdr + body(cell, rnd) + "    \\end{tabular}}\n\\end{table*}\n")
    t2 = ("\n% ----------------- results table ------------------\n\\begin{table*}[ht]\n    \\centering\n    \\small\n"
          "    \\setlength{\\tabcolsep}{3pt}\n    \\caption{\n        One-tailed two-sample t-test p-values (Method $>$ Random) for overall "
          "anomaly detection performance across four performance metrics.\n    }\n    \\label{tab:main_results_pvalues_n5}\n"
          "    \\begin{tabular}{lllcccc}\n" + hdr
          + body(lambda m, k: (f"$\\mathbf{{{'<.001' if pv(m,k)<.001 else fmt(pv(m,k))}}}$" if best[k] == m else fmt_p(pv(m, k))), rnd_p)
          + "    \\end{tabular}\n\\end{table*}\n")
    pd.DataFrame(
        [{"method": m, **{k: pv(m, k) for k in METRICS}} for m in order]
    ).to_csv(HERE / "main_results_pvalues.csv", index=False)
    out = HERE / "main_results_tables.tex"
    out.write_text(t1 + t2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
