"""
detector_transfer_stats.py — summarise detector-transfer results from the raw
prediction CSVs.

For each test folder (GPT, Opus) and each detector (g_safeguard, svm, xgboost),
compute F1 / Accuracy / ROC-AUC / Balanced-Accuracy per seed and report the
mean +/- std over seeds:

    <detector>            -> OLD trace (detector trained + tested in-distribution)
    <detector>_testonly   -> NEW trace (detector trained on old, applied to new)

Raw CSV columns: gt_label, pred_label, pred_score, test_id
"""

from __future__ import annotations

import csv
import glob
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# label -> test-folder name (raw-predictions dir is auto-detected inside each)
TEST_FOLDERS = {
    "GPT": "ojal_s gpt test",
    "Opus": "opus tests",
}
DETECTORS = ["g_safeguard", "svm", "xgboost"]


# ── metrics (self-contained, no sklearn dependency) ────────────────────────

def _f1(gt, pred):
    tp = sum(1 for g, p in zip(gt, pred) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gt, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gt, pred) if g == 1 and p == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _acc(gt, pred):
    return sum(1 for g, p in zip(gt, pred) if g == p) / len(gt) if gt else 0.0


def _bal_acc(gt, pred):
    tp = sum(1 for g, p in zip(gt, pred) if g == 1 and p == 1)
    tn = sum(1 for g, p in zip(gt, pred) if g == 0 and p == 0)
    npos = sum(1 for g in gt if g == 1)
    nneg = sum(1 for g in gt if g == 0)
    tpr = tp / npos if npos else 0.0
    tnr = tn / nneg if nneg else 0.0
    return (tpr + tnr) / 2


def _auc(gt, scores):
    # Mann-Whitney U with average ranks for ties.
    pos = [s for g, s in zip(gt, scores) if g == 1]
    neg = [s for g, s in zip(gt, scores) if g == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(r for g, r in zip(gt, ranks) if g == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _to_int_label(v):
    """Parse a label that may be '0'/'1', 'True'/'False', or '0.0'/'1.0'."""
    s = str(v).strip().lower()
    if s in ("true", "t"):
        return 1
    if s in ("false", "f"):
        return 0
    return int(round(float(s)))


def _load(path):
    gt, pred, score = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt.append(_to_int_label(r["gt_label"]))
            pred.append(_to_int_label(r["pred_label"]))
            score.append(float(r["pred_score"]))
    return gt, pred, score


def _mean_std(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return float("nan"), 0.0
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
    return m, sd


def variant_stats(pred_dir: Path, variant: str):
    """Mean/std of each metric over all seed CSVs of one variant folder."""
    folder = pred_dir / variant
    files = sorted(glob.glob(str(folder / f"{variant}_seed*_action_predictions.csv")))
    f1s, accs, aucs, bals = [], [], [], []
    for fp in files:
        gt, pred, score = _load(fp)
        f1s.append(_f1(gt, pred))
        accs.append(_acc(gt, pred))
        aucs.append(_auc(gt, score))
        bals.append(_bal_acc(gt, pred))
    return len(files), {
        "F1": _mean_std(f1s), "Acc": _mean_std(accs),
        "AUC": _mean_std(aucs), "BalAcc": _mean_std(bals),
    }


def _find_pred_dir(test_folder: Path) -> Path:
    hits = [Path(p) for p in glob.glob(str(test_folder / "*raw_final_predictions"))]
    if not hits:
        raise FileNotFoundError(f"no *raw_final_predictions dir under {test_folder}")
    return hits[0]


def main():
    metrics = ["F1", "Acc", "AUC", "BalAcc"]
    for label, folder in TEST_FOLDERS.items():
        test_folder = HERE / folder
        if not test_folder.exists():
            print(f"[skip] {label}: {test_folder} not found")
            continue
        pred_dir = _find_pred_dir(test_folder)
        print(f"\n{'='*78}\n{label} test  ({folder})\n{'='*78}")
        print(f"{'Detector':<14}{'Trace':<6}{'seeds':>6}"
              + "".join(f"{m:>14}" for m in metrics))
        print("-" * 78)
        for det in DETECTORS:
            for variant, trace in [(det, "old"), (f"{det}_testonly", "new")]:
                n, st = variant_stats(pred_dir, variant)
                if n == 0:
                    print(f"{det:<14}{trace:<6}{'-':>6}  (no files)")
                    continue
                cells = "".join(f"{st[m][0]:>8.3f}±{st[m][1]:<5.3f}" for m in metrics)
                print(f"{det:<14}{trace:<6}{n:>6}{cells}")
            print("-" * 78)


if __name__ == "__main__":
    main()
