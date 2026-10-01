"""
deepseek_seed_merge_auc.py — for the 3-MAS-seed deepseek test, compare the
in-distribution (method / OLD) AUC against the transfer AUC on the NEW deepseek
traces, where the three MAS seeds' `*_testonly` predictions are POOLED into one
set before computing AUC.

Why pool for AUC: each MAS seed produced a different set of new traces; AUC is a
rank metric on `pred_score`, so pooling the three seeds' scores gives a single
aggregate transfer AUC (thresholds/pred_label aren't needed).

Two seed levels:
  outer  = MAS run seed  (deepseek*_3seeds_1/2/3)  -> different NEW trace sets
  inner  = detector seed (*_seed1..5 CSVs)         -> different trained detectors

For each detector and each inner detector-seed k:
  method AUC (old) : from method/<det>_seed{k}   (should match across MAS seeds)
  merged AUC (new) : AUC over the 3 MAS seeds' <det>_testonly_seed{k} concatenated
Then report mean +/- std over the 5 detector seeds.
"""

from __future__ import annotations

import csv
import glob
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "Deepseek seed tests (only 3 random seeds for run)"
DETECTORS = ["g_safeguard", "svm", "xgboost"]
SEEDS = [1, 2, 3, 4, 5]


def _to_int(v):
    s = str(v).strip().lower()
    if s in ("true", "t"):
        return 1
    if s in ("false", "f"):
        return 0
    return int(round(float(s)))


def _load(path):
    gt, score = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt.append(_to_int(r["gt_label"]))
            score.append(float(r["pred_score"]))
    return gt, score


def _auc(gt, scores):
    pos = [1 for g in gt if g == 1]
    if not pos or len(pos) == len(gt):
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(r for g, r in zip(gt, ranks) if g == 1)
    n_pos = sum(1 for g in gt if g == 1)
    n_neg = len(gt) - n_pos
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _mean_std(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return float("nan"), 0.0
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
    return m, sd


def _mas_dirs():
    dirs = sorted(glob.glob(str(ROOT / "deepseek*3seeds_*")))
    return [Path(d) for d in dirs if Path(d).is_dir()]


def _pred_dir(mas_dir: Path) -> Path:
    hits = sorted(glob.glob(str(mas_dir / "*raw_final_predictions*")))
    return Path(hits[0])


def _csv(pred_dir: Path, variant: str, k: int) -> Path:
    return pred_dir / variant / f"{variant}_seed{k}_action_predictions.csv"


def main():
    mas_dirs = _mas_dirs()
    pred_dirs = [_pred_dir(d) for d in mas_dirs]
    print(f"MAS seeds found: {len(mas_dirs)}")
    for d in mas_dirs:
        print(f"  - {d.name}")

    print(f"\n{'='*90}")
    print(f"{'Detector':<14}{'method AUC (old)':>22}{'merged testonly AUC (new)':>28}"
          f"{'Δ (new-old)':>14}{'pooled N':>10}")
    print("-" * 90)

    for det in DETECTORS:
        method_aucs, merged_aucs = [], []
        method_consistency = []   # max spread of method AUC across MAS seeds, per k
        pooled_n = 0
        for k in SEEDS:
            # method (old): read from every MAS seed and check they agree
            m_vals = []
            for pd in pred_dirs:
                fp = _csv(pd, det, k)
                if fp.exists():
                    m_vals.append(_auc(*_load(fp)))
            m_vals = [v for v in m_vals if not math.isnan(v)]
            if m_vals:
                method_aucs.append(m_vals[0])
                method_consistency.append(max(m_vals) - min(m_vals))
            # merged testonly (new): pool the 3 MAS seeds' predictions
            gt_all, sc_all = [], []
            for pd in pred_dirs:
                fp = _csv(pd, f"{det}_testonly", k)
                if fp.exists():
                    g, s = _load(fp)
                    gt_all += g
                    sc_all += s
            if gt_all:
                merged_aucs.append(_auc(gt_all, sc_all))
                pooled_n = len(gt_all)

        mo, so = _mean_std(method_aucs)
        mn, sn = _mean_std(merged_aucs)
        print(f"{det:<14}{f'{mo:.3f} ± {so:.3f}':>22}{f'{mn:.3f} ± {sn:.3f}':>28}"
              f"{mn - mo:>+14.3f}{pooled_n:>10}")

    print("-" * 90)
    # sanity: is method AUC really identical across MAS seeds?
    max_spread = 0.0
    for det in DETECTORS:
        for k in SEEDS:
            vals = []
            for pd in pred_dirs:
                fp = _csv(pd, det, k)
                if fp.exists():
                    vals.append(_auc(*_load(fp)))
            vals = [v for v in vals if not math.isnan(v)]
            if len(vals) > 1:
                max_spread = max(max_spread, max(vals) - min(vals))
    print(f"method AUC max spread across the 3 MAS seeds: {max_spread:.2e} "
          f"({'identical (as expected)' if max_spread < 1e-9 else 'NOT identical — check'})")
    print("(mean ± std over the 5 detector-training seeds; testonly pooled over the 3 MAS seeds)")


if __name__ == "__main__":
    main()
