from pyod.models.ocsvm import OCSVM
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from benchmark_AD.methods.tabular.prepare_traces_tabular import prepare_traces_tabular
from benchmark_AD.utils import log_results, predict_scores

def ocsvm(args, X_train, y_train, X_test, y_test, train_ids, test_ids):
    ocsvm = OCSVM()
    ocsvm.fit(X_train)
    y_test_scores = ocsvm.decision_function(X_test)

    y_test_pred = predict_scores(args, y_test_scores, y_test)

    f1 = f1_score(y_test, y_test_pred)
    acc = accuracy_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_scores)
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)

    print(f"OCSVM F1 Score: {f1:.4f}")
    print(f"OCSVM Accuracy: {acc:.4f}")
    print(f"OCSVM AUC: {auc:.4f}")
    print(f"OCSVM Balanced Accuracy: {bal_acc:.4f}")

    log_results(args, 0, 1, f1, acc, auc, bal_acc, model=None, gt_labels=y_test, pred_labels=y_test_pred, pred_scores=y_test_scores, train_ids=train_ids, test_ids=test_ids)  # 0 for epoch


def run_ocsvm(args, train_df, test_df):
    print("Running OCSVM...")
    train_type = 'occ'
    X_train, y_train, X_test, y_test, train_ids, test_ids = prepare_traces_tabular(args, train_type, train_df, test_df)
    ocsvm(args, X_train, y_train, X_test, y_test, train_ids, test_ids)