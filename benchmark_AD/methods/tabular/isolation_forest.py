from pyod.models.iforest import IForest
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from benchmark_AD.methods.tabular.prepare_traces_tabular import prepare_traces_tabular
from benchmark_AD.utils import log_results, predict_scores

def isolation_forest(args, X_train, y_train, X_test, y_test, train_ids, test_ids):
    if_model = IForest()
    if_model.fit(X_train)
    IF_scores = if_model.decision_function(X_test)  # the higher, the more normal
    
    IF_preds = predict_scores(args, IF_scores, y_test)
    
    f1 = f1_score(y_test, IF_preds)
    acc = accuracy_score(y_test, IF_preds)
    auc = roc_auc_score(y_test, IF_scores)
    bal_acc = balanced_accuracy_score(y_test, IF_preds)

    # print the F1 score, accuracy, AUC and balanced accuracy of the IForest model
    print(f"IForest F1 Score: {f1:.4f}")
    print(f"IForest Accuracy: {acc:.4f}")
    print(f"IForest AUC: {auc:.4f}")
    print(f"IForest Balanced Accuracy: {bal_acc:.4f}")

    log_results(args, 0, 1, f1, acc, auc, bal_acc, model=None, gt_labels=y_test, pred_labels=IF_preds, pred_scores=IF_scores, train_ids=train_ids, test_ids=test_ids)  # 0 for epoch

def run_isolation_forest(args, train_df, test_df):
    print("Running Isolation Forest")
    
    train_type = "unsupervised"
    # read in data and convert raw traces to tabular format
    X_train, y_train, X_test, y_test, train_ids, test_ids = prepare_traces_tabular(args, train_type, train_df, test_df)
    isolation_forest(args, X_train, y_train, X_test, y_test, train_ids, test_ids)