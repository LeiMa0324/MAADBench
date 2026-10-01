from pyod.models.knn import KNN
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from benchmark_AD.methods.tabular.prepare_traces_tabular import prepare_traces_tabular
from benchmark_AD.utils import log_results, predict_scores

def knn(args, X_train, y_train, X_test, y_test, train_ids, test_ids):
    KNN_model = KNN()
    KNN_model.fit(X_train)
    KNN_scores = KNN_model.decision_function(X_test)  # the higher, the more abnormal
    
    KNN_preds = predict_scores(args, KNN_scores, y_test)
    
    f1 = f1_score(y_test, KNN_preds)
    acc = accuracy_score(y_test, KNN_preds)
    auc = roc_auc_score(y_test, KNN_scores)
    bal_acc = balanced_accuracy_score(y_test, KNN_preds)
    
    print(f"KNN F1 Score: {f1:.4f}")
    print(f"KNN Accuracy: {acc:.4f}")
    print(f"KNN AUC: {auc:.4f}")
    print(f"KNN Balanced Accuracy: {bal_acc:.4f}")

    log_results(args, 0, 1, f1, acc, auc, bal_acc, model=None, gt_labels=y_test, pred_labels=KNN_preds, pred_scores=KNN_scores, train_ids=train_ids, test_ids=test_ids)  # 0 for epoch


def run_knn(args, train_df, test_df):
    print("Running KNN")
    
    train_type = "unsupervised"
    # read in data and convert raw traces to tabular format
    X_train, y_train, X_test, y_test, train_ids, test_ids = prepare_traces_tabular(args, train_type, train_df, test_df)
    knn(args, X_train, y_train, X_test, y_test, train_ids, test_ids)