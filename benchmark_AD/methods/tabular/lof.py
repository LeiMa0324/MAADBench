from pyod.models.lof import LOF
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from benchmark_AD.methods.tabular.prepare_traces_tabular import prepare_traces_tabular
from benchmark_AD.utils import log_results, predict_scores

def lof(args, X_train, y_train, X_test, y_test, train_ids, test_ids):
    LOF_model = LOF()
    LOF_model.fit(X_train)
    LOF_scores = LOF_model.decision_function(X_test)  # the higher, the more normal
    
    LOF_preds = predict_scores(args, LOF_scores, y_test)
     
    f1 = f1_score(y_test, LOF_preds)
    acc = accuracy_score(y_test, LOF_preds)
    auc = roc_auc_score(y_test, LOF_scores)
    bal_acc = balanced_accuracy_score(y_test, LOF_preds)    
    
    # print the F1 score, accuracy, AUC and balanced accuracy of the LOF model
    print(f"LOF F1 Score: {f1}")
    print(f"LOF Accuracy: {acc}")
    print(f"LOF AUC: {auc}")
    print(f"LOF Balanced Accuracy: {bal_acc}")

    log_results(args, 0, 1, f1, acc, auc, bal_acc, model=None, gt_labels=y_test, pred_labels=LOF_preds, pred_scores=LOF_scores, train_ids=train_ids, test_ids=test_ids)  # 0 for epoch


def run_lof(args, train_df, test_df):
    print("Running LOF")
    
    train_type = "unsupervised"
    # read in data and convert raw traces to tabular format
    X_train, y_train, X_test, y_test, train_ids, test_ids = prepare_traces_tabular(args, train_type, train_df, test_df)
    lof(args, X_train, y_train, X_test, y_test, train_ids, test_ids)