from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from benchmark_AD.methods.tabular.prepare_traces_tabular import prepare_traces_tabular
from benchmark_AD.utils import log_results, predict_scores

def random_forest(args, X_train, y_train, X_test, y_test, train_ids, test_ids):
    rf_model = RandomForestClassifier(n_estimators=300, random_state=args.seed, n_jobs=-1)
    
    rf_model.fit(X_train, y_train)
    y_scores = rf_model.predict_proba(X_test)[:, 1]
    y_pred = predict_scores(args, y_scores, y_test, probability=True)

    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_scores)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")

    log_results(args, 0, 1, f1, acc, auc, bal_acc, model=None, gt_labels=y_test, pred_labels=y_pred, pred_scores=y_scores, train_ids=train_ids, test_ids=test_ids)  # 0 for epoch


def run_random_forest(args, train_df, test_df):
    print("Running Random Forest...")
    train_type = 'supervised'
    X_train, y_train, X_test, y_test, train_ids, test_ids = prepare_traces_tabular(args, train_type, train_df, test_df)
    random_forest(args, X_train, y_train, X_test, y_test, train_ids, test_ids)