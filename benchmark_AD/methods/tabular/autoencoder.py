from pyod.models.auto_encoder import AutoEncoder
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from benchmark_AD.methods.tabular.prepare_traces_tabular import prepare_traces_tabular
from benchmark_AD.utils import log_results, predict_scores

def autoencoder(args, X_train, y_train, X_test, y_test, train_ids, test_ids):
    epoch_num = getattr(args, "epochs", None) or 100
    ae = AutoEncoder(epoch_num=epoch_num, random_state=args.seed, device='cpu', preprocessing=False, batch_size=20)
    ae.fit(X_train)
    y_test_scores = ae.decision_function(X_test)

    y_test_pred = predict_scores(args, y_test_scores, y_test)

    f1 = f1_score(y_test, y_test_pred)
    acc = accuracy_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_scores)
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)

    print(f"AutoEncoder F1 Score: {f1:.4f}")
    print(f"AutoEncoder Accuracy: {acc:.4f}")
    print(f"AutoEncoder AUC: {auc:.4f}")
    print(f"AutoEncoder Balanced Accuracy: {bal_acc:.4f}")

    log_results(args, epoch_num, epoch_num+1, f1, acc, auc, bal_acc, model=None, gt_labels=y_test, pred_labels=y_test_pred, pred_scores=y_test_scores, train_ids=train_ids, test_ids=test_ids)  # epoch_num for epoch


def run_autoencoder(args, train_df, test_df):
    print("Running Autoencoder...")
    train_type = 'occ'
    X_train, y_train, X_test, y_test, train_ids, test_ids = prepare_traces_tabular(args, train_type, train_df, test_df)
    autoencoder(args, X_train, y_train, X_test, y_test, train_ids, test_ids)