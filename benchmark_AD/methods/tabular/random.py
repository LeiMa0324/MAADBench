"""Random baseline: no embeddings or torch required."""
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
from benchmark_AD.utils import log_results, predict_scores


def run_random(args, train_df, test_df):
    column = 'step_label' if args.dataset_level == 'action' else 'label'
    y = test_df[column].to_numpy()
    scores = np.random.default_rng(args.seed).random(len(y))
    predictions = predict_scores(args, scores, y, probability=True)
    log_results(args, 0, 1, f1_score(y, predictions), accuracy_score(y, predictions),
                roc_auc_score(y, scores), balanced_accuracy_score(y, predictions),
                gt_labels=y, pred_labels=predictions, pred_scores=scores,
                train_ids=train_df.row_id.to_numpy(), test_ids=test_df.row_id.to_numpy())
