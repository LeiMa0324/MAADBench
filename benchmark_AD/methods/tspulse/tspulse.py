import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

import math
import random
import sys
import tempfile

import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import random_split

from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from benchmark_AD.methods.tspulse.modeling_tspulse import TSPulseForReconstruction
from benchmark_AD.methods.tspulse.helpers import PatchMaskingDatasetWrapper
from benchmark_AD.methods.tspulse.time_series_anomaly_detection_pipeline import TimeSeriesAnomalyDetectionPipeline

from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.dataset import TSPulseFinetuneDataset

# from methods.tspulse.base import BaseDetector
# from methods.tspulse.modeling_tspulse import TSPulseForReconstruction
# from tsfm_public.models.tspulse.utils.helpers import PatchMaskingDatasetWrapper
# from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import TimeSeriesAnomalyDetectionPipeline
# from TSB_AD.utils.dataset import TSPulseFinetuneDataset


class TSPulsePipeline(BaseDetector):
    def __init__(
        self,
        model_path: str | None = None,
        batch_size: int = 256,
        aggr_win_size: int = 96,
        num_input_channels: int = 1,
        smoothing_window: int = 8,
        prediction_mode: str = "time",
        finetune_epochs: int = 20,
        finetune_validation: float = 0.2,
        finetune_lr: float = 1e-4,
        finetune_seed: int = 42,
        finetune_freeze_backbone: bool = False,
        finetune_decoder_mode: str = "common_channel",
        **kwargs,
    ):
        self._batch_size = batch_size
        self._headers = [f"x{i + 1}" for i in range(num_input_channels)]

        if model_path is None:
            model_path = "ibm-granite/granite-timeseries-tspulse-r1"

        if num_input_channels == 1:
            finetune_decoder_mode = "common_channel"

        if finetune_decoder_mode is None:
            if num_input_channels > 1:
                self.decoder_mode = "mix_channel"
            else:
                self.decoder_mode = "common_channel"
        else:
            self.decoder_mode = finetune_decoder_mode
        # Setting random seed
        random.seed(finetune_seed)
        np.random.seed(finetune_seed)
        # Loading the model to memory
        self._model = TSPulseForReconstruction.from_pretrained(
            model_path,
            num_input_channels=num_input_channels,
            decoder_mode=self.decoder_mode,
            scaling="revin",
            mask_type="user",
        )
        # Reading patch length from the loaded model instance
        p_length = self._model.config.patch_length
        if (aggr_win_size < p_length) or (aggr_win_size % p_length != 0):
            raise ValueError(f"Error: aggregation window must be greater than and multiple of patch_length={p_length}")
        prediction_mode_array = [s_.strip() for s_ in str(prediction_mode).split("+")]
        # Storing pipeline configuration parameters
        self._pipeline_config = {
            "timestamp_column": "timestamp",
            "target_columns": self._headers.copy(),
            "prediction_mode": prediction_mode_array.copy(),
            "aggregation_length": aggr_win_size,
            "smoothing_window": smoothing_window,
            "least_significant_scale": 0.0,
            "least_significant_score": 1.0,
        }
        # Storing model finetuning parameters
        self._finetune_params = {
            "finetune_epochs": finetune_epochs,
            "finetune_validation": finetune_validation,
            "finetune_lr": finetune_lr,
            "finetune_seed": finetune_seed,
            "finetune_freeze_backbone": finetune_freeze_backbone,
        }
        self._scorer = TimeSeriesAnomalyDetectionPipeline(
            self._model,
            target_columns=self._pipeline_config.get("target_columns"),
            prediction_mode=prediction_mode_array,
            aggregation_length=aggr_win_size,
            smoothing_window=self._pipeline_config.get("smoothing_window"),
            least_significant_scale=self._pipeline_config.get("least_significant_scale"),
            least_significant_score=self._pipeline_config.get("least_significant_score"),
        )

    def zero_shot(self, x, label=None):
        self.decision_scores_ = self.decision_function(x)

    def fit(self, X, y=None):
        try:
            print("Fine-tuning TSPulse.")
            validation_size = float(self._finetune_params.get("finetune_validation", 0.2))
            create_valid = True
            if X.shape[0] < 3000:  # 20% of this should be > context_len
                print("Data too small to create a validation set.")
                create_valid = False
                validation_size = 0.0

            if X.shape[0] < self._model.config.context_length:
                print("Skipping fine-tuning due to very short length")
                return

            tsTrain = X[: int((1 - validation_size) * len(X))]
            context_length = self._model.config.context_length
            train_dataset = PatchMaskingDatasetWrapper(
                TSPulseFinetuneDataset(tsTrain, window_size=context_length, return_dict=True),
                window_length=self._pipeline_config.get("aggregation_length"),
                patch_length=self._model.config.patch_length,
                window_position="last",
            )
            if len(train_dataset) < 100:
                print("Skipping fine-tuning due to very few training samples")
                return

            if create_valid:
                tsValid = X[int((1 - validation_size) * len(X)) :]
                valid_dataset = PatchMaskingDatasetWrapper(
                    TSPulseFinetuneDataset(tsValid, window_size=context_length, return_dict=True),
                    window_length=self._pipeline_config.get("aggregation_length"),
                    patch_length=self._model.config.patch_length,
                    window_position="last",
                )
            else:
                valid_dataset = train_dataset

            max_finetune_samples = 100_000
            if len(train_dataset) > max_finetune_samples:
                use_fraction = max_finetune_samples / len(train_dataset)
                # Randomly select use_fraction samples to make finetuning faster
                train_dataset, _ = random_split(train_dataset, [use_fraction, 1 - use_fraction])
                valid_dataset, _ = random_split(valid_dataset, [use_fraction, 1 - use_fraction])
                print(
                    f"Training samples are > max_finetune_samples ({max_finetune_samples}), using {round(use_fraction * 100)}% for faster fine-tuning."
                )

            freeze_backbone = self._finetune_params.get("finetune_freeze_backbone")
            # Freeze the backbone
            if freeze_backbone:
                # Freeze the backbone of the model
                for param in self._model.backbone.parameters():
                    param.requires_grad = False

            temp_dir = tempfile.mkdtemp()

            suggested_lr = self._finetune_params.get("finetune_lr", 1e-4)
            finetune_num_epochs: int = int(self._finetune_params.get("finetune_epochs", 20))
            if not create_valid:
                finetune_num_epochs = min(5, finetune_num_epochs)

            finetune_batch_size = self._batch_size
            if len(train_dataset) < 500:
                finetune_batch_size = 8
            num_workers = 4
            num_gpus = 1

            print(f"Fine-tune: Train samples = {len(train_dataset)}, Valid Samples = {len(valid_dataset)}")

            finetune_args = TrainingArguments(
                output_dir=temp_dir,
                overwrite_output_dir=True,
                learning_rate=suggested_lr,
                num_train_epochs=finetune_num_epochs,
                do_eval=True,
                eval_strategy="epoch",
                per_device_train_batch_size=finetune_batch_size,
                per_device_eval_batch_size=finetune_batch_size * 10,
                dataloader_num_workers=num_workers,
                report_to="tensorboard",
                save_strategy="epoch",
                logging_strategy="epoch",
                save_total_limit=1,
                logging_dir=temp_dir,  # Make sure to specify a logging directory
                load_best_model_at_end=True,  # Load the best model when training ends
                metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
                greater_is_better=False,  # For loss
            )

            # Create the early stopping callback
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=5,  # Number of epochs with no improvement after which to stop
                early_stopping_threshold=1e-5,  # Minimum improvement required to consider as improvement
            )

            # Optimizer and scheduler
            optimizer = AdamW(self._model.parameters(), lr=suggested_lr)
            scheduler = OneCycleLR(
                optimizer,
                suggested_lr,
                epochs=finetune_num_epochs,
                steps_per_epoch=math.ceil(len(train_dataset) / (finetune_batch_size * num_gpus)),
            )

            finetune_trainer = Trainer(
                model=self._model,
                args=finetune_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                callbacks=[early_stopping_callback],
                optimizers=(optimizer, scheduler),
            )

            # Fine tune
            finetune_trainer.train()

        except Exception as e:
            print("Error occured in finetune. Error =", e)
            sys.exit(-1)

    def decision_function(self, X):
        """
        Not used, present for API consistency by convention.
        """
        data = attach_timestamp_column(pd.DataFrame(X, columns=self._headers))
        score = self._scorer(data, batch_size=self._batch_size)
        if not isinstance(score, pd.DataFrame) or ("anomaly_score" not in score):
            raise ValueError("Error: expect anomaly_score column in the output!")

        score = score["anomaly_score"].values.ravel()
        norm_value = np.nanmax(np.asarray(score), axis=0, keepdims=True) + 1e-5
        anomaly_score = score / norm_value
        return anomaly_score


# ===========================================================================
# Data preparation (reused from your existing code)
# ===========================================================================

def prepare_sequences_for_transformer(df, embedding_model, max_sequence_length=100):
    sequences = []
    labels = []
    original_lengths = []
    trace_ids_list = []

    grouped = df.groupby('trace_id')
    print(f"Processing {len(grouped)} traces...\n")

    for trace_idx, (trace_id, group) in enumerate(grouped):
        group = group.sort_values('row_id').reset_index(drop=True)

        timesteps = []
        step_labels_in_trace = []

        previous_message = ""
        for _, row in group.iterrows():
            step_dict = row['step']
            message = row.get('message_context', '')
            step_label = row.get('step_label', 0)
            duration = row["step"]["call_statistic"].get('duration', -1)

            # Extract only current step's text
            current_text = message[len(previous_message):].strip()
            previous_message = message
            message = current_text

            step_labels_in_trace.append(int(step_label))

            text_with_duration = f"{message} [duration: {duration:.2f}s]"
            text_emb = embedding_model.encode(text_with_duration, normalize_embeddings=True)
            timesteps.append(text_emb)

        trace_array = np.stack(timesteps, axis=0)
        original_lengths.append(len(trace_array))  # Save before padding/truncating

        # Pad or truncate to fixed length
        if len(trace_array) < max_sequence_length:
            padding = np.zeros((max_sequence_length - len(trace_array), trace_array.shape[1]))
            trace_array = np.vstack([trace_array, padding])
        else:
            trace_array = trace_array[:max_sequence_length]

        sequences.append(trace_array)
        trace_ids_list.append(trace_id)
        labels.append(step_labels_in_trace)

    return sequences, labels, original_lengths, trace_ids_list


def pad_labels(labels_list, max_length=100):
    """Pad or truncate labels to fixed length. Padding value is 0 (normal)."""
    padded_labels = []
    for trace_labels in labels_list:
        if len(trace_labels) < max_length:
            padded = list(trace_labels) + [0] * (max_length - len(trace_labels))
        else:
            padded = list(trace_labels)[:max_length]
        padded_labels.append(padded)
    return np.array(padded_labels, dtype=np.float32)


def make_mask(original_lengths, max_length=100):
    """
    Create a flat boolean mask for all traces.
    True = real timestep, False = padding.
    Shape: (num_traces * max_length,)
    """
    mask = np.concatenate([
        np.concatenate([
            np.ones(min(length, max_length)),
            np.zeros(max(0, max_length - length))
        ])
        for length in original_lengths
    ])
    return mask.astype(bool)


# ===========================================================================
# TSPulse runner
# ===========================================================================

def run_tspulse(args, train_df, test_df):

    # ============================================================
    # Step 1: OCC filtering — keep only normal actions for training
    # ============================================================
    first_anomaly_step = train_df[train_df["step_label"] == 1].groupby('trace_id')['step_num'].min()

    filtered_rows = []
    for trace_id in train_df['trace_id'].unique():
        trace_df = train_df[train_df['trace_id'] == trace_id]

        if trace_id in first_anomaly_step.index:
            cutoff_step = first_anomaly_step[trace_id]
            filtered_trace = trace_df[trace_df['step_num'] < cutoff_step]
        else:
            filtered_trace = trace_df[trace_df["step_label"] == 0]

        filtered_rows.append(filtered_trace)

    train_df = pd.concat(filtered_rows, ignore_index=False).sort_index()

    # ============================================================
    # Step 2: Embed sequences
    # ============================================================
    embedding_model_dir = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(embedding_model_dir)

    print("Preparing training sequences...")
    train_sequences, train_labels, train_original_lengths, train_ids = prepare_sequences_for_transformer(
        train_df, embedding_model
    )

    print("Preparing test sequences...")
    test_sequences, test_labels, test_original_lengths, test_ids = prepare_sequences_for_transformer(
        test_df, embedding_model
    )

    # ============================================================
    # Step 3: Convert to numpy and flatten for TSPulse
    # TSPulse expects: (num_timesteps, num_channels)
    # We treat each embedding dimension as a channel
    # ============================================================
    train_sequences = np.array(train_sequences, dtype=np.float32)  # (num_train_traces, 100, 384)
    test_sequences = np.array(test_sequences, dtype=np.float32)    # (num_test_traces, 100, 384)

    num_channels = train_sequences.shape[-1]  # 384

    # Flatten: (num_traces, 100, 384) -> (num_traces * 100, 384)
    train_X = train_sequences.reshape(-1, num_channels)
    test_X = test_sequences.reshape(-1, num_channels)

    print(f"\nTSPulse input shapes:")
    print(f"  train_X: {train_X.shape}  (num_timesteps={train_X.shape[0]}, channels={train_X.shape[1]})")
    print(f"  test_X:  {test_X.shape}")

    # ============================================================
    # Step 4: Fit TSPulse on training data (normal only)
    # ============================================================
    print("\nFitting TSPulse...")
    model = TSPulsePipeline(
        num_input_channels=num_channels,
        finetune_epochs=20,
        finetune_validation=0.2,
        finetune_lr=1e-4,
        batch_size=256,
    )
    model.fit(train_X)

    # ============================================================
    # Step 5: Get anomaly scores on test data
    # ============================================================
    print("\nScoring test data...")
    scores = model.decision_function(test_X)  # shape: (num_test_timesteps,)

    # ============================================================
    # Step 6: Mask out padding and evaluate
    # ============================================================
    # Pad labels to match flattened sequences
    test_labels_padded = pad_labels(test_labels, max_length=100)
    test_labels_flat = test_labels_padded.flatten()  # (num_test_traces * 100,)

    # Create mask to remove padding
    test_mask = make_mask(test_original_lengths, max_length=100)

    # Apply mask — only evaluate on real timesteps
    scores_real = scores[test_mask]
    labels_real = test_labels_flat[test_mask]

    print(f"\nEvaluation on {len(scores_real)} real events "
          f"({int(np.sum(labels_real))} anomalous, {int(np.sum(1 - labels_real))} normal)")

    # Threshold: flag top-K events where K = number of true anomalies
    num_anomalies = int(np.sum(labels_real))
    thresh = np.sort(scores_real)[-num_anomalies] if num_anomalies > 0 else np.max(scores_real)
    preds = (scores_real >= thresh).astype(int)

    f1 = f1_score(labels_real, preds)
    acc = accuracy_score(labels_real, preds)
    auc = roc_auc_score(labels_real, scores_real)
    bal_acc = balanced_accuracy_score(labels_real, preds)

    print(f"\nEvent-level Metrics:")
    print(f"  F1-score:          {f1:.4f}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  AUC:               {auc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")

    return {
        'f1': f1,
        'accuracy': acc,
        'auc': auc,
        'balanced_accuracy': bal_acc,
        'scores': scores_real,
        'labels': labels_real,
        'preds': preds
    }