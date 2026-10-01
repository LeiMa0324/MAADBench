from benchmark_AD.utils import sequence_row_ids
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
from benchmark_AD.utils import log_results, predict_scores

# ===========================================================================
# Model
# ===========================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class SupervisedTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=128, n_heads=4, n_layers=2, max_len=100):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        """
        x:    (B, T, F)   — input features
        mask: (B, T)      — True = real timestep, False = padding
        returns logits: (B, T)
        """
        x = self.input_proj(x)
        x = self.pos_enc(x)
        # TransformerEncoderLayer expects True = ignore (padding), so invert
        src_key_padding_mask = ~mask
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.head(x).squeeze(-1)  # (B, T)
        return logits


# ===========================================================================
# Data preparation
# ===========================================================================

def prepare_sequences_for_transformer(df, embedding_model, max_sequence_length=100):
    sequences = []
    labels = []
    original_lengths = []
    trace_ids_list = []

    grouped = df.groupby('trace_id', sort=False)
    print(f"Processing {len(grouped)} traces...\n")

    for trace_idx, (trace_id, group) in enumerate(grouped):
        group = group.sort_values('step_num').reset_index(drop=True)

        timesteps = []
        step_labels_in_trace = []

        previous_message = ""
        for _, row in group.iterrows():
            step_dict = row['step']
            message = row.get('message_context', '')
            step_label = row.get('step_label', 0)
            duration = (row["step"].get("call_statistic") or {}).get('duration', -1)

            # Extract only current step's text
            current_text = message[len(previous_message):].strip()
            previous_message = message
            message = current_text

            step_labels_in_trace.append(int(step_label))

            text_with_duration = f"{message} [duration: {duration:.2f}s]"
            text_emb = embedding_model.encode(text_with_duration, normalize_embeddings=True)
            timesteps.append(text_emb)

        trace_array = np.stack(timesteps, axis=0)
        original_lengths.append(min(len(trace_array), 100))  # Save before padding/truncating

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
    """
    Pad or truncate labels to fixed length.
    Padding value is 0 (normal).
    """
    padded_labels = []
    for trace_labels in labels_list:
        if len(trace_labels) < max_length:
            padded = list(trace_labels) + [0] * (max_length - len(trace_labels))
        else:
            padded = list(trace_labels)[:max_length]
        padded_labels.append(padded)
    return np.array(padded_labels, dtype=np.float32)


def make_masks(original_lengths, max_length=100):
    """
    Create boolean masks: True = real timestep, False = padding.
    Shape: (num_traces, max_length)
    """
    masks = []
    for length in original_lengths:
        real_len = min(length, max_length)
        mask = [True] * real_len + [False] * (max_length - real_len)
        masks.append(mask)
    return np.array(masks, dtype=bool)


# ===========================================================================
# Training
# ===========================================================================

def train_sup_transformer(args, train_sequences, train_labels, test_sequences, test_labels,
                           original_lengths_train, original_lengths_test, train_ids, test_ids):

    batch_size = 32
    num_epochs = getattr(args, "epochs", None) or 50
    lr = 1e-4
    max_length = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pad labels to fixed length
    train_labels_padded = pad_labels(train_labels, max_length=max_length)
    test_labels_padded = pad_labels(test_labels, max_length=max_length)

    # Create masks (True = real, False = padding)
    train_masks = make_masks(original_lengths_train, max_length=max_length)
    test_masks = make_masks(original_lengths_test, max_length=max_length)

    # Convert to numpy arrays
    train_sequences = np.array(train_sequences, dtype=np.float32)
    test_sequences = np.array(test_sequences, dtype=np.float32)

    # Convert to torch tensors
    train_X = torch.from_numpy(train_sequences)                         # (N, 100, F)
    train_y = torch.from_numpy(train_labels_padded)                     # (N, 100)
    train_m = torch.from_numpy(train_masks)                             # (N, 100) bool

    test_X = torch.from_numpy(test_sequences)                           # (N, 100, F)
    test_y = torch.from_numpy(test_labels_padded)                       # (N, 100)
    test_m = torch.from_numpy(test_masks)                               # (N, 100) bool

    print(f"train_X shape: {train_X.shape}")
    print(f"train_y shape: {train_y.shape}")
    print(f"train_m shape: {train_m.shape}")

    # Create DataLoaders (include masks in dataset)
    train_dataset = TensorDataset(train_X, train_y, train_m)
    test_dataset = TensorDataset(test_X, test_y, test_m)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Build model
    feature_dim = train_X.shape[-1]
    model = SupervisedTransformer(feature_dim=feature_dim, max_len=max_length).to(device)

    # Class-weighted loss to handle imbalance
    all_train_labels = train_labels_padded[train_masks].flatten()
    num_pos = np.sum(all_train_labels)
    num_neg = len(all_train_labels) - num_pos
    pos_weight = num_neg / (num_pos + 1e-6)
    print(f"\nClass balance: {num_pos:.0f} anomalous, {num_neg:.0f} normal, pos_weight={pos_weight:.2f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for x, y, mask in train_loader:
            x = x.float().to(device)      # (B, T, F)
            y = y.float().to(device)      # (B, T)
            mask = mask.to(device)        # (B, T) bool

            optimizer.zero_grad()

            logits = model(x, mask)       # (B, T)

            # Only compute loss on real (non-padded) timesteps
            logits_real = logits[mask]    # (num_real_timesteps,)
            y_real = y[mask]              # (num_real_timesteps,)

            loss = loss_fn(logits_real, y_real)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for x, y, mask in test_loader:
            x = x.float().to(device)
            y = y.float().to(device)
            mask = mask.to(device)

            logits = model(x, mask)       # (B, T)
            probs = torch.sigmoid(logits) # (B, T)

            # Only evaluate on real timesteps
            probs_real = probs[mask]      # (num_real_timesteps,)
            y_real = y[mask]              # (num_real_timesteps,)

            all_labels.extend(y_real.cpu().numpy())
            all_scores.extend(probs_real.cpu().numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    all_preds = predict_scores(args, all_scores, all_labels, probability=True)

    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    print(f"\nEvent-level Metrics:")
    print(f"  F1-score:          {f1:.4f}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  AUC:               {auc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")

    log_results(args, epoch, num_epochs, f1, acc, auc, bal_acc, model=model, gt_labels=all_labels, pred_labels=all_preds, pred_scores=all_scores, train_ids=train_ids, test_ids=test_ids)  # epoch for epoch



# ===========================================================================
# Entry point
# ===========================================================================

def run_sup_transformer(args, train_df, test_df):
    train_ids = sequence_row_ids(train_df)
    test_ids = sequence_row_ids(test_df)

    embedding_model_dir = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(embedding_model_dir)

    train_sequences, train_labels, train_original_lengths, _ = prepare_sequences_for_transformer(train_df, embedding_model)
    test_sequences, test_labels, test_original_lengths, _ = prepare_sequences_for_transformer(test_df, embedding_model)

    train_sup_transformer(
        args,
        train_sequences, train_labels,
        test_sequences, test_labels,
        train_original_lengths, test_original_lengths,
        train_ids, test_ids
    )