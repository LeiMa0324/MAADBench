from benchmark_AD.utils import sequence_row_ids
import pandas as pd
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import numpy as np

from benchmark_AD.methods.g_safeguard.convert_lomas_traces import clean_text, extract_llm_outputs

def load_traces_to_df_gsm_hard(folder_path):
    '''
    read in each raw trace and extract the metadata from the filename and the trace itself.
    save them in a dataframe with columns: MAS, Task Dataset, label, id, trace.
    '''    
    folder = Path(folder_path)
    rows = []

    for file in folder.glob("*.json"):

        if not file.stem[0].isdigit():
            continue

        filename = file.stem
        parts = filename.split("_")

        # Extract metadata
        task_dataset = parts[1]
        mas = parts[2]
        label = parts[-1].lower()
        id = parts[5]
        temp = float(parts[7])

        # Convert label to numeric (your convention: 1=inlier, -1=outlier)
        if label == "abnormal" or label == "parseerror":
            if parts[-2].lower() == "normal":  # some traces that have parseerror might still be normal
                numeric_label = 0
            else:
                numeric_label = 1
        elif label == "normal":
            numeric_label = 0
        else:
            numeric_label = None
        
        # Load JSON trace
        with open(file, "r", encoding="utf-8") as f:
            trace_obj = json.load(f)

        rows.append({
            "MAS": mas,
            "Task Dataset": task_dataset,
            "Temperature": temp,
            "label": numeric_label,
            "id": id,
            "trace": trace_obj
        })

    return pd.DataFrame(rows)


def load_traces_to_df_EscapeRoom_v2(folder_path):
    folder = Path(folder_path)
    rows = []

    id_temp = 0  # each trace should have its own id
    for file in folder.rglob("*.json"):

        if not file.stem[0].isdigit():
            continue

        filename = file.stem
        parts = filename.split("_")

        # Extract metadata
        room_num = parts[2]
        difficulty = parts[3]
        id = id_temp
        temperature = float(parts[5])
        label = parts[-1].lower()
        model_name = file.parent.name

        id_temp += 1

        # Convert label to numeric (0=inlier, 1=outlier)
        if label == "escaped":
            numeric_label = 0
        elif label == "failed":
            numeric_label = 1
        else: 
            raise ValueError(f"Label {label} not recognized for EscapeRoom_v2 dataset")
    
        
        # Load JSON trace
        with open(file, "r", encoding="utf-8") as f:
            trace_obj = json.load(f)

        rows.append({
            "model_name": model_name,
            "room_number": room_num,
            "difficulty": difficulty,
            "temperature": temperature,
            "label": numeric_label,
            "id": id,
            "trace": trace_obj
        })

    return pd.DataFrame(rows)


def convert_lomas_traces_gsm_hard(PATH_TO_TRACES):
    df = load_traces_to_df_gsm_hard(PATH_TO_TRACES)
    embedding_model_dir = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(embedding_model_dir)

    agent_to_idx = {
        "cot_solver": 0,
        "pot_solver": 1,
        "judge": 2
    }
    num_agents = len(agent_to_idx)

    # cot and pot act first (in parallel), then judge responds
    event_order = ["cot_solver", "pot_solver", "judge"]

    sequences = []   # list of (N, d) arrays — the X matrices
    labels = []      # 0 = normal, 1 = anomalous

    for _, row in df.iterrows():
        trace = row["trace"]
        label = int(row["label"])

        # Get raw text outputs in event order
        raw_outputs, llm_durations = extract_llm_outputs(trace) 

        agent_outputs = {
            "cot_solver": f"{raw_outputs[0]} [duration: {llm_durations[0]:.2f}s]",
            "pot_solver": f"{raw_outputs[1]} [duration: {llm_durations[1]:.2f}s]",
            "judge":      f"{raw_outputs[2]} [duration: {llm_durations[2]:.2f}s]"
        }

        timesteps = []
        for agent_name in event_order:
            text = clean_text(agent_outputs[agent_name])

            # --- Text embedding: shape (384,) for MiniLM ---
            text_emb = embedding_model.encode(text, normalize_embeddings=True)

            # --- Agent ID: one-hot, shape (3,) ---
            agent_id = np.zeros(num_agents, dtype=np.float32)
            agent_id[agent_to_idx[agent_name]] = 1.0

            # --- Concatenate: shape (387,) ---
            token = np.concatenate([agent_id, text_emb], axis=0)
            timesteps.append(token)

        # X shape: (N=3, d=387)
        X = np.stack(timesteps, axis=0)

        sequences.append(X)
        labels.append(label)

    sequences = np.array(sequences)  # shape: (num_traces, 3, 387)
    labels = np.array(labels)

    print(f"Converted {len(sequences)} traces")
    print(f"Sequence shape: {sequences.shape}")  # (num_traces, N, d)
    print(f"Label distribution: {np.bincount(labels)}")

    return sequences, labels


def prepare_sequences_for_transformer(df, embedding_model):
    sequences = []
    attention_masks = []  # tells the transformer which positions are real vs padding
    labels = []
    trace_ids_list = []
    original_lengths  = []  # to keep track of the original lengths before padding

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
            step_label = row.get('step_label', '')
            duration = (row["step"].get("call_statistic") or {}).get('duration', -1)

            current_text = message[len(previous_message):].strip()
            previous_message = message
            message = current_text

            step_labels_in_trace.append(step_label)

            text_with_duration = f"{message} [duration: {duration:.2f}s]"
            text_emb = embedding_model.encode(text_with_duration, normalize_embeddings=True)

            step_label = np.array([float(step_label)], dtype=np.float32)

            token = np.concatenate([text_emb], axis=0)
            timesteps.append(token)
        trace_array = np.stack(timesteps, axis=0)
        original_lengths.append(min(len(trace_array), 100))

        # Pad or truncate to max_sequence_length: traces must be exactly the same length
        max_sequence_length = 100
        if len(trace_array) < max_sequence_length:
            padding = np.zeros((max_sequence_length - len(trace_array), trace_array.shape[1]))
            trace_array = np.vstack([trace_array, padding])
        else:
            trace_array = trace_array[:max_sequence_length]

        sequences.append(trace_array)
        trace_ids_list.append(trace_id)
        labels.append(step_labels_in_trace)
    
    return sequences, labels, original_lengths


def convert_lomas_traces_escape_room_v2(args, train_df, test_df):
    embedding_model_dir = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(embedding_model_dir)

    if args.occ_training == 'clean':
        # OCC: only select clean actions for training
        # Drop all actions that follow the first anomalous action. Including the anomalous action
        first_anomaly_step = train_df[train_df["step_label"] == 1].groupby('trace_id')['step_num'].min()
    
        filtered_rows = []
        for trace_id in train_df['trace_id'].unique():
            trace_df = train_df[train_df['trace_id'] == trace_id]
    
            if trace_id in first_anomaly_step.index:
                # Trace has anomalies: keep rows before first anomaly
                cutoff_step = first_anomaly_step[trace_id]
                filtered_trace = trace_df[trace_df['step_num'] < cutoff_step]
            else:
                # Trace has no anomalies: keep all rows with label=0
                filtered_trace = trace_df[trace_df["step_label"] == 0]
    
            filtered_rows.append(filtered_trace)
    
        train_df = pd.concat(filtered_rows, ignore_index=False).sort_index()
        
    if train_df.empty:
        raise ValueError("No clean sequence prefixes remain for training")

    train_ids = sequence_row_ids(train_df)
    test_ids = sequence_row_ids(test_df)  
    
    train_sequences, train_labels, train_original_lengths = prepare_sequences_for_transformer(train_df, embedding_model)
    test_sequences, test_labels, test_original_lengths = prepare_sequences_for_transformer(test_df, embedding_model)

    return train_sequences, train_labels, test_sequences, test_labels, train_original_lengths, test_original_lengths, train_ids, test_ids