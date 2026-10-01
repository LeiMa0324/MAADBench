import pandas as pd
from pathlib import Path
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

from benchmark_AD.utils import get_train_test_indexes_action_level


def clean_text(text):
    # Remove markdown fences
    text = text.replace("```json", "")
    text = text.replace("```", "")

    # Remove escaped newlines and real newlines
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")

    # Remove curly braces
    text = text.replace("{", "")
    text = text.replace("}", "")

    # remove brackets
    text = text.replace("[", "")
    text = text.replace("]", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove string quotes
    text = text.replace('"', "")

    # Remove *
    text = text.replace("*", "")

    return text


def extract_llm_outputs_gsm_hard(trace):
    llm_outputs = []
    for llm_output in trace["trace"]:  # each llm_output is the full output from 1 LLM call
        llm_reasoning = llm_output["output"]["reasoning"]  # only extract the reasoning of the LLM to prevent repetition
        
        # clean the text
        llm_reasoning = clean_text(llm_reasoning)
        llm_outputs.append(llm_reasoning)  # only extract the reasoning of the LLM to prevent repetition 

    return llm_outputs


def extract_llm_outputs_EscapeRoom_v2(trace):
    llm_outputs = []
    for llm_output in trace["trace"]:  # each llm_output is the full output from 1 LLM call
        llm_reasoning = llm_output.get("output_message", llm_output.get("message", str(llm_output.get("output", ""))))  # only extract the reasoning of the LLM to prevent repetition
        agent = llm_output["agent"]

        # combine agent and reasoning
        llm_reasoning = f"{agent}: {llm_reasoning}"

        # clean the text
        llm_reasoning = clean_text(llm_reasoning)
        llm_outputs.append(llm_reasoning)  # only extract the reasoning of the LLM to prevent repetition 

    return llm_outputs


def embed_full_trace(trace_text, model):
    return model.encode(trace_text, convert_to_numpy=True)


def convert_traces_by_puzzle(df):
    """Convert traces to tabular data at the puzzle level.

    Groups action steps by (room, puzzle_id), concatenates messages per puzzle
    into one embedding, aggregates step-level features, and determines puzzle
    label (failed if any action in the puzzle failed).
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    all_rows = []
    for _, row in df.iterrows():
        trace = row["trace"]["trace"]
        room_id = row.get("room_number", "")
        difficulty = row.get("difficulty", "")

        # Group steps by puzzle_id
        puzzles = {}
        for step in trace:
            pid = step.get("puzzle_id", "")
            if pid not in puzzles:
                puzzles[pid] = []
            puzzles[pid].append(step)

        for pid, steps in puzzles.items():
            features = {}
            features["room_id"] = room_id
            features["difficulty"] = difficulty
            features["puzzle_id"] = pid

            # Puzzle label: failed if any action in this puzzle failed
            verification_statuses = [s.get("verification", {}).get("status", "") for s in steps]
            puzzle_failed = int(any(s != "correct" for s in verification_statuses))
            features["label"] = puzzle_failed

            # Aggregate step-level scalar features
            num_steps = len(steps)
            features["num_steps"] = num_steps

            confidences = np.array([s.get("confidence", 0.0) for s in steps])
            attempts = np.array([s.get("attempt", 1) for s in steps])
            durations = np.array([s.get("call_statistic", {}).get("duration", -1) for s in steps])
            input_tokens = np.array([s.get("call_statistic", {}).get("input_tokens", -1) for s in steps])
            output_tokens = np.array([s.get("call_statistic", {}).get("output_tokens", -1) for s in steps])
            schema_errors = np.array([1 if s.get("schema_errors") else 0 for s in steps])
            action_failed_arr = np.array([0 if s.get("verification", {}).get("status") == "correct" else 1 for s in steps])

            for name, arr in [("confidence", confidences), ("attempt", attempts),
                              ("duration", durations), ("input_tokens", input_tokens),
                              ("output_tokens", output_tokens), ("schema_errors", schema_errors),
                              ("action_failed", action_failed_arr)]:
                features[f"{name}_mean"] = float(np.mean(arr))
                features[f"{name}_sum"] = float(np.sum(arr))
                features[f"{name}_std"] = float(np.std(arr))

            # Concatenate all messages in this puzzle and embed
            messages = []
            for s in steps:
                msg = f"{s.get('agent', '')}: {s.get('message', '')}"
                messages.append(clean_text(msg))
            full_message = " ".join(messages)
            features["message"] = full_message

            emb = model.encode(full_message, convert_to_numpy=True)
            for i in range(emb.shape[0]):
                features[f"emb_{i}"] = float(emb[i])

            all_rows.append(features)

    modeling_df = pd.DataFrame(all_rows)

    # Reorder columns
    meta_cols = ["room_id", "difficulty", "puzzle_id", "label", "model_name"]
    emb_cols = [c for c in modeling_df.columns if c.startswith("emb_")]
    other_cols = [c for c in modeling_df.columns if c not in meta_cols and c not in emb_cols and c != "message"]
    modeling_df = modeling_df[meta_cols + other_cols + ["message"] + emb_cols]

    contamination = modeling_df["label"].mean()
    print(f"the contamination is {contamination}")

    return modeling_df

        
def convert_traces_tabular(df):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    num_llm_calls = []
    llm_call_duration_mean = []
    llm_call_input_token_mean = []
    llm_call_output_token_mean = []

    llm_call_duration_sum = []
    llm_call_input_token_sum = []
    llm_call_output_token_sum = []

    llm_call_duration_min = []
    llm_call_input_token_min = []
    llm_call_output_token_min = []

    llm_call_duration_max = []
    llm_call_input_token_max = []
    llm_call_output_token_max = []

    llm_call_duration_std = []
    llm_call_input_token_std = []
    llm_call_output_token_std = []
    
    for trace_dict in df["trace"]:  # loop through each trace
        call_durations = []
        call_input_tokens = []
        call_output_tokens = []
        for call in trace_dict["trace"]:
            try:
                call_durations.append(call["call_statistic"]["duration"])
            except:
                call_durations.append(-1)  # if duration is missing, fill with -1
            try:
                call_input_tokens.append(call["call_statistic"]["input_tokens"])
            except:
                call_input_tokens.append(-1)  # if input_tokens is missing, fill with -1
            try:
                call_output_tokens.append(call["call_statistic"]["output_tokens"])
            except:
                call_output_tokens.append(-1)  # if output_tokens is missing, fill with -1

        # Convert to numpy array for statistics
        durations_arr = np.array(call_durations or [-1])
        input_tokens_arr = np.array(call_input_tokens or [-1])
        output_tokens_arr = np.array(call_output_tokens or [-1])

        num_llm_calls.append(len(call_durations))

        llm_call_duration_mean.append(float(np.mean(durations_arr)))
        llm_call_input_token_mean.append(float(np.mean(input_tokens_arr)))
        llm_call_output_token_mean.append(float(np.mean(output_tokens_arr)))

        llm_call_duration_min.append(float(np.min(durations_arr)))
        llm_call_input_token_min.append(float(np.min(input_tokens_arr)))
        llm_call_output_token_min.append(float(np.min(output_tokens_arr)))

        llm_call_duration_max.append(float(np.max(durations_arr)))
        llm_call_input_token_max.append(float(np.max(input_tokens_arr)))
        llm_call_output_token_max.append(float(np.max(output_tokens_arr)))

        llm_call_duration_std.append(float(np.std(durations_arr)))
        llm_call_input_token_std.append(float(np.std(input_tokens_arr)))
        llm_call_output_token_std.append(float(np.std(output_tokens_arr)))

    df["num_llm_calls"] = num_llm_calls

    df["llm_call_duration_mean"] = llm_call_duration_mean
    df["llm_call_input_token_mean"] = llm_call_input_token_mean
    df["llm_call_output_token_mean"] = llm_call_output_token_mean

    df["llm_call_duration_min"] = llm_call_duration_min
    df["llm_call_input_token_min"] = llm_call_input_token_min
    df["llm_call_output_token_min"] = llm_call_output_token_min

    df["llm_call_duration_max"] = llm_call_duration_max
    df["llm_call_input_token_max"] = llm_call_input_token_max
    df["llm_call_output_token_max"] = llm_call_output_token_max

    df["llm_call_duration_std"] = llm_call_duration_std
    df["llm_call_input_token_std"] = llm_call_input_token_std
    df["llm_call_output_token_std"] = llm_call_output_token_std

    llm_outputs = df["trace"].apply(extract_llm_outputs_EscapeRoom_v2)
    flattened_llm_outputs = [" ".join(sublist) for sublist in llm_outputs]
    df["raw_llm_outputs"] = flattened_llm_outputs

    # embed llm outputs
    df["trace_embedding"] = df["raw_llm_outputs"].apply(lambda x: embed_full_trace(x, model))

    # create tabular dataset
    embedding_matrix = np.vstack(df["trace_embedding"].values)
    embedding_cols = [f"emb_{i}" for i in range(embedding_matrix.shape[1])]
    embedding_df = pd.DataFrame(embedding_matrix, columns=embedding_cols, index=df.index)

    # combine with original df
    modeling_df = pd.concat([df, embedding_df], axis=1)

    return modeling_df


def convert_actions_tabular(df):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm_call_duration = []
    llm_call_input_token = []
    llm_call_output_token = []
    for _, row in df.iterrows():
        try:
            step_duration = row["step_trace"]["call_statistic"]["duration"]
        except:
            step_duration = -1  # if duration is missing, fill with -1
        try:
            step_input_tokens = row["step_trace"]["call_statistic"]["input_tokens"]
        except:
            step_input_tokens = -1  # if input_tokens is missing, fill with -1
        try:
            step_output_tokens = row["step_trace"]["call_statistic"]["output_tokens"]
        except:
            step_output_tokens = -1  # if output_tokens is missing, fill with -1
        
        llm_call_duration.append(step_duration)
        llm_call_input_token.append(step_input_tokens)
        llm_call_output_token.append(step_output_tokens)

    df["step_duration"] = llm_call_duration
    df["step_input_tokens"] = llm_call_input_token
    df["step_output_tokens"] = llm_call_output_token
    
    # combine input context and message context, then embed together
    # agent steps have a plain string input, tool call steps have a dict with the
    # values actually passed to the tool under "received_input" (agent steps have no such field)
    def stringify_input(x):
        if isinstance(x, str):
            return x
        return json.dumps(x.get("received_input", x) if isinstance(x, dict) else x)
    df["input_context"] = df["input"].apply(stringify_input)
    df["input_context"] = df["input_context"].apply(clean_text)
    df["message_context"] = df["message_context"].apply(clean_text)
    df["combined_context"] = df["input_context"] + " " + df["message_context"]
    df["trace_embedding"] = df["combined_context"].apply(lambda x: embed_full_trace(x, model))

    # create tabular dataset
    embedding_matrix = np.vstack(df["trace_embedding"].values)
    embedding_cols = [f"emb_{i}" for i in range(embedding_matrix.shape[1])]
    embedding_df = pd.DataFrame(embedding_matrix, columns=embedding_cols, index=df.index)

    # combine with original df
    modeling_df = pd.concat([df, embedding_df], axis=1)  

    return modeling_df


def prepare_traces_tabular(args, train_type, train_df, test_df):
    print("Converting traces to tabular format")
    # covert traces to format such that it can be modeled
    if args.dataset_level == "trace":
        modeling_train_df = convert_traces_tabular(train_df)
        modeling_test_df = convert_traces_tabular(test_df)
    elif args.dataset_level == "action":
        modeling_train_df = convert_actions_tabular(train_df)
        modeling_test_df = convert_actions_tabular(test_df)
    else:
        raise ValueError("dataset_level must be one of 'trace' or 'action'")

    # 3 training settings, test is always the same
    # 1. supervised:   train with all labels (normal + anomaly)
    # 2. unsupervised: train with all labels (same as supervised, methods don't use labels)
    # 3. occ:          train with normal only (anomaly removed)
    test_df = modeling_test_df
    if train_type == 'supervised' or train_type == 'unsupervised' or train_type == 'semi_supervised':
        train_df = modeling_train_df
    elif train_type == 'occ':
        if args.dataset_level == "action":
            label = "step_label"
        elif args.dataset_level == "trace":
            label = "label"
        if args.occ_training == "clean" and args.dataset_level == "trace":
            train_df = modeling_train_df[modeling_train_df[label] == 0].copy()
        elif args.occ_training == "clean":
            # Drop all actions that follow the first anomalous action. Including the anomalous action
            first_anomaly_step = modeling_train_df[modeling_train_df[label] == 1].groupby('trace_id')['step_num'].min()

            filtered_rows = []
            for trace_id in modeling_train_df['trace_id'].unique():
                trace_df = modeling_train_df[modeling_train_df['trace_id'] == trace_id]

                if trace_id in first_anomaly_step.index:
                    # Trace has anomalies: keep rows before first anomaly
                    cutoff_step = first_anomaly_step[trace_id]
                    filtered_trace = trace_df[trace_df['step_num'] < cutoff_step]
                else:
                    # Trace has no anomalies: keep all rows with label=0
                    filtered_trace = trace_df[trace_df[label] == 0]

                filtered_rows.append(filtered_trace)

            train_df = pd.concat(filtered_rows, ignore_index=False).sort_index()
        elif args.occ_training == "polluted":
            # keep all rows but change all labels to 0
            train_df = modeling_train_df.copy()
            train_df[label] = 0
        
    else:
        raise ValueError("train_type must be one of 'supervised', 'occ', or 'unsupervised'")   
    
    if train_df.empty:
        raise ValueError("No training samples remain after OCC filtering")
    data_dir = Path(args.run_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Save numeric features plus IDs; raw JSON lives in the original trace files.
    emb_cols = [c for c in train_df if c.startswith("emb_")]
    scalar_cols = (["step_duration", "step_input_tokens", "step_output_tokens"]
                   if args.dataset_level == "action" else
                   [c for c in train_df if c.startswith("llm_call_")] + ["num_llm_calls"])
    feature_cols = scalar_cols + emb_cols
    label = "step_label" if args.dataset_level == "action" else "label"
    for name, frame in [("train", train_df), ("test", test_df)]:
        frame[["row_id", label] + feature_cols].to_csv(data_dir / f"{name}_features.csv", index=False)
    train_ids, test_ids = train_df.row_id.to_numpy(), test_df.row_id.to_numpy()
    X_train, X_test = train_df[feature_cols].to_numpy(), test_df[feature_cols].to_numpy()
    y_train, y_test = train_df[label].to_numpy(), test_df[label].to_numpy()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    import joblib
    joblib.dump(scaler, data_dir / "scaler.joblib")
    (data_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))
    return X_train, y_train, X_test, y_test, train_ids, test_ids
