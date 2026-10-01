import pandas as pd
from pathlib import Path
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from benchmark_AD.utils import get_train_test_indexes

def load_traces_to_df_gsm_hard(folder_path):
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

        # Convert label to numeric (your convention: 0=inlier, 1=outlier)
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


def extract_ground_truth_answer_gsm_hard(trace):
        gt_answer = trace['question']['gold_answer']
        
        gt_answer = float(gt_answer) 
        
        return gt_answer


def extract_mas_answer_gsm_hard(trace):
        try:
            return float(trace['trace'][-1]['output']["content"]['final_answer'])
        except:
            return None
        

def count_llm_words(call):
    content = call["output"]["content"]
    text = str(content)  # convert dict to string
    
    # Remove escaped newlines and backslashes
    text = text.replace("\\n", " ")
    text = text.replace("\\\\", " ")

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    
    return len(text.split())  # count words


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
        llm_reasoning = llm_output["message"]  # only extract the reasoning of the LLM to prevent repetition
        agent = llm_output["agent"]

        # combine agent and reasoning
        llm_reasoning = f"{agent}: {llm_reasoning}"

        # clean the text
        llm_reasoning = clean_text(llm_reasoning)
        llm_outputs.append(llm_reasoning)  # only extract the reasoning of the LLM to prevent repetition 

    return llm_outputs


def embed_full_trace(trace_text, model):
    return model.encode(trace_text, convert_to_numpy=True)


def convert_traces_gsm_hard(df, args, train_type):
    df["mas_answer"] = df["trace"].apply(extract_mas_answer_gsm_hard)
    df["ground_truth_answer"] = df["trace"].apply(extract_ground_truth_answer_gsm_hard)

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
    

    # sum total duration of execution, total input tokens, and total output tokens
    for trace_dict in df["trace"]:
        call_durations = []
        call_input_tokens = []
        call_output_tokens = []
        for call in trace_dict["trace"]:
            call_durations.append(call["call_statistic"]["duration"])
            call_input_tokens.append(call["call_statistic"]["input_tokens"])
            call_output_tokens.append(call["call_statistic"]["output_tokens"])

        # Convert to numpy array for statistics
        durations_arr = np.array(call_durations)
        input_tokens_arr = np.array(call_input_tokens)
        output_tokens_arr = np.array(call_output_tokens)

        num_llm_calls.append(len(durations_arr))

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

    llm_outputs = df["trace"].apply(extract_llm_outputs_gsm_hard)
    flattened_llm_outputs = [" ".join(sublist) for sublist in llm_outputs]
    df["raw_llm_outputs"] = flattened_llm_outputs

    # embed the llm outputs
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    df["trace_embedding"] = df["raw_llm_outputs"].apply(lambda x: embed_full_trace(x, model)) 

    # create tabular dataset
    embedding_matrix = np.vstack(df["trace_embedding"].values)
    embedding_cols = [f"emb_{i}" for i in range(embedding_matrix.shape[1])]
    embedding_df = pd.DataFrame(embedding_matrix, columns=embedding_cols, index=df.index)

    modeling_df = pd.concat([
        df[[
            "label",
            "num_llm_calls",
            "llm_call_duration_mean",
            "llm_call_input_token_mean",
            "llm_call_output_token_mean",
            "llm_call_duration_min",
            "llm_call_input_token_min",
            "llm_call_output_token_min",
            "llm_call_duration_max",
            "llm_call_input_token_max",
            "llm_call_output_token_max",
            "llm_call_duration_std",
            "llm_call_input_token_std",
            "llm_call_output_token_std",
            "mas_answer",
        ]],
        embedding_df
    ], axis=1)

    # create a column for if the MAS was NaN or not
    modeling_df["mas_answer_is_nan"] = (modeling_df["mas_answer"].isna() | np.isinf(modeling_df["mas_answer"])).astype(int)
    # replace NaN values with 0 (since we have a separate column indicating if it was NaN or not)
    modeling_df["mas_answer"] = modeling_df["mas_answer"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    contamination = modeling_df["label"].mean()  # proportion of outliers in the data
    print(f"the contamination is {contamination}")
    
    labels = np.array(modeling_df["label"].values)
    train_inlier_indexes, train_inlier_labels, train_outlier_indexes, train_outlier_labels, test_indexes, test_labels = get_train_test_indexes(args, labels)
    
    X_test = modeling_df.drop(columns=["label"]).values[test_indexes]
    y_test = modeling_df["label"].values[test_indexes]
    
    if train_type == 'supervised' or train_type == 'unsupervised':
        # combine train_inlier_indexes and train_outlier_indexes for supervised learning
        training_indexes = np.concatenate([train_inlier_indexes, train_outlier_indexes])
        
        X_train = modeling_df.drop(columns=["label"]).values[training_indexes]
        y_train = modeling_df["label"].values[training_indexes]
        return X_train, y_train, X_test, y_test
    
    elif train_type == 'occ':
        # only use train_inlier_indexes for OCC training
        X_train = modeling_df.drop(columns=["label"]).values[train_inlier_indexes]
        y_train = modeling_df["label"].values[train_inlier_indexes]
        return X_train, y_train, X_test, y_test
    
    else:
        raise ValueError("train_type must be one of 'supervised', 'occ', or 'unsupervised'")
    

def convert_traces_EscapeRoom_v2(df, args, train_type):
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
    

    # sum total duration of execution, total input tokens, and total output tokens
    for trace_dict in df["trace"]:
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
        durations_arr = np.array(call_durations)
        input_tokens_arr = np.array(call_input_tokens)
        output_tokens_arr = np.array(call_output_tokens)

        num_llm_calls.append(len(durations_arr))

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

    # embed the llm outputs
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    df["trace_embedding"] = df["raw_llm_outputs"].apply(lambda x: embed_full_trace(x, model)) 

    # create tabular dataset
    embedding_matrix = np.vstack(df["trace_embedding"].values)
    embedding_cols = [f"emb_{i}" for i in range(embedding_matrix.shape[1])]
    embedding_df = pd.DataFrame(embedding_matrix, columns=embedding_cols, index=df.index)

    modeling_df = pd.concat([
        df[[
            "label",
            "num_llm_calls",
            "llm_call_duration_mean",
            "llm_call_input_token_mean",
            "llm_call_output_token_mean",
            "llm_call_duration_min",
            "llm_call_input_token_min",
            "llm_call_output_token_min",
            "llm_call_duration_max",
            "llm_call_input_token_max",
            "llm_call_output_token_max",
            "llm_call_duration_std",
            "llm_call_input_token_std",
            "llm_call_output_token_std",
        ]],
        embedding_df
    ], axis=1)
    
    contamination = modeling_df["label"].mean()  # proportion of outliers in the data
    print(f"the contamination is {contamination}")
    
    labels = np.array(modeling_df["label"].values)
    train_inlier_indexes, train_inlier_labels, train_outlier_indexes, train_outlier_labels, test_indexes, test_labels = get_train_test_indexes(args, labels)
    
    X_test = modeling_df.drop(columns=["label"]).values[test_indexes]
    y_test = modeling_df["label"].values[test_indexes]
    
    if train_type == 'supervised' or train_type == 'unsupervised':
        # combine train_inlier_indexes and train_outlier_indexes for supervised learning
        training_indexes = np.concatenate([train_inlier_indexes, train_outlier_indexes])
        
        X_train = modeling_df.drop(columns=["label"]).values[training_indexes]
        y_train = modeling_df["label"].values[training_indexes]
        return X_train, y_train, X_test, y_test
    
    elif train_type == 'occ':
        # only use train_inlier_indexes for OCC training
        X_train = modeling_df.drop(columns=["label"]).values[train_inlier_indexes]
        y_train = modeling_df["label"].values[train_inlier_indexes]
        return X_train, y_train, X_test, y_test
    
    else:
        raise ValueError("train_type must be one of 'supervised', 'occ', or 'unsupervised'")

        
def prepare_traces_tabular(args, train_type):
    print("Preparing Lomas traces in tabular format")

    # read in the data    
    if args.trace_dataset_name == "gsm_hard":
        df = load_traces_to_df_gsm_hard(args.path_traces)
    elif args.trace_dataset_name == "EscapeRoom_v2":
        df = load_traces_to_df_EscapeRoom_v2(args.path_traces)
    print(f"Read in {len(df)} traces")

    # filter the df based on requested temperature values
    temp_values = [float(t) for t in args.temp_llm]
    df = df[df["temperature"].isin(temp_values)]

    # convert raw traces to tabular data
    if args.trace_dataset_name == "gsm_hard":
        X_train, y_train, X_test, y_test = convert_traces_gsm_hard(df, args, train_type)
    elif args.trace_dataset_name == "EscapeRoom_v2":
        X_train, y_train, X_test, y_test = convert_traces_EscapeRoom_v2(df, args, train_type)

    # normalize 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test


