import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import copy

def drop_gold_answer(trace):
    trace = copy.deepcopy(trace)  # don't modify the original
    del trace['question']['gold_answer']
    return trace


def load_traces_to_df(folder_path):
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

def get_raw_traces(args, run_setting, test_df):
    
    if run_setting == "zero_shot":
        test_labels = test_df['step_label'].to_numpy().astype(int)

        # loop through traces combine each action message with duration
        test_traces = []
        for index, row in test_df.iterrows():
            message = row["message_context"]
            try:
                duration = row["step_trace"]['call_statistic']['duration']
            except:
                duration = -1
            message_combined = f"{message} [duration: {duration:.2f}s]"
            test_traces.append(message_combined)

    return test_traces, test_labels
