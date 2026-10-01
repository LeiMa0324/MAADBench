import json
import os
import re
import pandas as pd
from pathlib import Path


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


def extract_llm_outputs(trace):
    '''
    extracts and cleans the raw reasoning outputs from the LLM calls in the trace
    '''
    llm_outputs = []
    llm_durations = []
    for llm_output in trace["trace"]:  # each llm_output is the full output from 1 LLM call
        llm_reasoning = llm_output["output"]["reasoning"] 
        try:
            llm_durations.append(llm_output["call_statistic"]["duration"]) # only extract the reasoning of the LLM to prevent repetition
        except:
            llm_durations.append(-1)  # if duration is missing, fill with -1

        # clean the text
        llm_reasoning = clean_text(llm_reasoning)
        llm_outputs.append(llm_reasoning)  # only extract the reasoning of the LLM to prevent repetition 

    
    return llm_outputs, llm_durations


def clean_text(text):
    '''
    cleans the raw reasoning output from the LLM calls
    '''
    
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


def stringify_input(step_input):
    '''
    regular agent steps have a plain string input; tool call steps have a dict input,
    with the values actually passed to the tool under "received_input" (agent steps have no such field)
    '''
    if isinstance(step_input, str):
        return step_input
    return json.dumps(step_input.get("received_input", step_input))


def convert_lomas_traces_gsm_hard(PATH_TO_TRACES):
    '''
    convert each trace in df to the graph format blindguard needs 
    save the converted traces to the proper location
    '''

    df = load_traces_to_df_gsm_hard(PATH_TO_TRACES)
    
    dataset = []
    for _, row in df.iterrows():
        trace = row["trace"]
        label = int(row["label"])

        # map each agent to an index
        agent_to_idx = {
            "cot_solver": 0,
            "pot_solver": 1,
            "judge": 2
        }

        # ----- Adjacency matrix: only cot and pot talk to judge -----
        adj_matrix = [
            [0, 0, 1],  # cot -> judge
            [0, 0, 1],  # pot -> judge
            [0, 0, 1]   # judge -> judge # self loop to include judge output
        ]

        # ----- System prompts -----
        system_prompts = [
            clean_text(trace['config']['agents'][0]['prompt']),  # cot_solver
            clean_text(trace['config']['agents'][1]['prompt']),  # pot_solver
            clean_text(trace['config']['agents'][2]['prompt'])  # judge
        ]

        # ----- Communication data and runtime duration per agent(single turn) -----
    
        communication_data, llm_durations = extract_llm_outputs(trace)
        
        communication_data = [[
            (0, f"{communication_data[0]} [duration: {llm_durations[0]:.2f}s]"),  # cot_solver
            (1, f"{communication_data[1]} [duration: {llm_durations[1]:.2f}s]"),  # pot_solver
            (2, f"{communication_data[2]} [duration: {llm_durations[2]:.2f}s]")   # judge
        ]]

        # ----- attacker_idxes: label per agent/node: if graph is abnormal label all nodes as abnormal -----
        if label == 1:
            attacker_idxes = [0, 1, 2]
        else:
            attacker_idxes = []

        graph_data = {
            "adj_matrix": adj_matrix,
            "attacker_idxes": attacker_idxes,
            "system_prompts": system_prompts,
            "communication_data": communication_data
        }

        dataset.append(graph_data)  # save the graph
    
    print(f"Converted {len(dataset)} graphs")
    return dataset


def build_adj_matrix_escape_room_v2(trace, agent_to_idx):
    '''
    If agent A is followed by agent B in the trace, then A -> B (directed edge)
    '''
    num_agents = len(agent_to_idx)
    adj_matrix = [[0 for _ in range(num_agents)] for _ in range(num_agents)]  # init
    for i in range(len(trace) - 1):
        curr_agent = trace[i]["agent"]
        next_agent = trace[i + 1]["agent"]

        curr_idx = agent_to_idx[curr_agent]
        next_idx = agent_to_idx[next_agent]

        # add directed edge
        adj_matrix[curr_idx][next_idx] = 1

    # self loop for last output 
    last_agent = trace[-1]["agent"]
    last_idx = agent_to_idx[last_agent]
    adj_matrix[last_idx][last_idx] = 1

    return adj_matrix


def create_dataset_trace_level(df):
    dataset = []
    for _, row in df.iterrows():
        trace = row["trace"]
        label = int(row["label"])

        trace_steps = trace["trace"] or [{"agent": "execution", "input": "", "output_message": ""}]
        n_nodes = len(trace_steps)
           
        agent_prompt_lookup = {
            agent["role"]: clean_text(agent["prompt"])
            for agent in trace["config"]["agents"]
        }
        
        adj_matrix = [[0] * n_nodes for _ in range(n_nodes)]  # Create directed graph
        for i in range(n_nodes - 1):
            adj_matrix[i][i + 1] = 1  # i -> i+1
        adj_matrix[n_nodes - 1][n_nodes - 1] = 1  # self-loop on final node

        system_prompts = [
            f"{agent_prompt_lookup.get(step['agent'], '')} {clean_text(stringify_input(step.get('input', '')))}"
            for step in trace_steps
        ]

        communication_data_row = []
        for node_idx, step in enumerate(trace_steps):
            # v2 traces use "message", v3 traces (with/without tool calls) use "output_message"
            message = clean_text(step.get("output_message", step.get("message", "")))
            duration = step.get("call_statistic", {}).get("duration", -1)
            communication_data_row.append(
                (node_idx, f"{message} [duration: {duration:.2f}s]")
            )

        communication_data = [communication_data_row]

        if label == 1:
            attacker_idxes = list(range(n_nodes))
        else:
            attacker_idxes = []

        graph_data = {
            "adj_matrix": adj_matrix,
            "attacker_idxes": attacker_idxes,
            "system_prompts": system_prompts,
            "communication_data": communication_data
        }

        dataset.append(graph_data)  # save the graph
    
    print(f"Converted {len(dataset)} graphs")
    return dataset


def _build_graph_from_nodes(nodes, trace_data, args):
    """Helper function to build graph data for a single trace"""
    n_nodes = len(nodes)
    
    # Build adjacency matrix (sequential chain)
    adj_matrix = [[0] * n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes - 1):
        adj_matrix[i][i + 1] = 1  # i -> i+1
    adj_matrix[n_nodes - 1][n_nodes - 1] = 1  # self-loop on final node
    
    # Build agent prompt lookup
    agent_prompt_lookup = {
        agent["role"]: clean_text(agent["prompt"])
        for agent in nodes[0]["row"]["raw_trace"]["config"]["agents"]
    }
    
    # Extract system prompts and communication data
    system_prompts = []
    communication_data_row = []
    node_labels = []  # Node-level labels
    attacker_idxes = []

    for node_idx, node_info in enumerate(nodes):
        row = node_info["row"]
        step_num = node_info["step_num"]
        node_label = node_info["label"]
        
        agent_role = row["step_trace"]["agent"]
        step_input = clean_text(stringify_input(row["step_trace"].get("input", "")))
        # v2 traces use "message", v3 traces (with/without tool calls) use "output_message"
        message = clean_text(row["step_trace"].get("output_message", row["step_trace"].get("message", "")))
        try:
            duration = row["step_trace"]['call_statistic']['duration']
        except:
            duration = -1  # if duration is missing, fill with -1
        try:
            input_tokens = row["step_trace"]['call_statistic']['input_tokens']
        except:
            input_tokens = -1  # if input_tokens is missing, fill with -1
        try:
            output_tokens = row["step_trace"]['call_statistic']['output_tokens']
        except:
            output_tokens = -1  # if output_tokens is missing, fill with -1

        system_prompts.append(
            f"{agent_prompt_lookup.get(agent_role)} {step_input}"
        )
        
        communication_data_row.append(
            (node_idx, f"{message} [duration: {duration:.2f}s] [input_tokens: {input_tokens}] [output_tokens: {output_tokens}]")
        )
        
        if node_label == 1:
            attacker_idxes.append(node_idx)
    
    communication_data = [communication_data_row]
    
    graph_data = {
        "adj_matrix": adj_matrix,
        "system_prompts": system_prompts,
        "communication_data": communication_data,
        "attacker_idxes": attacker_idxes,  # Node-level labels instead of attacker_idxes
    }
    
    return graph_data


def create_dataset_action_level(df, args):
    dataset = []
    current_trace_id = None
    current_nodes = []
    current_trace_data = None
    for _, row in df.iterrows():
        trace_id = row["trace_id"]
        step_num = row["step_num"]
        label = int(row["step_label"])

        if trace_id != current_trace_id and current_nodes:
            graph_data = _build_graph_from_nodes(
                current_nodes, 
                current_trace_data,
                args
            )
            dataset.append(graph_data)
            current_nodes = []

        # Start tracking new trace
        if trace_id != current_trace_id:
            current_trace_id = trace_id
            current_trace_data = row["step_trace"]
        
        # Add node information
        current_nodes.append({
            "step_num": step_num,
            "row": row,
            "label": label
        })

    if current_nodes:
        graph_data = _build_graph_from_nodes(
            current_nodes, 
            current_trace_data,
            args
        )
        dataset.append(graph_data)
    
    print(f"Converted {len(dataset)} graphs with node-level labels")
    return dataset

        
        
        
def convert_lomas_traces_escape_room_v2(args, df):
    '''
    convert each trace in df to the graph format blindguard needs 
    save the converted traces to the proper location
    each action is treated as a node
    '''
    if args.dataset_level == "trace":
        dataset = create_dataset_trace_level(df)  # trace is label
    elif args.dataset_level == "action":
        dataset = create_dataset_action_level(df, args)  # action is label

    return dataset