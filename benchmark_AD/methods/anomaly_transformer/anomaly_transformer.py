import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch

from benchmark_AD.methods.anomaly_transformer.convert_lomas_traces import *
from benchmark_AD.methods.anomaly_transformer.solver import Solver


def pad_labels(labels_list, max_length=100):
    """
    Pad labels to fixed length.
    
    Args:
        labels_list: List of lists, each inner list is labels for one trace
        max_length: Target length (should match your win_size)
    
    Returns:
        numpy array of shape (num_traces, max_length)
    """
    padded_labels = []
    for trace_labels in labels_list:
        if len(trace_labels) < max_length:
            # Pad with 0s (normal)
            padded = trace_labels + [0] * (max_length - len(trace_labels))
        else:
            # Truncate if longer
            padded = trace_labels[:max_length]
        padded_labels.append(padded)
    
    return np.array(padded_labels, dtype=np.float32)


def train_anomaly_transformer(args, train_sequences, train_labels, test_sequences, test_labels, original_lengths_train, original_lengths_test, train_ids, test_ids):
    # hyperparameters from codebase
    
    if args.trace_dataset_name == 'gsm_hard':
        config = {
            'lr': 1e-4,
            'num_epochs': getattr(args, 'epochs', None) or 10,
            'k': 3,
            'win_size': 100,  # 100
            'input_c': 387,
            'output_c': 387,
            'batch_size': 256,
            'pretrained_model': 20,
            'mode': "train",
        }
    else:
        config = {
            'lr': 1e-4,
            'num_epochs': getattr(args, 'epochs', None) or 10,
            'k': 3,
            'win_size': 100,  # 100
            'input_c': 384,
            'output_c': 384,
            'batch_size': 32,
            'pretrained_model': 20,
            'mode': "train",
        }

    train_labels = pad_labels(train_labels, max_length=100)
    test_labels = pad_labels(test_labels, max_length=100)

    train_sequences = np.array(train_sequences)
    test_sequences = np.array(test_sequences)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    train_len = train_sequences.shape[0]
    # win_size = min(100, train_len // 100)
    win_size = 100
    config['win_size'] = win_size

    # scaler = StandardScaler()
    # scaler.fit(train_sequences)
    # flat_train = scaler.transform(train_sequences)
    # flat_test  = scaler.transform(test_sequences)

    train_X = torch.from_numpy(train_sequences)  
    train_y = torch.from_numpy(train_labels)     
    test_X = torch.from_numpy(test_sequences)    
    test_y = torch.from_numpy(test_labels)       
    
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    solver = Solver(config, train_data=train_sequences, train_labels=train_labels, test_data=test_sequences, test_labels=test_labels, args=args)
    
    solver.train_loader = train_loader
    solver.test_loader = test_loader
    
    solver.train(original_lengths_test, train_ids, test_ids)

def run_anomaly_transformer(args, train_df, test_df):
    print("Running Anomaly Transformer...")

    # convert Lomas traces to transformer format
    train_sequences, train_labels, test_sequences, test_labels, train_original_lengths, test_original_lengths, train_ids, test_ids = convert_lomas_traces_escape_room_v2(args, train_df, test_df)
    
    train_anomaly_transformer(args, train_sequences, train_labels, test_sequences, test_labels, train_original_lengths, test_original_lengths, train_ids, test_ids)