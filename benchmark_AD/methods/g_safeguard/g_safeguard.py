import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from benchmark_AD.methods.g_safeguard.convert_lomas_traces import *
from benchmark_AD.methods.g_safeguard.gen_training_dataset import gen_training_dataset
from benchmark_AD.methods.g_safeguard.data import AgentGraphDataset
from benchmark_AD.methods.g_safeguard.model import MyGAT
from benchmark_AD.utils import log_results, predict_scores


def test(model, test_loader, criterion, device, args, epoch, epoch_total, train_ids, test_ids):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    graph_level_preds = []
    graph_level_labels = []
    graph_level_scores = []

    with torch.no_grad():
        for data in test_loader:
            x, y, edge_index, edge_attr = data.x.to(device), data.y.to(device), data.edge_index.to(device), data.edge_attr.to(device)

            outputs = model(x, edge_index, edge_attr)
            loss = criterion(outputs, y.float().unsqueeze(-1))
            running_loss += loss.item()

            predicted = (torch.sigmoid(outputs) >= 0.5).view(-1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            # if all values in predicted are 0 then label the graph-level label as 0 else label the graph-level label as 1
            if args.dataset_level == "trace":
                gt_labels = (1 if sum(y) > 0 else 0)  # DH: if any node is anomalous, the graph is anomalous            
                pred_labels = (1 if sum(predicted) > 0 else 0)  # DH: if any node is predicted anomalous, the graph is predicted anomalous
                outputs = torch.sigmoid(outputs).mean().item() 
            elif args.dataset_level == "action":
                gt_labels = y.cpu().numpy()
                pred_labels = predicted.cpu().numpy()
                outputs = torch.sigmoid(outputs).cpu().numpy()
            graph_level_labels.append(gt_labels)
            graph_level_preds.append(pred_labels)
            graph_level_scores.append(outputs)  # DH: use mean node loss as

    # compute f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
    graph_level_labels = np.concatenate([np.atleast_1d(x).reshape(-1) for x in graph_level_labels])
    graph_level_preds = np.concatenate([np.atleast_1d(x).reshape(-1) for x in graph_level_preds])
    graph_level_scores = np.concatenate([np.atleast_1d(x).reshape(-1) for x in graph_level_scores])
    graph_level_preds = predict_scores(args, graph_level_scores, graph_level_labels, probability=True)
    f1 = f1_score(graph_level_labels, graph_level_preds)
    acc = accuracy_score(graph_level_labels, graph_level_preds)
    bal_acc = balanced_accuracy_score(graph_level_labels, graph_level_preds)
    auc = roc_auc_score(graph_level_labels, graph_level_scores)

    # print metrics
    print(f"Graph-level Metrics: F1-score: {f1:.4f}")
    print(f"Graph-level Metrics: Accuracy: {acc:.4f}")
    print(f"Graph-level Metrics: AUC: {auc:.4f}")
    print(f"Graph-level Metrics: Balanced Accuracy: {bal_acc:.4f}")

    log_results(args, epoch, epoch_total, f1, acc, auc, bal_acc, model=model, gt_labels=graph_level_labels, pred_labels=graph_level_preds, pred_scores=graph_level_scores, train_ids=train_ids, test_ids=test_ids)


def train(model: MyGAT, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in train_loader:
        x, y, edge_index, edge_attr = data.x.to(device), data.y.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        random_turns = random.choice(list(range(1, 5)))
        edge_attr[:, :random_turns, :]
        optimizer.zero_grad()
        outputs = model(x, edge_index=edge_index, edge_attr=edge_attr)
        loss = criterion(outputs, y.float().unsqueeze(-1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = (torch.sigmoid(outputs) >= 0.5).squeeze()
        total += y.size(0)
        correct += (predicted == y).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

def remove_anomalous_nodes(graph):
    labels = graph["labels"]
    anomalous_indices = np.where(labels == 1)[0]

    if len(anomalous_indices) == 0:
        # No anomalous nodes, return graph as-is
        return graph

    first_anomaly_idx = anomalous_indices[0]
    keep_indices = np.arange(first_anomaly_idx)  # keep only nodes before the first anomaly

    if len(keep_indices) == 0:
        return None  # # No normal nodes before the first anomaly, return None or empty graph
    
    new_graph = graph.copy()
    new_graph["features"] = graph["features"][keep_indices]
    new_graph["labels"] = graph["labels"][keep_indices]

    edge_index = graph["edge_index"]
    mask = np.isin(edge_index[0], keep_indices) & np.isin(edge_index[1], keep_indices)
    new_edge_index = edge_index[:, mask]

    index_mapping = np.full(len(labels), -1, dtype=int)
    index_mapping[keep_indices] = np.arange(len(keep_indices))

    # if only 1 node remains, there is no edge
    new_edge_index = index_mapping[new_edge_index]
    new_graph["edge_index"] = new_edge_index

    new_graph["edge_attr"] = graph["edge_attr"][mask]

    num_nodes = len(keep_indices)
    new_adj_matrix = np.zeros((num_nodes, num_nodes), dtype=graph["adj_matrix"].dtype)

    # Fill in adjacency for kept nodes
    for i, old_i in enumerate(keep_indices):
        for j, old_j in enumerate(keep_indices):
            new_adj_matrix[i, j] = graph["adj_matrix"][old_i, old_j]
    
    new_graph["adj_matrix"] = new_adj_matrix

    if "attacker_idxes" in graph:
        new_attackers = [idx for idx in graph["attacker_idxes"] if idx in keep_indices]
        new_graph["attacker_idxes"] = new_attackers
    
    return new_graph


def clean_graph_dataset(args, graph_dataset, perc_outliers):
    n_graphs = len(graph_dataset)
    num_labeled_graphs = max(1, int(np.ceil(n_graphs * perc_outliers)))

    # Randomly select which graphs to keep anomalies in
    np.random.seed(args.seed)
    labeled_graph_indices = np.random.choice(n_graphs, size=num_labeled_graphs, replace=False)
    
    cleaned_dataset = []
    for graph_idx, graph in enumerate(graph_dataset):
        if graph_idx in labeled_graph_indices:
            # Keep this graph as-is (with anomalies)
            cleaned_dataset.append(graph)
        else:
            # Remove all anomalies from this graph
            cleaned_graph = remove_anomalous_nodes(graph)

            # Only keep graphs that still have nodes remaining
            if cleaned_graph is not None and len(cleaned_graph["labels"]) > 0:
                cleaned_dataset.append(cleaned_graph)

    return cleaned_dataset


def run_g_safeguard_on_graphs(args, graph_dataset_train, graph_dataset_test, train_ids, test_ids):
    '''
    run g_safeguard on the graph dataset 
    '''
    print("Running g_safeguard on the graphs")

    # Hyperparameters from g-safeguard codebase 
    hidden_dim = 1024
    dropout = 0.2
    num_heads = 8
    num_layers = 2
    epochs = getattr(args, "epochs", None) or 20
    lr = 0.001
    weight_decay = 0.0002
    batch_size = 32
    device = 0    

    
    if args.method == 'g_safeguard':
        # supervised: only train with inliers and outliers
        train_dataset = graph_dataset_train
        val_dataset = graph_dataset_test

    elif args.method == 'g_safeguard_semi_supervised':
        # Semi-supervised training: only keep small number of anomalous nodes
        cleaned_graphs = clean_graph_dataset(args, graph_dataset_train, args.perc_outliers_train)  # only keep clean nodes before anomalous node occurs
        train_dataset = cleaned_graphs
        val_dataset = graph_dataset_test

    train_dataset = AgentGraphDataset(train_dataset, phase="train")
    val_dataset = AgentGraphDataset(val_dataset, phase="val")
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_dataset, shuffle=False)
    example = train_dataset[0]
    in_channels = example.x.size(1)
    edge_dim = example.edge_attr.size()[1:]

    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    gnn = MyGAT(in_channels, hidden_dim, out_channels=1, heads=num_heads, num_layers=num_layers, edge_dim=edge_dim)
    gnn.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(gnn.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    best_acc = 0.0

    for i in range(epochs): 
        train_loss, train_acc = train(gnn, trainloader, criterion, optimizer, device=device)
        test(gnn, testloader, criterion, device=device, args=args, epoch=i, epoch_total=epochs, train_ids=train_ids, test_ids=test_ids)        
        
        scheduler.step()
        print(f"Epoch {i}/{epochs} || Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")


def run_g_safeguard(args, train_df, test_df):
    print("Running g_safeguard")
    
    train_ids = train_df["row_id"].values
    test_ids = test_df["row_id"].values
    
    # convert traces to g_safeguard format
    converted_train = convert_lomas_traces_escape_room_v2(args, train_df)  # same function as in g_safeguard
    converted_test = convert_lomas_traces_escape_room_v2(args, test_df)  # same function as in g_safeguard

    # create g_safeguard graphs
    graph_dataset_train = gen_training_dataset(args, converted_train)
    graph_dataset_test = gen_training_dataset(args, converted_test)

    # run g_safeguard on the graphs
    run_g_safeguard_on_graphs(args, graph_dataset_train, graph_dataset_test, train_ids, test_ids)

