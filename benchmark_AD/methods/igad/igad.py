import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from benchmark_AD.methods.g_safeguard.convert_lomas_traces import *
from benchmark_AD.methods.g_safeguard.gen_training_dataset import gen_training_dataset
from benchmark_AD.methods.g_safeguard.data import AgentGraphDataset
from benchmark_AD.methods.igad.model import IGAD
from benchmark_AD.utils import log_results, predict_scores


def train(epoch, model, trainloader, optimizer, criterion, device, label_priors):
    model.train()
    epoch_loss = 0
    epoch_time = 0
    total = 0
    j = 0
    for data in trainloader:
        
        # IGAD expects a sparse adjacency matrix.
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        values = torch.ones(edge_index.size(1)).to(data.x.device)
        adj = torch.sparse_coo_tensor(
            edge_index,
            values,
            (num_nodes, num_nodes)
        ).to(device)

        # node features
        feats = data.x.to(device)

        # graph pool: average
        num_graphs = data.num_graphs
        graph_pool = torch.zeros((num_graphs, num_nodes), device=feats.device)

        for g in range(num_graphs):
            node_mask = (data.batch == g)
            n = node_mask.sum()
            graph_pool[g, node_mask] = 1./n
        graph_pool = graph_pool.to(device)

        # graph batch
        graph_indicator = data.batch.to(device)
        
        # graph labels
        graph_labels = torch.zeros(num_graphs, device=data.y.device)
        for g in range(num_graphs):
            node_mask = (data.batch == g)
            if torch.any(data.y[node_mask] == 1):
                graph_labels[g] = 1
        labels = graph_labels

        optimizer.zero_grad()

        outputs = model(adj, feats, graph_pool, graph_indicator)
        loss = criterion(outputs + label_priors, labels.to(torch.long).to(device))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(trainloader), model

def compute_priors(num1, num2, device):
    y_prior = torch.log(torch.tensor([num1+1e-8, num2+1e-8], requires_grad = False)).to(device)
    return y_prior


def test(epoch, model, testloader, criterion, device, label_priors):
    model.eval()
    graph_level_labels = []
    graph_level_scores = []

    with torch.no_grad():
        for data in testloader:
            # IGAD expects a sparse adjacency matrix.
            num_nodes = data.num_nodes
            edge_index = data.edge_index
            values = torch.ones(edge_index.size(1)).to(data.x.device)
            adj = torch.sparse_coo_tensor(
                edge_index,
                values,
                (num_nodes, num_nodes)
            ).to(device)

            # node features
            feats = data.x.to(device)
            
            # graph pool: average 
            num_graphs = data.num_graphs
            graph_pool = torch.zeros((num_graphs, num_nodes), device=feats.device)

            for g in range(num_graphs):
                node_mask = (data.batch == g)
                n = node_mask.sum()
                graph_pool[g, node_mask] = 1./n
            graph_pool = graph_pool.to(device)

            # graph batch
            graph_indicator = data.batch.to(device)
            
            # graph labels
            graph_labels = torch.zeros(num_graphs, device=data.y.device)
            for g in range(num_graphs):
                node_mask = (data.batch == g)
                if torch.any(data.y[node_mask] == 1):
                    graph_labels[g] = 1
            labels = graph_labels

            outputs = model(adj, feats, graph_pool, graph_indicator)
            outputs = nn.functional.softmax(outputs, dim=1)
            anomaly_score = outputs[:, 1]  # anomaly score is the probability of being an anomaly (class 1)
            graph_level_scores.append(anomaly_score.item())
            graph_level_labels.append(labels.item())
    
    return graph_level_scores, graph_level_labels


def eval_metrics(graph_level_scores, graph_level_labels, args, epoch, epoch_total, model, train_ids, test_ids):
    y_pred = predict_scores(args, graph_level_scores, graph_level_labels, probability=True)
    
    # compute f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
    f1 = f1_score(graph_level_labels, y_pred)
    acc = accuracy_score(graph_level_labels, y_pred)
    bal_acc = balanced_accuracy_score(graph_level_labels, y_pred)
    auc = roc_auc_score(graph_level_labels, graph_level_scores)

    # print metrics
    print(f"Graph-level Metrics: F1-score: {f1:.4f}")
    print(f"Graph-level Metrics: Accuracy: {acc:.4f}")
    print(f"Graph-level Metrics: AUC: {auc:.4f}")
    print(f"Graph-level Metrics: Balanced Accuracy: {bal_acc:.4f}")

    log_results(args, epoch, epoch_total, f1, acc, auc, bal_acc, model=model, gt_labels=graph_level_labels, pred_labels=y_pred, pred_scores=graph_level_scores, train_ids=train_ids, test_ids=test_ids)  # epoch for epoch


def train_igad_on_graphs(args, graph_dataset_train, graph_dataset_test, train_ids, test_ids):

    # hyperparamters from igad codebase
    dim_1 = 32
    f_hidden_dim = 64
    f_output_dim = 32
    t_hidden_dim = 32
    t_output_dim = 16
    graph_pooling_type = 'average'
    n_subgraphs = 5
    size_subgraphs = 10
    max_step = 6
    normalize = True
    dropout = 0.2
    lr = 1e-3
    batch_size = 128
    epochs = getattr(args, "epochs", None) or 100
    num_layers = 2
    hard = False
    device = args.device


    # supervised: only train with inliers and outliers
    train_dataset = graph_dataset_train
    val_dataset = graph_dataset_test

    train_labels = []
    for graph in train_dataset:
        node_labels = np.array(graph['labels'])
        if np.all(node_labels == 0):
            train_labels.append(0)
        elif np.all(node_labels == 1):
            train_labels.append(1)
        else:
            raise ValueError("Graph contains mixed node labels.")

    graph_labels = np.array(train_labels)
    
    num_y_0 = train_labels.count(0)
    num_y_1 = train_labels.count(1)
    label_priors = compute_priors(num_y_0, num_y_1, device)

    train_dataset = AgentGraphDataset(train_dataset, phase="train")
    val_dataset = AgentGraphDataset(val_dataset, phase="val")

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = args.device

    example = train_dataset[0]

    features_dim = example.x.size(1)
    
    model = IGAD(features_dim,
				 dim_1,
				 f_hidden_dim,
				 f_output_dim,
				 t_hidden_dim,
				 t_output_dim,
				 graph_pooling_type,
				 n_subgraphs,
				 size_subgraphs,
				 max_step,
				 normalize,
				 dropout,
				 device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        loss, model = train(epoch, model, trainloader, optimizer, criterion, device, label_priors)
        print(f"Epoch {epoch}, Loss: {loss}")

        # evaluate the model
        graph_level_scores, graph_level_labels = test(epoch, model, testloader, criterion, device, label_priors)
        eval_metrics(graph_level_scores, graph_level_labels, args, epoch, epochs+1, model, train_ids, test_ids)


def run_igad(args, train_df, test_df):
    print("Running IGAD")

    train_ids = train_df["row_id"].values
    test_ids = test_df["row_id"].values
    
    # convert traces to g_safeguard format
    converted_train = convert_lomas_traces_escape_room_v2(args, train_df)  # same function as in g_safeguard
    converted_test = convert_lomas_traces_escape_room_v2(args, test_df)  # same function as in g_safeguard

    # create g_safeguard graphs
    graph_dataset_train = gen_training_dataset(args, converted_train)
    graph_dataset_test = gen_training_dataset(args, converted_test)

    train_igad_on_graphs(args, graph_dataset_train, graph_dataset_test, train_ids, test_ids)