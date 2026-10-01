import numpy as np
import torch
import torch.nn as nn
import tqdm
from benchmark_AD.methods.g_safeguard import model
from benchmark_AD.methods.g_safeguard.convert_lomas_traces import *
from benchmark_AD.methods.g_safeguard.gen_training_dataset import gen_training_dataset
from benchmark_AD.methods.g_safeguard.data import AgentGraphDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
from benchmark_AD.utils import log_results, predict_scores

from benchmark_AD.methods.ggad.model import Model


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


def train_ggad_on_graphs(args, graph_dataset_train, graph_dataset_test, train_ids, test_ids):
    lr = 1e-3
    weight_decay = 0
    seed = args.seed
    embedding_dim = 300
    num_epoch = getattr(args, "epochs", None) or 100  # 100
    drop_prob = 0
    readout = 'avg'
    auc_test_rounds = 256
    negsamp_ratio = 1
    mean = 0 
    var = 0
    device = args.device
    batch_size = 1  # method is build to single graph: we adopt it for multiple graphs

    # Semi-supervised training: only keep small number of anomalous nodes
    cleaned_graphs = clean_graph_dataset(args, graph_dataset_train, args.perc_outliers_train)  # only keep clean nodes before anomalous node occurs
        
    train_dataset = cleaned_graphs
    
    val_dataset = graph_dataset_test

    train_dataset = AgentGraphDataset(train_dataset, phase="train")
    val_dataset = AgentGraphDataset(val_dataset, phase="val")

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    example = train_dataset[0]
    in_channels = example.x.size(1)
    print(f"Input channels: {in_channels}")

    device_id = 0
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    model = Model(in_channels, embedding_dim, 'prelu', negsamp_ratio, readout, var, mean).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([negsamp_ratio])).to(device)
    xent = nn.CrossEntropyLoss()  

    for epoch in range(num_epoch):
        print(f"Epoch {epoch+1}/{num_epoch}")
        model.train()
        total_loss = 0
        num_batches = 0
        for data in trainloader:
            x, edge_index, edge_attr, labels = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device), data.y.to(device)
            optimizer.zero_grad()
            
            num_nodes = x.size(0)
            x = x.unsqueeze(0)
            
            # convert edge_index to adj matric
            adj = torch.zeros(num_nodes, num_nodes)
            adj[edge_index[0], edge_index[1]] = 1
            adj = adj.unsqueeze(0)

            # get the abnormal and normal node indices from the labels
            abnormal_label_idx = torch.where(labels == 1)[0]
            normal_label_idx = torch.where(labels == 0)[0]

            # put everything on the same device
            adj = adj.to(device)
            abnormal_label_idx = abnormal_label_idx.to(device)
            normal_label_idx = normal_label_idx.to(device)

            # Train model
            train_flag = True
            emb, emb_combine, logits, emb_con, emb_abnormal = model(x, adj, abnormal_label_idx, normal_label_idx, train_flag)
            
            lbl = torch.unsqueeze(torch.cat((torch.zeros(len(normal_label_idx)), torch.ones(len(emb_con)))),1).unsqueeze(0)
            # put lbl on same device as logits
            lbl = lbl.to(device)

            loss_bce = b_xent(logits, lbl)
            loss_bce = torch.mean(loss_bce)

            emb = emb.squeeze(0)

            emb_inf = torch.norm(emb, dim=-1, keepdim=True)
            emb_inf = torch.pow(emb_inf, -1)
            emb_inf[torch.isinf(emb_inf)] = 0.
            emb_norm = emb * emb_inf

            sim_matrix = torch.mm(emb_norm, emb_norm.T)
            # raw_adj = torch.squeeze(raw_adj)
            similar_matrix = sim_matrix * adj

            r_inv = torch.pow(torch.sum(adj, 0), -1)
            r_inv[torch.isinf(r_inv)] = 0.
            affinity = torch.sum(similar_matrix, 0) * r_inv

            affinity_normal_mean = torch.mean(affinity[normal_label_idx])
            affinity_abnormal_mean = torch.mean(affinity[abnormal_label_idx])

            confidence_margin = 0.7
            # skip margin loss if no anomalous nodes in the batch
            if len(abnormal_label_idx) > 0:
                affinity_abnormal_mean = torch.mean(affinity[abnormal_label_idx])
                confidence_margin = 0.7
                loss_margin = (confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)

            else:
                loss_margin = torch.tensor(0.0, device=device)

            diff_attribute = torch.pow(emb_con - emb_abnormal, 2)
            loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))

            loss = 1 * loss_margin + 1 * loss_bce + 1 * loss_rec

            print(f"Batch loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()

    # testing 
    model.eval()
    scores_all = []
    labels_all = []
    for data in testloader:
        x, edge_index, edge_attr, labels = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device), data.y.to(device)

        num_nodes = x.size(0)
        x = x.unsqueeze(0)
        
        # convert edge_index to adj matric
        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1
        adj = adj.unsqueeze(0)

        # get the abnormal and normal node indices from the labels
        abnormal_label_idx = torch.where(labels == 1)[0]
        normal_label_idx = torch.where(labels == 0)[0]

        # put everything on the same device
        adj = adj.to(device)
        abnormal_label_idx = abnormal_label_idx.to(device)
        normal_label_idx = normal_label_idx.to(device)
        train_flag = False

        emb, emb_combine, logits, emb_con, emb_abnormal = model(x, adj, abnormal_label_idx, normal_label_idx, train_flag)

        scores = np.squeeze(logits.cpu().detach().numpy())
        labels = np.squeeze(labels.cpu().detach().numpy())
        
        if scores.ndim == 0:
            scores_all.append(np.atleast_1d(scores))
            labels_all.append(np.atleast_1d(labels))
        else:
            scores_all.append(scores)
            labels_all.append(labels)

    # get preds
    scores_all = np.concatenate(scores_all)
    labels_all = np.concatenate(labels_all)
    y_pred = predict_scores(args, scores_all, labels_all)

    # Metrics
    f1 = f1_score(labels_all, y_pred)
    acc = accuracy_score(labels_all, y_pred)
    bal_acc = balanced_accuracy_score(labels_all, y_pred)

    # For AUC we use raw loss scores
    auc = roc_auc_score(labels_all, scores_all)

    print(f"F1 Score: {f1:.4f}")
        
    log_results(args, epoch, num_epoch, f1, acc, auc, bal_acc, model=model, gt_labels=labels_all, pred_labels=y_pred, pred_scores=scores_all, train_ids=train_ids, test_ids=test_ids)  # epoch for epoch


def run_ggad(args, train_df, test_df):
    print("Running GGAD")

    train_ids = train_df["row_id"].values
    test_ids = test_df["row_id"].values
    
    # convert traces to g_safeguard format
    converted_train = convert_lomas_traces_escape_room_v2(args, train_df)  # same function as in g_safeguard
    converted_test = convert_lomas_traces_escape_room_v2(args, test_df)  # same function as in g_safeguard

    # create g_safeguard graphs
    graph_dataset_train = gen_training_dataset(args, converted_train)
    graph_dataset_test = gen_training_dataset(args, converted_test)

    train_ggad_on_graphs(args, graph_dataset_train, graph_dataset_test, train_ids, test_ids)

