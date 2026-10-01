import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from benchmark_AD.methods.g_safeguard.convert_lomas_traces import *
from benchmark_AD.methods.g_safeguard.gen_training_dataset import gen_training_dataset
from benchmark_AD.methods.g_safeguard.data import AgentGraphDataset
from benchmark_AD.methods.blindguard.TAM import TAMModel
from benchmark_AD.utils import log_results, predict_scores


def eval_model_metrics(loss_per_graph, label_per_graph, args, epoch, epoch_total, model, train_ids, test_ids):
    # anomalies will have a higher loss
    loss_per_graph = np.array(loss_per_graph)
    label_per_graph = np.array(label_per_graph)
    
    y_pred = predict_scores(args, loss_per_graph, label_per_graph)

    # Metrics
    f1 = f1_score(label_per_graph, y_pred)
    acc = accuracy_score(label_per_graph, y_pred)
    bal_acc = balanced_accuracy_score(label_per_graph, y_pred)

    # For AUC we use raw loss scores
    auc = roc_auc_score(label_per_graph, loss_per_graph)

    # print metrics
    print(f"Graph-level Metrics: F1-score: {f1:.4f}")
    print(f"Graph-level Metrics: Accuracy: {acc:.4f}")
    print(f"Graph-level Metrics: AUC: {auc:.4f}")
    print(f"Graph-level Metrics: Balanced Accuracy: {bal_acc:.4f}")

    log_results(args, epoch, epoch_total, f1, acc, auc, bal_acc, model=model, gt_labels=label_per_graph, pred_labels=y_pred, pred_scores=loss_per_graph, train_ids=train_ids, test_ids=test_ids)


def test(model, test_loader, device, args):
    model.eval()
    total_loss = 0
    num_batches = 0

    loss_per_graph = []  # DH: we want graph-level prediction
    label_per_graph = []  # DH: we want graph-level label (0 if all nodes are normal, 1 if any node is anomalous)
    
    with torch.no_grad():
        for data in test_loader:
            x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
            y = data.y.to(device)  # DH: Get per node label: all 0 for inliers, all 1 for outliers. num_nodes (3) * batch_size (12)
            node_emb = model.encode(x, edge_index)
            num_nodes = x.size(0)
            adj = torch.eye(num_nodes, device=device)
            adj[edge_index[0], edge_index[1]] = 1.0
            loss1, _ = model.max_message(node_emb, adj)
            reg_loss = model.reg_edge(node_emb, adj)
            loss = loss1 + 1 * reg_loss

            if args.dataset_level == "trace":
                avg_loss_per_node = loss.item()
                loss_per_graph.append(avg_loss_per_node)  # DH
                label_per_graph.append(1 if sum(y) > 0 else 0)  # DH: if any node is anomalous, the graph is anomalous
            
            elif args.dataset_level == "action":
                loss1_per_node = model.max_message_per_node(node_emb, adj)
                reg_loss_per_node = model.reg_edge_per_node(node_emb, adj)
                loss_per_node = loss1_per_node + 1 * reg_loss_per_node
                
                loss_per_graph.append(loss_per_node.cpu().numpy())  # DH: for action-level, we keep the node-level losses
                label_per_graph.append(y.cpu().numpy())  # DH: for action-level, we keep the node-level labels and losses

            else:
                raise ValueError(f"dataset_level {args.dataset_level} not recognized. Must be 'trace' or 'action'.")
            
            total_loss += loss.item()
            num_batches += 1

    loss_per_graph = np.concatenate([np.atleast_1d(x) for x in loss_per_graph])
    label_per_graph = np.concatenate([np.atleast_1d(x) for x in label_per_graph])
        
    return total_loss / max(num_batches, 1), loss_per_graph, label_per_graph


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    # anamaly_ori_idx = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0]).to(device)  # assumes the graph has exactly 8 nodes. 3/8: ratio to make anomalous. Non-hardcoded added below

    for data in train_loader:
        x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        optimizer.zero_grad()

        # DH: non-hardcoded version of anamaly_ori_idx #
        num_nodes = x.size(0)
        fraction_anomalous = 3 / 8  # same as original
        num_anomalous = max(1, int(num_nodes * fraction_anomalous))
        anomaly_idx = torch.zeros(num_nodes, device=x.device)  # mask
        anomaly_idx[:num_anomalous] = 1

        anamaly_ori_idx = anomaly_idx
        # DH: non-hardcoded version of anamaly_ori_idx #

        anamaly_idx = anamaly_ori_idx[torch.randperm(len(anamaly_ori_idx))]
        noise = torch.randn_like(x)
        noise = F.normalize(noise, dim=1)
        noise_magnitude = 0.8 * torch.norm(x, dim=1, keepdim=True)
        noised_embeddings = x + noise * noise_magnitude * anamaly_idx.unsqueeze(1)
        x = noised_embeddings
        node_emb = model.encode(x, edge_index)
        loss = model.neg_all(node_emb, anamaly_idx)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1), model


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


def clean_graph_dataset(graph_dataset):
    cleaned_dataset = []
    for graph in graph_dataset:
        cleaned_graph = remove_anomalous_nodes(graph)

        # Only keep graphs that still have nodes remaining
        if cleaned_graph is not None and len(cleaned_graph["labels"]) > 0:
            cleaned_dataset.append(cleaned_graph)

    return cleaned_dataset


def run_blindguard_on_graphs(args, graph_dataset_train, graph_dataset_test, train_ids, test_ids):

    # Hyperparameters from blindguard codebase 
    batch_size = 32
    hidden_dim = 1024
    latent_dim = 512
    # dropout = 0.2
    # num_heads = 8
    # num_layers = 2
    # temperature = 0.1
    epochs = getattr(args, "epochs", None) or 50
    lr = 0.001
    weight_decay = 0.0002
    device = 0

    if args.occ_training == "clean":
        # only train with inliers
        cleaned_graphs = clean_graph_dataset(graph_dataset_train)  # only keep clean nodes before anomalous node occurs
        train_dataset = cleaned_graphs
    elif args.occ_training == "polluted":
        # train with all nodes labeled as inliers
        polluted_graphs = graph_dataset_train.copy()
        for graph in graph_dataset_train:
            graph["labels"] = graph["labels"] * 0
        train_dataset = polluted_graphs

    val_dataset = graph_dataset_test

    train_dataset = AgentGraphDataset(train_dataset, phase="train")
    val_dataset = AgentGraphDataset(val_dataset, phase="val")

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    example = train_dataset[0]
    edge_attr = example.edge_attr
    in_channels = edge_attr.size(-1)

    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"

    model = TAMModel(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        out_channels=latent_dim,
        dropout=0,
        readout='avg'
    )
    model.to(args.device)

    for param in model.parameters():
        param.data = param.data.float()
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    best_loss = float('inf')

    for epoch in range(epochs):
        train_loss, model = train(model, trainloader, optimizer, device=device)
        test_loss, loss_per_graph, label_per_graph = test(model, testloader, device=device, args=args)

        eval_model_metrics(loss_per_graph, label_per_graph, args, epoch, epochs, model, train_ids, test_ids) # DH: Eval metrics for graph-level anomaly detection 

        scheduler.step()
        
        print(f"Epoch {epoch}/{epochs} || Training Loss: {train_loss:.4f} || Test Loss: {test_loss:.4f} || Samples in batch: {1}")


def run_blindguard(args, train_df, test_df):
    print("Running blindguard")
    
    train_ids = train_df["row_id"].values
    test_ids = test_df["row_id"].values
    
    # convert traces to g_safeguard format
    converted_train = convert_lomas_traces_escape_room_v2(args, train_df)  # same function as in g_safeguard
    converted_test = convert_lomas_traces_escape_room_v2(args, test_df)  # same function as in g_safeguard

    # create g_safeguard graphs
    graph_dataset_train = gen_training_dataset(args, converted_train)
    graph_dataset_test = gen_training_dataset(args, converted_test)

    # run blindguard on the graphs
    run_blindguard_on_graphs(args, graph_dataset_train, graph_dataset_test, train_ids, test_ids)
    
