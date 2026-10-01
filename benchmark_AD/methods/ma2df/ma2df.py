import torch
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
import torch.nn.functional as F

from benchmark_AD.methods.g_safeguard.convert_lomas_traces import *
from benchmark_AD.methods.g_safeguard.gen_training_dataset import gen_training_dataset
from benchmark_AD.methods.g_safeguard.data import AgentGraphDataset
from benchmark_AD.utils import log_results, predict_scores


def edge_index_to_set(edge_index):
    """
    edge_index: Tensor shape [2, num_edges]
    returns: set of (u, v) tuples
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    edges = set()
    for i in range(edge_index.shape[1]):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        edges.add((u, v))
    return edges


def build_clean_union_graph(trainloader):
    """
    Returns:
        clean_edge_set (set of edges)
    """
    clean_edge_set = set()
    node_semantic_sum = None
    node_counts = 0

    for batch in trainloader:
        graph = batch  # batch_size = 1
        edge_index = graph.edge_index
        edge_set = edge_index_to_set(edge_index)
        clean_edge_set |= edge_set  # union

        node_feats = graph.x
        n = node_feats.shape[0]
        dim = node_feats.shape[1]

        if node_semantic_sum is None:
            node_semantic_sum = node_feats.clone()
            node_counts = torch.ones(n, device=node_feats.device)

        else:
            current_size = node_semantic_sum.shape[0]
            if n > current_size:  # add padding: graph nodes can be different
                padding = torch.zeros(n - current_size, dim, device=node_feats.device)
                node_semantic_sum = torch.cat([node_semantic_sum, padding], dim=0)
                count_padding = torch.zeros(n - current_size, device=node_feats.device)
                node_counts = torch.cat([node_counts, count_padding], dim=0)

            node_semantic_sum[:n] += node_feats
            node_counts[:n] += 1

    normal_semantic_mean = node_semantic_sum / node_counts.clamp(min=1).unsqueeze(-1)
    return clean_edge_set, normal_semantic_mean


def jaccard_similarity(edge_set_1, edge_set_2):
    intersection = len(edge_set_1 & edge_set_2)
    union = len(edge_set_1 | edge_set_2)

    if union == 0:
        return 1.0  # both empty

    return intersection / union


def compute_anomaly_scores(clean_edge_set, testloader, normal_avg_edge_att, args, train_ids, test_ids):
    scores_structure = []
    scores_semantics = []
    gt_labels = []

    for batch in testloader:
        graph = batch  # batch_size = 1
        edge_index = graph.edge_index
        edge_set = edge_index_to_set(edge_index)

        # Structure MA2DF solution 
        j_score = jaccard_similarity(clean_edge_set, edge_set)  # smaller means more anomalous
        scores_structure.append(j_score)

        # Semantics added to make MA2DF applicable to other scenarios
        node_feats = graph.x
        n = node_feats.shape[0]
        reference = normal_avg_edge_att
        if n > len(reference):
            reference = torch.cat([reference, reference.mean(dim=0, keepdim=True).expand(n - len(reference), -1)])
        cos_sim = F.cosine_similarity(node_feats, reference[:n], dim=1)  # smaller means more anomalous, crop unioned avg train graph to fit current test graph
        scores_semantics.append(cos_sim.mean().item())

        label = batch.y
        gt_labels.append(1 if sum(label) > 0 else 0)

    # convert scores to labels. Smaller jaccard similarity means more anomalous
    scores_structure = np.array(scores_structure)
    scores_semantics = np.array(scores_semantics)

    # Normalize to [0,1]
    scores_structure = (scores_structure - scores_structure.min()) / (scores_structure.max() - scores_structure.min() + 1e-8)
    scores_semantics = (scores_semantics - scores_semantics.min()) / (scores_semantics.max() - scores_semantics.min() + 1e-8)
        
    alpha_ = 0.5  # to run original MA2DF set alpha_ to 1
    final_scores = alpha_ * np.array(scores_structure) + (1 - alpha_) * np.array(scores_semantics)

    pred_labels = predict_scores(args, -final_scores, gt_labels)

    # Metrics
    f1 = f1_score(gt_labels, pred_labels)
    acc = accuracy_score(gt_labels, pred_labels)
    bal_acc = balanced_accuracy_score(gt_labels, pred_labels)

    # For AUC we use raw loss scores
    auc = roc_auc_score(gt_labels, -final_scores)

    # print metrics
    print(f"Graph-level Metrics: F1-score: {f1:.4f}")
    print(f"Graph-level Metrics: Accuracy: {acc:.4f}")
    print(f"Graph-level Metrics: AUC: {auc:.4f}")
    print(f"Graph-level Metrics: Balanced Accuracy: {bal_acc:.4f}")

    log_results(args, 0, 1, f1, acc, auc, bal_acc, model=None, gt_labels=gt_labels, pred_labels=pred_labels, pred_scores=-final_scores, train_ids=train_ids, test_ids=test_ids)  # 0 for epoch 
   
    
    return final_scores, gt_labels, pred_labels


def run_ma2df(args, train_df, test_df):
    if args.dataset_level != "trace":
        raise ValueError("MA2DF produces trace-level predictions")
    if args.occ_training == "clean":
        train_df = train_df[train_df.label == 0].copy()
    if train_df.empty:
        raise ValueError("MA2DF has no clean training traces")
    train_graphs = gen_training_dataset(args, convert_lomas_traces_escape_room_v2(args, train_df))
    test_graphs = gen_training_dataset(args, convert_lomas_traces_escape_room_v2(args, test_df))
    trainloader = DataLoader(AgentGraphDataset(train_graphs), batch_size=1, shuffle=False)
    testloader = DataLoader(AgentGraphDataset(test_graphs, phase="val"), batch_size=1, shuffle=False)
    clean_edges, mean_features = build_clean_union_graph(trainloader)
    compute_anomaly_scores(clean_edges, testloader, mean_features, args,
                           train_df.row_id.to_numpy(), test_df.row_id.to_numpy())
