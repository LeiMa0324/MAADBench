import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch
from torch_geometric.loader import DataLoader
from scipy.sparse import coo_matrix
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from benchmark_AD.methods.g_safeguard.convert_lomas_traces import *
from benchmark_AD.methods.g_safeguard.gen_training_dataset import gen_training_dataset
from benchmark_AD.methods.rqgnn import model
from benchmark_AD.methods.g_safeguard.data import AgentGraphDataset
from benchmark_AD.methods.rqgnn import utils
import benchmark_AD.methods.rqgnn.lossfunc
from benchmark_AD.utils import log_results, predict_scores


def extract_rqgnn_inputs(dataset):
    adj_list = []
    feats_list = []
    label_list = []

    for data in dataset:
        n_nodes = data.x.size(0)
        
        # --- Adjacency matrix (scipy sparse) ---
        edge_index = data.edge_index.numpy()  # shape (2, n_edges)
        rows = edge_index[0]
        cols = edge_index[1]
        values = np.ones(len(rows), dtype=np.float32)
        adj = coo_matrix((values, (rows, cols)), shape=(n_nodes, n_nodes)).tolil()
        adj_list.append(adj)

        # --- Node features ---
        x = data.x
        feats_list.append(x)

        # --- Graph-level label (derived from node labels) ---
        node_labels = np.array(data.y.numpy() if hasattr(data, 'y') and data.y is not None else data['labels'])

        if np.all(node_labels == 0):
            label_list.append(0)
        elif np.all(node_labels == 1):
            label_list.append(1)
        else:
            raise ValueError(f"Graph has mixed node labels: {node_labels}")
    
    return adj_list, feats_list, label_list

def train_rqgnn_on_graphs(args, graph_dataset_train, graph_dataset_test, train_ids, test_ids):
    # hyperparameters from codebase
    lr = 5e-3
    batch_size = 512
    nepoch = getattr(args, "epochs", None) or 100
    hdim = 64
    width = 6
    depth = 6
    dropout = 0.4
    normalize= 1
    beta = 0.999
    gamma = 1.5
    decay = 0
    patience = 50
    nclass = 2
    device = 0

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
    
    train_dataset = AgentGraphDataset(train_dataset, phase="train")
    val_dataset = AgentGraphDataset(val_dataset, phase="val")

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    
    ny_0 = train_labels.count(0)
    ny_1  = train_labels.count(1)

    example = train_dataset[0]
    featuredim = example.x.size(1)

    gad = model.GADGNN(featuredim, hdim, nclass, width, depth, dropout, normalize)
    optimizer = optim.Adam(gad.parameters(), lr=lr, weight_decay=decay)

    patiencecount = 0

    print("Starts training...")
    for epoch in range(nepoch):
        gad.train()
        
        # get our graphs into RQGNN format
        adj_train, feats_train, label_train = extract_rqgnn_inputs(train_dataset)
        batchsize = batch_size
        
        train_batches = utils.generate_batches(adj_train, feats_train, label_train, batchsize, True)
        epoch_loss = 0

        for train_batch in train_batches:
            optimizer.zero_grad()
            outputs = gad(train_batch)
            loss = benchmark_AD.methods.rqgnn.lossfunc.CB_loss(train_batch.label_list, outputs, [ny_0, ny_1], nclass, beta, gamma)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print('Epoch: {}, loss: {}'.format(epoch, epoch_loss / len(train_batches)))

        # Evaluate the model
        gad.eval()

        # get our graphs into RQGNN format
        adj_val, feats_val, label_val = extract_rqgnn_inputs(val_dataset)
        batchsize = batch_size
        
        val_batches = utils.generate_batches(adj_val, feats_val, label_val, batchsize, False)
        truths = torch.Tensor()
        scores = torch.Tensor()

        for i, val_batch in enumerate(val_batches):
            outputs = gad(val_batch)
            outputs = nn.functional.softmax(outputs, dim=1)
            scores = torch.cat((scores, outputs[:, 1].detach().cpu()), dim=0)
            truths = torch.cat((truths, val_batch.label_list.detach().cpu()), dim=0)
        
        # get preds
        preds_binary = torch.as_tensor(predict_scores(args, scores.numpy(), truths.numpy(), probability=True))

        f1 = f1_score(truths, preds_binary)
        acc = accuracy_score(truths, preds_binary)
        auc = roc_auc_score(truths, scores)
        bal_acc = balanced_accuracy_score(truths, preds_binary)
        
        # print metrics 
        print("F1:", f1)
        print("Accuracy:", acc)
        print("AUC:", auc)
        print("Balanced Accuracy:", bal_acc)
        
        log_results(args, epoch, nepoch, f1, acc, auc, bal_acc, model=gad, gt_labels=truths.numpy(), pred_labels=preds_binary.numpy(), pred_scores=scores.numpy(), train_ids=train_ids, test_ids=test_ids)

def run_rqgnn(args, train_df, test_df):
    print("Running RQGNN")

    train_ids = train_df["row_id"].values
    test_ids = test_df["row_id"].values
    
    # convert traces to g_safeguard format
    converted_train = convert_lomas_traces_escape_room_v2(args, train_df)  # same function as in g_safeguard
    converted_test = convert_lomas_traces_escape_room_v2(args, test_df)  # same function as in g_safeguard

    # create g_safeguard graphs
    graph_dataset_train = gen_training_dataset(args, converted_train)
    graph_dataset_test = gen_training_dataset(args, converted_test)

    train_rqgnn_on_graphs(args, graph_dataset_train, graph_dataset_test, train_ids, test_ids)