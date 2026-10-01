import os 
import json
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def gen_model_training_set(language_dataset, embedding_model): 
    dataset = []
    for meta_data in tqdm(language_dataset, desc="Generate training data"): 
        adj_matrix = meta_data["adj_matrix"]
        attacker_idxes = meta_data["attacker_idxes"]
        system_prompts = meta_data["system_prompts"]
        communication_data = meta_data["communication_data"]

        adj_matrix_np = np.array(adj_matrix)
        labels = np.array([1 if i in attacker_idxes else 0 for i in range(len(adj_matrix))])

        num_nodes = len(labels)
        if num_nodes == 1:
            print(f"Warning: Graph with only 1 node found! attacker_idxes={attacker_idxes}, adj_matrix={adj_matrix_np}")

        system_prompts_embedding = []
        for i in range(len(system_prompts)): 
            system_prompts_embedding.append(embedding_model.encode(system_prompts[i]))
        system_prompts_embedding = np.array(system_prompts_embedding)

        # edge_embedding
        edge_index = adj_matrix_np.nonzero()
        edge_index = np.array(edge_index)
        communication_embeddings = [[] for _ in range(len(adj_matrix))]
        for i in range(len(communication_data)):
            turn_i_data = communication_data[i]
            turn_i_embeddings = [None] * len(turn_i_data)
            for agent_idx, c_data in turn_i_data: 
                i_turns_agent_idx_embedding = embedding_model.encode(c_data)
                turn_i_embeddings[agent_idx] = i_turns_agent_idx_embedding
            for agent_idx in range(len(turn_i_embeddings)): 
                communication_embeddings[agent_idx].append(turn_i_embeddings[agent_idx])
        
        communication_embeddings = np.array(communication_embeddings)
        # edge attr = the source node's own output, i.e. what actually gets passed to the destination as its input
        edge_attr = np.array(communication_embeddings[edge_index[0]], copy=True)
        
        data = {}
        data["adj_matrix"] = adj_matrix_np
        data["features"] = system_prompts_embedding
        data["labels"] = labels    
        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["attacker_idxes"] = attacker_idxes
        
        dataset.append(data)
    return dataset


def gen_training_dataset(args, traces):
    embedding_model_dir = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(embedding_model_dir)

    dataset = gen_model_training_set(traces, embedding_model)

    return dataset

