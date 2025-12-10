import numpy as np
import scipy.sparse as sp
import torch
import json
import os

def encode_onehot(labels):
    """
    Convert labels to one-hot encoding
    
    Args:
        labels: List of labels
        
    Returns:
        One-hot encoded labels
    """
    classes = sorted(set(labels))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data(data_dir='./Data/relation', valid_company_file='./valid_company.txt'):
    """
    Load dataset and build graph
    
    Args:
        data_dir: Directory containing relation data
        valid_company_file: Path to file with valid company tickers
        
    Returns:
        adj: Normalized adjacency matrix as torch.FloatTensor
    """
    print('Loading dataset and building graph...')
    
    # Check if files exist
    connection_file = os.path.join(data_dir, 'NYSE_connections.json')
    tic_wiki_file = os.path.join(data_dir, 'NYSE_wiki.csv')
    sel_path_file = os.path.join(data_dir, 'selected_wiki_connections.csv')
    
    if not os.path.exists(connection_file):
        raise FileNotFoundError(f"Connection file not found: {connection_file}")
    if not os.path.exists(tic_wiki_file):
        raise FileNotFoundError(f"Wiki file not found: {tic_wiki_file}")
    if not os.path.exists(sel_path_file):
        raise FileNotFoundError(f"Selected path file not found: {sel_path_file}")
    if not os.path.exists(valid_company_file):
        raise FileNotFoundError(f"Valid company file not found: {valid_company_file}")
    
    # Load valid companies
    with open(valid_company_file, 'r') as f:
        valid_company_list = [company.strip() for company in f.readlines()]
    
    COMPANY_NUM = len(valid_company_list)
    print(f"Found {COMPANY_NUM} valid companies")

    # Read ticker to wiki index mapping
    idx_labels = np.genfromtxt(tic_wiki_file, dtype=str, delimiter=',', skip_header=False)
    
    # Build mapping from wiki index to company position
    idx_map = {}
    for idx in idx_labels:
        if idx[1] != 'unknown' and idx[0] in valid_company_list:
            idx_map[idx[1]] = valid_company_list.index(idx[0])
            
    # Read selected paths/connections
    sel_paths = np.genfromtxt(sel_path_file, dtype=str, delimiter=' ', skip_header=False)
    sel_paths = set(sel_paths[:, 0])

    # Read connections
    with open(connection_file, 'r') as fin:
        connections = json.load(fin)
    
    # Extract edges based on selected paths
    edges_unordered = []
    for key1, conns in connections.items():
        for key2, paths in conns.items():
            if key1 in idx_map and key2 in idx_map:
                for p in paths:
                    path_key = '_'.join(p)
                    if path_key in sel_paths:
                        edges_unordered.append([key1, key2])
                        break  # One match is enough to create an edge
    
    if not edges_unordered:
        print("Warning: No edges found in the graph!")
        # Create empty adjacency matrix
        adj = sp.eye(COMPANY_NUM, dtype=np.float32)
    else:
        # Map wiki indices to company indices
        edges_unordered = np.array(edges_unordered)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
        
        # Create sparse adjacency matrix
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), 
                           shape=(COMPANY_NUM, COMPANY_NUM), 
                           dtype=np.float32)
    
    # Normalize adjacency matrix
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    
    print(f"Graph built with shape {adj.size()}")
    return adj

def normalize_adj(mx):
    """
    Row-normalize sparse matrix: symmetric normalization
    
    Args:
        mx: scipy sparse matrix
        
    Returns:
        Normalized matrix
    """
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    """
    Row-normalize sparse matrix
    
    Args:
        mx: scipy sparse matrix
        
    Returns:
        Normalized matrix
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    """
    Calculate accuracy of predictions
    
    Args:
        output: Model output logits
        labels: True labels
        
    Returns:
        Accuracy value
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def edge_index_from_adj(adj):
    """
    Convert adjacency matrix to edge index format
    
    Args:
        adj: Adjacency matrix (torch.Tensor)
        
    Returns:
        edge_index: Edge index (torch.LongTensor)
    """
    # Get indices of non-zero elements
    edge_index = adj.nonzero(as_tuple=True)
    return edge_index

if __name__ == '__main__':
    adj = load_data()
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Number of edges: {torch.nonzero(adj).shape[0]}")