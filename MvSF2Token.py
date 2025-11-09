import argparse
import torch
import scipy.sparse as sp
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import dgl
from sklearn.metrics import f1_score
import scipy.sparse as sp
import numpy as np
import networkx as nx
from numpy import *
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_score
import time
import torch
import torch.nn.functional as F
import networkx as nx

def semantic_FC(features: torch.Tensor,
                                   target_indices: torch.Tensor,
                                   meta_adj: torch.Tensor,
                                   d_oth: int) -> torch.Tensor:
    """
    SVD-based Semantic Feature Extraction for meta-path P
    """
    device = features.device
    N, d = features.shape

    # Identify non-target indices
    all_idx = torch.arange(N, device=device)
    mask = torch.ones(N, dtype=torch.bool, device=device)
    mask[target_indices] = False
    oth_indices = all_idx[mask]  # [N_oth]

    X_oth = features[oth_indices]  # [N_oth, d]
    U, S, Vh = torch.linalg.svd(X_oth, full_matrices=False)  # U:[N_oth,N_oth], Vh:[d,d]
    # Project X_oth onto top-d_oth right singular vectors
    V = Vh.transpose(-2, -1) 
    W = V[:, :d_oth]          
    X_oth_reduced = X_oth @ W  

    X_tilde = torch.zeros(N, d_oth, device=device)
    X_tilde[oth_indices] = X_oth_reduced

    # For each target node, aggregate neighbors' reduced features
    N_t = target_indices.shape[0]
    d0 = d + d_oth
    X_p = torch.zeros(N_t, d0, device=device)

    for idx_i, v in enumerate(target_indices.tolist()):
        nbrs = torch.where(meta_adj[v] > 0)[0]
        nbrs = nbrs[mask[nbrs]]
        if nbrs.numel() > 0:
            avg_tilde = X_tilde[nbrs].mean(dim=0)
        else:
            avg_tilde = torch.zeros(d_oth, device=device)
        X_p[idx_i] = torch.cat([features[v], avg_tilde], dim=0)

    return X_p

def compute_local_clustering(adj: torch.Tensor) -> torch.Tensor:
    """
    Compute the local clustering coefficient for each node in the graph represented by adj.
    adj: [N, N] adjacency matrix (binary or weighted).
    Returns: coh vector of shape [N], where coh[i] is the clustering coefficient of node i.
    """
    # Convert to NetworkX graph for clustering
    G = nx.from_numpy_array(adj.cpu().numpy())
    coh_dict = nx.clustering(G)
    coh = torch.tensor([coh_dict[i] for i in range(adj.size(0))], device=adj.device)
    return coh


def multi_view_FC(adj: torch.Tensor, 
                                    features: torch.Tensor,
                                    K: int,
                                    theta: float) -> torch.Tensor:
    """
    Multi-view Feature Construction
    """
    N, d = features.shape
    device = features.device

    coh = compute_local_clustering(adj)  # [N]
    nodes_features = torch.zeros(N, K+1, d, device=device)
    # 0-hop view: self feature
    nodes_features[:, 0, :] = features 
    B = (adj > 0).float()
    # One-hot initial reach for each node
    prev_reach = torch.eye(N, device=device)

    # Iterate over hops
    for k in range(1, K+1):
        # Compute k-hop reachability
        reach = (prev_reach @ B) > 0
        prev_reach = reach.float()
        for i in range(N):
            idx = torch.where(reach[i])[0]  # nodes in k-hop view of i
            if idx.nelement() == 0:
                continue

            # Feature similarity sim(v, u)
            sims = F.cosine_similarity(
                features[i].unsqueeze(0).expand(len(idx), -1),
                features[idx],
                dim=1
            )  # [num_neighbors]
            # Structural cohesion coh(u)
            cohs = coh[idx]  # [num_neighbors]

            # Adaptive weights
            exp_sim = torch.exp(sims)
            exp_coh = torch.exp(cohs)
            w_s = exp_coh / (exp_sim + exp_coh)
            w_c = 1.0 - w_s

            # Joint score JS_k(u)
            js = w_s * sims + w_c * cohs
            retained = idx[js >= theta]
            if retained.nelement() == 0:
                agg = torch.zeros(d, device=device)
            else:
                agg = features[retained].sum(dim=0)
            nodes_features[i, k, :] = agg

    return nodes_features #[4780, 4, 1232]

