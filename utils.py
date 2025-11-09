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


# Training settings
def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='imdb', help='dataset name')
    # parser.add_argument('--device', type=int, default=1, help='Device cuda id')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=888, 
                        help='Random seed.')
    parser.add_argument('--metapath_num', type=int, default=2, 
                        help='numbers of metapath.')
    # model parameters
    parser.add_argument('--class_num', type=int, default=3)
    parser.add_argument('--hops', type=int, default=4,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--pe_dim', type=int, default=5,
                        help='position embedding size')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=64,
                        help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')
    parser.add_argument('--readout', type=str, default="mean")
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help='the value the balance the loss.')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=4780,
                        help='Batch size')
    parser.add_argument('--group_epoch_gap', type=int, default=20,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--tot_updates',  type=int, default=1000,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001, 
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=50, 
                        help='Patience for early stopping')
    
    # model saving
    parser.add_argument('--save_path', type=str, default='./model/',
                        help='The path for the model to save')
    parser.add_argument('--model_name', type=str, default='imdb',
                        help='The name for the model to save')
    parser.add_argument('--embedding_path', type=str, default='./pretrain_result/',
                        help='The path for the embedding to save')
    
    parser.add_argument('--embedding_tensor_name', type=str, help='embedding tensor name')
    parser.add_argument('--EmbeddingPath', type=str, default='./pretrain_result/', help='embedding path')
    parser.add_argument('--topk', type=int, default=400, help='the number of nodes selected.')

    return parser.parse_args()



def re_features(adj, features, K): #[4780, 4780], [4780, 1232], 3
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])
    for i in range(features.shape[0]):
        nodes_features[i, 0, 0, :] = features[i] 
    x = features + torch.zeros_like(features) 
    for i in range(K): 
        x = torch.matmul(adj, x)
        for index in range(features.shape[0]):
            nodes_features[index, 0, i + 1, :] = x[index] 
    nodes_features = nodes_features.squeeze()
    return nodes_features

def f1_score_calculation(y_pred, y_true):  
    F1_scores = []
    for i in range(y_pred.shape[0]): 
        y_pred_i = y_pred[i].reshape(1, -1)
        y_true_i = y_true[i].reshape(1, -1)
        pre = torch.sum(torch.multiply(y_pred_i, y_true_i)) / (torch.sum(y_pred_i) + 1E-9)
        rec = torch.sum(torch.multiply(y_pred_i, y_true_i)) / (torch.sum(y_true_i) + 1E-9)
        F1 = 2 * pre * rec / (pre + rec + 1E-9)
        F1_scores.append(F1.item())  
    return sum(F1_scores) / len(F1_scores) 



def evaluation(comm_find, comm):
    
    comm_find = comm_find.reshape(-1)
    comm = comm.reshape(-1)
    
    return normalized_mutual_info_score(comm, comm_find), adjusted_rand_score(comm, comm_find), jaccard_score(comm, comm_find)



def NMI_score(comm_find, comm):

    score = normalized_mutual_info_score(comm, comm_find)
    #print("q, nmi:", score)
    return score

def ARI_score(comm_find, comm):

    score = adjusted_rand_score(comm, comm_find)
    #print("q, ari:", score)

    return score

def JAC_score(comm_find, comm):

    score = jaccard_score(comm, comm_find)
    #print("q, jac:", score)
    return score

def cosin_similarity(query_tensor, emb_tensor):
    similarity = torch.stack([torch.cosine_similarity(query_tensor[i].reshape(1, -1), emb_tensor, dim=1) for i in range(len(query_tensor))], 0)
    return similarity
    
def dot_similarity(query_tensor, emb_tensor):
    similarity = torch.mm(query_tensor, emb_tensor.t()) # (query_num, node_num)
    similarity = torch.nn.Softmax(dim=1)(similarity)
    return similarity


def transform_coo_to_csr(adj):
    row=adj._indices()[0] 
    col=adj._indices()[1]
    data=adj._values() 
    shape=adj.size()  
    adj=sp.csr_matrix((data, (row, col)), shape=shape)
    return adj 
	           
def transform_csr_to_coo(adj, size=None):
    adj = adj.tocoo()
    adj = torch.sparse.LongTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                              torch.LongTensor(adj.data.astype(np.int32)),
                              torch.Size([size, size]))
    return adj

def transform_sp_csr_to_coo(adj, batch_size, node_num): 
    # chunks
    node_index = [i for i in range(node_num)]
    divide_index = [node_index[i:i+batch_size] for i in range(0, len(node_index), batch_size)] 

    # adj of each chunks, in the format of sp_csr
    # print("start mini batch: adj of each chunks")
    adj_sp_csr = [adj[divide_index[i]][:, divide_index[i]] for i in range(len(divide_index))]
    # print("start mini batch: minus adj of each chunks")
    minus_adj_sp_csr = [sp.csr_matrix(torch.ones(item.shape))-item for item in adj_sp_csr]

    # adj_tensor_coo = [transform_csr_to_coo(item).to_dense() for item in adj_sp_csr]
    # minus_adj_tensor_coo = [transform_csr_to_coo(item).to_dense() for item in minus_adj_sp_csr]
    # print("start mini batch: back to torch coo adj")
    adj_tensor_coo = [transform_csr_to_coo(adj_sp_csr[i], len(divide_index[i])).to_dense() for i in range(len(divide_index))]
    # print("start mini batch: back to torch coo minus adj")
    minus_adj_tensor_coo = [transform_csr_to_coo(minus_adj_sp_csr[i], len(divide_index[i])).to_dense() for i in range(len(divide_index))]

    return adj_tensor_coo, minus_adj_tensor_coo 

