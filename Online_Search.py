
import time
import utils
import MvSF2Token
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from VASGhormer_and_layers import VASGhormer
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data
from utils import *


def epoch_evaluate(community_embedding, class_prediction, query, labels ,G,args): 

    node_num_in_community = labels.sum(dim=1).int().tolist() 
    # start = time.time()
    query_feature = community_embedding[query] 
    query_score = cosin_similarity(query_feature, community_embedding) 

    y_pred = torch.zeros_like(query_score)
    print("evaluating...")
    for i in tqdm(range(query_score.shape[0])): 
        selected_candidates = McCommunitySearch([query[i].tolist()],class_prediction, query_score[i].tolist(),G)  
        for j in range(len(selected_candidates)):
            y_pred[i][selected_candidates[j]] = 1
        
    # end = time.time()
    # print("The search using time: {:.4f}".format(end-start)) 
    f1_score = f1_score_calculation(y_pred.int(), labels.int())
    return  f1_score

def subgraph_density_controled(candidate_score, graph_score): 
    
    weight_gain = (sum(candidate_score)-sum(graph_score)*(len(candidate_score)**1)/(len(graph_score)**1))/(len(candidate_score)**0.68)
    return weight_gain


# Multi-Constrained Community Search
def McCommunitySearch(query_index, class_prediction, graph_score , G): 
    filtered_nodes = [node for node in range(0,len(graph_score)) if graph_score[node] > 0.5]
    
    top_values = torch.topk(class_prediction[query_index], k=2).values  
    max_value, second_max_value = top_values[0][0], top_values[0][1] 
    if max_value - second_max_value > 0.3:
        query_class = torch.argmax(class_prediction[query_index], dim=1) 
        filtered_nodes = [
            node for node in filtered_nodes
            if torch.argmax(class_prediction[node]) == query_class
        ]

    high_score_subgraph = G.subgraph(filtered_nodes).copy()

    if query_index[0] in high_score_subgraph:
        connected_subgraph = nx.node_connected_component(high_score_subgraph, query_index[0])
        result_nodes = list(connected_subgraph)
    else:
        result_nodes = query_index

    return result_nodes




if __name__ == "__main__":
    
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)

    node_class = torch.load(f'./dataset/{args.dataset}/{args.dataset}_node_class.pt') 
    query_node=torch.load(f'./dataset/{args.dataset}/query_node.pt')
    true_community = torch.load(f'./dataset/{args.dataset}/{args.dataset}_true_community_of_query.pt')
    home_adj = torch.load(f'./dataset/IMDB/imdb_con_adj.pt') 
    G = nx.from_numpy_array(home_adj.to_dense().numpy())
    test_query_node=query_node[200:600]
    test_true_community = true_community.to_dense()[200:600]

    MvSF_path = f'./dataset/{args.dataset}/{args.dataset}_MvSFs.pt'
    if not os.path.exists(MvSF_path):
        adjs = torch.load(f'./dataset/{args.dataset}/{args.dataset}_adjs.pt') 
        features = torch.load(f'./dataset/{args.dataset}/{args.dataset}_node_features.pt')  
        processed_features=[]
        for i in range(0,args.metapath_num):
            processed_features.append(MvSF2Token.multi_view_FC(adjs[i].to_dense().float(), features[i].to_dense(), args.hops, 0.2)) 
        processed_features = torch.stack(processed_features, dim=1) 
        torch.save(processed_features, f'./dataset/{args.dataset}/{args.dataset}_MvSFs.pt') 
    else:
        processed_features = torch.load(MvSF_path)
    
    data_loader = Data.DataLoader(processed_features, batch_size=args.batch_size, shuffle = False)

    model = VASGhormer(input_dim=processed_features.shape[3], config=args).to(args.device)
    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))
    model.load_state_dict(torch.load(args.save_path + args.model_name + '.pth'))


    all_community_emb=[]
    all_class_prediction = []
    for index, item in enumerate(data_loader):
        nodes_features = item.to(args.device) 
        community_emb, class_prediction = model.forward(nodes_features, args.metapath_num) 
        all_community_emb.append(community_emb)
        all_class_prediction.append(class_prediction)

    all_community_emb = torch.cat(all_community_emb, dim=0) 
    all_class_prediction = torch.cat(all_class_prediction, dim=0)

    result_f1= epoch_evaluate(all_community_emb, all_class_prediction, test_query_node,test_true_community, G,args)
    print( "F1: {:.4f} ".format(result_f1))
    print("Finished Online Search!")


