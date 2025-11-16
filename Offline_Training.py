import time
import utils
import MvSF2Token
import random as pyrandom  # 确保使用的是 Python 内置的 random 模块
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from VASGhormer_and_layers import VASGhormer
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data
from utils import *
from Online_Search import epoch_evaluate  

if __name__ == "__main__":

    args = parse_args()
    print(args)
    pyrandom.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)

    node_class = torch.load(f'./dataset/{args.dataset}/{args.dataset}_node_class.pt')   # [N,3]
    home_adj = torch.load(f'./dataset/{args.dataset}/{args.dataset}_con_adj.pt')       # [N,N] 稀疏
    adjs = torch.load(f'./dataset/{args.dataset}/{args.dataset}_adjs.pt')               # [M,N,N] 稀疏
    G_global = nx.from_numpy_array(home_adj.to_dense().numpy())

    samples_path = f'./dataset/{args.dataset}/{args.dataset}_samples.pt'

    if not os.path.exists(samples_path):

        num_nodes = node_class.shape[0]
        sample_num = args.sample_num
        hops = args.hops
        M = adjs.shape[0]

        node_labels = node_class.argmax(dim=1).tolist()
        all_nodes = set(range(num_nodes))

        G_mps = []
        for m in range(M):
            G_m = nx.from_numpy_array(adjs[m].to_dense().numpy())
            G_mps.append(G_m)

        global_samples = torch.zeros((num_nodes, 2, sample_num), dtype=torch.long)

        mp_samples = torch.zeros((num_nodes, M, 2, sample_num), dtype=torch.long)

        def sample_for_graph(G, i, node_label):
            neighbors = set(G.neighbors(i))

            # ------- 正样本 -------
            positive_candidates = [n for n in neighbors if node_labels[n] == node_label and n != i]

            if len(positive_candidates) < sample_num:
                pos_samples = pyrandom.choices(positive_candidates or [i], k=sample_num)
            else:
                pos_samples = pyrandom.sample(positive_candidates, sample_num)

            # ------- 负样本 -------
            hops_nodes = set(nx.single_source_shortest_path_length(G, i, cutoff=hops).keys())
            negative_candidates = list((all_nodes - hops_nodes) | 
                                    {n for n in all_nodes if node_labels[n] != node_label})
            negative_candidates = [n for n in negative_candidates if n != i]

            if len(negative_candidates) < sample_num:
                neg_samples = pyrandom.choices(negative_candidates or [i], k=sample_num)
            else:
                neg_samples = pyrandom.sample(negative_candidates, sample_num)

            return pos_samples, neg_samples


        for i in tqdm(range(num_nodes), desc="Sampling on all graphs"):

            node_label = node_labels[i]

            pos_g, neg_g = sample_for_graph(G_global, i, node_label)
            global_samples[i, 0] = torch.tensor(pos_g)
            global_samples[i, 1] = torch.tensor(neg_g)

            for m in range(M):
                pos_m, neg_m = sample_for_graph(G_mps[m], i, node_label)
                mp_samples[i, m, 0] = torch.tensor(pos_m)
                mp_samples[i, m, 1] = torch.tensor(neg_m)

        torch.save({
            "global_samples": global_samples,
            "mp_samples": mp_samples
        }, samples_path)

    else:
        data = torch.load(samples_path)
        global_samples = data["global_samples"]
        mp_samples = data["mp_samples"]




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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
                    optimizer,
                    warmup_updates=args.warmup_updates,
                    tot_updates=args.tot_updates,
                    lr=args.peak_lr,
                    end_lr=args.end_lr,
                    power=1.0)
    print("starting training...")
    # t_start = time.time()

    # Initialize variables for early stopping and saving the best model
    best_f1 = 0.0  # Stores the highest F1 score
    no_improve_count = 0  # Tracks the number of epochs without improvement
    best_model_path = f'{args.save_path}/{args.model_name}_best.pth'  # Path to save the best model
    query_node=torch.load(f'./dataset/{args.dataset}/query_node.pt') 
    true_community = torch.load(f'./dataset/{args.dataset}/{args.dataset}_true_community_of_query.pt')
    test_query_node=query_node[:200]
    test_true_community = true_community.to_dense()[:200]
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        all_community_emb = []
        all_class_prediction = []
        for index, item in enumerate(data_loader):
            nodes_features = item.to(args.device)

            optimizer.zero_grad()
            community_emb, class_prediction, loss_train = model.trainModel(
                nodes_features, args.metapath_num, node_class, global_samples, mp_samples, args
            )
            epoch_loss += loss_train.item()
            loss_train.backward()
            optimizer.step()
            lr_scheduler.step()
            all_community_emb.append(community_emb)
            all_class_prediction.append(class_prediction)
        all_community_emb = torch.cat(all_community_emb, dim=0)
        all_class_prediction = torch.cat(all_class_prediction, dim=0)

        # Evaluate the model
        model.eval()
        result_f1 = epoch_evaluate(
            all_community_emb, all_class_prediction, test_query_node, test_true_community, G_global, args
        )
        print(
            'Epoch: {:03d}  '.format(epoch + 1),
            'loss_train: {:.4f}  '.format(epoch_loss),
            "F1: {:.4f} ".format(result_f1),
        )

        # Check if the current F1 score is the best
        if result_f1 > best_f1:
            best_f1 = result_f1
            no_improve_count = 0  # Reset the no improvement counter
            torch.save(model.state_dict(), best_model_path)  # Save the best model
            print(f"Saved the best model with F1 score: {best_f1:.4f}")
        else:
            no_improve_count += 1  # Increment the no improvement counter

        # Stop training if no improvement for 10 consecutive epochs
        if no_improve_count >= 10:
            print("No improvement for 10 consecutive epochs. Stopping training.")
            break

    print("Training finished. Best F1 score: {:.4f}".format(best_f1))


    print("Finish offline training process!")



