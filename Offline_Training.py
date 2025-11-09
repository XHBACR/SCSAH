
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
from Online_Search import epoch_evaluate  

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
    test_query_node=query_node[:200]
    test_true_community = true_community.to_dense()[:200]

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
    
    # adj_batch, minus_adj_batch = [], []
    # for i in range(0,args.metapath_num):
    #     adj_, minus_adj_ = transform_sp_csr_to_coo(transform_coo_to_csr(adjs[i]), args.batch_size, features.shape[1])
    #     adj_batch.append(adj_)
    #     minus_adj_batch.append(minus_adj_)
    
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

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        all_community_emb = []
        all_class_prediction = []
        for index, item in enumerate(data_loader):
            nodes_features = item.to(args.device)
            adj_set, minus_adj_set = [], []

            optimizer.zero_grad()
            community_emb, class_prediction, loss_train = model.trainModel(
                nodes_features, adj_set, minus_adj_set, args.metapath_num, node_class
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
            all_community_emb, all_class_prediction, test_query_node, test_true_community, G, args
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
    


