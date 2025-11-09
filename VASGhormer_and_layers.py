
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GCNConv

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)



class NodeClassificationLayer(nn.Module):
    def __init__(self, input_dim, class_num):
        super(NodeClassificationLayer, self).__init__()
        self.fc = nn.Linear(input_dim, class_num) 
        self.softmax = nn.Softmax(dim=-1) 

    def forward(self, node_emb):
        logits = self.fc(node_emb)  
        probabilities = self.softmax(logits) 
        return probabilities 


class SemanticAttention(nn.Module):
    def __init__(self, input_dim):
        super(SemanticAttention, self).__init__()
        self.attention_vector = nn.Parameter(torch.randn(input_dim))

    def forward(self, low_level_emb , high_level_emb):

        weights = torch.matmul(torch.cat((low_level_emb, high_level_emb), dim=-1), self.attention_vector) 

        weights = F.softmax(weights, dim=1)

        low_level_agg = torch.sum(low_level_emb * weights.unsqueeze(-1), dim=1)
        high_level_agg = torch.sum(high_level_emb * weights.unsqueeze(-1), dim=1)
        return low_level_agg, high_level_agg 


class Zoom_aware_Transformer(nn.Module):
    def __init__(
        self,
        hops, 
        input_dim,
        n_layers=6,
        num_heads=8,
        hidden_dim=64,
        dropout_rate=0.0,
        attention_dropout_rate=0.1
    ):
        super().__init__()

        self.seq_len = hops+1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = 2 * hidden_dim
        self.num_heads = num_heads
        
        self.n_layers = n_layers

        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)
        self.hop_num = hops

        encoders = [EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads, self.hop_num)
                    for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

   

        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))

        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)

        self.scaling = nn.Parameter(torch.ones(1) * 0.5)


        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data): 

        # print(batched_data.shape)
        tensor = self.att_embeddings_nope(batched_data) 

        for enc_layer in self.layers:
            tensor = enc_layer(tensor)
        
        output = self.final_ln(tensor)

        # print(output.shape)
        target = output[:,0,:].unsqueeze(1).repeat(1,self.seq_len-1,1)
        split_tensor = torch.split(output, [1, self.seq_len-1], dim=1)

        node_tensor = split_tensor[0] 
        neighbor_tensor = split_tensor[1] 
    
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2)) 
        
        layer_atten = F.softmax(layer_atten, dim=1) 
    
        neighbor_tensor = neighbor_tensor * layer_atten 
        
        return node_tensor, neighbor_tensor 
    



class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class Zoom_aware_MSA(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, hop_num):
        super(Zoom_aware_MSA, self).__init__()

        self.num_heads = num_heads
        self.hop_num = hop_num

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

        self.zoom_bias = nn.Parameter(torch.randn(2 * hop_num + 1, 1))

    def forward(self, q, k, v, attn_bias=None): 
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        zoom_encoding = torch.zeros(self.hop_num+1, self.hop_num+1)

        for i in range(self.hop_num+1):
            for j in range(self.hop_num+1):
                zoom_encoding[i, j] = self.zoom_bias[i - j + self.hop_num]  


        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k) 
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v) 

        q = q.transpose(1, 2)               
        v = v.transpose(1, 2)                 
        k = k.transpose(1, 2).transpose(2, 3)  

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k)+Zoom)V
        q = q * self.scale  
        x = torch.matmul(q, k) +zoom_encoding
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3) 
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn] 

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, hop_num):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = Zoom_aware_MSA(
            hidden_size, attention_dropout_rate, num_heads, hop_num)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class VASGhormer(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.config = config

        self.Linear1 = nn.Linear(input_dim, self.config.hidden_dim)
        self.encoder = Zoom_aware_Transformer(hops=config.hops, 
                        input_dim=input_dim, 
                        n_layers=config.n_layers,
                        num_heads=config.n_heads,
                        hidden_dim=config.hidden_dim,
                        dropout_rate=config.dropout,
                        attention_dropout_rate=config.attention_dropout)
        if config.readout == "sum":
            self.readout = global_add_pool
        elif config.readout == "mean":
            self.readout = global_mean_pool
        elif config.readout == "max":
            self.readout = global_max_pool
        else:
            raise ValueError("Invalid pooling type.")
        
        
        self.SemanticAttention= SemanticAttention(config.hidden_dim*2)
        self.classification_layer = NodeClassificationLayer(config.hidden_dim, config.class_num)

        self.marginloss = nn.MarginRankingLoss(0.5)
        self.classificationloss = nn.CrossEntropyLoss()

    #Offline Training
    def trainModel(self, x , adj_, minus_adj , metapath_num, class_label): 
        shuf_index = torch.randperm(x.shape[0])
        sem_con_loss = 0.0
        glo_con_loss = 0.0
        cls_loss = 0.0
        low_level_emb=[]
        high_level_emb=[]

        for i in range(0, metapath_num):
            node_tensor, neighbor_tensor = self.encoder(x[:,i,:,:])
            neighbor_tensor = self.readout(neighbor_tensor, torch.tensor([0]).to(self.config.device)) 
            node_tensor=node_tensor.squeeze() 
            neighbor_tensor=neighbor_tensor.squeeze() 
            node_tensor_shuf = node_tensor[shuf_index] 
            neighbor_tensor_shuf = neighbor_tensor[shuf_index] 
            logits_aa = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor, dim = -1)) 
            logits_bb = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor_shuf, dim = -1))
            logits_ab = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor_shuf, dim = -1))
            logits_ba = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor, dim = -1))
            # ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
            ones = torch.ones(logits_aa.size(0)).cpu() 
            sem_con_loss += self.marginloss(logits_aa, logits_ba, ones) 
            sem_con_loss += self.marginloss(logits_bb, logits_ab, ones)
            low_level_emb.append(node_tensor)
            high_level_emb.append(neighbor_tensor)
        low_level_emb = torch.stack(low_level_emb, dim=1) 
        high_level_emb = torch.stack(high_level_emb, dim=1) 
        node_emb, community_emb = self.SemanticAttention(low_level_emb, high_level_emb) 
        node_tensor_shuf = node_emb[shuf_index]  
        neighbor_tensor_shuf = community_emb[shuf_index] 
        logits_aa = torch.sigmoid(torch.sum(node_emb * community_emb, dim = -1)) 
        logits_bb = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor_shuf, dim = -1))
        logits_ab = torch.sigmoid(torch.sum(node_emb * neighbor_tensor_shuf, dim = -1))
        logits_ba = torch.sigmoid(torch.sum(node_tensor_shuf * community_emb, dim = -1))
        # ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        ones = torch.ones(logits_aa.size(0)).cpu() 
        glo_con_loss += self.marginloss(logits_aa, logits_ba, ones) 
        glo_con_loss += self.marginloss(logits_bb, logits_ab, ones)
        con_loss= 0.7*sem_con_loss + glo_con_loss
        
        class_prediction = self.classification_layer(node_emb) 
        cls_loss=self.classificationloss(class_prediction, class_label)
        
        total_loss=con_loss + cls_loss
        return community_emb, class_prediction, total_loss


    #propagation of VASGhormer
    def forward(self, x,  metapath_num): 
        shuf_index = torch.randperm(x.shape[0])
        TotalLoss = 0.0
        low_level_emb=[]
        high_level_emb=[]
        for i in range(0, metapath_num):
            node_tensor, neighbor_tensor = self.encoder(x[:,i,:,:]) 
            neighbor_tensor = self.readout(neighbor_tensor, torch.tensor([0]).to(self.config.device))
            node_tensor=node_tensor.squeeze() 
            neighbor_tensor=neighbor_tensor.squeeze()
            
            low_level_emb.append(node_tensor)
            high_level_emb.append(neighbor_tensor)
        low_level_emb = torch.stack(low_level_emb, dim=1) 
        high_level_emb = torch.stack(high_level_emb, dim=1) 
        node_emb, community_emb = self.SemanticAttention(low_level_emb, high_level_emb) 
        class_prediction = self.classification_layer(node_emb)
        return community_emb, class_prediction

       

