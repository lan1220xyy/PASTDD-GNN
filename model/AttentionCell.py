import torch
import torch.nn as nn
import copy
import math


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
    


class AttentionToEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_k = d_model
        self.linears = clones(nn.Linear(d_model, d_model), 2)

    def forward(self, query, key, node_embedding):
        # X:[B,N,D]
        query, key = [
            lin(x)
            for lin, x in zip(self.linears, (query, key))
        ]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        p_attn = scores.softmax(dim=-1)
        return torch.matmul(p_attn, node_embedding)
