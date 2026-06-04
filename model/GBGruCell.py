import torch
import torch.nn as nn
from model.GBGCN import GBGCN

class GBGruCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(GBGruCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = GBGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = GBGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)
        # self.time_aware = nn.Sequential(nn.Linear(embed_dim, dim_out),nn.Sigmoid())
        self.time_aware = nn.Sequential(nn.Linear(embed_dim, dim_out),nn.ReLU())
        # self.time_aware = nn.Sequential(nn.Linear(embed_dim, dim_out))

    def forward(self, x, state, time_embeddings, node_embeddings, static_node_embedings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        # time_embeddings: B, d
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, static_node_embedings))  # B N IN ->  B N OUT
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, static_node_embedings))
        if time_embeddings == None:
            h = r * state + (1 - r) * hc
            print('没有time_aware')
        else:
            # time = self.time_aware(time_embeddings).unsqueeze(1).repeat(1, self.node_num, 1)
            time = self.time_aware(time_embeddings)
            h = r * state * time + (1 - r) * hc * (1 - time)  # B N IN  B N IN
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)