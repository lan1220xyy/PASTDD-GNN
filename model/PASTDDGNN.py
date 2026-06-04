import torch
import torch.nn as nn

from model.GBGruCell import GBGruCell
from model.AttentionCell import AttentionToEmbedding
class Graph_Based_GRU(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(Graph_Based_GRU, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.gbgru_cells = nn.ModuleList()
        self.gbgru_cells.append(GBGruCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.gbgru_cells.append(GBGruCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, periodic_embeddings, node_embeddings, static_node_embedings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        # shape of periodic_embeddings: (B, T, D)
        assert x.shape[2] == self.node_num
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.gbgru_cells[i](current_inputs[:, t, :, :], state, periodic_embeddings[:, t, :],
                                                node_embeddings,
                                                static_node_embedings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.gbgru_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)
    
class PASTDDGNN(nn.Module):
    def __init__(self, args, period, unique_periods):
        super(PASTDDGNN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.lag = args.lag
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.rnn_units = args.rnn_units
        self.cheb_k = args.cheb_k
        self.embed_dim = args.embed_dim
        self.embed_d_model = args.embed_d_model
        self.lamb = args.lamb

        self.unique_periods = unique_periods
        self.min_period = min(self.unique_periods)
        self.period = period.long()  # [N,3]
        self.k = args.k

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)
        self.period2idx = {p: i for i, p in enumerate(self.unique_periods)}

        self.period_embeddings = nn.ModuleList([])
        self.is_decomposed = [] 

        for p in self.unique_periods:
            if p > self.min_period and p % self.min_period == 0:
                self.period_embeddings.append(nn.Embedding(p // self.min_period, self.embed_dim))
                self.is_decomposed.append(True)
            else:
                self.period_embeddings.append(nn.Embedding(p, self.embed_dim))
                self.is_decomposed.append(False)

        node_period_idx = torch.zeros_like(self.period)
        for p, idx in self.period2idx.items():
            node_period_idx[self.period == p] = idx
        self.register_buffer("node_period_idx", node_period_idx)

        self.periodic_mlp = nn.Sequential(nn.Linear(self.embed_dim, self.embed_d_model),nn.ReLU(),nn.Linear(self.embed_d_model, self.embed_d_model))
        self.data_mlp = nn.Sequential(nn.Linear(1, self.embed_d_model),nn.ReLU(),nn.Linear(self.embed_d_model, self.embed_d_model))
        self.trend_embed = nn.Sequential(nn.Conv2d(self.lag, 1, (1, 1)), nn.Linear(self.embed_d_model, self.embed_d_model))
        self.trend_cap = AttentionToEmbedding(self.embed_d_model)

        self.encoder = Graph_Based_GRU(self.num_node, self.input_dim, self.rnn_units, self.cheb_k, self.embed_dim,
                                    self.num_layers)
        # predictor
        self.end_conv = nn.Conv2d(self.input_dim, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim),
                                  bias=True)

    def forward(self, source, targets):
        node_embeddings = self.node_embeddings  # [N,d]

        time_id = source[:, :, -1, self.input_dim:].long() 
        B, T = time_id.shape[0], time_id.shape[1]

        # ===== global period position =====
        periods = torch.tensor(self.unique_periods, device=time_id.device)  # [P]
        periodic_idx = time_id % periods  # [B, T, P]
        periodic_idx_expand = periodic_idx.unsqueeze(2).expand(B, T, self.num_node, periods.shape[0])
        index = self.node_period_idx.unsqueeze(0).unsqueeze(0).expand(B, T, self.num_node, self.k)

        # gather:[B,T,N,K] 
        node_periodic_idx = torch.gather(
            periodic_idx_expand,
            dim=-1,
            index=index
        )

        periodic_embeddings = torch.zeros(B, T, self.num_node, self.k, self.embed_dim, device=time_id.device)

        for p_idx, (p, emb_layer, decomposed) in enumerate(
                zip(self.unique_periods, self.period_embeddings, self.is_decomposed)):
            mask = (self.node_period_idx == p_idx)  # [N, K]
            if not mask.any():
                continue
            idx = node_periodic_idx[:, :, mask]  # [B, T, num_selected]

            if decomposed:
                base_embedding_layer = self.period_embeddings[0]  # p0's embedding
                pos_base = idx % self.min_period  
                segment_idx = idx // self.min_period  
                base_emb = base_embedding_layer(pos_base) 
                scale_emb = emb_layer(segment_idx)  
                emb = base_emb + scale_emb  
            else:
                emb = emb_layer(idx)

            periodic_embeddings[:, :, mask] = emb

        period_emb = periodic_embeddings.sum(dim=3)

        input_data = source[:, :, :, :self.input_dim]
        all_data = self.data_mlp(input_data)
        periodic_data = self.periodic_mlp(period_emb)
        dyc_data= all_data - periodic_data
        
        dyc = self.trend_embed(dyc_data).squeeze(1)
        dyc_node_embeddings = self.trend_cap(dyc, dyc, self.node_embeddings)
        node_embeddings = node_embeddings + period_emb[:, -1, :, :] + self.lamb*dyc_node_embeddings

        init_state = self.encoder.init_hidden(input_data.shape[0])
        output, _ = self.encoder(input_data, init_state, period_emb, node_embeddings, self.node_embeddings)  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv(output)  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C

        return output
    