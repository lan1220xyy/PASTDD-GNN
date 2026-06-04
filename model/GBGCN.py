import torch
import torch.nn.functional as F
import torch.nn as nn

class GBGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(GBGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings, static_node_embedings):
        # x shaped[B, N, C], node_embeddings shaped [B, N, D] -> supports shaped [B, N, N]
        # output shape [B, N, C]
        node_num = x.shape[-2]
        batch_size = x.shape[0]
        if node_embeddings.dim() == 2:
            node_embeddings = node_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, N, D)
        supports = F.softmax(F.relu(torch.bmm(node_embeddings, node_embeddings.transpose(-2, -1))), dim=-1)
        support_set = [torch.eye(node_num).to(supports.device).unsqueeze(0).repeat(batch_size, 1, 1), supports]
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', static_node_embedings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(static_node_embedings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("kbnm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out

        return x_gconv