
import torch
import math

def time_sampled_masking(adj: torch.Tensor, masking_ratio: float, ratio_highest_time, eps: float = 1e-6):
    """
    adj Tensor of shape [B, T, N, N]
    highest_ratio: how many top time steps to consider as important
    ratio: ratio to sample from top_k (0 < ratio <= 1)
    eps: small constant to avoid division by zero
    """
    B, T, N, _ = adj.shape
    k = int(math.ceil(ratio_highest_time * T))
    change_scores = torch.zeros(B, T, device=adj.device)  # [B, T]

    for t in range(T):
        A_curr = adj[:, t]
        left_score = torch.zeros(B, device=adj.device)
        right_score = torch.zeros(B, device=adj.device)

        if t > 0:
            A_prev = adj[:, t - 1]
            left_diff = torch.abs((A_curr - A_prev) / (A_prev + eps))
            left_score = left_diff.sum(dim=(1, 2))  # or .pow(2).sum(...).sqrt() for Frobenius

        if t < T - 1:
            A_next = adj[:, t + 1]
            right_diff = torch.abs((A_next - A_curr) / (A_curr + eps))
            right_score = right_diff.sum(dim=(1, 2))

        if t == 0:
            change_scores[:, t] = right_score
        elif t == T - 1:
            change_scores[:, t] = left_score
        else:
            change_scores[:, t] = (left_score + right_score) / 2

    topk_scores, topk_indices = torch.topk(change_scores, k, dim=1)  # [B, top_k]

    num_select = max(1, int(math.ceil(masking_ratio * T)))
    sampled_indices = []
    for b in range(B):
        idx = torch.randperm(k, device=adj.device)[:num_select]
        sampled = topk_indices[b][idx]
        sampled_indices.append(sampled)

    # shape: [B, num_select], each row = selected important time indices
    sampled_indices = torch.stack(sampled_indices, dim=0)
    return sampled_indices


def adj_sampled_masking(masking_ratio, ratio_highest_weights, adj, device):
    """
    Args:
        masking_ratio: 掩码比例
        ratio_highest_weights: 选择高weights节点的比例
        adj: [N, N] adjacency矩阵

    Returns:
        masked_indices: [num_masked] 被掩码的节点索引
    """
    # attn: [N, N]
    N, _ = adj.shape

    # 计算节点重要性
    instance_weights = torch.sum(adj, dim=-2) - torch.diagonal(adj, dim1=-2, dim2=-1)  # [N]

    # 选择高权重节点（k = ratio_highest_attention * N）
    k = int(math.ceil(ratio_highest_weights * N))
    _, topk_indices = torch.topk(instance_weights, k=k, dim=-1)  # [k]

    # 随机选择掩码位置（选 masking_ratio * N 个位置）
    num_masked = int(math.ceil(masking_ratio * N))
    rand_perm = torch.randperm(k, device=device)  # [k]
    sampled_indices_in_topk = rand_perm[:num_masked]  # [num_masked]

    # 从 topk_indices 中按采样索引获取真正的 masked 节点索引
    masked_indices = topk_indices[sampled_indices_in_topk]  # [num_masked]

    return masked_indices  # 返回的是 GPU Tensor: [num_masked]


def combined_time_node_masking(adj: torch.Tensor,
                               time_masking_ratio: float,
                               ratio_highest_time: float,
                               node_masking_ratio: float,
                               ratio_highest_weights: float,
                               eps: float = 1e-6):
    """
    对[B,T,N,N]的adj tensor进行时间-节点联合掩码

    Args:
        adj: [B, T, N, N] attention tensor
        time_masking_ratio: 时间掩码比例
        ratio_highest_time: 选择多少比例的top时间步
        node_masking_ratio: 节点掩码比例
        ratio_highest_weights: 选择多少比例的top节点
        eps: 防止除零的小常数

    Returns:
        time_masked_indices: [B, num_time_masked] 被掩码的时间步索引
        node_masked_indices: [B, num_time_masked, num_node_masked] 在每个被掩码时间步中被掩码的节点索引
    """

    # 使用time_sampled_masking选择重要时间步
    time_masked_indices = time_sampled_masking(
        adj=adj,
        masking_ratio=time_masking_ratio,
        ratio_highest_time=ratio_highest_time,
        eps=eps
    )  # [B, num_time_masked]

    B, num_time_masked = time_masked_indices.shape
    _, T, N, _ = adj.shape

    # 第二步：在每个被掩码的时间步中选择重要节点进行掩码
    node_masked_indices_list = []

    for b in range(B):
        batch_node_indices = []
        for i in range(num_time_masked):
            t_idx = time_masked_indices[b, i].item()  # 当前被掩码的时间步

            # 提取该时间步的attention矩阵 [N, N]
            attn_at_t = adj[b, t_idx]  # [N, N]

            # 使用单样本attention掩码函数
            masked_nodes = adj_sampled_masking(
                masking_ratio=node_masking_ratio,
                ratio_highest_weights=ratio_highest_weights,
                adj=attn_at_t,
                device=adj.device
            )  # [num_node_masked]

            batch_node_indices.append(masked_nodes)  # [num_node_masked]

        # 将该batch的所有时间步的节点索引stack起来
        batch_node_indices = torch.stack(batch_node_indices, dim=0)  # [num_time_masked, num_node_masked]
        node_masked_indices_list.append(batch_node_indices)

    # 将所有batch的结果stack起来
    node_masked_indices = torch.stack(node_masked_indices_list, dim=0)  # [B, num_time_masked, num_node_masked]

    return time_masked_indices, node_masked_indices


def mask_input_data(X: torch.Tensor,
                    time_masked_indices: torch.Tensor,
                    node_masked_indices: torch.Tensor) -> torch.Tensor:
    """
    使用已生成的掩码索引对输入数据X[B,T,N,1]进行置零掩码

    Args:
        X: [B, T, N, 1] 输入数据
        time_masked_indices: [B, num_time_masked] 时间掩码索引
        node_masked_indices: [B, num_time_masked, num_node_masked] 节点掩码索引

    Returns:
        masked_X: [B, T, N, 1] 掩码后的输入数据
    """
    masked_X = X.clone()
    B, num_time_masked, num_node_masked = node_masked_indices.shape

    for b in range(B):
        for i in range(num_time_masked):
            t_idx = time_masked_indices[b, i].item()
            node_indices = node_masked_indices[b, i]  # [num_node_masked]

            # 置零掩码
            masked_X[b, t_idx, node_indices, :] = 0.0

    return masked_X


def create_boolean_mask(X: torch.Tensor,
                        time_masked_indices: torch.Tensor,
                        node_masked_indices: torch.Tensor) -> torch.Tensor:
    """
    为输入数据X创建[B,T,N,1]的布尔掩码张量

    Args:
        X: [B, T, N, 1] 输入数据
        time_masked_indices: [B, num_time_masked] 时间掩码索引
        node_masked_indices: [B, num_time_masked, num_node_masked] 节点掩码索引

    Returns:
        mask: [B, T, N, 1] 布尔掩码张量，True表示被掩码的位置
    """
    B, T, N, _ = X.shape
    mask = torch.zeros(B, T, N, 1, dtype=torch.bool, device=X.device)

    num_time_masked, num_node_masked = time_masked_indices.shape[1], node_masked_indices.shape[2]

    for b in range(B):
        for i in range(num_time_masked):
            t_idx = time_masked_indices[b, i].item()
            node_indices = node_masked_indices[b, i]  # [num_node_masked]

            # 标记被掩码的时间步和节点
            mask[b, t_idx, node_indices, :] = True

    return mask

if __name__ == '__main__':
    B, T, N = 2, 10, 10
    X = torch.rand(B, T, N, 1) * 10  # 放大数值便于观察
    attn = torch.rand(B, T, N, N)
    print(X)
    print(attn)
    time,node = combined_time_node_masking(attn,0.25,0.5,0.25, 0.5)
    print(time)
    print(node)
    masked_x = mask_input_data(X,time,node)
    print(masked_x)