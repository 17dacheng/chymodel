"""
几何特征提取和处理模块 - 重构版本
合并相关类，简化架构
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, MessagePassing
import numpy as np
from torch_cluster import knn
from dataclasses import dataclass


class DDGAttention(nn.Module):
    """
    基于 GearBind 的 DDGAttention 实现
    用于残基级别的几何特征注意力机制
    """
    
    def __init__(self, input_dim, output_dim, value_dim=16, query_key_dim=16, num_heads=12):
        super(DDGAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_heads = num_heads

        self.query = nn.Linear(input_dim, query_key_dim*num_heads, bias=False)
        self.key   = nn.Linear(input_dim, query_key_dim*num_heads, bias=False)
        self.value = nn.Linear(input_dim, value_dim*num_heads, bias=False)

        self.out_transform = nn.Linear(
            in_features = (num_heads*value_dim) + (num_heads*(3+3+1)),
            out_features = output_dim,
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def _alpha_from_logits(self, logits, mask, inf=1e5):
        """
        Args:
            logits: Logit matrices, (N, L_i, L_j, num_heads).
            mask:   Masks, (N, L).
        Returns:
            alpha:  Attention weights.
        """
        N, L, _, _ = logits.size()
        mask_row = mask.view(N, L, 1, 1).expand_as(logits)      # (N, L, *, *)
        mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)     # (N, L, L, *)
        
        logits = torch.where(mask_pair, logits, logits-inf)
        alpha = torch.softmax(logits, dim=2)  # (N, L, L, num_heads)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha

    def _heads(self, x, n_heads, n_ch):
        """
        Args:
            x:  (..., num_heads * num_channels)
        Returns:
            (..., num_heads, num_channels)
        """
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def forward(self, x, pos_CA, pos_CB, frame, mask):
        """
        DDGAttention前向传播
        Args:
            x: (N, L, hidden_dim) 残基特征
            pos_CA: (N, L, 3) CA原子位置
            pos_CB: (N, L, 3) CB原子位置  
            frame: (N, L, 3, 3) 局部坐标系
            mask: (N, L) 残基掩码
        """
        # Attention logits
        query = self._heads(self.query(x), self.num_heads, self.query_key_dim)    # (N, L, n_heads, head_size)
        key = self._heads(self.key(x), self.num_heads, self.query_key_dim)      # (N, L, n_heads, head_size)
        logits_node = torch.einsum('blhd, bkhd->blkh', query, key)
        alpha = self._alpha_from_logits(logits_node, mask)  # (N, L, L, n_heads)

        value = self._heads(self.value(x), self.num_heads, self.value_dim)  # (N, L, n_heads, head_size)
        feat_node = torch.einsum('blkh, bkhd->blhd', alpha, value).flatten(-2)
        
        # 几何特征计算（基于 GearBind）
        rel_pos = pos_CB.unsqueeze(1) - pos_CA.unsqueeze(2)  # (N, L, L, 3)
        atom_pos_bias = torch.einsum('blkh, blkd->blhd', alpha, rel_pos)  # (N, L, n_heads, 3)
        feat_distance = atom_pos_bias.norm(dim=-1, keepdim=True)
        feat_points = torch.einsum('blij, blhj->blhi', frame, atom_pos_bias)  # (N, L, n_heads, 3)
        feat_direction = feat_points / (feat_points.norm(dim=-1, keepdim=True) + 1e-10)
        
        feat_spatial = torch.cat([
            feat_points.flatten(-2),      # (N, L, n_heads * 3)
            feat_distance.flatten(-2),    # (N, L, n_heads * 1)  
            feat_direction.flatten(-2),    # (N, L, n_heads * 3)
        ], dim=-1)

        feat_all = torch.cat([feat_node, feat_spatial], dim=-1)

        feat_all = self.out_transform(feat_all)  # (N, L, F)
        feat_all = torch.where(mask.unsqueeze(-1), feat_all, torch.zeros_like(feat_all))
        
        if x.shape[-1] == feat_all.shape[-1]:
            # 使用clone()避免梯度计算问题
            x_clone = x.clone()
            x_updated = self.layer_norm(x_clone + feat_all)
        else:
            x_updated = self.layer_norm(feat_all)

        return x_updated


@dataclass
class InterfaceGraphData:
    """界面图数据容器"""
    # 节点特征
    node_features: torch.Tensor  # [num_nodes, node_feat_dim]
    # 边索引
    edge_index: torch.Tensor  # [2, num_edges]
    # 边特征
    edge_features: torch.Tensor  # [num_edges, edge_feat_dim]
    # 边类型（多关系）
    edge_types: torch.Tensor  # [num_edges] 0: covalent, 1: spatial_knn, 2: spatial_radius
    # 节点位置（几何信息）
    node_positions: torch.Tensor  # [num_nodes, 3]
    # 批量索引
    batch: torch.Tensor  # [num_nodes]
    # 原子名称列表
    atom_names: list  # [num_nodes] 每个节点对应的原子名称
    # 突变位点掩码
    is_mutation: torch.Tensor  # [num_nodes] 标记哪些节点是突变位点
    # 残基索引列表（用于原子到残基的映射）
    residue_indices: list  # [num_nodes] 每个原子对应的残基标识符
    
    def to(self, device):
        """将所有Tensor属性移动到指定设备"""
        return InterfaceGraphData(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.long().to(device),
            edge_features=self.edge_features.to(device),
            edge_types=self.edge_types.long().to(device),
            node_positions=self.node_positions.to(device),
            batch=self.batch.long().to(device),
            atom_names=self.atom_names,
            is_mutation=self.is_mutation.to(device),
            residue_indices=self.residue_indices
        )


def nearest(query, key, query2graph, key2graph):
    """
    Find the nearest key for each query.
    """
    device = query.device
    num_query_graphs = query2graph.max().item() + 1
    
    nearest_indices = []
    for graph_id in range(num_query_graphs):
        query_mask = query2graph == graph_id
        key_mask = key2graph == graph_id
        
        if not query_mask.any() or not key_mask.any():
            continue
            
        query_points = query[query_mask]
        key_points = key[key_mask]
        
        distances = torch.cdist(query_points, key_points)
        _, local_nearest = distances.min(dim=1)
        
        key_indices = torch.where(key_mask)[0]
        global_nearest = key_indices[local_nearest]
        nearest_indices.append(global_nearest)
    
    if nearest_indices:
        result = torch.cat(nearest_indices)
    else:
        result = torch.tensor([], dtype=torch.long, device=device)
    
    return result

# 简化版本的几何GNN
class SimplifiedGeometricGNN(nn.Module):
    """
    简化的几何GNN
    统一了 GeometricMessagePassing 和 GeometricGNN 的功能
    """
    
    def __init__(self, node_feat_dim=128, edge_feat_dim=128, hidden_dim=128, num_heads=4):
        super(SimplifiedGeometricGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 特征投影
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 统一的几何处理器
        self.geometric_processor = UnifiedResidueGeometry(hidden_dim, num_heads)
        
        # 消息传递层
        self.msg_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.update_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 输出投影 - 增强几何特征表达能力  
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128)  # 提升到128维几何特征输出
        )
    
    def forward(self, graph_data: InterfaceGraphData) -> torch.Tensor:
        """简化的前向传播"""
        # 初始特征投影
        x = self.node_proj(graph_data.node_features)
        edge_attr = self.edge_proj(graph_data.edge_features)
        
        # 简化的消息传递
        row, col = graph_data.edge_index
        
        # 消息计算
        msg_input = torch.cat([x[row], x[col] + edge_attr], dim=-1)
        messages = self.msg_proj(msg_input)
        
        # 聚合消息（简单求和）
        aggr_out = torch.zeros_like(x)
        aggr_out.index_add_(0, row, messages)
        aggr_out.index_add_(0, col, messages)
        
        # 更新节点特征
        update_input = torch.cat([x, aggr_out], dim=-1)
        updated_x = self.update_proj(update_input)
        
        # 残基级几何处理（如果有足够的数据）
        if len(graph_data.residue_indices) > 0:
            residue_features, pos_CA, pos_CB, atom2residue, frames = self.geometric_processor.aggregate_atoms_to_residues(
                InterfaceGraphData(
                    node_features=updated_x,
                    edge_index=graph_data.edge_index,
                    edge_features=graph_data.edge_features,
                    edge_types=graph_data.edge_types,
                    node_positions=graph_data.node_positions,
                    batch=graph_data.batch,
                    atom_names=graph_data.atom_names,
                    is_mutation=graph_data.is_mutation,
                    residue_indices=graph_data.residue_indices
                )
            )
            
            if residue_features.shape[0] > 0:
                # 重新组织为batch格式
                batch_size = int(graph_data.batch.max().item() + 1)
                if batch_size == 1:
                    residue_features_batch = residue_features.unsqueeze(0)
                    pos_CA_batch = pos_CA.unsqueeze(0)
                    pos_CB_batch = pos_CB.unsqueeze(0)
                    frames_batch = frames.unsqueeze(0)
                    mask = torch.ones(1, residue_features.shape[0], dtype=torch.bool, device=updated_x.device)
                    
                    # 应用基于 GearBind DDGAttention 的位置感知注意力
                    enhanced_residue = self.geometric_processor.apply_positional_attention(
                        residue_features_batch, pos_CA_batch, pos_CB_batch, frames_batch, mask
                    )
                    
                    # 将增强特征映射回原子级（避免原地操作）
                    enhanced_expanded = torch.zeros_like(updated_x)
                    for i, res_idx in enumerate(torch.unique(atom2residue)):
                        if res_idx < enhanced_residue.shape[1]:
                            enhanced_feat = enhanced_residue[0, i]
                            atom_mask = atom2residue == res_idx
                            if atom_mask.any():
                                enhanced_expanded[atom_mask] = enhanced_feat * 0.1
                    
                    # 使用加法而不是原地操作
                    updated_x = updated_x + enhanced_expanded
        
        # 全局池化
        if graph_data.batch.dtype != torch.int64:
            graph_data.batch = graph_data.batch.long()
        
        graph_rep = global_mean_pool(updated_x, graph_data.batch)
        
        # 输出投影
        output = self.output_proj(graph_rep)
        
        return output