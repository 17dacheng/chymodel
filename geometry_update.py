"""
增强的几何特征处理模块 - 基于GearBind优化
集成多头注意力机制和更丰富的几何特征处理
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, MessagePassing
import numpy as np
from torch_cluster import knn
from dataclasses import dataclass
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean


class EnhancedDDGAttention(nn.Module):
    """
    增强版DDGAttention - 基于GearBind但优化性能
    专门用于蛋白质界面几何特征的注意力机制
    """
    
    def __init__(self, input_dim, output_dim, value_dim=32, query_key_dim=32, num_heads=16):
        super(EnhancedDDGAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_heads = num_heads

        # 增强的投影层
        self.query = nn.Linear(input_dim, query_key_dim*num_heads, bias=False)
        self.key   = nn.Linear(input_dim, query_key_dim*num_heads, bias=False)
        self.value = nn.Linear(input_dim, value_dim*num_heads, bias=False)

        # 空间位置编码
        self.spatial_encoding = nn.Sequential(
            nn.Linear(3, query_key_dim),
            nn.ReLU(),
            nn.Linear(query_key_dim, query_key_dim)
        )

        # 增强的输出变换
        self.out_transform = nn.Sequential(
            nn.Linear(
                in_features = (num_heads*value_dim) + (num_heads*(3+3+1+3)),  # 增加空间编码
                out_features = output_dim * 2,  # 更大的中间层
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        # 层归一化和残差连接
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.1)

    def _alpha_from_logits(self, logits, mask, inf=1e5):
        """计算注意力权重"""
        N, L, _, _ = logits.size()
        mask_row = mask.view(N, L, 1, 1).expand_as(logits)
        mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)
        
        logits = torch.where(mask_pair, logits, logits-inf)
        alpha = torch.softmax(logits, dim=2)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha

    def _heads(self, x, n_heads, n_ch):
        """重塑为多头格式"""
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def forward(self, x, pos_CA, pos_CB, frame, mask):
        """
        增强版DDGAttention前向传播
        Args:
            x: (N, L, hidden_dim) 残基特征
            pos_CA: (N, L, 3) CA原子位置
            pos_CB: (N, L, 3) CB原子位置  
            frame: (N, L, 3, 3) 局部坐标系
            mask: (N, L) 残基掩码
        """
        # 增强的注意力计算
        query = self._heads(self.query(x), self.num_heads, self.query_key_dim)
        key = self._heads(self.key(x), self.num_heads, self.query_key_dim)
        
        # 添加空间位置信息到query和key
        spatial_feat = self.spatial_encoding(pos_CA)  # (N, L, query_key_dim)
        query = query + self._heads(spatial_feat, self.num_heads, self.query_key_dim)
        
        logits_node = torch.einsum('blhd, bkhd->blkh', query, key)
        alpha = self._alpha_from_logits(logits_node, mask)

        # 值聚合
        value = self._heads(self.value(x), self.num_heads, self.value_dim)
        feat_node = torch.einsum('blkh, bkhd->blhd', alpha, value).flatten(-2)
        
        # 增强的几何特征计算
        rel_pos = pos_CB.unsqueeze(1) - pos_CA.unsqueeze(2)  # (N, L, L, 3)
        atom_pos_bias = torch.einsum('blkh, blkd->blhd', alpha, rel_pos)
        
        # 更多几何特征
        feat_distance = atom_pos_bias.norm(dim=-1, keepdim=True)
        feat_points = torch.einsum('blij, blhj->blhi', frame, atom_pos_bias)
        feat_direction = feat_points / (feat_points.norm(dim=-1, keepdim=True) + 1e-10)
        
        # 添加角度特征
        angle_features = torch.atan2(feat_points[..., 1], feat_points[..., 0]).unsqueeze(-1)
        
        feat_spatial = torch.cat([
            feat_points.flatten(-2),      # (N, L, n_heads * 3)
            feat_distance.flatten(-2),    # (N, L, n_heads * 1)  
            feat_direction.flatten(-2),  # (N, L, n_heads * 3)
            angle_features.flatten(-2),   # (N, L, n_heads * 1)
        ], dim=-1)

        # 组合所有特征
        feat_all = torch.cat([feat_node, feat_spatial], dim=-1)
        feat_all = self.out_transform(feat_all)
        
        # 应用掩码
        feat_all = torch.where(mask.unsqueeze(-1), feat_all, torch.zeros_like(feat_all))
        
        # 残差连接和归一化
        if x.shape[-1] == feat_all.shape[-1]:
            feat_all = self.layer_norm(x + self.dropout(feat_all))
        else:
            feat_all = self.layer_norm(feat_all)

        return feat_all


@dataclass
class InterfaceGraphData:
    """界面图数据容器"""
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    edge_types: torch.Tensor
    node_positions: torch.Tensor
    batch: torch.Tensor
    atom_names: list
    is_mutation: torch.Tensor
    residue_indices: list
    
    def to(self, device):
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
    """高效的最近邻搜索"""
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


class GeometricMessagePassing(MessagePassing):
    """
    增强的几何消息传递层
    集成距离、角度等几何信息
    """
    
    def __init__(self, hidden_dim, num_heads=8):
        super(GeometricMessagePassing, self).__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 边特征编码
        self.edge_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim),  # 3D距离 + 3D方向 + 边类型
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 消息函数
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # source + target + edge
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 更新函数
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # current + aggregated
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 注意力权重
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, x, pos, edge_index, edge_attr, edge_types):
        """增强的消息传递"""
        # 计算几何边特征
        row, col = edge_index
        distance_vec = pos[row] - pos[col]
        distance = distance_vec.norm(dim=-1, keepdim=True)
        direction = distance_vec / (distance + 1e-8)
        edge_type_onehot = F.one_hot(edge_types, num_classes=3).float()
        
        geometric_edge = torch.cat([distance, direction, edge_type_onehot], dim=-1)
        edge_features = self.edge_encoder(geometric_edge)
        
        # 消息传递
        return self.propagate(edge_index, x=x, edge_features=edge_features)
    
    def message(self, x_j, x_i, edge_features):
        """计算消息"""
        message_input = torch.cat([x_j, x_i, edge_features], dim=-1)
        return self.message_net(message_input)
    
    def update(self, aggr_out, x):
        """更新节点特征"""
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_net(update_input)


class EnhancedResidueGeometry(nn.Module):
    """增强的残基几何处理器"""
    
    def __init__(self, hidden_dim, num_heads=16):
        super(EnhancedResidueGeometry, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 多层DDGAttention
        self.attention_layers = nn.ModuleList([
            EnhancedDDGAttention(hidden_dim, hidden_dim, num_heads=num_heads)
            for _ in range(3)  # 3层注意力
        ])
        
        # 前馈网络
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(0.1)
            ) for _ in range(3)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(6)
        ])
        
    def aggregate_atoms_to_residues(self, graph_data):
        """原子到残基聚合"""
        device = graph_data.node_features.device
        
        # 创建原子到残基映射
        atom2residue = torch.tensor(graph_data.residue_indices, dtype=torch.long, device=device)
        unique_residues = torch.unique(atom2residue)
        
        # 简化的残基特征聚合
        residue_features = []
        pos_CA_list = []
        pos_CB_list = []
        frames_list = []
        
        for res_id in unique_residues:
            atom_mask = atom2residue == res_id
            if atom_mask.any():
                # 聚合特征
                res_feat = graph_data.node_features[atom_mask].mean(dim=0)
                residue_features.append(res_feat)
                
                # 简化的位置信息（使用第一个CA原子）
                ca_mask = atom_mask & (torch.tensor([name == 'CA' for name in graph_data.atom_names], device=device))
                if ca_mask.any():
                    pos_CA_list.append(graph_data.node_positions[ca_mask][0])
                else:
                    pos_CA_list.append(graph_data.node_positions[atom_mask][0])
                
                # 伪CB位置（用于兼容）
                pos_CB_list.append(graph_data.node_positions[atom_mask][0] + torch.tensor([0.0, 1.0, 0.0], device=device))
                
                # 简化的坐标系
                frames_list.append(torch.eye(3, device=device))
        
        if not residue_features:
            # 返回空的默认值
            return (
                torch.zeros(1, 1, self.hidden_dim, device=device),
                torch.zeros(1, 1, 3, device=device),
                torch.zeros(1, 1, 3, device=device),
                torch.zeros(1, 1, dtype=torch.long, device=device),
                torch.zeros(1, 1, 3, 3, device=device)
            )
        
        residue_features = torch.stack(residue_features)
        pos_CA = torch.stack(pos_CA_list)
        pos_CB = torch.stack(pos_CB_list)
        frames = torch.stack(frames_list)
        
        # 创建残基到原子的映射
        new_atom2residue = torch.arange(len(unique_residues), device=device).repeat_interleave(
            torch.tensor([torch.sum(atom2residue == res_id).item() for res_id in unique_residues], device=device)
        )
        
        return residue_features, pos_CA, pos_CB, new_atom2residue, frames
    
    def apply_positional_attention(self, residue_features, pos_CA, pos_CB, frames, mask):
        """应用位置注意力"""
        x = residue_features
        
        # 多层注意力+前馈网络
        for i in range(len(self.attention_layers)):
            # 注意力层
            attn_out = self.attention_layers[i](x, pos_CA, pos_CB, frames, mask)
            x = self.layer_norms[2*i](x + attn_out)
            
            # 前馈网络
            ffn_out = self.ffn_layers[i](x)
            x = self.layer_norms[2*i + 1](x + ffn_out)
        
        return x


class EnhancedGeometricGNN(nn.Module):
    """
    增强的几何GNN - 融合GearBind的核心优势
    """
    
    def __init__(self, node_feat_dim=128, edge_feat_dim=128, hidden_dim=256, num_heads=8):
        super(EnhancedGeometricGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 多层几何消息传递
        self.geom_layers = nn.ModuleList([
            GeometricMessagePassing(hidden_dim, num_heads)
            for _ in range(3)
        ])
        
        # 残基级几何处理
        self.residue_geometry = EnhancedResidueGeometry(hidden_dim, num_heads)
        
        # 全局特征聚合
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 输出投影 - 更强的表达能力
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256),  # 增强到256维
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256)  # 最终256维几何特征输出
        )
    
    def forward(self, graph_data: InterfaceGraphData) -> torch.Tensor:
        """增强版前向传播"""
        # 输入投影
        x = self.node_proj(graph_data.node_features)
        edge_attr = self.edge_proj(graph_data.edge_features)
        
        # 多层几何消息传递
        for geom_layer in self.geom_layers:
            x = geom_layer(x, graph_data.node_positions, graph_data.edge_index, 
                          edge_attr, graph_data.edge_types)
        
        # 残基级几何处理
        residue_features, pos_CA, pos_CB, atom2residue, frames = self.residue_geometry.aggregate_atoms_to_residues(
            graph_data
        )
        
        if residue_features.shape[0] > 0:
            # 重组为batch格式
            batch_size = int(graph_data.batch.max().item() + 1)
            if batch_size == 1:
                residue_features_batch = residue_features.unsqueeze(0)
                pos_CA_batch = pos_CA.unsqueeze(0)
                pos_CB_batch = pos_CB.unsqueeze(0)
                frames_batch = frames.unsqueeze(0)
                mask = torch.ones(1, residue_features.shape[0], dtype=torch.bool, device=x.device)
                
                # 应用增强的位置注意力
                enhanced_residue = self.residue_geometry.apply_positional_attention(
                    residue_features_batch, pos_CA_batch, pos_CB_batch, frames_batch, mask
                )
                
                # 将增强特征映射回原子级
                enhanced_expanded = torch.zeros_like(x)
                for i, res_idx in enumerate(torch.unique(atom2residue)):
                    if res_idx < enhanced_residue.shape[1]:
                        enhanced_feat = enhanced_residue[0, i]
                        atom_mask = atom2residue == res_idx
                        if atom_mask.any():
                            enhanced_expanded[atom_mask] = enhanced_feat * 0.2  # 适度贡献
                
                x = x + enhanced_expanded
        
        # 全局池化
        if graph_data.batch.dtype != torch.int64:
            graph_data.batch = graph_data.batch.long()
        
        graph_rep = global_mean_pool(x, graph_data.batch)
        
        # 输出投影
        output = self.output_proj(graph_rep)
        
        return output