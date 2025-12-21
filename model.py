import torch
import torch.nn as nn
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB import PDBParser
from torch_geometric.nn import global_mean_pool, MessagePassing
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any, Union
from dataclasses import dataclass
from feature_extractor import RealFeatureExtractor
from pdb import set_trace
# 导入AtomPositionGather相关模块
import sys
sys.path.append('/home/chengwang/code/chymodel/refer')

def nearest(query, key, query2graph, key2graph):
    """
    Find the nearest key for each query.
    
    Args:
        query: tensor of shape [N_query, d]
        key: tensor of shape [N_key, d] 
        query2graph: tensor of shape [N_query]
        key2graph: tensor of shape [N_key]
        
    Returns:
        indices of nearest keys for each query
    """
    device = query.device
    num_query_graphs = query2graph.max().item() + 1
    
    nearest_indices = []
    for graph_id in range(num_query_graphs):
        query_mask = query2graph == graph_id
        key_mask = key2graph == graph_id
        
        if not query_mask.any() or not key_mask.any():
            continue
            
        query_points = query[query_mask]  # [n_query_in_graph, d]
        key_points = key[key_mask]        # [n_key_in_graph, d]
        
        # Compute distances between all query-key pairs
        distances = torch.cdist(query_points, key_points)  # [n_query_in_graph, n_key_in_graph]
        
        # Find nearest key for each query
        _, local_nearest = distances.min(dim=1)  # [n_query_in_graph]
        
        # Map local indices back to global indices
        key_indices = torch.where(key_mask)[0]
        global_nearest = key_indices[local_nearest]
        nearest_indices.append(global_nearest)
    
    if nearest_indices:
        result = torch.cat(nearest_indices)
    else:
        result = torch.tensor([], dtype=torch.long, device=device)
    
    return result

class KNNMutationSite(nn.Module):
    """KNN突变位点选择模块"""
    
    def __init__(self, k=256):
        super(KNNMutationSite, self).__init__()
        self.k = k

    def forward(self, node_positions, atom_names, is_mutation, batch):
        """
        选择离突变点最近的k个节点
        
        Args:
            node_positions: [num_nodes, 3] 节点位置
            atom_names: list[str] 原子名称列表  
            is_mutation: [num_nodes] 突变位点掩码
            batch: [num_nodes] 批次索引
            
        Returns:
            node_mask: [num_nodes] 节点选择掩码
        """
        device = node_positions.device
        
        # 找到突变位点的CA原子
        mutation_mask = is_mutation.clone()
        ca_mask = torch.tensor([name == "CA" for name in atom_names], 
                              device=device, dtype=torch.bool)
        mutation_ca_mask = mutation_mask & ca_mask
        
        if not mutation_ca_mask.any():
            # 如果没有找到突变CA原子，返回所有节点
            return torch.ones(len(atom_names), dtype=torch.bool, device=device)
        
        # 获取突变CA原子位置
        center_positions = node_positions[mutation_ca_mask]
        mut2graph = batch[mutation_ca_mask]
        
        # 计算每个节点到中心位置的距离
        center_indices = nearest(node_positions, center_positions, batch, mut2graph)
        dist_to_center = ((node_positions - center_positions[center_indices])**2).sum(-1)
        dist_to_center[mutation_ca_mask] = 0.0  # 突变位点距离设为0
        
        # 为每个batch选择最近的k个节点
        num_graphs = batch.max().item() + 1
        selected_indices = []
        
        for graph_id in range(num_graphs):
            graph_mask = batch == graph_id
            graph_nodes = torch.where(graph_mask)[0]
            
            if len(graph_nodes) <= self.k:
                # 如果节点数不超过k，选择所有节点
                selected_indices.append(graph_nodes)
            else:
                # 选择距离最小的k个节点
                graph_distances = dist_to_center[graph_mask]
                _, local_selected = torch.topk(graph_distances, self.k, largest=False)
                global_selected = graph_nodes[local_selected]
                selected_indices.append(global_selected)
        
        # 创建节点掩码
        node_mask = torch.zeros(len(atom_names), dtype=torch.bool, device=device)
        if selected_indices:
            all_selected = torch.cat(selected_indices)
            node_mask[all_selected] = True
            
        return node_mask

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
            edge_index=self.edge_index.long().to(device),  # 确保int64
            edge_features=self.edge_features.to(device),
            edge_types=self.edge_types.long().to(device),  # 确保int64
            node_positions=self.node_positions.to(device),
            batch=self.batch.long().to(device),  # 确保int64
            atom_names=self.atom_names,  # 非Tensor属性保持不变
            is_mutation=self.is_mutation.to(device),
            residue_indices=self.residue_indices  # 非Tensor属性保持不变
        )


class GeometricGNN(nn.Module):
    """修复维度匹配的几何GNN - 压缩特征维度版本"""
    
    def __init__(self,
                 node_feat_dim: int = 64,   # 压缩到64维
                 edge_feat_dim: int = 64,   # 压缩到64维
                 hidden_dim: int = 64,      # 压缩到64维
                 num_relation_types: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super(GeometricGNN, self).__init__()
        
        # 修正输入维度匹配
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),  # 64 -> 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # 边特征投影修正为正确维度
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),  # 64 -> 64
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 多关系消息传递层
        self.relation_weights = nn.Parameter(torch.ones(num_relation_types, hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.relation_weights)
        
        # 几何感知的GNN层
        self.gnn_layers = nn.ModuleList([
            GeometricMessagePassing(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(3)
        ])
        
        # 几何注意力（类似GearBind的边级消息传递）
        self.geometric_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 全局几何信息提取
        self.global_geom_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # 包含坐标信息
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 突变位点特征提取
        self.mutation_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
    
    def forward(self, graph_data: InterfaceGraphData) -> torch.Tensor:
        """前向传播"""
        # 初始节点和边特征
        x = self.node_proj(graph_data.node_features)  # [num_nodes, hidden_dim]
        edge_attr = self.edge_proj(graph_data.edge_features)  # [num_edges, hidden_dim]
        
        # 多关系消息传递
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, graph_data.edge_index, edge_attr, graph_data.edge_types, graph_data)
            
            # 添加几何坐标信息到节点特征
            if i == 1:  # 在中间层注入几何信息
                geom_info = self.global_geom_proj(
                    torch.cat([x, graph_data.node_positions], dim=-1)
                )
                x = x + 0.1 * geom_info
        
        # 几何注意力（边级消息传递）
        batch_nodes = []
        for batch_idx in torch.unique(graph_data.batch):
            mask = graph_data.batch == batch_idx
            if mask.sum() > 0:
                batch_x = x[mask]
                attn_output, _ = self.geometric_attention(batch_x, batch_x, batch_x)
                batch_nodes.append(attn_output)
        
        if batch_nodes:
            x_attn = torch.cat(batch_nodes, dim=0)
            x = x + 0.5 * x_attn  # 残差连接
        
        # 突变位点注意力机制
        mutation_nodes = []
        for batch_idx in torch.unique(graph_data.batch):
            batch_mask = (graph_data.batch == batch_idx) & graph_data.is_mutation
            ca_mask = batch_mask & torch.tensor([atom_name == "CA" for atom_name in graph_data.atom_names], 
                                              device=x.device, dtype=torch.bool)
            
            if ca_mask.sum() > 0:
                # 提取突变位点的CA原子特征
                mutation_x = x[ca_mask].unsqueeze(0)  # [1, num_mutations, hidden_dim]
                # 对突变位点进行自注意力
                mutation_attn, _ = self.mutation_attention(mutation_x, mutation_x, mutation_x)
                mutation_nodes.append(mutation_attn.squeeze(0).mean(dim=0))  # 平均池化
        
        # 全局池化得到图级别表示
        # 确保batch索引是int64类型
        if graph_data.batch.dtype != torch.int64:
            graph_data.batch = graph_data.batch.long()
        
        graph_rep = global_mean_pool(x, graph_data.batch)  # [batch_size, hidden_dim]
        
        # 如果存在突变位点特征，将其整合到图表示中
        if mutation_nodes:
            mutation_rep = torch.stack(mutation_nodes, dim=0)  # [num_mutations, hidden_dim]
            
            # 获取实际batch size
            actual_batch_size = graph_rep.shape[0]
            
            # 如果mutation_rep的batch size不匹配，需要进行padding或截取
            if mutation_rep.shape[0] != actual_batch_size:
                # 创建匹配维度的张量
                aligned_rep = torch.zeros(actual_batch_size, mutation_rep.shape[1], 
                                         device=x.device, dtype=mutation_rep.dtype)
                
                # 将有效的mutation_rep复制到对应位置
                min_size = min(mutation_rep.shape[0], actual_batch_size)
                aligned_rep[:min_size] = mutation_rep[:min_size]
                
                # 使用对齐后的特征
                graph_rep = graph_rep + 0.3 * aligned_rep
            else:
                graph_rep = graph_rep + 0.3 * mutation_rep
        
        output = self.output_proj(graph_rep)  # [batch_size, hidden_dim//4]
        
        return output


class PositionAwareLayer(nn.Module):
    """残基级位置感知层 - 基于GearBind实现，处理残基而非原子"""
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(PositionAwareLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.query_key_dim = hidden_dim // num_heads
        self.value_dim = hidden_dim // num_heads
        
        # Attention projections
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Output transformation
        spatial_dim = num_heads * 7  # 3(points) + 1(distance) + 3(direction)
        self.out_transform = nn.Sequential(
            nn.Linear(hidden_dim + spatial_dim, hidden_dim),  # feat_node + feat_spatial
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def _heads(self, tensor, num_heads, head_dim):
        """将张量重塑为多头注意力格式"""
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, num_heads, head_dim)
    
    def _alpha_from_logits(self, logits, mask):
        """从logits计算注意力权重"""
        if mask is not None:
            # mask: [batch_size, seq_len]
            batch_size, seq_len, seq_len2, num_heads = logits.shape
            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)  # [batch_size, seq_len, seq_len]
            mask_expanded = mask_2d.unsqueeze(-1).expand(-1, -1, -1, num_heads)
            logits = logits.masked_fill(~mask_expanded, float('-inf'))
        
        alpha = torch.softmax(logits, dim=2)  # 在key维度上softmax
        return alpha
    
    def forward(self, x, pos_CA, pos_CB, frame, mask):
        # x: [batch_size, num_residues, hidden_dim] - 残基级特征
        # pos_CA: [batch_size, num_residues, 3] - 残基CA原子位置
        # pos_CB: [batch_size, num_residues, 3] - 残基CB原子位置
        # frame: [batch_size, num_residues, 3, 3] - 残基局部坐标系
        # mask: [batch_size, num_residues] - 残基掩码
        
        batch_size, num_residues, _ = x.shape
        
        # Attention logits
        query = self._heads(self.query(x), self.num_heads, self.query_key_dim)    # (N, L, n_heads, head_size)
        key = self._heads(self.key(x), self.num_heads, self.query_key_dim)      # (N, L, n_heads, head_size)
        logits_node = torch.einsum('blhd, bkhd->blkh', query, key)
        alpha = self._alpha_from_logits(logits_node, mask)  # (N, L, L, n_heads)

        value = self._heads(self.value(x), self.num_heads, self.value_dim)  # (N, L, n_heads, head_size)
        feat_node = torch.einsum('blkh, bkhd->blhd', alpha, value).flatten(-2)  # (N, L, hidden_dim)
        
        # 位置相关特征 - 现在基于残基位置
        rel_pos = pos_CB.unsqueeze(2) - pos_CA.unsqueeze(1)  # (N, L, L, 3)
        atom_pos_bias = torch.einsum('blkh, blkd->blhd', alpha, rel_pos)  # (N, L, n_heads, 3)
        feat_distance = atom_pos_bias.norm(dim=-1, keepdim=True)  # (N, L, n_heads, 1)
        feat_points = torch.einsum('blij, blhj->blhi', frame, atom_pos_bias)  # (N, L, n_heads, 3)
        feat_direction = feat_points / (feat_points.norm(dim=-1, keepdim=True) + 1e-10)
        
        # 确保所有张量维度一致
        feat_points_flat = feat_points.flatten(-2)  # (N, L, n_heads * 3)
        feat_distance_flat = feat_distance.flatten(-2)  # (N, L, n_heads * 1)
        feat_direction_flat = feat_direction.flatten(-2)  # (N, L, n_heads * 3)
        
        feat_spatial = torch.cat([
            feat_points_flat,
            feat_distance_flat,
            feat_direction_flat,
        ], dim=-1)  # (N, L, n_heads * 7)

        feat_all = torch.cat([feat_node, feat_spatial], dim=-1)  # (N, L, hidden_dim + n_heads * 7)

        feat_all = self.out_transform(feat_all)  # (N, L, hidden_dim)
        feat_all = torch.where(mask.unsqueeze(-1), feat_all, torch.zeros_like(feat_all))
        if x.shape[-1] == feat_all.shape[-1]:
            x_updated = self.layer_norm(x + feat_all)
        else:
            x_updated = self.layer_norm(feat_all)

        return x_updated


class AtomPositionGather(nn.Module):
    """原子到残基的聚合模块 - 类似GearBind的AtomPositionGather"""
    
    def __init__(self, hidden_dim):
        super(AtomPositionGather, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 原子名称到ID的映射
        self.atom_name2id = {
            'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6, 
            'CE': 7, 'CZ': 8, 'OG': 9, 'OD1': 10, 'OD2': 11, 'OE1': 12,
            'OE2': 13, 'ND1': 14, 'ND2': 15, 'NE1': 16, 'NE2': 17,
            'NZ': 18, 'SG': 19, 'SD': 20
        }
        self.num_atom_types = len(self.atom_name2id)
    
    def forward(self, graph_data):
        """
        将原子级特征聚合为残基级特征
        
        Args:
            graph_data: InterfaceGraphData，包含原子级信息
            
        Returns:
            residue_features: [num_residues, hidden_dim] 残基级特征
            residue_positions_CA: [num_residues, 3] CA原子位置
            residue_positions_CB: [num_residues, 3] CB原子位置
            residue_frames: [num_residues, 3, 3] 局部坐标系
            atom2residue: [num_atoms] 原子到残基的映射
            residue_mask: [num_residues] 残基完整性掩码
        """
        device = graph_data.node_features.device
        num_atoms = graph_data.node_features.shape[0]
        
        # 1. 构建原子到残基的映射
        atom2residue, residue_info = self._build_atom2residue_mapping(graph_data)
        num_residues = len(residue_info)
        
        if num_residues == 0:
            # 如果没有有效残基，返回空张量
            return (torch.empty(0, self.hidden_dim, device=device),
                   torch.empty(0, 3, device=device),
                   torch.empty(0, 3, device=device), 
                   torch.empty(0, 3, 3, device=device),
                   torch.empty(0, dtype=torch.long, device=device),
                   torch.empty(0, dtype=torch.bool, device=device))
        
        # 2. 验证残基完整性（包含N、CA、C原子）
        residue_mask = self._validate_residue_completeness(atom2residue, graph_data, num_residues)
        
        # 3. 聚合原子特征到残基特征（只使用CA原子特征作为残基代表）
        ca_mask = torch.tensor([name == "CA" for name in graph_data.atom_names], device=device)
        ca_indices = torch.where(ca_mask)[0]
        
        if len(ca_indices) == 0:
            # 如果没有CA原子，使用第一个原子作为代表
            residue_features = torch.zeros(num_residues, self.hidden_dim, device=device)
        else:
            ca_atom2residue = atom2residue[ca_mask]
            ca_features = graph_data.node_features[ca_mask]
            
            # 为每个残基分配CA原子特征
            residue_features = torch.zeros(num_residues, self.hidden_dim, device=device)
            for i in range(num_residues):
                mask = ca_atom2residue == i
                if mask.any():
                    residue_features[i] = ca_features[mask].mean(dim=0)
        
        # 4. 提取CA和CB原子位置
        residue_positions_CA, residue_positions_CB = self._extract_atom_positions(
            atom2residue, graph_data, num_residues
        )
        
        # 5. 构建局部坐标系
        residue_frames = self._build_residue_frames(residue_positions_CA, residue_positions_CB, num_residues)
        
        return (residue_features, residue_positions_CA, residue_positions_CB, 
                residue_frames, atom2residue, residue_mask)
    
    def _build_atom2residue_mapping(self, graph_data):
        """构建原子到残基的映射"""
        device = graph_data.node_features.device
        atom2residue = torch.zeros(len(graph_data.atom_names), dtype=torch.long, device=device)
        residue_info = {}
        current_residue_idx = 0
        
        for i, residue_idx in enumerate(graph_data.residue_indices):
            if residue_idx not in residue_info:
                residue_info[residue_idx] = current_residue_idx
                current_residue_idx += 1
            atom2residue[i] = residue_info[residue_idx]
        
        return atom2residue, residue_info
    
    def _validate_residue_completeness(self, atom2residue, graph_data, num_residues):
        """验证残基是否包含完整的N、CA、C原子"""
        device = atom2residue.device
        residue_mask = torch.zeros(num_residues, dtype=torch.bool, device=device)
        
        # 检查每个残基是否有N、CA、C原子
        for res_idx in range(num_residues):
            atom_mask = atom2residue == res_idx
            if atom_mask.sum() >= 3:  # 至少3个原子
                residue_atoms = [graph_data.atom_names[i] for i, mask in enumerate(atom_mask) if mask.item()]
                if "N" in residue_atoms and "CA" in residue_atoms and "C" in residue_atoms:
                    residue_mask[res_idx] = True
        
        return residue_mask
    
    def _extract_atom_positions(self, atom2residue, graph_data, num_residues):
        """提取CA和CB原子位置"""
        device = atom2residue.device
        residue_positions_CA = torch.zeros(num_residues, 3, device=device)
        residue_positions_CB = torch.zeros(num_residues, 3, device=device)
        
        # 提取CA原子位置
        ca_mask = torch.tensor([name == "CA" for name in graph_data.atom_names], device=device)
        ca_indices = torch.where(ca_mask)[0]
        for atom_idx in ca_indices:
            res_idx = atom2residue[atom_idx].item()
            if res_idx < num_residues:
                residue_positions_CA[res_idx] = graph_data.node_positions[atom_idx]
        
        # 提取CB原子位置（如果没有CB，使用CA位置）
        cb_mask = torch.tensor([name == "CB" for name in graph_data.atom_names], device=device)
        cb_indices = torch.where(cb_mask)[0]
        for atom_idx in cb_indices:
            res_idx = atom2residue[atom_idx].item()
            if res_idx < num_residues:
                residue_positions_CB[res_idx] = graph_data.node_positions[atom_idx]
        
        # 对于没有CB原子的残基，使用CA位置
        no_cb_mask = (residue_positions_CB.abs().sum(dim=1) < 1e-6)
        residue_positions_CB[no_cb_mask] = residue_positions_CA[no_cb_mask]
        
        return residue_positions_CA, residue_positions_CB
    
    def _build_residue_frames(self, pos_CA, pos_CB, num_residues):
        """构建残基的局部坐标系"""
        frames = torch.eye(3).unsqueeze(0).repeat(num_residues, 1, 1).to(pos_CA.device)
        
        # 简化的坐标系构建，使用CA和CB位置
        for i in range(num_residues):
            if i < len(pos_CA) - 1:
                # 使用CA-CB向量作为x轴
                e1 = pos_CB[i] - pos_CA[i]
                e1_norm = e1.norm()
                if e1_norm > 1e-6:
                    e1 = e1 / e1_norm
                
                # 构建正交基
                if e1_norm > 1e-6:
                    # 选择一个随机向量与e1叉积得到e2
                    temp_vec = torch.tensor([0, 0, 1], dtype=pos_CA.dtype, device=pos_CA.device)
                    e2 = torch.cross(e1, temp_vec)
                    e2_norm = e2.norm()
                    if e2_norm < 1e-6:  # 如果e1和temp_vec平行
                        temp_vec = torch.tensor([0, 1, 0], dtype=pos_CA.dtype, device=pos_CA.device)
                        e2 = torch.cross(e1, temp_vec)
                        e2_norm = e2.norm()
                    
                    if e2_norm > 1e-6:
                        e2 = e2 / e2_norm
                        e3 = torch.cross(e1, e2)
                        frames[i] = torch.stack([e1, e2, e3], dim=1)
        
        return frames


class GeometricMessagePassing(MessagePassing):
    """几何感知的消息传递层 - 原子级图卷积 + 残基级注意力"""
    
    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.1):
        super(GeometricMessagePassing, self).__init__()
        
        # 存储输出维度
        self.out_channels = out_channels
        
        # 确保输入输出维度匹配
        self.msg_proj = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),  # 128 -> 64
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.update_proj = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),  # 128 -> 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_channels)
        )
        
        # 原子到残基的聚合模块
        self.atom_gather = AtomPositionGather(out_channels)
        
        # 残基级位置感知层
        self.position_aware = PositionAwareLayer(
            hidden_dim=out_channels,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.attention = nn.MultiheadAttention(
            out_channels, num_heads, dropout=dropout, batch_first=True
        )
        
        # 添加num_heads属性以便在message方法中使用
        self.num_heads = num_heads
        
        # 初始化关系权重矩阵 [num_relation_types, hidden_dim, hidden_dim]
        self.relation_weights = nn.Parameter(torch.ones(3, self.out_channels, self.out_channels))
        nn.init.xavier_uniform_(self.relation_weights)
    
    def forward(self, x, edge_index, edge_attr, edge_types, graph_data=None):
        # 1. 原子级图卷积
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_types=edge_types)
        
        # 更新原子级节点特征
        update_input = torch.cat([x, aggr_out], dim=-1)
        updated_atoms = self.update_proj(update_input)  # [num_atoms, hidden_dim]
        
        # 2. 如果有图数据，应用残基级注意力
        if graph_data is not None:
            # 确保graph_data有residue_indices属性
            if not hasattr(graph_data, 'residue_indices'):
                # 如果没有残基索引信息，直接返回原子级特征
                return updated_atoms
            
        # 使用AtomPositionGather将原子级特征聚合为残基级特征
        (residue_features, residue_positions_CA, residue_positions_CB, 
         residue_frames, atom2residue, residue_mask) = self.atom_gather(graph_data)
        
        if residue_features.shape[0] == 0:
            # 如果没有有效残基，直接返回原子级特征
            return updated_atoms
        

        
        # 3. 残基级注意力处理
        # 重新组织数据以适应batch处理
        batch_size = int(graph_data.batch.max().item() + 1)
        
        # 将残基特征按batch分组
        residue_features_list = []
        residue_positions_CA_list = []
        residue_positions_CB_list = []
        residue_frames_list = []
        residue_masks_list = []
        
        for batch_idx in range(batch_size):
                # 找到属于当前batch的原子
                atom_mask = graph_data.batch == batch_idx
                
                if atom_mask.any():
                    # 获取这些原子对应的残基
                    atom2residue_batch = atom2residue[atom_mask]
                    unique_residues = torch.unique(atom2residue_batch)
                    
                    # 过滤完整残基
                    valid_residues = unique_residues[residue_mask[unique_residues]]
                    
                    if len(valid_residues) > 0:
                        # 提取残基级特征
                        batch_residue_features = residue_features[valid_residues]
                        batch_residue_positions_CA = residue_positions_CA[valid_residues]
                        batch_residue_positions_CB = residue_positions_CB[valid_residues]
                        batch_residue_frames = residue_frames[valid_residues]
                        batch_residue_mask = residue_mask[valid_residues]
                        
                        residue_features_list.append(batch_residue_features)
                        residue_positions_CA_list.append(batch_residue_positions_CA)
                        residue_positions_CB_list.append(batch_residue_positions_CB)
                        residue_frames_list.append(batch_residue_frames)
                        residue_masks_list.append(batch_residue_mask)
        
        if residue_features_list:
            # 简化处理：只处理第一个batch（对于单个样本的情况）
            if len(residue_features_list) == 1:
                residue_features_single = residue_features_list[0]
                pos_CA_single = residue_positions_CA_list[0]
                pos_CB_single = residue_positions_CB_list[0]
                frames_single = residue_frames_list[0]
                masks_single = residue_masks_list[0]
                
                # 调整为batch格式
                batch_features = residue_features_single.unsqueeze(0)  # [1, num_residues, hidden_dim]
                batch_pos_CA = pos_CA_single.unsqueeze(0)           # [1, num_residues, 3]
                batch_pos_CB = pos_CB_single.unsqueeze(0)           # [1, num_residues, 3]
                batch_frames = frames_single.unsqueeze(0)           # [1, num_residues, 3, 3]
                batch_masks = masks_single.unsqueeze(0)            # [1, num_residues]
                
                # 应用残基级位置感知注意力
                enhanced_batch_features = self.position_aware(
                    batch_features, batch_pos_CA, batch_pos_CB, 
                    batch_frames, batch_masks
                )
                
                # 将残基级特征映射回原子级
                updated_atoms = self._map_residue_to_atoms(
                    updated_atoms, enhanced_batch_features, batch_masks,
                    atom2residue, graph_data, 1
                )
            else:
                # 将不同batch的残基特征padding到相同长度
                max_residues = max(feat.shape[0] for feat in residue_features_list)
                
                padded_features = torch.zeros(len(residue_features_list), max_residues, 
                                            self.out_channels, device=x.device)
                padded_pos_CA = torch.zeros(len(residue_features_list), max_residues, 3, device=x.device)
                padded_pos_CB = torch.zeros(len(residue_features_list), max_residues, 3, device=x.device)
                padded_frames = torch.zeros(len(residue_features_list), max_residues, 3, 3, device=x.device)
                padded_masks = torch.zeros(len(residue_features_list), max_residues, dtype=torch.bool, device=x.device)
                
                for i, (feat, pos_CA, pos_CB, frames, mask) in enumerate(zip(
                    residue_features_list, residue_positions_CA_list, 
                    residue_positions_CB_list, residue_frames_list, residue_masks_list)):
                    seq_len = feat.shape[0]
                    padded_features[i, :seq_len] = feat
                    padded_pos_CA[i, :seq_len] = pos_CA
                    padded_pos_CB[i, :seq_len] = pos_CB
                    padded_frames[i, :seq_len] = frames
                    padded_masks[i, :seq_len] = mask
                
                # 应用残基级位置感知注意力
                enhanced_residue_features = self.position_aware(
                    padded_features, padded_pos_CA, padded_pos_CB, 
                    padded_frames, padded_masks
                )
                
                # 4. 将残基级特征映射回原子级（只更新CA原子）
                updated_atoms = self._map_residue_to_atoms(
                    updated_atoms, enhanced_residue_features, padded_masks,
                    atom2residue, graph_data, len(residue_features_list)
                )
        
        return updated_atoms
    
    def _map_residue_to_atoms(self, atom_features, residue_features, residue_masks, 
                             atom2residue, graph_data, batch_size):
        """将残基级特征映射回原子级，只更新CA原子"""
        device = atom_features.device
        updated_atoms = atom_features.clone()
        
        # 构建原始残基索引到本地残基索引的映射
        for batch_idx in range(batch_size):
            atom_mask = graph_data.batch == batch_idx
            
            if atom_mask.any():
                atom2residue_batch = atom2residue[atom_mask]
                unique_residues = torch.unique(atom2residue_batch)
                

                
                # 为每个原始残基索引找到对应的本地索引
                local_residue_idx = 0
                residue_idx_map = {}
                for orig_res_idx in unique_residues:
                    if orig_res_idx < len(residue_masks[batch_idx]) and residue_masks[batch_idx, orig_res_idx]:
                        residue_idx_map[orig_res_idx.item()] = local_residue_idx
                        local_residue_idx += 1
                
                # 更新原子特征
                for orig_res_idx, local_idx in residue_idx_map.items():
                    if local_idx < residue_features[batch_idx].shape[0]:
                        # 获取残基的增强特征
                        enhanced_feat = residue_features[batch_idx, local_idx]
                        
                        # 找到该残基的CA原子
                        for atom_idx in torch.where(atom_mask)[0]:
                            if (atom2residue[atom_idx].item() == orig_res_idx and 
                                graph_data.atom_names[atom_idx] == "CA"):
                                # 更新CA原子特征
                                updated_atoms[atom_idx] = updated_atoms[atom_idx] + enhanced_feat * 0.5
                                break
        
        return updated_atoms
    
    def _compute_frames(self, positions):
        """计算局部坐标系"""
        batch_size, seq_len, _ = positions.shape
        frames = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1).to(positions.device)
        
        # 简化处理：对每个残基计算局部坐标系
        for b in range(batch_size):
            for i in range(seq_len):
                if i < seq_len - 2:  # 确保有足够的前后残基
                    pos_prev = positions[b, i]
                    pos_curr = positions[b, i + 1] 
                    pos_next = positions[b, i + 2]
                    
                    # 简化的坐标系计算
                    e1 = pos_curr - pos_prev
                    e1_norm = e1.norm()
                    if e1_norm > 1e-6:
                        e1 = e1 / e1_norm
                    
                    e2 = pos_next - pos_curr
                    e2_norm = e2.norm()
                    if e2_norm > 1e-6:
                        e2 = e2 - torch.dot(e2, e1) * e1
                        e2_norm = e2.norm()
                        if e2_norm > 1e-6:
                            e2 = e2 / e2_norm
                    
                    e3 = torch.cross(e1, e2)
                    e3_norm = e3.norm()
                    if e3_norm > 1e-6:
                        e3 = e3 / e3_norm
                    
                    frames[b, i + 1] = torch.stack([e1, e2, e3], dim=1)
        
        return frames
    
    def message(self, x_i, x_j, edge_attr, edge_types):
        # 计算消息
        msg_input = torch.cat([x_i, x_j + edge_attr], dim=-1)
        message = self.msg_proj(msg_input)
        
        # 简化的关系权重处理 - 使用逐元素乘法而不是矩阵乘法
        # 为每种关系类型创建权重索引
        relation_weights_selected = self.relation_weights[edge_types]  # [num_edges, hidden_dim, hidden_dim]
        
        # 只使用关系权重矩阵的对角线元素来减少内存使用
        if relation_weights_selected.shape[-1] == self.out_channels:
            # 取对角线部分 [num_edges, hidden_dim]
            diagonal_weights = torch.diagonal(relation_weights_selected, dim1=-2, dim2=-1)
            weighted_message = message * diagonal_weights
        else:
            # 如果维度不匹配，直接使用消息
            weighted_message = message
        
        return weighted_message
    
    def update(self, aggr_out, x):
        # 这个方法现在在forward中处理
        return aggr_out


class InterfaceFeatureExtractor:
    """界面几何特征提取器 - 压缩特征维度版本，集成KNN突变位点选择"""
    
    def __init__(self, cutoff_distance: float = 8.0, k_neighbors: int = 20, knn_mutation_k: int = 256):
        self.cutoff_distance = cutoff_distance
        self.k_neighbors = k_neighbors
        self.knn_mutation_k = knn_mutation_k
        
        # 按照GearBind模型设置维度 - 压缩到64维
        self.node_feat_dim = 64   # 最终节点特征维度
        self.edge_feat_dim = 64   # 边特征维度
        self.initial_atom_feat_dim = 32  # 初始原子特征维度
        
        # KNN突变位点选择模块
        self.mutation_site_selector = KNNMutationSite(k=knn_mutation_k)
        
    def extract_interface_graph(self, wt_structure, mt_structure, 
                               interface_chain_a: str, interface_chain_b: str, 
                               mutation: str) -> InterfaceGraphData:
        """
        提取复合物界面图 - 使用全部残基
        
        Args:
            wt_structure: 野生型结构
            mt_structure: 突变型结构
            interface_chain_a: 界面链A（如抗体链）
            interface_chain_b: 界面链B（如抗原链）
            mutation: 突变信息（如 "K15I"）
            
        Returns:
            InterfaceGraphData对象
        """
        interface_graphs = []
        
        for structure, is_mutant in [(wt_structure, False), (mt_structure, True)]:
            # 提取全部残基（不再仅提取界面残基）
            all_residues = self._get_all_residues(structure)
            
            # 构建图
            graph_data = self._build_interface_graph(all_residues, is_mutant, mutation)
            interface_graphs.append(graph_data)
        
        return interface_graphs
    
    def _get_all_residues(self, structure):
        """获取结构中的全部残基"""
        all_residues = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if is_aa(residue, standard=True):
                        all_residues.append(residue)
        
        return all_residues
    
    def _build_interface_graph(self, all_residues, is_mutant, mutation):
        """构建界面图 - 使用KNN选择离突变点最近的256个残基"""
        # 解析突变信息
        mutation_pos = None
        mutation_chain = None
        if mutation and len(mutation) >= 3:
            try:
                mutation_pos = int(mutation[1:-1])  # 提取位置数字
                mutation_chain = mutation[0]  # 提取链ID
            except:
                mutation_pos = None
                mutation_chain = None
        
        # 提取节点特征（原子级别）
        node_features = []
        node_positions = []
        atom_names = []
        residue_indices = []
        is_mutation_list = []
        
        for residue in all_residues:
            # 为每个残基的原子创建节点
            for atom in residue:
                # 原子类型特征
                atom_feat = self._atom_to_feature(atom)
                
                # 原子位置
                position = atom.coord
                
                # 判断是否为突变位点的CA原子
                is_mutation_node = False
                if (mutation_pos is not None and 
                    mutation_chain is not None and
                    residue.id[1] == mutation_pos and 
                    residue.parent.id == mutation_chain and
                    atom.name == "CA"):
                    is_mutation_node = True
                
                node_features.append(atom_feat)
                node_positions.append(position)
                atom_names.append(atom.name)
                residue_indices.append(f"{residue.parent.id}_{residue.id[1]}")
                is_mutation_list.append(is_mutation_node)
        
        # 转换为tensor
        node_positions_tensor = torch.tensor(node_positions, dtype=torch.float32)
        is_mutation_tensor = torch.tensor(is_mutation_list, dtype=torch.bool)
        
        # 构建批量索引（每个结构一个图）
        batch = torch.zeros(len(node_features), dtype=torch.long)
        if is_mutant:
            batch = torch.ones(len(node_features), dtype=torch.long)
        batch = batch.long()  # 确保是int64
        
        # 使用KNN选择离突变点最近的节点
        if self.knn_mutation_k > 0 and len(node_features) > self.knn_mutation_k:
            # 应用KNN突变位点选择
            node_mask = self.mutation_site_selector(
                node_positions_tensor, 
                atom_names, 
                is_mutation_tensor, 
                batch
            )
            
            # 过滤数据
            node_features = [node_features[i] for i in range(len(node_features)) if node_mask[i]]
            node_positions = [node_positions[i] for i in range(len(node_positions)) if node_mask[i]]
            atom_names = [atom_names[i] for i in range(len(atom_names)) if node_mask[i]]
            residue_indices = [residue_indices[i] for i in range(len(residue_indices)) if node_mask[i]]
            is_mutation_list = [is_mutation_list[i] for i in range(len(is_mutation_list)) if node_mask[i]]
            
            # 更新batch索引
            original_indices = torch.where(node_mask)[0]
            batch = batch[original_indices]
        
        # 构建边
        edge_index, edge_features, edge_types = self._build_edges(
            node_positions, residue_indices
        )
        
        return InterfaceGraphData(
            node_features=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            edge_features=edge_features,
            edge_types=edge_types,
            node_positions=torch.tensor(node_positions, dtype=torch.float32),
            batch=batch,
            atom_names=atom_names,
            is_mutation=torch.tensor(is_mutation_list, dtype=torch.bool),
            residue_indices=residue_indices  # 添加残基索引
        )
    
    def _atom_to_feature(self, atom, residue_frame=None, atom_positions_in_residue=None):
        """
        原子转换为特征向量 - 压缩版本，只保留原子类型和位置特征
        """
        # 原子类型one-hot编码
        atom_types = ['C', 'N', 'O', 'S', 'H', 'P', 'CA', 'CB', 'CG', 'CD', 'CE', 'CZ']
        
        atom_name = atom.name.strip()
        if atom_name in atom_types:
            idx = atom_types.index(atom_name)
        else:
            idx = 0  # 默认C
        
        # 基础特征：原子类型 + 位置特征 - 修改为64维
        base_feature = np.zeros(64)
        base_feature[idx] = 1.0
        
        # 原子位置信息（归一化坐标）- 保留原始坐标用于位置计算
        try:
            atom_pos = np.array(atom.coord, dtype=np.float32)
            base_feature[12:15] = atom_pos / 10.0  # 归一化到[-1, 1]范围
        except:
            base_feature[12:15] = 0.0
        
        # 主链/侧链标识
        backbone_atoms = ['N', 'CA', 'C', 'O']
        is_backbone = 1.0 if atom_name in backbone_atoms else 0.0
        base_feature[15] = is_backbone
        
        # 简化的残基类型编码
        if hasattr(atom, 'parent') and hasattr(atom.parent, 'resname'):
            aa_types = 'ACDEFGHIKLMNPQRSTVWY'
            resname = atom.parent.resname
            if len(resname) == 3 and resname in aa_types:
                aa_idx = aa_types.index(resname)
                base_feature[16 + aa_idx] = 1.0
        
        # 填充剩余特征（36-63）使用位置编码
        try:
            atom_pos = np.array(atom.coord, dtype=np.float32)
            for i in range(7):  # 使用7个频率
                freq = (i + 1) * np.pi / 5.0
                base_feature[36 + i*4] = np.sin(atom_pos[0] * freq)
                base_feature[37 + i*4] = np.cos(atom_pos[0] * freq)
                base_feature[38 + i*4] = np.sin(atom_pos[1] * freq)
                base_feature[39 + i*4] = np.cos(atom_pos[1] * freq)
                if 36 + i*4 + 3 < 64:  # 确保不越界
                    base_feature[36 + i*4 + 2] = np.sin(atom_pos[2] * freq)
                    base_feature[36 + i*4 + 3] = np.cos(atom_pos[2] * freq)
        except:
            base_feature[36:64] = 0.0
        
        return base_feature
    
    def _build_edges(self, node_positions, residue_indices):
        """构建图的边 - 压缩边特征维度到64维"""
        edge_sources = []
        edge_targets = []
        edge_features_list = []
        edge_types_list = []
        
        positions = np.array(node_positions)
        n_nodes = len(positions)
        
        # 计算距离矩阵
        dist_matrix = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                dist_matrix[i, j] = np.linalg.norm(positions[i] - positions[j])
        
        # 构建多种类型的边
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                dist = dist_matrix[i, j]
                
                # 确定边类型（根据GearBind论文）
                try:
                    res_i = residue_indices[i].split('_')[0]
                    res_j = residue_indices[j].split('_')[0]
                except:
                    res_i = "A"
                    res_j = "A"
                
                # 同一残基内的原子
                if res_i == res_j:
                    if dist < 2.0:  # 共价键距离
                        edge_type = 0  # 共价键
                    elif dist < 5.0:
                        edge_type = 2  # 空间径向关系
                    else:
                        continue
                elif dist < 5.0:
                    edge_type = 2  # 空间径向关系（radial edges）
                elif dist < 8.0:
                    edge_type = 1  # K近邻关系（KNN edges）
                else:
                    continue
                
                edge_sources.extend([i, j])
                edge_targets.extend([j, i])
                
                # 压缩边特征到64维
                edge_feat = np.zeros(self.edge_feat_dim)
                
                # 1. 基础距离特征 (0-2)
                edge_feat[0] = dist
                edge_feat[1] = 1.0 / dist if dist > 0 else 0
                edge_feat[2] = np.log(dist + 1.0)
                
                # 2. 方向特征 (3-5)
                if dist > 0:
                    direction = (positions[j] - positions[i]) / dist
                    edge_feat[3:6] = direction
                
                # 3. 序列距离特征 (6-8)
                seq_dist = abs(i - j)
                edge_feat[6] = seq_dist
                edge_feat[7] = 1.0 / (seq_dist + 1.0)
                edge_feat[8] = np.log(seq_dist + 1.0)
                
                # 4. 简化的残基类型交互特征 (9-20)
                aa_types = 'ACDEFGHIKLMNPQRSTVWY'
                aa_idx_i = aa_types.find(res_i) if res_i in aa_types else 0
                aa_idx_j = aa_types.find(res_j) if res_j in aa_types else 0
                edge_feat[9 + aa_idx_i] = 1.0
                edge_feat[9 + aa_idx_j] = 1.0  # 简化：共享相同位置
                
                # 5. 简化的几何编码特征 (21-63) - 减少傅里叶编码数量
                for k in range(7):  # 从20减少到7
                    freq = (k + 1) * np.pi
                    idx = 21 + k * 6
                    if idx + 5 < 64:
                        edge_feat[idx] = np.sin(dist * freq)
                        edge_feat[idx + 1] = np.cos(dist * freq)
                        edge_feat[idx + 2] = np.sin(dist * freq * 2)
                        edge_feat[idx + 3] = np.cos(dist * freq * 2)
                        edge_feat[idx + 4] = np.sin(dist * freq * 3)
                        edge_feat[idx + 5] = np.cos(dist * freq * 3)
                
                edge_features_list.extend([edge_feat, edge_feat])
                edge_types_list.extend([edge_type, edge_type])
        
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_features = torch.tensor(edge_features_list, dtype=torch.float32)
        edge_types = torch.tensor(edge_types_list, dtype=torch.long)
        # 确保所有索引都是int64
        edge_index = edge_index.long()
        edge_types = edge_types.long()
        # 确保所有索引都是int64
        edge_index = edge_index.long()
        edge_types = edge_types.long()
        
        return edge_index, edge_features, edge_types
    
    def _build_residue_frames(self, all_residues):
        """
        为每个残基构建局部坐标系（类似AtomPositionGather的功能）
        返回: dict{residue_key: (3x3_rotation_matrix, 3d_origin)}
        """
        residue_frames = {}
        residue_atom_positions = {}
        
        for residue in all_residues:
            residue_key = f"{residue.parent.id}_{residue.id[1]}"
            
            # 收集主链原子位置
            backbone_atoms = {}
            for atom in residue:
                if atom.name in ['N', 'CA', 'C']:
                    backbone_atoms[atom.name] = np.array(atom.coord, dtype=np.float32)
                residue_atom_positions[residue_key] = {
                    atom.name: np.array(atom.coord, dtype=np.float32) 
                    for atom in residue
                }
            
            # 只有当拥有完整主链（N, CA, C）时才构建坐标系
            if len(backbone_atoms) == 3:
                try:
                    # 使用Gram-Schmidt算法构建局部坐标系（类似AtomPositionGather）
                    N_pos = backbone_atoms['N']
                    CA_pos = backbone_atoms['CA'] 
                    C_pos = backbone_atoms['C']
                    
                    # 构建三个基向量
                    # x轴: CA -> N
                    e0 = N_pos - CA_pos
                    e0_norm = np.linalg.norm(e0)
                    if e0_norm > 1e-6:
                        e0 = e0 / e0_norm
                    
                    # xy平面中的向量: CA -> C
                    e1 = C_pos - CA_pos
                    
                    # 正交化e1使其垂直于e0
                    dot_product = np.dot(e0, e1)
                    e1 = e1 - dot_product * e0
                    e1_norm = np.linalg.norm(e1)
                    if e1_norm > 1e-6:
                        e1 = e1 / e1_norm
                    
                    # z轴: e0 x e1 (右手坐标系)
                    e2 = np.cross(e0, e1)
                    e2_norm = np.linalg.norm(e2)
                    if e2_norm > 1e-6:
                        e2 = e2 / e2_norm
                    
                    # 构建旋转矩阵（从局部坐标系到全局坐标系）
                    # 每列是局部坐标系的基向量在全局坐标系中的表示
                    rotation_matrix = np.column_stack([e0, e1, e2])
                    
                    residue_frames[residue_key] = rotation_matrix
                    
                except Exception as e:
                    print(f"警告: 构建残基 {residue_key} 的局部坐标系失败: {e}")
                    residue_frames[residue_key] = np.eye(3)  # 使用单位矩阵作为默认值
            else:
                # 如果缺少主链原子，使用单位矩阵
                residue_frames[residue_key] = np.eye(3)
        
        return residue_frames, residue_atom_positions


class CHYModelWithGeometric(nn.Module):    
    def __init__(self, 
                 esm_dim: int = 1280,
                 foldx_dim: int = 22,
                 geometric_dim: int = 16,  # 几何特征维度
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super(CHYModelWithGeometric, self).__init__()
        
        self.esm_dim = esm_dim
        self.foldx_dim = foldx_dim
        self.geometric_dim = geometric_dim
        
        # FoldX特征投影
        self.foldx_projection = nn.Sequential(
            nn.Linear(foldx_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 128),  # 输出128维特征
            nn.ReLU()
        )
        
        # ESM特征处理
        self.esm_projection = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 几何特征处理（GearBind-inspired）- 压缩维度
        self.geometric_gnn_wt = GeometricGNN(
            node_feat_dim=64,   # 64维输入
            edge_feat_dim=64,   # 64维边特征
            hidden_dim=64,       # 64维隐藏层
            num_relation_types=3,
            num_heads=4,
            dropout=dropout
        )
        
        # 为突变体也创建一个GNN（需要这个变量）
        self.geometric_gnn_mt = GeometricGNN(
            node_feat_dim=64,   # 64维输入
            edge_feat_dim=64,   # 64维边特征
            hidden_dim=64,       # 64维隐藏层
            num_relation_types=3,
            num_heads=4,
            dropout=dropout
        )
        
        # 特征融合层 - 调整输入维度
        self.feature_fusion = nn.Sequential(
            nn.Linear(
                hidden_dim + 128 + 16 + 16,  # seq_rep + foldx_expanded + wt_geom_rep + mt_geom_rep = 512 + 128 + 16 + 16 = 672
                hidden_dim * 2
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # 对称性处理（类似GearBind的Anti-symmetric处理）
        self.antisymmetric_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        # 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                esm_embeddings: torch.Tensor,
                foldx_features: torch.Tensor,
                wt_graph_data: InterfaceGraphData,
                mt_graph_data: InterfaceGraphData,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            esm_embeddings: [batch_size, seq_len, esm_dim] ESM嵌入
            foldx_features: [batch_size, foldx_dim] FoldX能量项
            wt_graph_data: 野生型界面图数据
            mt_graph_data: 突变型界面图数据
            attention_mask: [batch_size, seq_len] 注意力掩码
            
        Returns:
            ddg_predictions: [batch_size, 1] ΔΔG预测值
        """
        batch_size = esm_embeddings.shape[0]
        
        # 1. 处理FoldX特征
        foldx_proj = self.foldx_projection(foldx_features)  # [batch_size, esm_dim//2]
        
        # 2. 处理ESM特征
        esm_proj = self.esm_projection(esm_embeddings)  # [batch_size, seq_len, hidden_dim]
        
        # 3. Transformer编码
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                src_key_padding_mask = (attention_mask == 0)
            else:
                src_key_padding_mask = (attention_mask.squeeze(1) == 0)
        
        transformer_output = self.transformer_encoder(
            esm_proj,
            src_key_padding_mask=src_key_padding_mask if attention_mask is not None else None
        )  # [batch_size, seq_len, hidden_dim]
        
        # 全局平均池化
        seq_rep = transformer_output.mean(dim=1)  # [batch_size, hidden_dim]
        
        # 4. 处理几何特征（WT和MT分别处理）
        wt_geom_rep = self.geometric_gnn_wt(wt_graph_data)  # [2, hidden_dim//4] (WT=0, MT=1)
        mt_geom_rep = self.geometric_gnn_mt(mt_graph_data)  # [2, hidden_dim//4] (WT=0, MT=1)
        
        # 确保所有特征都有相同的batch size
        batch_size = seq_rep.shape[0]
        
        # 处理几何特征 - 确保batch维度匹配
        if wt_geom_rep.shape[0] >= batch_size:
            # 如果几何特征有足够的batch维度，直接使用
            wt_geom_rep = wt_geom_rep[:batch_size]
            mt_geom_rep = mt_geom_rep[:batch_size]
        else:
            # 否则扩展到匹配batch size
            # 使用WT几何特征作为基础，为每个样本复制
            wt_geom_rep = wt_geom_rep[0:1].expand(batch_size, -1)
            mt_geom_rep = mt_geom_rep[0:1].expand(batch_size, -1)  # 同样使用WT部分
        
        # 扩展几何特征到合适维度（如果需要）
        if wt_geom_rep.shape[-1] < 16:
            # 将16维扩展到合适维度
            geom_expansion = nn.Linear(wt_geom_rep.shape[-1], 16).to(seq_rep.device)
            wt_geom_rep = geom_expansion(wt_geom_rep)
            mt_geom_rep = geom_expansion(mt_geom_rep)
        
        # 扩展FoldX特征到合适维度
        if foldx_proj.shape[-1] < 128:
            foldx_expansion = nn.Linear(foldx_proj.shape[-1], 128).to(seq_rep.device)
            foldx_expanded = foldx_expansion(foldx_proj)
        else:
            foldx_expanded = foldx_proj
        
        # 拼接所有特征
        combined_features = torch.cat([
            seq_rep,  # 序列特征 [batch_size, hidden_dim=512]
            foldx_expanded,  # FoldX特征 [batch_size, 128]
            wt_geom_rep,  # WT几何特征 [batch_size, 16]
            mt_geom_rep   # MT几何特征 [batch_size, 16]
        ], dim=-1)  # [batch_size, 512 + 128 + 16 + 16 = 672]
        
        

        
        fused_features = self.feature_fusion(combined_features)  # [batch_size, hidden_dim]
        
        # 6. 对称性处理（ΔΔG应该是反对称的）
        antisym_features = self.antisymmetric_layer(fused_features)
        # 通过WT和MT特征的差异来实现反对称性
        geom_diff = mt_geom_rep - wt_geom_rep  # [batch_size, 16]
        
        # 扩展几何差异到与antisym_features相同的维度
        if geom_diff.shape[-1] != antisym_features.shape[-1]:
            geom_diff_expansion = nn.Linear(geom_diff.shape[-1], antisym_features.shape[-1]).to(antisym_features.device)
            geom_diff = geom_diff_expansion(geom_diff)
        
        antisym_features = antisym_features + geom_diff
        
        # 7. 回归预测
        ddg_pred = self.regression_head(antisym_features)  # [batch_size, 1]
        
        return ddg_pred


class DDGModelTester:
    """集成几何特征的模型测试器 - 压缩特征维度版本"""
    
    def __init__(self, pdb_base_path: str,
                 weights_path: str = "/home/chengwang/weights/esm2",
                 cache_dir: str = "./test_cache",
                 model_checkpoint: Optional[str] = None,
                 use_geometric: bool = True):
        self.pdb_base_path = Path(pdb_base_path)
        self.weights_path = Path(weights_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_geometric = use_geometric
        
        # 初始化特征提取器
        self.feature_extractor = RealFeatureExtractor(
            pdb_base_path=str(pdb_base_path),
            cache_dir=str(cache_dir),
            use_esm=True,
            weights_path=str(weights_path)
        )
        
        # 初始化几何特征提取器
        if use_geometric:
            self.geometric_extractor = InterfaceFeatureExtractor()
        
        # 初始化模型
        if use_geometric:
            self.model = CHYModelWithGeometric()
        
        # 加载模型检查点
        if model_checkpoint and Path(model_checkpoint).exists():
            self.load_model_checkpoint(model_checkpoint)
        
        print(f"模型测试器初始化完成:")
        print(f"  - PDB路径: {pdb_base_path}")
        print(f"  - 权重路径: {weights_path}")
        print(f"  - 缓存目录: {cache_dir}")
        print(f"  - 使用几何特征: {use_geometric}")
        print(f"  - 模型检查点: {model_checkpoint if model_checkpoint else '无'}")
        print("-" * 50)
    
    def load_model_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print(f"模型加载成功: {checkpoint_path}")
            
        except Exception as e:
            print(f"警告: 加载模型检查点失败: {e}")
            print("使用随机初始化的模型")
    
    def extract_geometric_features(self, pdb_id: str, chain: str, mutation: str):
        """从实际PDB文件中提取野生型和突变体结构的界面特征"""
        try:            
            # 构建文件路径
            mutation_str = mutation[0] + chain + mutation[1:]
            pdb_base_path = Path("/home/chengwang/data/SKEMPI/PDBs_fixed")
            mutation_folder = pdb_base_path / f"{pdb_id}_{mutation_str}"
            
            if not mutation_folder.exists():
                print(f"  警告: 突变文件夹不存在 {mutation_folder}")
                return None, None
            
            # 查找野生型和突变体PDB文件
            wt_pdb_file = None
            mt_pdb_file = None
            
            # 查找野生型文件（WT_开头）
            for pdb_file in mutation_folder.glob("WT_*.pdb"):
                wt_pdb_file = pdb_file
                break
            
            # 查找突变体文件（非WT_开头的pdb文件）
            for pdb_file in mutation_folder.glob("*_Repair_1.pdb"):
                if not pdb_file.name.startswith("WT_"):
                    mt_pdb_file = pdb_file
                    break
            
            # 如果找不到标准命名，尝试其他策略
            if wt_pdb_file is None or mt_pdb_file is None:
                print(f"  警告: 未找到完整的野生型/突变体PDB文件对")
                print(f"    WT文件: {wt_pdb_file}")
                print(f"    MT文件: {mt_pdb_file}")
                set_trace()
                return None, None
                        
            # 使用Bio.PDB解析结构            
            parser = PDBParser(QUIET=True)
            try:
                wt_structure = parser.get_structure('WT', str(wt_pdb_file))
                mt_structure = parser.get_structure('MT', str(mt_pdb_file))
            except Exception as e:
                print(f"  错误: PDB文件解析失败 {e}")
                return None, None
            
            # 提取全部残基（不再仅提取界面残基）
            wt_all_residues = self._extract_all_residues(wt_structure, chain)
            mt_all_residues = self._extract_all_residues(mt_structure, chain)
            
            if not wt_all_residues or not mt_all_residues:
                print(f"  警告: 无法提取残基")
                return None, None
            
            # print(f"  野生型残基数: {len(wt_all_residues)}")
            # print(f"  突变体残基数: {len(mt_all_residues)}")
            
            # 构建图数据
            wt_graph = self._build_interface_graph_from_structure(wt_structure, wt_all_residues, is_mutant=False, mutation=mutation)
            mt_graph = self._build_interface_graph_from_structure(mt_structure, mt_all_residues, is_mutant=True, mutation=mutation)
            
            # print(f"  野生型图: 节点={wt_graph.node_features.shape[0]}, 边={wt_graph.edge_index.shape[1]}")
            # print(f"  突变体图: 节点={mt_graph.node_features.shape[0]}, 边={mt_graph.edge_index.shape[1]}")
            
            return wt_graph, mt_graph
            
        except Exception as e:
            print(f"  错误: 提取几何特征失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _extract_all_residues(self, structure, target_chain):
        """提取全部残基"""
        try:
            # 获取所有链
            all_chains = list(structure.get_chains())
            if len(all_chains) < 1:
                print(f"  警告: 结构没有链")
                return []
            
            # 找到目标链
            target_chain_obj = None
            for chain_obj in all_chains:
                if chain_obj.id == target_chain:
                    target_chain_obj = chain_obj
                    break
            
            if not target_chain_obj:
                print(f"  警告: 未找到目标链 {target_chain}")
                return []
            
            # 获取目标链的所有残基
            all_residues = []
            for residue in target_chain_obj:
                if is_aa(residue, standard=True):
                    all_residues.append(residue)
            
            return all_residues
            
        except Exception as e:
            print(f"  错误: 提取残基失败 {e}")
            return []
    
    def _build_interface_graph_from_structure(self, structure, all_residues, is_mutant, mutation):
        """从结构中使用KNN选择最近残基构建图"""        
        if not all_residues:
            set_trace()
        
        # 解析突变信息
        mutation_pos = None
        mutation_chain = None
        if mutation and len(mutation) >= 3:
            try:
                mutation_pos = int(mutation[1:-1])  # 提取位置数字
                mutation_chain = mutation[0]  # 提取链ID
            except:
                mutation_pos = None
                mutation_chain = None
        
        # 提取节点特征和位置
        node_features = []
        node_positions = []
        atom_names = []
        residue_indices = []
        is_mutation_list = []
        
        # 为每个残基构建局部坐标系（类似AtomPositionGather）
        residue_frames, residue_atom_positions = self.geometric_extractor._build_residue_frames(all_residues)
        
        for residue in all_residues:
            residue_key = f"{residue.parent.id}_{residue.id[1]}"
            residue_frame = residue_frames.get(residue_key, None)
            atom_positions_in_residue = residue_atom_positions.get(residue_key, {})
            
            # 为每个残基的原子创建节点
            for atom in residue:
                # 原子特征（使用改进的_atom_to_feature）
                atom_feat = self.geometric_extractor._atom_to_feature(
                    atom, 
                    residue_frame=residue_frame, 
                    atom_positions_in_residue=atom_positions_in_residue
                )
                node_features.append(atom_feat)
                
                # 原子位置
                position = atom.coord
                node_positions.append(position)
                atom_names.append(atom.name)
                residue_indices.append(f"{residue.parent.id}_{residue.id[1]}")
                
                # 判断是否为突变位点的CA原子
                is_mutation_node = False
                if (mutation_pos is not None and 
                    mutation_chain is not None and
                    residue.id[1] == mutation_pos and 
                    residue.parent.id == mutation_chain and
                    atom.name == "CA"):
                    is_mutation_node = True
                
                is_mutation_list.append(is_mutation_node)
        
        # 转换为tensor和numpy数组
        node_features = np.array(node_features, dtype=np.float32)
        node_positions = np.array(node_positions, dtype=np.float32)
        node_positions_tensor = torch.tensor(node_positions, dtype=torch.float32)
        is_mutation_tensor = torch.tensor(is_mutation_list, dtype=torch.bool)
        
        # 构建批量索引（每个结构一个图）
        batch = torch.zeros(len(node_features), dtype=torch.long)
        if is_mutant:
            batch = torch.ones(len(node_features), dtype=torch.long)
        batch = batch.long()  # 确保是int64
        
        # 使用KNN选择离突变点最近的节点
        if self.geometric_extractor.knn_mutation_k > 0 and len(node_features) > self.geometric_extractor.knn_mutation_k:
            # 应用KNN突变位点选择
            node_mask = self.geometric_extractor.mutation_site_selector(
                node_positions_tensor, 
                atom_names, 
                is_mutation_tensor, 
                batch
            )
            
            # 过滤数据
            node_features = node_features[node_mask.numpy()]
            node_positions = node_positions[node_mask.numpy()]
            atom_names = [atom_names[i] for i in range(len(atom_names)) if node_mask[i]]
            residue_indices = [residue_indices[i] for i in range(len(residue_indices)) if node_mask[i]]
            is_mutation_list = [is_mutation_list[i] for i in range(len(is_mutation_list)) if node_mask[i]]
            
            # 更新batch索引
            original_indices = torch.where(node_mask)[0]
            batch = batch[original_indices]
        
        # 构建边
        edge_index, edge_features, edge_types = self._build_edges_from_positions_ddgtester(
            node_positions, residue_indices
        )
        
        return InterfaceGraphData(
            node_features=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            edge_features=edge_features,
            edge_types=edge_types,
            node_positions=torch.tensor(node_positions, dtype=torch.float32),
            batch=batch,
            atom_names=atom_names,
            is_mutation=torch.tensor(is_mutation_list, dtype=torch.bool),
            residue_indices=residue_indices  # 添加残基索引
        )
        

    def _build_edges_from_positions_ddgtester(self, node_positions, residue_indices):
        """基于原子位置构建边 - 压缩到64维边特征"""
        edge_sources = []
        edge_targets = []
        edge_features_list = []
        edge_types_list = []
        
        n_nodes = len(node_positions)
        
        # 计算距离矩阵
        dist_matrix = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                dist_matrix[i, j] = np.linalg.norm(node_positions[i] - node_positions[j])
        
        # 构建多种类型的边
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                dist = dist_matrix[i, j]
                
                # 确定边类型（根据GearBind论文）
                try:
                    res_i = residue_indices[i].split('_')[0]
                    res_j = residue_indices[j].split('_')[0]
                except:
                    res_i = "A"
                    res_j = "A"
                
                # 同一残基内的原子
                if res_i == res_j:
                    if dist < 2.0:  # 共价键距离
                        edge_type = 0  # 共价键
                    elif dist < 5.0:
                        edge_type = 2  # 空间径向关系
                    else:
                        continue
                elif dist < 5.0:
                    edge_type = 2  # 空间径向关系（radial edges）
                elif dist < 8.0:
                    edge_type = 1  # K近邻关系（KNN edges）
                else:
                    continue
                
                edge_sources.extend([i, j])
                edge_targets.extend([j, i])
                
                # 构建64维边特征
                edge_feat = np.zeros(64)  # 压缩到64维
                
                # 1. 基础距离特征 (0-2)
                edge_feat[0] = dist
                edge_feat[1] = 1.0 / dist if dist > 0 else 0
                edge_feat[2] = np.log(dist + 1.0)
                
                # 2. 方向特征 (3-5)
                if dist > 0:
                    direction = (node_positions[j] - node_positions[i]) / dist
                    edge_feat[3:6] = direction
                
                # 3. 序列距离特征 (6-8)
                seq_dist = abs(i - j)
                edge_feat[6] = seq_dist
                edge_feat[7] = 1.0 / (seq_dist + 1.0)
                edge_feat[8] = np.log(seq_dist + 1.0)
                
                # 4. 简化的残基类型交互特征 (9-28)
                aa_types = 'ACDEFGHIKLMNPQRSTVWY'
                aa_idx_i = aa_types.find(res_i) if res_i in aa_types else 0
                aa_idx_j = aa_types.find(res_j) if res_j in aa_types else 0
                edge_feat[9 + aa_idx_i] = 1.0
                edge_feat[19 + aa_idx_j] = 1.0  # 简化位置分配
                
                # 5. 简化的几何编码特征 (29-63)
                for k in range(6):  # 从21减少到6
                    freq = (k + 1) * np.pi
                    idx = 29 + k * 6
                    if idx + 5 < 64:
                        edge_feat[idx] = np.sin(dist * freq)
                        edge_feat[idx + 1] = np.cos(dist * freq)
                        edge_feat[idx + 2] = np.sin(dist * freq * 2)
                        edge_feat[idx + 3] = np.cos(dist * freq * 2)
                        edge_feat[idx + 4] = np.sin(dist * freq * 3)
                        edge_feat[idx + 5] = np.cos(dist * freq * 3)
                
                edge_features_list.extend([edge_feat, edge_feat])
                edge_types_list.extend([edge_type, edge_type])
        
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_features = torch.tensor(edge_features_list, dtype=torch.float32)
        edge_types = torch.tensor(edge_types_list, dtype=torch.long)
        # 确保所有索引都是int64
        edge_index = edge_index.long()
        edge_types = edge_types.long()
        # 确保所有索引都是int64
        edge_index = edge_index.long()
        edge_types = edge_types.long()
        
        return edge_index, edge_features, edge_types
    

    

    def test_from_csv(self, csv_path, pdb_col="#Pdb_origin", mutation_col="Mutation(s)_cleaned", limit=None):
        """
        从CSV文件读取突变数据并进行测试

        参数:
            csv_path: CSV文件路径
            pdb_col: PDB ID列名
            mutation_col: 突变信息列名
            limit: 限制测试的行数
        """
        # 读取CSV文件，注意分隔符可能是制表符`\t`或逗号`,`, 请根据实际文件调整
        df = pd.read_csv(csv_path, sep='\t')  # 如果文件是制表符分隔
        
        results = []
        
        if limit:
            df = df.head(limit)
        
        for index, row in df.iterrows():
            pdb_id = row[pdb_col]
            mutation_str = row[mutation_col].strip()  # 去除可能的空格
            
            # 正确解析突变字符串格式，如"KI15I" -> chain="I", mutation="K15I"
            chain = mutation_str[1]  # 第一个字符是链ID，例如 "I"
            mutation = mutation_str[0] + mutation_str[2:]  # 剩余部分是突变信息，例如 "K15I"                
            print(f"处理: PDB={pdb_id}, 链={chain}, 突变={mutation}")
            
            # 测试单个突变
            result = self.test_single_mutation(pdb_id, chain, mutation)
            result['csv_index'] = index
            result['original_mutation'] = mutation_str
            results.append(result)
        
        return results

    
    def test_single_mutation(self, pdb_id: str, chain: str, mutation: str) -> Dict[str, Any]:
        """测试单个突变"""
        print(f"\n测试突变: {pdb_id}_{chain}_{mutation}")
        
        # 提取ESM和FoldX特征
        seq_feature, energy_feature = self.feature_extractor.extract_features(
            pdb_id, chain, mutation
        )
        
        # 提取几何特征
        if self.use_geometric:
            wt_graph, mt_graph = self.extract_geometric_features(pdb_id, chain, mutation)
        else:
            wt_graph, mt_graph = None, None
        
        # 准备模型输入
        if seq_feature.ndim == 1:
            seq_feature = seq_feature.reshape(1, -1)
        
        # 转换为张量
        esm_embeddings = torch.tensor(seq_feature, dtype=torch.float32).unsqueeze(0)
        foldx_features = torch.tensor(energy_feature, dtype=torch.float32).unsqueeze(0)
        attention_mask = torch.ones(1, 1, dtype=torch.float32)
        
        # 前向传播
        with torch.no_grad():
            if self.use_geometric and wt_graph is not None and mt_graph is not None:
                ddg_pred = self.model(
                    esm_embeddings, 
                    foldx_features, 
                    wt_graph, 
                    mt_graph, 
                    attention_mask
                )
            else:
                # 如果没有几何特征或使用非几何模型，创建默认的空图数据
                if hasattr(self.model, 'forward') and 'wt_graph_data' in self.model.forward.__code__.co_varnames:
                    # 如果模型需要几何参数但几何特征缺失，创建空的图数据
                    dummy_graph_data = InterfaceGraphData(
                        node_features=torch.zeros(1, 64),  # 压缩到64维
                        edge_index=torch.zeros(2, 1, dtype=torch.long),
                        edge_features=torch.zeros(1, 64),   # 压缩到64维
                        edge_types=torch.zeros(1, dtype=torch.long),
                        node_positions=torch.zeros(1, 3),
                        batch=torch.zeros(1, dtype=torch.long),
                        atom_names=[""],
                        is_mutation=torch.zeros(1, dtype=torch.bool)
                    )
                    ddg_pred = self.model(
                        esm_embeddings, 
                        foldx_features, 
                        dummy_graph_data, 
                        dummy_graph_data, 
                        attention_mask
                    )
                else:
                    ddg_pred = self.model(esm_embeddings, foldx_features, attention_mask)
        
        result = {
            'pdb_id': pdb_id,
            'chain': chain,
            'mutation': mutation,
            'predicted_ddg': ddg_pred.item(),
            'sequence_feature_shape': seq_feature.shape,
            'energy_feature_shape': energy_feature.shape,
            'energy_values': energy_feature.tolist(),
            'status': 'success',
            'use_geometric': self.use_geometric
        }
        
        print(f"  预测ΔΔG: {ddg_pred.item():.3f} kcal/mol")
        
        return result
    
    def save_results(self, results, output_file):
        """保存测试结果到CSV文件"""
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 保存到CSV
        output_path = Path(output_file)
        df.to_csv(output_path, index=False)
        
        print(f"\n已保存 {len(results)} 个结果到: {output_path}")
        
        # 显示统计信息
        if 'predicted_ddg' in df.columns:
            predictions = df['predicted_ddg'].values
            print(f"预测ΔΔG统计:")
            print(f"  平均值: {np.mean(predictions):.3f} kcal/mol")
            print(f"  标准差: {np.std(predictions):.3f} kcal/mol")
            print(f"  范围: [{np.min(predictions):.3f}, {np.max(predictions):.3f}] kcal/mol")


def load_pretrained_geometric_model(model_path: str, device: str = 'cpu') -> Union[CHYModelWithGeometric]:
    """加载预训练模型"""
    model = CHYModelWithGeometric()
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model.to(device)
        print(f"从 {model_path} 加载预训练模型成功")
        
    except Exception as e:
        print(f"警告: 加载预训练模型失败: {e}")
        print("使用随机初始化的模型")
    
    return model


# 使用示例
if __name__ == "__main__":    
    # 测试带几何特征的版本 - 压缩特征维度
    tester = DDGModelTester(
        pdb_base_path="/home/chengwang/data/SKEMPI/PDBs_fixed",
        weights_path="/home/chengwang/weights/esm2",
        cache_dir="./dataset_cache",
        model_checkpoint=None,
        use_geometric=True  # 测试几何版本
    )
    
    csv_path = "/home/chengwang/code/chymodel/s1131.csv"
    if Path(csv_path).exists():
        # 修改test_from_csv方法以适应新的CSV格式
        csv_results = tester.test_from_csv(
            csv_path, 
            pdb_col="#Pdb_origin",  # 使用原始PDB ID列
            mutation_col="Mutation(s)_cleaned",  # 使用清理后的突变列
            limit=1  # 只测试1行数据
        )
        
        # 保存结果
        tester.save_results(csv_results, "test_results_s1131.csv")
        print(f"结果已保存到: test_results_s1131.csv")
    else:
        print(f"CSV文件不存在: {csv_path}")
        print("跳过CSV测试")