"""
几何特征提取和处理模块
从 model.py 中提取的几何特征相关类和函数
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, MessagePassing
import numpy as np
from dataclasses import dataclass


def build_unified_residue_frames(all_residues):
    """
    模块级统一的残基坐标系构建函数
    供所有需要构建坐标系的模块调用
    
    Args:
        all_residues: Bio.PDB.Residue对象的列表
        
    Returns:
        tuple: (residue_frames, residue_atom_positions)
            residue_frames: dict{residue_key: torch.Tensor[3, 3]}
            residue_atom_positions: dict{residue_key: dict{atom_name: torch.Tensor[3]}}
    """
    # 创建一个临时实例来调用方法
    extractor = InterfaceFeatureExtractor()
    return extractor._build_unified_residue_frames(all_residues)


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


class PositionAwareLayer(nn.Module):
    """残基级位置感知层 - 基于GearBind实现，处理残基而非原子"""
    
    def __init__(self, hidden_dim, num_heads=4):
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
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
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
        
        # 按照GearBind标准的37维原子类型映射
        self.atom_name2id = {
            'N': 0, 'CA': 1, 'C': 2, 'CB': 3, 'O': 4, 'CG': 5, 'CG1': 6, 'CG2': 7, 
            'OG': 8, 'OG1': 9, 'SG': 10, 'CD': 11, 'CD1': 12, 'CD2': 13, 'ND1': 14, 
            'ND2': 15, 'NE1': 16, 'NE2': 17, 'NZ': 18, 'OE1': 19, 'OE2': 20, 'OD1': 21,
            'OD2': 22, 'CE': 23, 'CE1': 24, 'CE2': 25, 'CE3': 26, 'CZ': 27, 'CZ2': 28, 
            'CZ3': 29, 'CH2': 30, 'NH1': 31, 'NH2': 32, 'OH': 33, 'SD': 34, 'FE': 35
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
        """构建残基的局部坐标系 - 简化版本用于快速处理"""
        frames = torch.eye(3).unsqueeze(0).repeat(num_residues, 1, 1).to(pos_CA.device)
        
        # 使用CA-CB向量作为主要方向的简化坐标系构建
        for i in range(num_residues):
            if i < len(pos_CA):
                # 使用CA-CB向量作为x轴（如果没有CB，使用下一个CA）
                if torch.norm(pos_CB[i]) > 1e-6:
                    e1 = pos_CB[i] - pos_CA[i]
                elif i < len(pos_CA) - 1:
                    e1 = pos_CA[i+1] - pos_CA[i]
                else:
                    e1 = torch.tensor([1.0, 0.0, 0.0], device=pos_CA.device)
                
                e1_norm = torch.norm(e1)
                if e1_norm > 1e-6:
                    e1 = e1 / e1_norm
                    
                    # 构建正交基
                    temp_vec = torch.tensor([0.0, 0.0, 1.0], device=pos_CA.device)
                    e2 = torch.cross(e1, temp_vec)
                    e2_norm = torch.norm(e2)
                    if e2_norm < 1e-6:
                        temp_vec = torch.tensor([0.0, 1.0, 0.0], device=pos_CA.device)
                        e2 = torch.cross(e1, temp_vec)
                        e2_norm = torch.norm(e2)
                    
                    if e2_norm > 1e-6:
                        e2 = e2 / e2_norm
                        e3 = torch.cross(e1, e2)
                        frames[i] = torch.stack([e1, e2, e3], dim=1)
        
        return frames


class GeometricMessagePassing(MessagePassing):
    """几何感知的消息传递层 - 原子级图卷积 + 残基级注意力"""
    
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(GeometricMessagePassing, self).__init__()
        
        # 存储输出维度
        self.out_channels = out_channels
        
        # 确保输入输出维度匹配
        self.msg_proj = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),  # 128 -> 64
            nn.LayerNorm(out_channels),
            nn.ReLU()
        )
        
        self.update_proj = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),  # 128 -> 64
            nn.LayerNorm(out_channels),
            nn.ReLU()
        )
        
        # 原子到残基的聚合模块
        self.atom_gather = AtomPositionGather(out_channels)
        
        # 残基级位置感知层
        self.position_aware = PositionAwareLayer(
            hidden_dim=out_channels,
            num_heads=num_heads
        )
        
        self.attention = nn.MultiheadAttention(
            out_channels, num_heads, batch_first=True
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


class GeometricGNN(nn.Module):
    """修复维度匹配的几何GNN - 压缩特征维度版本"""
    
    def __init__(self,
                 node_feat_dim: int = 64,   # 压缩到64维
                 edge_feat_dim: int = 64,   # 压缩到64维
                 hidden_dim: int = 64,      # 压缩到64维
                 num_relation_types: int = 3,
                 num_heads: int = 4):
        super(GeometricGNN, self).__init__()
        
        # 修正输入维度匹配
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),  # 64 -> 64
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 边特征投影修正为正确维度
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),  # 64 -> 64
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 多关系消息传递层
        self.relation_weights = nn.Parameter(torch.ones(num_relation_types, hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.relation_weights)
        
        # 添加BatchNorm用于GNN层之间的正则化
        self.gnn_batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(3)
        ])
        
        # 几何感知的GNN层
        self.gnn_layers = nn.ModuleList([
            GeometricMessagePassing(hidden_dim, hidden_dim, num_heads)
            for _ in range(3)
        ])
        
        # 几何注意力（类似GearBind的边级消息传递）
        self.geometric_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # 全局几何信息提取
        self.global_geom_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # 包含坐标信息
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 突变位点特征提取
        self.mutation_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
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
            
            # 应用BatchNorm在每个GNN层之后
            if x.dim() == 2:  # 确保是2D张量才能使用BatchNorm1d
                x = self.gnn_batch_norms[i](x)
            
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
        
    
    # 是否需要加入突变残疾指示？
    def _atom_to_feature(self, atom, atom_name_to_id, residue_feature=None):
        """
        原子转换为特征向量 - 按照GearBind模式
        
        Args:
            atom: Bio.PDB Atom对象
            atom_name_to_id: dict 原子名称到ID的映射
            residue_feature: torch.Tensor 可选的残基特征
        
        Returns:
            torch.Tensor 原子特征向量
        """
        # 1. 原子类型one-hot编码 (37维，按照GearBind标准)
        atom_types = [
            'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
            'CD1', 'CD2', 'ND1', 'ND2', 'NE1', 'NE2', 'NZ', 'OE1', 'OE2', 'OD1',
            'OD2', 'CE', 'CE1', 'CE2', 'CE3', 'CZ', 'CZ2', 'CZ3', 'CH2', 'NH1',
            'NH2', 'OH', 'SD', 'SG', 'FE'
        ]
        
        atom_name = atom.name.strip()
        if atom_name in atom_name_to_id:
            atom_type_idx = atom_name_to_id[atom_name]
        elif atom_name in atom_types:
            atom_type_idx = atom_types.index(atom_name)
        else:
            atom_type_idx = 0  # 默认为N原子
        
        atom_type_onehot = torch.zeros(37, device='cpu')
        atom_type_onehot[atom_type_idx] = 1.0
        
        # 2. 原子位置归一化特征
        atom_pos = torch.tensor(atom.coord, dtype=torch.float32, device='cpu')
        normalized_pos = atom_pos / 10.0  # 归一化到[-1, 1]范围
        
        # 3. 主链/侧链标识
        backbone_atoms = ['N', 'CA', 'C', 'O']
        is_backbone = 1.0 if atom_name in backbone_atoms else 0.0
        
        # 4. 残基类型特征（如果提供）
        if residue_feature is not None:
            residue_feat = residue_feature.to('cpu')
        else:
            # 简单的残基类型编码
            aa_types = 'ACDEFGHIKLMNPQRSTVWY'
            if hasattr(atom, 'parent') and hasattr(atom.parent, 'resname'):
                resname = atom.parent.resname
                if len(resname) == 3 and resname in aa_types:
                    aa_idx = aa_types.index(resname)
                    residue_feat = torch.zeros(20, device='cpu')
                    residue_feat[aa_idx] = 1.0
                else:
                    residue_feat = torch.zeros(20, device='cpu')
            else:
                residue_feat = torch.zeros(20, device='cpu')
        
        # 5. 位置编码（使用正弦/余弦编码）
        pos_encoding = torch.zeros(64 - 37 - 3 - 1 - 20, device='cpu')  # 剩余维度用于位置编码
        for i in range(pos_encoding.shape[0] // 2):
            freq = (i + 1) * np.pi / 5.0
            pos_encoding[i*2] = torch.sin(atom_pos[0] * freq)
            pos_encoding[i*2+1] = torch.cos(atom_pos[0] * freq)
        
        # 6. 组合所有特征
        atom_feature = torch.cat([
            atom_type_onehot,      # 37维 原子类型
            normalized_pos,        # 3维 位置
            torch.tensor([is_backbone], device='cpu'),  # 1维 主链标识
            residue_feat,          # 20维 残基类型
            pos_encoding           # 剩余维度 位置编码
        ], dim=0)
        
        # 确保特征维度为64
        if atom_feature.shape[0] < 64:
            padding = torch.zeros(64 - atom_feature.shape[0], device='cpu')
            atom_feature = torch.cat([atom_feature, padding], dim=0)
        elif atom_feature.shape[0] > 64:
            atom_feature = atom_feature[:64]
        
        return atom_feature
    
    
    def _build_unified_residue_frames(self, all_residues):
        """
        统一的残基坐标系构建方法 - 供其他模块调用
        按照GearBind标准使用N-CA-C坐标系
        """
        residue_frames = {}
        residue_atom_positions = {}
        
        for residue in all_residues:
            residue_key = f"{residue.parent.id}_{residue.id[1]}"
            
            # 收集所有原子位置
            atom_positions = {}
            backbone_atoms = {}
            
            for atom in residue:
                pos = torch.tensor(atom.coord, dtype=torch.float32)
                atom_positions[atom.name] = pos
                
                if atom.name in ['N', 'CA', 'C']:
                    backbone_atoms[atom.name] = pos
            
            residue_atom_positions[residue_key] = atom_positions
            
            # 只有当拥有完整主链（N, CA, C）时才构建标准坐标系
            if len(backbone_atoms) == 3:
                N_pos = backbone_atoms['N']
                CA_pos = backbone_atoms['CA'] 
                C_pos = backbone_atoms['C']
                
                # 构建标准右手坐标系
                # x轴: CA -> N
                e_x = N_pos - CA_pos
                e_x_norm = torch.norm(e_x)
                if e_x_norm > 1e-6:
                    e_x = e_x / e_x_norm
                
                # xy平面中的向量: CA -> C
                e_xy = C_pos - CA_pos
                
                # 正交化e_xy使其垂直于e_x
                dot_product = torch.dot(e_x, e_xy)
                e_y = e_xy - dot_product * e_x
                e_y_norm = torch.norm(e_y)
                if e_y_norm > 1e-6:
                    e_y = e_y / e_y_norm
                
                # z轴: e_x × e_y (右手坐标系)
                e_z = torch.cross(e_x, e_y)
                e_z_norm = torch.norm(e_z)
                if e_z_norm > 1e-6:
                    e_z = e_z / e_z_norm
                
                # 构建旋转矩阵 [3, 3]
                rotation_matrix = torch.stack([e_x, e_y, e_z], dim=1)
                residue_frames[residue_key] = rotation_matrix
            else:
                # 如果缺少主链原子，使用单位矩阵
                residue_frames[residue_key] = torch.eye(3)
        
        return residue_frames, residue_atom_positions