"""
几何特征提取和处理模块 - 重构版本
合并相关类，简化架构
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, MessagePassing
import numpy as np
from dataclasses import dataclass


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


class UnifiedResidueGeometry(nn.Module):
    """
    统一的残基几何处理类
    合并了 AtomPositionGather 和 PositionAwareLayer 的功能
    """
    
    def __init__(self, hidden_dim=96, num_heads=4):
        super(UnifiedResidueGeometry, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # GearBind标准的37维原子类型映射
        self.atom_name2id = {
            'N': 0, 'CA': 1, 'C': 2, 'CB': 3, 'O': 4, 'CG': 5, 'CG1': 6, 'CG2': 7, 
            'OG': 8, 'OG1': 9, 'SG': 10, 'CD': 11, 'CD1': 12, 'CD2': 13, 'ND1': 14, 
            'ND2': 15, 'NE1': 16, 'NE2': 17, 'NZ': 18, 'OE1': 19, 'OE2': 20, 'OD1': 21,
            'OD2': 22, 'CE': 23, 'CE1': 24, 'CE2': 25, 'CE3': 26, 'CZ': 27, 'CZ2': 28, 
            'CZ3': 29, 'CH2': 30, 'NH1': 31, 'NH2': 32, 'OH': 33, 'SD': 34, 'FE': 35
        }
        
        # 位置感知注意力机制（原 PositionAwareLayer 的功能）
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        spatial_dim = num_heads * 7  # 3(points) + 1(distance) + 3(direction)
        self.out_transform = nn.Sequential(
            nn.Linear(hidden_dim + spatial_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def build_residue_frames(self, all_residues):
        """构建统一的残基坐标系"""
        residue_frames = {}
        residue_atom_positions = {}
        
        for residue in all_residues:
            residue_key = f"{residue.parent.id}_{residue.id[1]}"
            atom_positions = {}
            backbone_atoms = {}
            
            for atom in residue:
                pos = torch.tensor(atom.coord, dtype=torch.float32)
                atom_positions[atom.name] = pos
                
                if atom.name in ['N', 'CA', 'C']:
                    backbone_atoms[atom.name] = pos
            
            residue_atom_positions[residue_key] = atom_positions
            
            if len(backbone_atoms) == 3:
                N_pos = backbone_atoms['N']
                CA_pos = backbone_atoms['CA'] 
                C_pos = backbone_atoms['C']
                
                # 构建标准右手坐标系
                e_x = N_pos - CA_pos
                e_x_norm = torch.norm(e_x)
                if e_x_norm > 1e-6:
                    e_x = e_x / e_x_norm
                
                e_xy = C_pos - CA_pos
                dot_product = torch.dot(e_x, e_xy)
                e_y = e_xy - dot_product * e_x
                e_y_norm = torch.norm(e_y)
                if e_y_norm > 1e-6:
                    e_y = e_y / e_y_norm
                
                e_z = torch.cross(e_x, e_y)
                e_z_norm = torch.norm(e_z)
                if e_z_norm > 1e-6:
                    e_z = e_z / e_z_norm
                
                rotation_matrix = torch.stack([e_x, e_y, e_z], dim=1)
                residue_frames[residue_key] = rotation_matrix
            else:
                residue_frames[residue_key] = torch.eye(3)
        
        return residue_frames, residue_atom_positions
    
    def atom_to_feature(self, atom, residue_feature=None):
        """原子转换为特征向量 - GearBind模式"""
        # 37维原子类型one-hot编码
        atom_name = atom.name.strip()
        atom_type_idx = self.atom_name2id.get(atom_name, 0)
        
        atom_type_onehot = torch.zeros(37, device='cpu')
        atom_type_onehot[atom_type_idx] = 1.0
        
        # 位置归一化
        atom_pos = torch.tensor(atom.coord, dtype=torch.float32, device='cpu')
        normalized_pos = atom_pos / 10.0
        
        # 主链/侧链标识
        backbone_atoms = ['N', 'CA', 'C', 'O']
        is_backbone = 1.0 if atom_name in backbone_atoms else 0.0
        
        # 残基类型特征
        if residue_feature is not None:
            residue_feat = residue_feature.to('cpu')
        else:
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
        
        # 位置编码
        pos_encoding = torch.zeros(64 - 37 - 3 - 1 - 20, device='cpu')
        for i in range(pos_encoding.shape[0] // 2):
            freq = (i + 1) * np.pi / 5.0
            pos_encoding[i*2] = torch.sin(atom_pos[0] * freq)
            pos_encoding[i*2+1] = torch.cos(atom_pos[0] * freq)
        
        # 组合特征
        atom_feature = torch.cat([
            atom_type_onehot,      # 37维
            normalized_pos,        # 3维
            torch.tensor([is_backbone], device='cpu'),  # 1维
            residue_feat,          # 20维
            pos_encoding           # 剩余维度
        ], dim=0)
        
        # 确保特征维度为96 (从64提升到96)
        target_dim = 96
        if atom_feature.shape[0] < target_dim:
            padding = torch.zeros(target_dim - atom_feature.shape[0], device='cpu')
            atom_feature = torch.cat([atom_feature, padding], dim=0)
        elif atom_feature.shape[0] > target_dim:
            atom_feature = atom_feature[:target_dim]
        
        return atom_feature
    
    def aggregate_atoms_to_residues(self, graph_data):
        """原子级特征聚合为残基级特征"""
        device = graph_data.node_features.device
        
        # 构建原子到残基的映射
        atom2residue = torch.zeros(len(graph_data.atom_names), dtype=torch.long, device=device)
        residue_info = {}
        current_residue_idx = 0
        
        for i, residue_idx in enumerate(graph_data.residue_indices):
            if residue_idx not in residue_info:
                residue_info[residue_idx] = current_residue_idx
                current_residue_idx += 1
            atom2residue[i] = residue_info[residue_idx]
        
        num_residues = len(residue_info)
        
        # 聚合特征（使用CA原子作为代表）
        ca_mask = torch.tensor([name == "CA" for name in graph_data.atom_names], device=device)
        ca_indices = torch.where(ca_mask)[0]
        
        residue_features = torch.zeros(num_residues, self.hidden_dim, device=device)
        residue_positions_CA = torch.zeros(num_residues, 3, device=device)
        residue_positions_CB = torch.zeros(num_residues, 3, device=device)
        
        if len(ca_indices) > 0:
            ca_atom2residue = atom2residue[ca_mask]
            ca_features = graph_data.node_features[ca_mask]
            
            for i in range(num_residues):
                mask = ca_atom2residue == i
                if mask.any():
                    residue_features[i] = ca_features[mask].mean(dim=0)
        
        # 提取位置
        for atom_idx in ca_indices:
            res_idx = atom2residue[atom_idx].item()
            if res_idx < num_residues:
                residue_positions_CA[res_idx] = graph_data.node_positions[atom_idx]
        
        cb_mask = torch.tensor([name == "CB" for name in graph_data.atom_names], device=device)
        cb_indices = torch.where(cb_mask)[0]
        for atom_idx in cb_indices:
            res_idx = atom2residue[atom_idx].item()
            if res_idx < num_residues:
                residue_positions_CB[res_idx] = graph_data.node_positions[atom_idx]
        
        # 对于没有CB的残基，使用CA位置
        no_cb_mask = (residue_positions_CB.abs().sum(dim=1) < 1e-6)
        residue_positions_CB[no_cb_mask] = residue_positions_CA[no_cb_mask]
        
        return residue_features, residue_positions_CA, residue_positions_CB, atom2residue
    
    def apply_positional_attention(self, residue_features, pos_CA, pos_CB, mask):
        """应用残基级位置感知注意力"""
        batch_size, num_residues, _ = residue_features.shape
        
        # 构建残基坐标系
        frames = self._build_frames(pos_CA, pos_CB)
        
        # 注意力计算
        query = self.query(residue_features).view(batch_size, num_residues, self.num_heads, -1)
        key = self.key(residue_features).view(batch_size, num_residues, self.num_heads, -1)
        value = self.value(residue_features).view(batch_size, num_residues, self.num_heads, -1)
        
        logits = torch.einsum('blhd,bkhd->blkh', query, key)
        if mask is not None:
            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask_expanded = mask_2d.unsqueeze(-1).expand(-1, -1, -1, self.num_heads)
            logits = logits.masked_fill(~mask_expanded, float('-inf'))
        
        alpha = torch.softmax(logits, dim=2)
        feat_node = torch.einsum('blkh,bkhd->blhd', alpha, value).flatten(-2)
        
        # 位置相关特征
        rel_pos = pos_CB.unsqueeze(2) - pos_CA.unsqueeze(1)
        atom_pos_bias = torch.einsum('blkh,blkd->blhd', alpha, rel_pos)
        feat_distance = atom_pos_bias.norm(dim=-1, keepdim=True)
        feat_points = torch.einsum('blij,blhj->blhi', frames, atom_pos_bias)
        feat_direction = feat_points / (feat_points.norm(dim=-1, keepdim=True) + 1e-10)
        
        # 组合空间特征
        feat_spatial = torch.cat([
            feat_points.flatten(-2),
            feat_distance.flatten(-2),
            feat_direction.flatten(-2),
        ], dim=-1)
        
        feat_all = torch.cat([feat_node, feat_spatial], dim=-1)
        feat_all = self.out_transform(feat_all)
        feat_all = torch.where(mask.unsqueeze(-1), feat_all, torch.zeros_like(feat_all))
        
        return self.layer_norm(residue_features + feat_all)
    
    def _build_frames(self, pos_CA, pos_CB):
        """构建残基坐标系"""
        batch_size, num_residues, _ = pos_CA.shape
        frames = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_residues, 1, 1).to(pos_CA.device)
        
        for i in range(num_residues):
            for b in range(batch_size):
                if i < num_residues - 1:
                    e1 = pos_CB[b, i] - pos_CA[b, i]
                    e1_norm = torch.norm(e1)
                    if e1_norm > 1e-6:
                        e1 = e1 / e1_norm
                        
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
                            frames[b, i] = torch.stack([e1, e2, e3], dim=1)
        
        return frames


class UnifiedGeometricProcessor(nn.Module):
    """
    统一的几何处理器
    合并了 KNNMutationSite, InterfaceFeatureExtractor, 和部分几何处理功能
    """
    
    def __init__(self, hidden_dim=96, cutoff_distance=8.0, k_neighbors=20, knn_mutation_k=256, num_heads=4):
        super(UnifiedGeometricProcessor, self).__init__()
        
        self.cutoff_distance = cutoff_distance
        self.k_neighbors = k_neighbors
        self.knn_mutation_k = knn_mutation_k
        self.hidden_dim = hidden_dim
        
        # 统一的残基几何处理器
        self.residue_geometry = UnifiedResidueGeometry(hidden_dim, num_heads)
        
        # KNN突变位点选择参数（集成到这里）
        self.knn_k = knn_mutation_k
    
    def extract_geometric_features(self, all_residues, mutation=None):
        """提取几何特征的主要接口"""
        # 1. 构建残基坐标系
        residue_frames, residue_atom_positions = self.residue_geometry.build_residue_frames(all_residues)
        
        # 2. 提取原子特征
        node_features = []
        node_positions = []
        atom_names = []
        residue_indices = []
        is_mutation_list = []
        
        for residue in all_residues:
            for atom in residue:
                # 原子特征
                atom_feat = self.residue_geometry.atom_to_feature(atom)
                node_features.append(atom_feat.detach().cpu().numpy())
                
                # 原子位置
                position = atom.coord
                node_positions.append(position)
                atom_names.append(atom.name)
                residue_indices.append(f"{residue.parent.id}_{residue.id[1]}")
                
                # 判断是否为突变位点
                is_mutation_node = False
                if mutation and len(mutation) >= 3:
                    mutation_pos = int(mutation[1:-1])
                    mutation_chain = mutation[0]
                    if (residue.id[1] == mutation_pos and 
                        residue.parent.id == mutation_chain and
                        atom.name == "CA"):
                        is_mutation_node = True
                
                is_mutation_list.append(is_mutation_node)
        
        return {
            'node_features': np.array(node_features, dtype=np.float32),
            'node_positions': np.array(node_positions, dtype=np.float32),
            'atom_names': atom_names,
            'residue_indices': residue_indices,
            'is_mutation_list': is_mutation_list,
            'residue_frames': residue_frames,
            'residue_atom_positions': residue_atom_positions
        }
    
    def apply_knn_selection(self, node_positions_tensor, atom_names, is_mutation_tensor, batch):
        """应用KNN突变位点选择"""
        device = node_positions_tensor.device
        
        # 找到突变位点的CA原子
        ca_mask = torch.tensor([name == "CA" for name in atom_names], device=device, dtype=torch.bool)
        mutation_ca_mask = is_mutation_tensor & ca_mask
        
        if not mutation_ca_mask.any():
            return torch.ones(len(atom_names), dtype=torch.bool, device=device)
        
        # 计算距离并选择最近的k个节点
        center_positions = node_positions_tensor[mutation_ca_mask]
        mut2graph = batch[mutation_ca_mask]
        
        center_indices = nearest(node_positions_tensor, center_positions, batch, mut2graph)
        dist_to_center = ((node_positions_tensor - center_positions[center_indices])**2).sum(-1)
        dist_to_center[mutation_ca_mask] = 0.0
        
        # 为每个batch选择最近的k个节点
        num_graphs = batch.max().item() + 1
        selected_indices = []
        
        for graph_id in range(num_graphs):
            graph_mask = batch == graph_id
            graph_nodes = torch.where(graph_mask)[0]
            
            if len(graph_nodes) <= self.knn_k:
                selected_indices.append(graph_nodes)
            else:
                graph_distances = dist_to_center[graph_mask]
                _, local_selected = torch.topk(graph_distances, self.knn_k, largest=False)
                global_selected = graph_nodes[local_selected]
                selected_indices.append(global_selected)
        
        # 创建节点掩码
        node_mask = torch.zeros(len(atom_names), dtype=torch.bool, device=device)
        if selected_indices:
            all_selected = torch.cat(selected_indices)
            node_mask[all_selected] = True
        
        return node_mask


# 简化版本的几何GNN
class SimplifiedGeometricGNN(nn.Module):
    """
    简化的几何GNN
    统一了 GeometricMessagePassing 和 GeometricGNN 的功能
    """
    
    def __init__(self, node_feat_dim=96, edge_feat_dim=96, hidden_dim=96, num_heads=4):
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
            try:
                residue_features, pos_CA, pos_CB, atom2residue = self.geometric_processor.aggregate_atoms_to_residues(
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
                        mask = torch.ones(1, residue_features.shape[0], dtype=torch.bool, device=updated_x.device)
                        
                        # 应用位置感知注意力
                        enhanced_residue = self.geometric_processor.apply_positional_attention(
                            residue_features_batch, pos_CA_batch, pos_CB_batch, mask
                        )
                        
                        # 将增强特征映射回原子级（简单版本）
                        for i, res_idx in enumerate(torch.unique(atom2residue)):
                            if res_idx < enhanced_residue.shape[1]:
                                enhanced_feat = enhanced_residue[0, i]
                                atom_mask = atom2residue == res_idx
                                if atom_mask.any():
                                    updated_x[atom_mask] = updated_x[atom_mask] + enhanced_feat * 0.1
            except Exception as e:
                # 如果几何处理失败，继续使用原子级特征
                pass
        
        # 全局池化
        if graph_data.batch.dtype != torch.int64:
            graph_data.batch = graph_data.batch.long()
        
        graph_rep = global_mean_pool(updated_x, graph_data.batch)
        
        # 输出投影
        output = self.output_proj(graph_rep)
        
        return output


# 模块级函数
def build_unified_residue_frames(all_residues):
    """模块级统一的残基坐标系构建函数"""
    processor = UnifiedResidueGeometry()
    return processor.build_residue_frames(all_residues)