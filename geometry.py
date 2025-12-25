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


class UnifiedResidueGeometry(nn.Module):
    """
    统一的残基几何处理类
    基于 GearBind 的几何特征提取方法
    """
    
    def __init__(self, hidden_dim=96, num_heads=4):
        super(UnifiedResidueGeometry, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 37维原子类型映射
        self.atom_name2id = {
            'N': 0, 'CA': 1, 'C': 2, 'CB': 3, 'O': 4, 'CG': 5, 'CG1': 6, 'CG2': 7, 
            'OG': 8, 'OG1': 9, 'SG': 10, 'CD': 11, 'CD1': 12, 'CD2': 13, 'ND1': 14, 
            'ND2': 15, 'NE1': 16, 'NE2': 17, 'NZ': 18, 'OE1': 19, 'OE2': 20, 'OD1': 21,
            'OD2': 22, 'CE': 23, 'CE1': 24, 'CE2': 25, 'CE3': 26, 'CZ': 27, 'CZ2': 28, 
            'CZ3': 29, 'CH2': 30, 'NH1': 31, 'NH2': 32, 'OH': 33, 'SD': 34, 'FE': 35
        }
        
        # DDGAttention 机制（基于 GearBind）
        self.attention = DDGAttention(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            value_dim=16,
            query_key_dim=16,
            num_heads=num_heads
        )
    
    def from_3_points(self, p_x_axis, origin, p_xy_plane, eps=1e-10):
        """
        从三个点构建局部坐标系
        基于 GearBind 的 from_3_points 实现
        Args:
            p_x_axis: [*, 3] coordinates (N atom)
            origin: [*, 3] coordinates (CA atom) 
            p_xy_plane: [*, 3] coordinates (C atom)
            eps: Small epsilon value
        Returns:
            旋转矩阵: [* , 3, 3]
        """
        e_x = p_x_axis - origin
        e_x_norm = torch.norm(e_x, dim=-1, keepdim=True)
        e_x = e_x / (e_x_norm + eps)
        
        e_xy = p_xy_plane - origin
        dot_product = torch.sum(e_x * e_xy, dim=-1, keepdim=True)
        e_y = e_xy - dot_product * e_x
        e_y_norm = torch.norm(e_y, dim=-1, keepdim=True)
        e_y = e_y / (e_y_norm + eps)
        
        e_z = torch.cross(e_x, e_y, dim=-1)
        e_z_norm = torch.norm(e_z, dim=-1, keepdim=True)
        e_z = e_z / (e_z_norm + eps)
        
        # 组合为旋转矩阵: [* , 3, 3]
        rotation_matrix = torch.cat([e_x, e_y, e_z], dim=-1)
        rotation_matrix = rotation_matrix.view(*e_x.shape[:-1], 3, 3)
        
        return rotation_matrix

    def build_residue_frames(self, all_residues):
        """基于 GearBind 方法构建残基坐标系"""
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
                    
                # 使用 from_3_points 方法构建局部坐标系
                frame = self.from_3_points(
                    N_pos.unsqueeze(0),  # [1, 3]
                    CA_pos.unsqueeze(0), # [1, 3] 
                    C_pos.unsqueeze(0)   # [1, 3]
                ).transpose(-1, -2)     # [1, 3, 3] -> [1, 3, 3]
                    
                residue_frames[residue_key] = frame.squeeze(0)  # [3, 3]
            else:
                residue_frames[residue_key] = torch.eye(3)
            
        return residue_frames, residue_atom_positions
    
    def atom_to_feature(self, atom, residue_feature=None):
        """原子转换为特征向量"""
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
        """
        基于 GearBind 方法聚合原子特征到残基级别
        """
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
        
        # 初始化原子位置数组 (基于 GearBind 的 atom_pos)
        atom_pos = torch.full((num_residues, len(self.atom_name2id), 3), float("inf"), 
                            dtype=torch.float, device=device)
        atom_pos_mask = torch.zeros((num_residues, len(self.atom_name2id)), 
                                    dtype=torch.bool, device=device)
        
        # 填充原子位置信息
        for i, (atom_name, residue_idx) in enumerate(zip(graph_data.atom_names, graph_data.residue_indices)):
            if atom_name in self.atom_name2id:
                res_idx = residue_info[residue_idx]
                atom_idx = self.atom_name2id[atom_name]
                atom_pos[res_idx, atom_idx] = graph_data.node_positions[i]
                atom_pos_mask[res_idx, atom_idx] = True
        
        # 获取 CA 位置
        pos_CA = atom_pos[:, self.atom_name2id["CA"]]  # [num_residues, 3]
        
        # 计算 CB 位置（基于 GearBind 方法）
        pos_CB = torch.where(
            atom_pos_mask[:, self.atom_name2id["CB"], None].expand(-1, 3),
            atom_pos[:, self.atom_name2id["CB"]],
            pos_CA  # 对于甘氨酸等缺少CB原子的残基，使用CA位置
        )
        
        # 构建局部坐标系 (基于 GearBind 的 frame 构建方法)
        frames = torch.eye(3).unsqueeze(0).repeat(num_residues, 1, 1).to(device)
        for res_idx in range(num_residues):
            if (atom_pos_mask[res_idx, self.atom_name2id["N"]] and 
                atom_pos_mask[res_idx, self.atom_name2id["CA"]] and 
                atom_pos_mask[res_idx, self.atom_name2id["C"]]):
                
                frame = self.from_3_points(
                    atom_pos[res_idx, self.atom_name2id["N"]].unsqueeze(0),  # [1, 3]
                    atom_pos[res_idx, self.atom_name2id["CA"]].unsqueeze(0), # [1, 3] 
                    atom_pos[res_idx, self.atom_name2id["C"]].unsqueeze(0)   # [1, 3]
                ).transpose(-1, -2)  # [1, 3, 3]
                
                frames[res_idx] = frame.squeeze(0)
        
        # 聚合原子特征到残基特征（使用CA原子）
        ca_mask = torch.tensor([name == "CA" for name in graph_data.atom_names], device=device)
        ca_indices = torch.where(ca_mask)[0]
        
        residue_features = torch.zeros(num_residues, self.hidden_dim, device=device)
        
        if len(ca_indices) > 0:
            ca_atom2residue = atom2residue[ca_mask]
            ca_features = graph_data.node_features[ca_mask]
                
            for i in range(num_residues):
                mask = ca_atom2residue == i
                if mask.any():
                    residue_features[i] = ca_features[mask].mean(dim=0)
        
        return residue_features, pos_CA, pos_CB, atom2residue, frames
    
    def apply_positional_attention(self, residue_features, pos_CA, pos_CB, frames, mask):
        """
        基于 GearBind DDGAttention 的位置感知注意力
        """
        # 使用 DDGAttention 进行几何特征融合
        enhanced_features = self.attention(residue_features, pos_CA, pos_CB, frames, mask)
        
        return enhanced_features


class UnifiedGeometricProcessor(nn.Module):
    """
    统一的几何处理器
    合并了 KNNMutationSite, InterfaceFeatureExtractor, 和部分几何处理功能
    """
    
    def __init__(self, hidden_dim=96, knn_mutation_k=512, num_heads=4):
        super(UnifiedGeometricProcessor, self).__init__()
        
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
        
        center_positions = node_positions_tensor[mutation_ca_mask]
        mut2graph = batch[mutation_ca_mask]
        
        k_select = min(self.knn_k, len(node_positions_tensor))
        if len(center_positions) > 0 and len(node_positions_tensor) > k_select:
            # 直接调用knn处理所有中心点
            knn_indices = knn(node_positions_tensor, center_positions, k_select, batch, mut2graph)
            # knn_indices shape: [num_centers, k]
            
            # 合并所有选中的索引
            selected = knn_indices.flatten()
            # 包含突变位点本身
            mutation_indices = torch.where(mutation_ca_mask)[0]
            all_selected = torch.unique(torch.cat([selected, mutation_indices]))
        else:
            all_selected = torch.arange(len(atom_names), device=device)
        
        node_mask = torch.zeros(len(atom_names), dtype=torch.bool, device=device)
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


# 模块级函数
def build_unified_residue_frames(all_residues):
    """模块级统一的残基坐标系构建函数"""
    processor = UnifiedResidueGeometry()
    return processor.build_residue_frames(all_residues)