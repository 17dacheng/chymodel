import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Optional, Any, Union
from dataclasses import dataclass
from feature_extractor import RealFeatureExtractor
# 导入几何特征模块
from geometry import (
    InterfaceGraphData, SimplifiedGeometricGNN, UnifiedGeometricProcessor, UnifiedResidueGeometry,
    nearest, build_unified_residue_frames
)
# 导入Bio.PDB相关模块
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from pdb import set_trace


class CHYModelWithGeometric(nn.Module):    
    def __init__(self, 
                 esm_dim: int = 1280,
                 foldx_dim: int = 22,
                 geometric_dim: int = 16,  # 几何特征维度
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 3):
        super(CHYModelWithGeometric, self).__init__()
        
        self.esm_dim = esm_dim
        self.foldx_dim = foldx_dim
        self.geometric_dim = geometric_dim
        
        # FoldX特征投影 - 降低维度
        self.foldx_projection = nn.Sequential(
            nn.Linear(foldx_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 64),  # 降低到64维特征
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # ESM特征处理 - 降低维度
        self.esm_projection = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim // 4),  # 降低到128维
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim // 4,  # 降低到128维
            nhead=num_heads // 2,     # 减少注意力头数
            dim_feedforward=hidden_dim // 2,
            dropout=0,  # 不使用dropout
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 几何特征处理（GearBind-inspired）- 增强特征维度
        self.geometric_gnn_wt = SimplifiedGeometricGNN(
            node_feat_dim=96,   # 96维原子特征输入
            edge_feat_dim=96,   # 96维边特征
            hidden_dim=96,       # 96维隐藏层
            num_heads=4
        )
        
        # 为突变体也创建一个GNN（需要这个变量）
        self.geometric_gnn_mt = SimplifiedGeometricGNN(
            node_feat_dim=96,   # 96维原子特征输入
            edge_feat_dim=96,   # 96维边特征
            hidden_dim=96,       # 96维隐藏层
            num_heads=4
        )
        
        # 特征融合层 - 降低特征维度
        self.feature_fusion = nn.Sequential(
            nn.Linear(
                hidden_dim // 4 + 64 + 128 + 128,  # seq_rep + foldx_expanded + wt_geom_rep + mt_geom_rep = 128 + 64 + 128 + 128 = 448
                hidden_dim // 2
            ),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )
        
        # 对称性处理（类似GearBind的Anti-symmetric处理）
        self.antisymmetric_layer = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.Tanh()
        )
        
        # 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1)
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
            elif isinstance(module, nn.BatchNorm1d):
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
        forward_start_time = time.time()
        batch_size = esm_embeddings.shape[0]
        
        # 1. 处理FoldX特征
        foldx_start = time.time()
        foldx_proj = self.foldx_projection(foldx_features)  # [batch_size, esm_dim//2]
        foldx_time = time.time() - foldx_start
        
        # 2. 处理ESM特征
        esm_start = time.time()
        esm_proj = self.esm_projection(esm_embeddings)  # [batch_size, seq_len, hidden_dim//4]
        esm_time = time.time() - esm_start
        
        # 3. Transformer编码
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                src_key_padding_mask = (attention_mask == 0)
            else:
                src_key_padding_mask = (attention_mask.squeeze(1) == 0)
        
        transformer_output = self.transformer_encoder(
            esm_proj,
            src_key_padding_mask=src_key_padding_mask if attention_mask is not None else None
        )  # [batch_size, seq_len, hidden_dim//4=128]
        
        # 全局平均池化
        seq_rep = transformer_output.mean(dim=1)  # [batch_size, hidden_dim//4]
        
        # 4. 处理几何特征（WT和MT分别处理）
        geom_start = time.time()
        wt_geom_rep = self.geometric_gnn_wt(wt_graph_data)  # [2, hidden_dim//4] (WT=0, MT=1)
        mt_geom_rep = self.geometric_gnn_mt(mt_graph_data)  # [2, hidden_dim//4] (WT=0, MT=1)
        geom_time = time.time() - geom_start
        
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
        
        # 确保几何特征维度正确（当前已经是128维）
        # 几何特征已经通过SimplifiedGeometricGNN输出为128维
        
        # 确保FoldX特征维度正确
        if foldx_proj.shape[-1] != 64:
            foldx_expansion = nn.Linear(foldx_proj.shape[-1], 64).to(seq_rep.device)
            foldx_expanded = foldx_expansion(foldx_proj)
        else:
            foldx_expanded = foldx_proj
        
        # 拼接所有特征
        combined_features = torch.cat([
            seq_rep,  # 序列特征 [batch_size, hidden_dim//4=128]
            foldx_expanded,  # FoldX特征 [batch_size, 64]
            wt_geom_rep,  # WT几何特征 [batch_size, 128]
            mt_geom_rep   # MT几何特征 [batch_size, 128]
        ], dim=-1)  # [batch_size, 128 + 64 + 128 + 128 = 448]
                
        fused_features = self.feature_fusion(combined_features)  # [batch_size, hidden_dim//4]
        
        # 6. 对称性处理（ΔΔG应该是反对称的）
        antisym_features = self.antisymmetric_layer(fused_features)
        # 通过WT和MT特征的差异来实现反对称性
        geom_diff = mt_geom_rep - wt_geom_rep  # [batch_size, 128]
        
        # 扩展几何差异到与antisym_features相同的维度
        if geom_diff.shape[-1] != antisym_features.shape[-1]:
            geom_diff_expansion = nn.Linear(geom_diff.shape[-1], antisym_features.shape[-1]).to(antisym_features.device)
            geom_diff = geom_diff_expansion(geom_diff)
        
        antisym_features = antisym_features + geom_diff
        
        # 7. 回归预测
        ddg_pred = self.regression_head(antisym_features)  # [batch_size, 1]
        
        # 打印各部分处理时间
        total_forward_time = time.time() - forward_start_time
        print(f"前向传播时间统计:")
        print(f"  - FoldX特征处理: {foldx_time:.3f}秒")
        print(f"  - ESM特征处理: {esm_time:.3f}秒")
        print(f"  - 几何特征处理: {geom_time:.3f}秒")
        print(f"  - 总前向传播时间: {total_forward_time:.3f}秒")
        
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
        
        # 自动检测设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化特征提取器
        self.feature_extractor = RealFeatureExtractor(
            pdb_base_path=str(pdb_base_path),
            cache_dir=str(cache_dir),
            use_esm=True,
            weights_path=str(weights_path)
        )
        
        # 初始化几何特征提取器
        if use_geometric:
            self.geometric_extractor = UnifiedGeometricProcessor()
        
        # 初始化模型
        if use_geometric:
            self.model = CHYModelWithGeometric()
            self.model = self.model.to(self.device)
        
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
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"模型加载成功: {checkpoint_path}")
    
    def extract_geometric_features(self, pdb_id: str, chain: str, mutation: str):
        """从实际PDB文件中提取野生型和突变体结构的界面特征"""
        start_time = time.time()
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
        wt_structure = parser.get_structure('WT', str(wt_pdb_file))
        mt_structure = parser.get_structure('MT', str(mt_pdb_file))
        
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
        
        geometric_time = time.time() - start_time
        print(f"  几何特征提取时间: {geometric_time:.3f}秒")
        
        return wt_graph, mt_graph
        
    
    def _extract_all_residues(self, structure, target_chain):
        """提取全部残基"""
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
    
    def _build_interface_graph_from_structure(self, structure, all_residues, is_mutant, mutation):
        """从结构中使用KNN选择最近残基构建图"""        
        if not all_residues:
            set_trace()
        
        # 解析突变信息
        mutation_pos = None
        mutation_chain = None
        if mutation and len(mutation) >= 3:
            mutation_pos = int(mutation[1:-1])  # 提取位置数字
            mutation_chain = mutation[0]  # 提取链ID
        
        # 提取节点特征和位置
        node_features = []
        node_positions = []
        atom_names = []
        residue_indices = []
        is_mutation_list = []
        
        # 使用模块级统一的坐标系构建逻辑
        residue_frames, residue_atom_positions = build_unified_residue_frames(all_residues)
        
        # 创建UnifiedResidueGeometry实例（只需要创建一次）
        atom_position_gather = UnifiedResidueGeometry(hidden_dim=96)
        
        for residue in all_residues:
            # 为每个残基的原子创建节点
            for atom in residue:
                # 原子特征（使用GearBind风格的特征提取）
                atom_feat = self.geometric_extractor.residue_geometry.atom_to_feature(
                    atom, 
                    residue_feature=None  # 可以在此处添加残基特征
                )

                # 确保atom_feat是numpy数组
                if torch.is_tensor(atom_feat):
                    atom_feat = atom_feat.detach().cpu().numpy()                
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
        if self.geometric_extractor.knn_k > 0 and len(node_features) > self.geometric_extractor.knn_k:
            # 应用KNN突变位点选择
            node_mask = self.geometric_extractor.apply_knn_selection(
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
        
        # GPU张量创建
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32, device=device)
        node_positions_tensor = torch.tensor(node_positions, dtype=torch.float32, device=device)
        is_mutation_tensor = torch.tensor(is_mutation_list, dtype=torch.bool, device=device)
        batch_tensor = batch.to(device) if hasattr(batch, 'to') else torch.tensor(batch, dtype=torch.long, device=device)
        
        return InterfaceGraphData(
            node_features=node_features_tensor,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_types=edge_types,
            node_positions=node_positions_tensor,
            batch=batch_tensor,
            atom_names=atom_names,
            is_mutation=is_mutation_tensor,
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
                res_i = residue_indices[i].split('_')[0]
                res_j = residue_indices[j].split('_')[0]
                
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
                
                # 构建96维边特征 (从64提升到96)
                edge_feat = np.zeros(96)  # 提升到96维
                
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
                
                # 5. 增强的几何编码特征 (29-95) - 扩展到96维
                for k in range(8):  # 从6增加到8，提供更丰富的几何编码
                    freq = (k + 1) * np.pi
                    idx = 29 + k * 8  # 每个频率使用8维而不是6维
                    if idx + 7 < 96:
                        edge_feat[idx] = np.sin(dist * freq)
                        edge_feat[idx + 1] = np.cos(dist * freq)
                        edge_feat[idx + 2] = np.sin(dist * freq * 2)
                        edge_feat[idx + 3] = np.cos(dist * freq * 2)
                        edge_feat[idx + 4] = np.sin(dist * freq * 3)
                        edge_feat[idx + 5] = np.cos(dist * freq * 3)
                        edge_feat[idx + 6] = np.sin(dist * freq * 4)  # 新增更高频编码
                        edge_feat[idx + 7] = np.cos(dist * freq * 4)  # 新增更高频编码
                
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
        csv_start_time = time.time()
        # 读取CSV文件，注意分隔符可能是制表符`\t`或逗号`,`, 请根据实际文件调整
        df = pd.read_csv(csv_path, sep='\t')  # 如果文件是制表符分隔
        
        results = []
        total_mutations = len(df) if not limit else limit
        print(f"开始批量测试，共 {total_mutations} 个突变")
        
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
        
        total_csv_time = time.time() - csv_start_time
        avg_time = total_csv_time / total_mutations if total_mutations > 0 else 0
        print(f"\n批量测试完成:")
        print(f"  - 总时间: {total_csv_time:.3f}秒")
        print(f"  - 平均每个突变: {avg_time:.3f}秒")
        print(f"  - 处理突变数: {total_mutations}")
        
        return results

    
    def test_single_mutation(self, pdb_id: str, chain: str, mutation: str) -> Dict[str, Any]:
        """测试单个突变"""
        total_start_time = time.time()
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
        
        # 转换为张量并移动到设备
        esm_embeddings = torch.tensor(seq_feature, dtype=torch.float32).unsqueeze(0).to(self.device)
        foldx_features = torch.tensor(energy_feature, dtype=torch.float32).unsqueeze(0).to(self.device)
        attention_mask = torch.ones(1, 1, dtype=torch.float32).to(self.device)
        
        # 确保图数据在正确的设备上
        if self.use_geometric and wt_graph is not None and mt_graph is not None:
            # 移动图数据到GPU
            for graph in [wt_graph, mt_graph]:
                for attr in ['node_features', 'edge_index', 'edge_features', 'edge_types', 
                            'node_positions', 'batch', 'is_mutation']:
                    if hasattr(graph, attr):
                        tensor = getattr(graph, attr)
                        if isinstance(tensor, torch.Tensor):
                            setattr(graph, attr, tensor.to(self.device))
        
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
                        node_features=torch.zeros(1, 96, device=self.device),  # 提升到96维
                        edge_index=torch.zeros(2, 1, dtype=torch.long, device=self.device),
                        edge_features=torch.zeros(1, 96, device=self.device),   # 提升到96维
                        edge_types=torch.zeros(1, dtype=torch.long, device=self.device),
                        node_positions=torch.zeros(1, 3, device=self.device),
                        batch=torch.zeros(1, dtype=torch.long, device=self.device),
                        atom_names=[""],
                        is_mutation=torch.zeros(1, dtype=torch.bool, device=self.device)
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
        
        total_time = time.time() - total_start_time
        result = {
            'pdb_id': pdb_id,
            'chain': chain,
            'mutation': mutation,
            'predicted_ddg': ddg_pred.item(),
            'sequence_feature_shape': seq_feature.shape,
            'energy_feature_shape': energy_feature.shape,
            'energy_values': energy_feature.tolist(),
            'status': 'success',
            'use_geometric': self.use_geometric,
            'total_time': total_time
        }
        
        print(f"  预测ΔΔG: {ddg_pred.item():.3f} kcal/mol")
        print(f"  总测试时间: {total_time:.3f}秒")
        
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
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    print(f"从 {model_path} 加载预训练模型成功")
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