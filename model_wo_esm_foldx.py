import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Optional, Any, Union
from torch_cluster import radius
# 导入几何特征模块
from geometry import (
    InterfaceGraphData, SimplifiedGeometricGNN, UnifiedGeometricProcessor, 
    UnifiedResidueGeometry, build_unified_residue_frames
)
# 导入Bio.PDB相关模块
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from pdb import set_trace


class CHYModelWithGeometric(nn.Module):    
    def __init__(self, hidden_dim: int = 512):
        super(CHYModelWithGeometric, self).__init__()
        
        
        # 几何特征处理 - 使用单个共享的GNN实例
        self.geometric_gnn = SimplifiedGeometricGNN(
            node_feat_dim=96,   # 96维原子特征输入
            edge_feat_dim=96,   # 96维边特征
            hidden_dim=96,       # 96维隐藏层
            num_heads=4
        )
        
        # 反对称几何特征处理MLP
        self.geometric_antisymmetric_mlp = nn.Sequential(
            nn.Linear(
                128 + 128,  # mt_geom_rep + wt_geom_rep = 256
                hidden_dim // 2  # 256维 -> 256维，充分提取反对称信息
            ),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # 256维 -> 128维
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )
        
        # 最终特征融合层 - 直接处理反对称几何特征
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 4),  # 128维 -> 128维
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
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
                wt_graph_data: InterfaceGraphData,
                mt_graph_data: InterfaceGraphData) -> torch.Tensor:
        """
        前向传播
        
        Args:
            wt_graph_data: 野生型界面图数据
            mt_graph_data: 突变型界面图数据
            
        Returns:
            ddg_predictions: [batch_size, 1] ΔΔG预测值
        """
        forward_start_time = time.time()
        
        # 1. 处理几何特征（WT和MT使用共享的GNN实例分别处理）
        geom_start = time.time()
        wt_geom_rep = self.geometric_gnn(wt_graph_data)  # [2, hidden_dim//4] (WT=0, MT=1)
        mt_geom_rep = self.geometric_gnn(mt_graph_data)  # [2, hidden_dim//4] (WT=0, MT=1)
        geom_time = time.time() - geom_start
        
        # 确保几何特征 - 确保batch维度匹配
        batch_size = max(wt_geom_rep.shape[0], mt_geom_rep.shape[0])
        
        if wt_geom_rep.shape[0] < batch_size:
            # 扩展到匹配batch size
            wt_geom_rep = wt_geom_rep[0:1].expand(batch_size, -1)
        if mt_geom_rep.shape[0] < batch_size:
            # 扩展到匹配batch size
            mt_geom_rep = mt_geom_rep[0:1].expand(batch_size, -1)
        
        # 确保几何特征维度正确（当前已经是128维）
        # 几何特征已经通过SimplifiedGeometricGNN输出为128维
        
        # 2. 几何特征反对称性处理
        # 创建两种组合用于反对称计算
        geom_mt_wt = torch.cat([mt_geom_rep, wt_geom_rep], dim=-1)  # [batch_size, 256]
        geom_wt_mt = torch.cat([wt_geom_rep, mt_geom_rep], dim=-1)  # [batch_size, 256]
        
        # 通过MLP处理两种几何组合
        mlp_geom_mt_wt = self.geometric_antisymmetric_mlp(geom_mt_wt)  # [batch_size, hidden_dim//4=128]
        mlp_geom_wt_mt = self.geometric_antisymmetric_mlp(geom_wt_mt)  # [batch_size, hidden_dim//4=128]
        
        # 反对称性计算: geom_antisymmetric = mlp([MT, WT]) - mlp([WT, MT])
        geometric_antisymmetric = mlp_geom_mt_wt - mlp_geom_wt_mt  # [batch_size, hidden_dim//4=128]
        
        # 3. 最终特征融合 - 直接处理反对称几何特征
        final_features = self.final_fusion(geometric_antisymmetric)  # [batch_size, hidden_dim//4=128]
        
        # 4. 回归预测
        ddg_pred = self.regression_head(final_features)  # [batch_size, 1]
        
        # 打印各部分处理时间
        total_forward_time = time.time() - forward_start_time
        # print(f"前向传播时间统计:")
        # print(f"  - 几何特征处理: {geom_time:.3f}秒")
        # print(f"  - 总前向传播时间: {total_forward_time:.3f}秒")
        
        return ddg_pred


class DDGModelTesterGeometric:
    """仅使用几何特征的模型测试器"""
    
    def __init__(self, pdb_base_path: str,
                 cache_dir: str = "./test_cache",
                 model_checkpoint: Optional[str] = None):
        self.pdb_base_path = Path(pdb_base_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 自动检测设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化几何特征提取器
        self.geometric_extractor = UnifiedGeometricProcessor()
        
        # 初始化模型
        self.model = CHYModelWithGeometric()
        self.model = self.model.to(self.device)
        
        # 加载模型检查点
        if model_checkpoint and Path(model_checkpoint).exists():
            self.load_model_checkpoint(model_checkpoint)
        
        print(f"几何特征模型测试器初始化完成:")
        print(f"  - PDB路径: {pdb_base_path}")
        print(f"  - 缓存目录: {cache_dir}")
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
        # start_time = time.time()
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
        
        # 构建图数据
        wt_graph = self._build_interface_graph_from_structure(wt_structure, wt_all_residues, is_mutant=False, mutation=mutation)
        mt_graph = self._build_interface_graph_from_structure(mt_structure, mt_all_residues, is_mutant=True, mutation=mutation)
        # geometric_time = time.time() - start_time
        # print(f"  几何特征提取时间: {geometric_time:.3f}秒")
        
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
                # 原子特征提取
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
        """基于原子位置构建边 - 完全避免O(n²)计算"""        
        n_nodes = len(node_positions)
        if n_nodes == 0:
            return (torch.zeros(2, 0, dtype=torch.long), 
                   torch.zeros(0, 96, dtype=torch.float32),
                   torch.zeros(0, dtype=torch.long))
        
        # 转换为torch tensor
        node_positions_tensor = torch.tensor(node_positions, dtype=torch.float32)
        batch = torch.zeros(n_nodes, dtype=torch.long)  # 所有节点属于一个图
        cutoff_distance = 5.0
        
            
        # 使用radius搜索找到所有距离在cutoff内的原子对
        row, col = radius(node_positions_tensor, node_positions_tensor, 
                        r=cutoff_distance, batch_x=batch, batch_y=batch)
        
        # 过滤自连接
        mask = row != col
        row = row[mask]
        col = col[mask]
        
        # 过滤重复边，只保留row < col的边，然后创建双向边
        unique_mask = row < col
        row = row[unique_mask]
        col = col[unique_mask]
        
        # 批量计算距离
        distances = torch.norm(node_positions_tensor[row] - node_positions_tensor[col], dim=1)
        
        # 构建边特征和类型（向量化）
        edge_features = self._create_edge_features_vectorized(row, col, distances, node_positions, residue_indices)
        edge_types = self._create_edge_types_vectorized(row, col, distances, residue_indices)
        
        # 创建双向边
        edge_index = torch.stack([
            torch.cat([row, col]),
            torch.cat([col, row])
        ], dim=0)
        edge_features = torch.cat([edge_features, edge_features], dim=0)
        edge_types = torch.cat([edge_types, edge_types], dim=0)
        return edge_index, edge_features, edge_types
    
    def _create_edge_features_vectorized(self, row, col, distances, node_positions, residue_indices):
        """向量化边特征创建"""
        num_edges = len(row)
        edge_features = torch.zeros(num_edges, 96, dtype=torch.float32)
        
        # 1. 基础距离特征 (0-2) - 向量化
        edge_features[:, 0] = distances
        edge_features[:, 1] = 1.0 / (distances + 1e-6)
        edge_features[:, 2] = torch.log(distances + 1.0)
        
        # 2. 方向特征 (3-5) - 批量计算
        node_positions_tensor = torch.tensor(node_positions, dtype=torch.float32)
        directions = (node_positions_tensor[col] - node_positions_tensor[row]) / distances.unsqueeze(1)
        edge_features[:, 3:6] = directions
        
        # 3. 序列距离特征 (6-8) - 向量化
        seq_dists = (row - col).abs().float()
        edge_features[:, 6] = seq_dists
        edge_features[:, 7] = 1.0 / (seq_dists + 1.0)
        edge_features[:, 8] = torch.log(seq_dists + 1.0)
        
        # 4. 残基类型特征 (9-28) - 批量处理
        aa_types = 'ACDEFGHIKLMNPQRSTVWY'
        for i, (r, c) in enumerate(zip(row.tolist(), col.tolist())):
            res_i = residue_indices[r].split('_')[0]
            res_j = residue_indices[c].split('_')[0]
            aa_idx_i = min(aa_types.find(res_i) if res_i in aa_types else 0, 9)
            aa_idx_j = min(aa_types.find(res_j) if res_j in aa_types else 0, 9)
            edge_features[i, 9 + aa_idx_i] = 1.0
            edge_features[i, 19 + aa_idx_j] = 1.0
        
        # 5. 简化的几何编码 (29-95) - 减少频率数量，向量化
        for k in range(4):  # 从8减少到4个频率，减少计算量
            freq = (k + 1) * np.pi
            start_idx = 29 + k * 16
            if start_idx + 15 < 96:
                sin_vals = torch.sin(distances * freq)
                cos_vals = torch.cos(distances * freq)
                edge_features[:, start_idx] = sin_vals
                edge_features[:, start_idx + 1] = cos_vals
                edge_features[:, start_idx + 8] = sin_vals * 2
                edge_features[:, start_idx + 9] = cos_vals * 2
        
        return edge_features
    
    def _create_edge_types_vectorized(self, row, col, distances, residue_indices):
        """向量化边类型创建"""
        edge_types = torch.zeros(len(row), dtype=torch.long)
        
        for i, (r, c) in enumerate(zip(row.tolist(), col.tolist())):
            res_i = residue_indices[r].split('_')[0]
            res_j = residue_indices[c].split('_')[0]
            dist = distances[i].item()
            
            if res_i == res_j:
                edge_types[i] = 0 if dist < 2.0 else 2
            else:
                edge_types[i] = 2
        
        return edge_types

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
        
        # 提取几何特征
        wt_graph, mt_graph = self.extract_geometric_features(pdb_id, chain, mutation)
        
        if wt_graph is None or mt_graph is None:
            # 如果几何特征提取失败，返回错误结果
            return {
                'pdb_id': pdb_id,
                'chain': chain,
                'mutation': mutation,
                'predicted_ddg': 0.0,
                'status': 'failed_geometric_extraction',
                'use_geometric': True,
                'total_time': time.time() - total_start_time
            }
        
        # 确保图数据在正确的设备上
        for graph in [wt_graph, mt_graph]:
            for attr in ['node_features', 'edge_index', 'edge_features', 'edge_types', 
                        'node_positions', 'batch', 'is_mutation']:
                if hasattr(graph, attr):
                    tensor = getattr(graph, attr)
                    if isinstance(tensor, torch.Tensor):
                        setattr(graph, attr, tensor.to(self.device))
        
        # 前向传播
        with torch.no_grad():
            ddg_pred = self.model(
                wt_graph, 
                mt_graph
            )
        
        total_time = time.time() - total_start_time
        result = {
            'pdb_id': pdb_id,
            'chain': chain,
            'mutation': mutation,
            'predicted_ddg': ddg_pred.item(),
            'status': 'success',
            'use_geometric': True,
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


def load_pretrained_geometricmodel(model_path: str, device: str = 'cpu') -> CHYModelWithGeometric:
    """加载预训练的仅几何特征模型"""
    model = CHYModelWithGeometric()
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    print(f"从 {model_path} 加载预训练几何特征模型成功")
    return model


# 使用示例
if __name__ == "__main__":    
    # 测试仅几何特征的版本
    tester = DDGModelTesterGeometric(
        pdb_base_path="/home/chengwang/data/SKEMPI/PDBs_fixed",
        cache_dir="./test_cache",
        model_checkpoint=None
    )
    
    csv_path = "/home/chengwang/code/chymodel/s1131.csv"
    if Path(csv_path).exists():
        # 测试仅几何特征版本
        csv_results = tester.test_from_csv(
            csv_path, 
            pdb_col="#Pdb_origin",  # 使用原始PDB ID列
            mutation_col="Mutation(s)_cleaned",  # 使用清理后的突变列
            limit=1  # 只测试1行数据
        )
        
        # 保存结果
        tester.save_results(csv_results, "test_results_geometric.csv")
        print(f"结果已保存到: test_results_geometric.csv")
    else:
        print(f"CSV文件不存在: {csv_path}")
        print("跳过CSV测试")