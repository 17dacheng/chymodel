"""
包含数据集定义和数据加载相关函数
"""
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from typing import Dict, Any
# 导入模型定义和特征提取器
from model import DDGModelTester, InterfaceGraphData
from feature_extractor import RealFeatureExtractor


def custom_collate_fn(batch):
    """自定义collate函数，处理InterfaceGraphData对象"""
    if len(batch) == 0:
        return {}
    
    # 获取所有键
    keys = batch[0].keys()
    collated_batch = {}
    
    for key in keys:
        if key in ['wt_graph', 'mt_graph']:
            # 处理InterfaceGraphData对象
            graphs = [item[key] for item in batch if item[key] is not None]
            if len(graphs) > 0:
                # 将图数据列表合并
                collated_batch[key] = collate_graph_data(graphs)
            else:
                collated_batch[key] = None
        else:
            # 使用默认的tensor处理
            values = [item[key] for item in batch if item[key] is not None]
            if len(values) > 0:
                collated_batch[key] = torch.utils.data.default_collate(values)
            else:
                collated_batch[key] = None
    
    return collated_batch


def collate_graph_data(graphs):
    """合并多个InterfaceGraphData对象"""
    if len(graphs) == 0:
        return None
    
    # 合并所有图
    all_node_features = []
    all_edge_index = []
    all_edge_features = []
    all_edge_types = []
    all_node_positions = []
    all_batch = []
    all_atom_names = []
    all_is_mutation = []
    all_residue_indices = []
    
    offset = 0
    
    for i, graph in enumerate(graphs):
        # 节点特征
        all_node_features.append(graph.node_features)
        all_node_positions.append(graph.node_positions)
        all_atom_names.extend(graph.atom_names)
        all_is_mutation.append(graph.is_mutation)
        all_residue_indices.extend(graph.residue_indices)
        
        # 批次索引
        batch_indices = torch.ones(graph.node_features.shape[0]) * i
        all_batch.append(batch_indices)
        
        # 边索引需要调整
        edge_index = graph.edge_index + offset
        all_edge_index.append(edge_index)
        all_edge_features.append(graph.edge_features)
        all_edge_types.append(graph.edge_types)
        
        offset += graph.node_features.shape[0]
    
    # 合并所有张量
    return InterfaceGraphData(
        node_features=torch.cat(all_node_features, dim=0),
        edge_index=torch.cat(all_edge_index, dim=1),
        edge_features=torch.cat(all_edge_features, dim=0),
        edge_types=torch.cat(all_edge_types, dim=0),
        node_positions=torch.cat(all_node_positions, dim=0),
        batch=torch.cat(all_batch, dim=0),
        atom_names=all_atom_names,
        is_mutation=torch.cat(all_is_mutation, dim=0),
        residue_indices=all_residue_indices
    )


class SKEMPIDataset(Dataset):
    """SKEMPI数据集类"""
    
    def __init__(self, data_path: str, pdb_base_path: str, 
                 cache_dir: str = "./dataset_cache_optimized",
                 use_geometric_features: bool = True):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            pdb_base_path: PDB文件基础路径
            cache_dir: 缓存目录
            use_geometric_features: 是否使用几何特征
        """
        self.data_path = data_path
        self.pdb_base_path = pdb_base_path
        self.use_geometric_features = use_geometric_features
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 加载数据
        self.data = pd.read_csv(data_path, sep='\t')
        print(f"加载数据集: {len(self.data)} 个样本")
        
        # 初始化特征提取器
        self.feature_extractor = RealFeatureExtractor(
            pdb_base_path=pdb_base_path,
            cache_dir=str(self.cache_dir),
            use_esm=True
        )
        
        # 初始化几何特征提取器
        if use_geometric_features:
            self.geometric_tester = DDGModelTester(
                pdb_base_path=pdb_base_path,
                cache_dir=self.cache_dir,
                use_geometric=True
            )
        else:
            self.geometric_tester = None
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        row = self.data.iloc[idx]
        
        # 解析突变信息
        pdb_id = row['#Pdb_origin']
        chain = row['Partner1']
        mutation_str = row['Mutation(s)_cleaned']
        ddg_value = row['ddG']
        
        # 解析突变字符串，例如"KI15I" -> chain="I", mutation="K15I"
        if len(mutation_str) >= 3:
            actual_chain = mutation_str[1]  # 第二个字符是链ID
            mutation = mutation_str[0] + mutation_str[2:]  # 突变信息
        else:
            actual_chain = chain
            mutation = mutation_str
        
        # 提取真实特征
        seq_feat, energy_feat = self.feature_extractor.extract_features(
            pdb_id, actual_chain, mutation
        )
        
        esm_embedding = torch.tensor(seq_feat, dtype=torch.float32).unsqueeze(0)  # [1, esm_dim]
        foldx_features = torch.tensor(energy_feat, dtype=torch.float32)  # [foldx_dim]
        attention_mask = torch.ones(1, dtype=torch.float32)  # [1]
        
        # 提取几何特征
        if self.use_geometric_features and self.geometric_tester is not None:
            wt_graph, mt_graph = self.geometric_tester.extract_geometric_features(
                pdb_id, actual_chain, mutation
            )
            if wt_graph is None or mt_graph is None:
                print(f"{pdb_id} {actual_chain} {mutation} 几何特征提取失败")
                sys.exit()
        else:
            wt_graph = None
            mt_graph = None
        
        ddg_target = torch.tensor([ddg_value], dtype=torch.float32)
        
        result = {
            'esm_embeddings': esm_embedding,
            'foldx_features': foldx_features,
            'attention_mask': attention_mask,
            'ddg': ddg_target,
            'pdb_id': pdb_id,
            'chain': actual_chain,
            'mutation': mutation
        }
        
        # 添加几何特征
        if self.use_geometric_features:
            result['wt_graph'] = wt_graph
            result['mt_graph'] = mt_graph
        
        return result
    

def create_dataloader(
    data_path: str,
    pdb_base_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    cache_dir: str = "./dataset_cache",
    use_geometric_features: bool = True
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_path: 数据文件路径
        pdb_base_path: PDB文件基础路径
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 数据加载工作进程数
        cache_dir: 缓存目录
        use_geometric_features: 是否使用几何特征
    
    Returns:
        DataLoader对象
    """
    # 当使用CUDA时，自动设置num_workers=0以避免fork问题
    if torch.cuda.is_available() and num_workers > 0:
        print(f"警告: 检测到CUDA环境，自动将num_workers从{num_workers}设置为0以避免multiprocessing问题")
        num_workers = 0
    
    dataset = SKEMPIDataset(
        data_path=data_path,
        pdb_base_path=pdb_base_path,
        cache_dir=cache_dir,
        use_geometric_features=use_geometric_features
    )
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=custom_collate_fn if use_geometric_features else None
    )


def get_data_statistics(data_path: str) -> Dict[str, Any]:
    """
    获取数据集统计信息
    
    Args:
        data_path: 数据文件路径
    
    Returns:
        包含统计信息的字典
    """
    data = pd.read_csv(data_path, sep='\t')
    
    stats = {
        'total_samples': len(data),
        'unique_complexes': data['#Pdb_origin'].nunique(),
        'unique_chains': data['Partner1'].nunique(),
        'unique_mutations': data['Mutation(s)_cleaned'].nunique(),
        'ddg_mean': data['ddG'].mean(),
        'ddg_std': data['ddG'].std(),
        'ddg_min': data['ddG'].min(),
        'ddg_max': data['ddG'].max(),
        'ddg_range': data['ddG'].max() - data['ddG'].min()
    }
    
    return stats


def print_data_statistics(data_path: str):
    """打印数据集统计信息"""
    stats = get_data_statistics(data_path)
    
    print(f"数据集统计信息 ({data_path}):")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  唯一复合物数: {stats['unique_complexes']}")
    print(f"  唯一链数: {stats['unique_chains']}")
    print(f"  唯一突变数: {stats['unique_mutations']}")
    print(f"  ΔΔG统计:")
    print(f"    均值: {stats['ddg_mean']:.3f}")
    print(f"    标准差: {stats['ddg_std']:.3f}")
    print(f"    范围: [{stats['ddg_min']:.3f}, {stats['ddg_max']:.3f}]")