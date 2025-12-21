"""
ComplexDDG数据处理模块
包含数据集定义和数据加载相关函数
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from pdb import set_trace


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
                try:
                    collated_batch[key] = torch.utils.data.default_collate(values)
                except:
                    # 如果默认collate失败，直接使用列表
                    collated_batch[key] = values
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
    
    offset = 0
    edge_offset = 0
    
    for i, graph in enumerate(graphs):
        # 节点特征
        all_node_features.append(graph.node_features)
        all_node_positions.append(graph.node_positions)
        all_atom_names.extend(graph.atom_names)
        all_is_mutation.append(graph.is_mutation)
        
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
        is_mutation=torch.cat(all_is_mutation, dim=0)
    )

# 导入模型定义和特征提取器
from model import DDGModelTester, InterfaceGraphData
from feature_extractor import RealFeatureExtractor


class SKEMPIDataset(Dataset):
    """SKEMPI数据集类"""
    
    def __init__(self, data_path: str, pdb_base_path: str, 
                 use_dummy_features: bool = False, cache_dir: str = "./dataset_cache_optimized",
                 use_geometric_features: bool = True):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            pdb_base_path: PDB文件基础路径
            use_dummy_features: 是否使用虚拟特征（用于测试）
            cache_dir: 缓存目录
            use_geometric_features: 是否使用几何特征
        """
        self.data_path = data_path
        self.pdb_base_path = pdb_base_path
        self.use_dummy_features = use_dummy_features
        self.use_geometric_features = use_geometric_features
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 加载数据
        self.data = pd.read_csv(data_path, sep='\t')
        print(f"加载数据集: {len(self.data)} 个样本")
        
        # 初始化特征提取器
        if not use_dummy_features:
            self.feature_extractor = RealFeatureExtractor(
                pdb_base_path=pdb_base_path,
                cache_dir=str(self.cache_dir),
                use_esm=True
            )
        else:
            self.feature_extractor = None
        
        # 初始化几何特征提取器
        if use_geometric_features and not use_dummy_features:
            self.geometric_tester = DDGModelTester(
                pdb_base_path=pdb_base_path,
                cache_dir=str(self.cache_dir) + "_geometric",
                use_geometric=True
            )
        else:
            self.geometric_tester = None
            
        # 预计算特征（可选）
        self.precomputed_features = None
        
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
        
        if self.use_dummy_features:
            # 使用虚拟特征用于测试
            esm_embedding = torch.randn(1, 1280)  # [seq_len, esm_dim]
            foldx_features = torch.randn(22)  # [foldx_dim]
            attention_mask = torch.ones(1)  # [seq_len]
            
            # 创建虚拟的几何图数据
            if self.use_geometric_features:
                wt_graph = self._create_dummy_graph()
                mt_graph = self._create_dummy_graph()
            else:
                wt_graph = None
                mt_graph = None
        else:
            # 提取真实特征
            if self.precomputed_features is not None:
                # 使用预计算的特征
                esm_embedding = self.precomputed_features['esm'][idx]
                foldx_features = self.precomputed_features['foldx'][idx]
                attention_mask = self.precomputed_features['mask'][idx]
            else:
                # 实时提取特征
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
                    # 如果几何特征提取失败，创建虚拟图
                    wt_graph = self._create_dummy_graph()
                    mt_graph = self._create_dummy_graph()
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
    
    def _create_dummy_graph(self) -> InterfaceGraphData:
        """创建虚拟的几何图数据"""
        num_nodes = 10
        return InterfaceGraphData(
            node_features=torch.randn(num_nodes, 64),  # 优化后的维度
            edge_index=torch.randint(0, num_nodes, (2, 20)),
            edge_features=torch.randn(20, 64),  # 优化后的维度
            edge_types=torch.randint(0, 3, (20,)),
            node_positions=torch.randn(num_nodes, 3),
            batch=torch.zeros(num_nodes, dtype=torch.long),
            atom_names=['CA'] * num_nodes,
            is_mutation=torch.zeros(num_nodes, dtype=torch.bool)
        )
    
    def precompute_features(self):
        """预计算所有特征以加速训练"""
        print("预计算数据集特征...")
        self.precomputed_features = {
            'esm': [],
            'foldx': [],
            'mask': []
        }
        
        for i in range(len(self.data)):
            if i % 10 == 0:
                print(f"预计算进度: {i}/{len(self.data)}")
            
            sample = self.__getitem__(i)
            self.precomputed_features['esm'].append(sample['esm_embeddings'])
            self.precomputed_features['foldx'].append(sample['foldx_features'])
            self.precomputed_features['mask'].append(sample['attention_mask'])
        
        print("特征预计算完成")


def create_dataloader(
    data_path: str,
    pdb_base_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    use_dummy_features: bool = False,
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
        use_dummy_features: 是否使用虚拟特征
        cache_dir: 缓存目录
        use_geometric_features: 是否使用几何特征
    
    Returns:
        DataLoader对象
    """
    dataset = SKEMPIDataset(
        data_path=data_path,
        pdb_base_path=pdb_base_path,
        use_dummy_features=use_dummy_features,
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


def split_dataset_by_complex(data_path: str, train_ratio: float = 0.8, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按复合物划分数据集
    
    Args:
        data_path: 数据文件路径
        train_ratio: 训练集比例
        random_seed: 随机种子
    
    Returns:
        (train_data, val_data): 训练集和验证集
    """
    # 加载数据
    data = pd.read_csv(data_path, sep='\t')
    
    # 获取所有唯一复合物
    complex_ids = data['#Pdb_origin'].unique()
    np.random.seed(random_seed)
    np.random.shuffle(complex_ids)
    
    # 划分复合物
    split_idx = int(len(complex_ids) * train_ratio)
    train_complexes = complex_ids[:split_idx]
    val_complexes = complex_ids[split_idx:]
    
    # 划分数据
    train_data = data[data['#Pdb_origin'].isin(train_complexes)]
    val_data = data[data['#Pdb_origin'].isin(val_complexes)]
    
    print(f"数据集划分:")
    print(f"  训练集: {len(train_data)} 个样本, {len(train_complexes)} 个复合物")
    print(f"  验证集: {len(val_data)} 个样本, {len(val_complexes)} 个复合物")
    
    return train_data, val_data


def create_kfold_splits(data_path: str, n_splits: int = 5, random_seed: int = 42):
    """
    创建K折交叉验证划分
    
    Args:
        data_path: 数据文件路径
        n_splits: 折数
        random_seed: 随机种子
    
    Returns:
        生成器，产生(train_data, val_data)元组
    """
    from sklearn.model_selection import KFold
    
    # 加载数据
    data = pd.read_csv(data_path, sep='\t')
    
    # 按复合物划分
    complex_ids = data['#Pdb_origin'].unique()
    
    # 初始化KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    for fold, (train_complex_idx, val_complex_idx) in enumerate(kf.split(complex_ids)):
        train_complexes = complex_ids[train_complex_idx]
        val_complexes = complex_ids[val_complex_idx]
        
        # 划分数据
        train_data = data[data['#Pdb_origin'].isin(train_complexes)]
        val_data = data[data['#Pdb_origin'].isin(val_complexes)]
        
        yield fold, train_data, val_data


def save_data_splits(train_data: pd.DataFrame, val_data: pd.DataFrame, 
                    output_dir: str, fold: Optional[int] = None):
    """
    保存数据划分
    
    Args:
        train_data: 训练数据
        val_data: 验证数据
        output_dir: 输出目录
        fold: 折数（可选）
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if fold is not None:
        train_path = output_path / f'fold_{fold+1}_train.csv'
        val_path = output_path / f'fold_{fold+1}_val.csv'
    else:
        train_path = output_path / 'train.csv'
        val_path = output_path / 'val.csv'
    
    train_data.to_csv(train_path, sep='\t', index=False)
    val_data.to_csv(val_path, sep='\t', index=False)
    
    print(f"数据划分已保存:")
    print(f"  训练集: {train_path}")
    print(f"  验证集: {val_path}")


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


# 使用示例
if __name__ == "__main__":
    # 示例：测试数据集功能
    print("测试数据集功能...")
    
    data_path = "/home/chengwang/code/chymodel/s1131.csv"
    pdb_base_path = "/home/chengwang/data/SKEMPI/PDBs_fixed"
    
    # 打印数据统计
    print_data_statistics(data_path)
    
    # 测试数据加载器
    print("\n测试数据加载器...")
    try:
        dataloader = create_dataloader(
            data_path=data_path,
            pdb_base_path=pdb_base_path,
            batch_size=4,
            shuffle=True,
            use_dummy_features=True,  # 使用虚拟特征进行测试
            num_workers=0
        )
        
        print(f"数据加载器创建成功，数据集大小: {len(dataloader.dataset)}")
        
        # 获取一个批次的数据
        for batch in dataloader:
            print(f"批次大小: {batch['esm_embeddings'].shape[0]}")
            print(f"ESM嵌入形状: {batch['esm_embeddings'].shape}")
            print(f"FoldX特征形状: {batch['foldx_features'].shape}")
            print(f"ΔΔG目标形状: {batch['ddg'].shape}")
            break
            
    except Exception as e:
        print(f"数据加载器测试失败: {e}")
    
    # 测试数据划分
    print("\n测试数据划分...")
    try:
        train_data, val_data = split_dataset_by_complex(data_path, train_ratio=0.8)
        print(f"数据划分测试成功")
        
        # 测试K折划分
        print("\n测试K折划分...")
        for fold, train_fold, val_fold in create_kfold_splits(data_path, n_splits=5):
            print(f"折 {fold+1}: 训练集={len(train_fold)}, 验证集={len(val_fold)}")
            if fold >= 2:  # 只测试前3折
                break
                
    except Exception as e:
        print(f"数据划分测试失败: {e}")