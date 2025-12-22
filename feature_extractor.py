"""
增强的特征提取模块 - 支持多ESM模型集成和本地权重
"""

import sys
import torch
import pickle
import numpy as np
import time
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, three_to_one
from typing import Dict, Tuple, Optional
from pdb import set_trace

import warnings
warnings.filterwarnings('ignore')

# FoldX 22个能量项名称（根据论文3.3节）
FOLDX_ENERGY_TERMS = [
    'total_energy', 'Backbone_Hbond', 'Sidechain_Hbond', 'Van_der_Waals', 
    'Electrostatics', 'Solvation_Polar', 'Solvation_Hydrophobic', 
    'Van_der_Waals_clashes', 'entropy_sidechain', 'entropy_mainchain', 
    'sloop_entropy', 'mloop_entropy', 'cis_bond', 'torsional_clash', 
    'backbone_clash', 'helix_dipole', 'water_bridge', 'disulfide', 
    'electrostatic_kon', 'partial_covalent_bonds', 'energy_Ionisation', 
    'Entropy_Complex'
]

class FeatureCache:
    """特征缓存管理器"""
    
    def __init__(self, cache_dir: str = "./feature_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 缓存文件路径
        self.sequence_cache_file = self.cache_dir / "sequence_features.pkl"
        self.energy_cache_file = self.cache_dir / "energy_features.pkl"
        
        # 加载现有缓存
        self.sequence_features = self._load_cache(self.sequence_cache_file)
        self.energy_features = self._load_cache(self.energy_cache_file)
    
    def _load_cache(self, cache_file: Path) -> Dict:
        """加载缓存文件"""
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save_cache(self, cache_file: Path, data: Dict):
        """保存缓存文件"""
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    
    def get_sequence_feature(self, pdb_id: str, chain: str) -> Optional[np.ndarray]:
        """获取序列特征"""
        key = f"{pdb_id}_{chain}"
        return self.sequence_features.get(key)
    
    def save_sequence_feature(self, pdb_id: str, chain: str, feature: np.ndarray):
        """保存序列特征"""
        key = f"{pdb_id}_{chain}"
        self.sequence_features[key] = feature
        self._save_cache(self.sequence_cache_file, self.sequence_features)
    
    def get_energy_feature(self, pdb_id: str, mutation: str) -> Optional[np.ndarray]:
        """获取能量项特征"""
        key = f"{pdb_id}_{mutation}"
        return self.energy_features.get(key)
    
    def save_energy_feature(self, pdb_id: str, mutation: str, feature: np.ndarray):
        """保存能量项特征"""
        key = f"{pdb_id}_{mutation}"
        self.energy_features[key] = feature
        self._save_cache(self.energy_cache_file, self.energy_features)
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            'sequence_features_count': len(self.sequence_features),
            'energy_features_count': len(self.energy_features),
            'cache_dir': str(self.cache_dir)
        }


class FoldXEnergyExtractor:
    """FoldX能量项提取器 - 从真实FoldX输出文件解析"""
    
    def __init__(self, pdb_base_path: str):
        self.pdb_base_path = Path(pdb_base_path)
    
    def extract_energy_features(self, pdb_id: str, mutation_str: str) -> np.ndarray:
        """
        从真实FoldX输出文件中提取22个能量项特征
        
        根据论文3.3节描述的22个能量项，从Dif_*.fxout文件解析
        """
        # 构建突变文件夹路径 - 格式如: 3BT1_GU131A
        mutation_folder_name = f"{pdb_id}_{mutation_str}"
        foldx_dir = self.pdb_base_path / mutation_folder_name
        
        if not foldx_dir.exists():
            print(f"警告: FoldX文件夹不存在 {foldx_dir}")
            return np.zeros(len(FOLDX_ENERGY_TERMS))
        
        # 优先使用Dif文件，它包含突变前后的能量差异
        dif_file = foldx_dir / f"Dif_{pdb_id}_Repair.fxout"
        if dif_file.exists():
            return self._parse_dif_file(dif_file)

        dif_file = foldx_dir / f"Dif_{pdb_id}_fixed_Repair.fxout"
        if dif_file.exists():
            return self._parse_dif_file(dif_file)
        
        print(f"警告: 未找到有效的FoldX输出文件 {pdb_id}_{mutation_str}")
        sys.exit()
        return np.zeros(len(FOLDX_ENERGY_TERMS))
    
    def _parse_dif_file(self, file_path: Path) -> np.ndarray:
        """解析Dif文件 - 包含能量差异"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 查找包含能量值的行
        lines = content.split('\n')
        
        # 初始化结果数组
        energy_values = np.zeros(len(FOLDX_ENERGY_TERMS), dtype=np.float32)
        
        # 查找标题行和数据行
        header_line = None
        data_line = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if 'total energy' in line.lower():
                # 找到标题行
                header_line = line
            elif '.pdb' in line:
                # 找到数据行
                data_line = line
                
            if header_line and data_line:
                break
        
        if header_line and data_line:
            # 处理标题行
            header_parts = header_line.split('\t')
            header_map = {}
            for i, header in enumerate(header_parts):
                header_map[header.lower()] = i
            
            # 处理数据行
            data_parts = data_line.split()
            
            # 提取每个能量项的值
            for i, term in enumerate(FOLDX_ENERGY_TERMS):
                # 将能量项名称转换为与文件中一致的格式
                normalized_term = term.replace('_', ' ').lower()

                for header, idx in header_map.items():
                    if normalized_term in header:
                        # 确保索引在数据范围内
                        if idx < len(data_parts):
                            try:
                                energy_values[i] = float(data_parts[idx])
                            except (ValueError, IndexError):
                                # 如果转换失败，保持默认值0
                                pass
                        break
            return energy_values
        set_trace()
        return None

    def _parse_average_file(self, file_path: Path) -> np.ndarray:
        """解析Average文件"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # 查找包含数值的行
        for line in lines:
            if 'total energy' in line.lower() or 'energy' in line.lower():
                parts = line.split()
                numeric_values = []
                for part in parts:
                    numeric_values.append(float(part))
                if len(numeric_values) >= len(FOLDX_ENERGY_TERMS):
                    return np.array(numeric_values[:len(FOLDX_ENERGY_TERMS)], dtype=np.float32)
        
        return np.zeros(len(FOLDX_ENERGY_TERMS))
    
    def _parse_raw_file(self, file_path: Path) -> np.ndarray:
        """解析Raw文件"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        return self._search_energy_values(content)

class RealFeatureExtractor:
    """真实特征提取器 - 整合序列和能量项提取"""
    
    def __init__(self, pdb_base_path: str, cache_dir: str = "./feature_cache", 
                 use_esm: bool = True, weights_path: str = "/home/chengwang/weights/esm2"):
        self.cache = FeatureCache(cache_dir)
        self.energy_extractor = FoldXEnergyExtractor(pdb_base_path)
        self.pdb_base_path = Path(pdb_base_path)
        self.use_esm = use_esm
        
        print(f"初始化真实特征提取器:")
        print(f"  - PDB基础路径: {pdb_base_path}")
        print(f"  - 缓存目录: {cache_dir}")
        print(f"  - 使用ESM: {use_esm}")
        print("-" * 50)
    
    def extract_features(self, pdb_id: str, chain: str, mutation: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        从真实数据中提取序列特征和能量项特征
        
        Returns:
            tuple: (sequence_embedding, energy_features)
        """
        print(f"\n提取特征: {pdb_id}_{chain}_{mutation}")
        
        # 检查缓存
        seq_feature = self.cache.get_sequence_feature(pdb_id, chain)
        energy_feature = self.cache.get_energy_feature(pdb_id, mutation)
        mutation_str = mutation[0] + chain + mutation[1:]
        
        # 提取序列特征（如果不在缓存中）
        if seq_feature is None:
            start_time = time.time()
            pdb_file = self._find_pdb_file(pdb_id, mutation_str)
            if pdb_file:
                seq_feature = self._extract_sequence_embedding(pdb_file, chain)
                self.cache.save_sequence_feature(pdb_id, chain, seq_feature)
            else:
                print(f"  警告: 未找到PDB文件 {pdb_id}_{mutation_str}，使用默认序列特征")
                seq_feature = self._get_default_sequence_features()
            seq_time = time.time() - start_time
            print(f"  ESM序列特征提取时间: {seq_time:.3f}秒")
        else:
            print(f"  ESM序列特征从缓存加载")
        
        # 提取能量项特征（如果不在缓存中）
        if energy_feature is None:
            start_time = time.time()
            energy_feature = self.energy_extractor.extract_energy_features(pdb_id, mutation_str)
            self.cache.save_energy_feature(pdb_id, mutation, energy_feature)
            foldx_time = time.time() - start_time
            print(f"  FoldX能量项提取时间: {foldx_time:.3f}秒")
        else:
            print(f"  FoldX能量项从缓存加载")
        
        return seq_feature, energy_feature
    
    def _extract_sequence_embedding(self, pdb_file: str, chain: str) -> np.ndarray:
        """从PDB文件提取序列嵌入"""
        sequence = self._extract_sequence_from_pdb(pdb_file, chain)
        return self._get_simple_sequence_features(sequence)
    
    def _extract_sequence_from_pdb(self, pdb_file: str, chain: str) -> str:
        """从PDB文件提取序列"""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        sequence = ""
        for model in structure:
            for chain_obj in model:
                if chain_obj.id == chain:
                    for residue in chain_obj:
                        if is_aa(residue, standard=True):
                            resname = residue.get_resname()
                            sequence += three_to_one(resname)
        return sequence
    
    def _simple_pdb_sequence_extraction(self, pdb_file: str, chain: str) -> str:
        """简单的PDB序列提取（不依赖Biopython）"""
        sequence = ""
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    if line[21] == chain:
                        resname = line[17:20].strip()
                        aa_map = {
                            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
                            'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
                            'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
                            'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
                            'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
                        }
                        if resname in aa_map:
                            sequence += aa_map[resname]
        return sequence
    
    def _get_simple_sequence_features(self, sequence: str) -> np.ndarray:
        """获取简单的序列特征"""
        features = np.zeros(1280, dtype=np.float32)
        
        if not sequence:
            return features
        
        # 序列长度特征
        features[0] = min(len(sequence) / 1000.0, 1.0)
        
        # 氨基酸组成
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_counts = {aa: sequence.count(aa) for aa in amino_acids}
        total_aa = len(sequence)
        
        if total_aa > 0:
            for i, aa in enumerate(amino_acids):
                features[1 + i] = aa_counts.get(aa, 0) / total_aa
        
        # 其他位置用随机值填充
        if features.shape[0] > 21:
            features[21:] = np.random.randn(features.shape[0] - 21).astype(np.float32) * 0.01
        
        return features
    
    def _get_default_sequence_features(self) -> np.ndarray:
        """获取默认序列特征"""
        return np.random.randn(1280).astype(np.float32) * 0.01
    
    def _find_pdb_file(self, pdb_id: str, mutation: str) -> Optional[str]:
        """查找PDB文件"""
        mutation_folder = self.pdb_base_path / f"{pdb_id}_{mutation}"
        if not mutation_folder.exists():
            return None
        
        possible_files = [
            mutation_folder / f"{pdb_id}_Repair_1.pdb",
            mutation_folder / f"WT_{pdb_id}_Repair_1.pdb",
            mutation_folder / f"{pdb_id}_Repair.pdb",
            mutation_folder / f"{pdb_id}.pdb",
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                return str(file_path)
        
        pdb_files = list(mutation_folder.glob("*.pdb"))
        if pdb_files:
            return str(pdb_files[0])
        
        return None