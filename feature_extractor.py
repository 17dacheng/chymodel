"""
增强的特征提取模块 - 支持多ESM模型集成和本地权重
"""

import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForMaskedLM
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
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"警告: 无法加载缓存文件 {cache_file}: {e}")
        return {}
    
    def _save_cache(self, cache_file: Path, data: Dict):
        """保存缓存文件"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"错误: 无法保存缓存文件 {cache_file}: {e}")
    
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
        try:
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
            
            # # 尝试其他文件
            # avg_file = foldx_dir / f"Average_{pdb_id}_Repair.fxout"
            # if avg_file.exists():
            #     return self._parse_average_file(avg_file)
            # raw_file = foldx_dir / f"Raw_{pdb_id}_Repair.fxout"
            # if raw_file.exists():
            #     return self._parse_raw_file(raw_file)
            
            print(f"警告: 未找到有效的FoldX输出文件 {pdb_id}_{mutation_str}")
            set_trace()
            return np.zeros(len(FOLDX_ENERGY_TERMS))
            
        except Exception as e:
            print(f"错误: 提取 {pdb_id}_{mutation_str} 能量项时出错: {e}")
            return np.zeros(len(FOLDX_ENERGY_TERMS))
    
    def _parse_dif_file(self, file_path: Path) -> np.ndarray:
        """解析Dif文件 - 包含能量差异"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # 查找包含能量值的行
            lines = content.split('\n')
            energy_values = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#') or 'total' in line.lower():
                    continue
                
                # 尝试提取数值
                parts = line.split()
                numeric_parts = []
                
                for part in parts:
                    try:
                        # 处理科学计数法
                        if 'e' in part.lower():
                            value = float(part)
                        else:
                            # 尝试直接转换
                            value = float(part)
                        numeric_parts.append(value)
                    except ValueError:
                        continue
                
                if len(numeric_parts) >= len(FOLDX_ENERGY_TERMS):
                    energy_values = numeric_parts[:len(FOLDX_ENERGY_TERMS)]
                    break
            
            if energy_values:
                return np.array(energy_values, dtype=np.float32)
            else:
                # 如果解析失败，尝试从文件内容中搜索数值
                return self._search_energy_values(content)
                
        except Exception as e:
            print(f"错误: 解析Dif文件 {file_path} 时出错: {e}")
            return np.zeros(len(FOLDX_ENERGY_TERMS))
    
    def _parse_average_file(self, file_path: Path) -> np.ndarray:
        """解析Average文件"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # 查找包含数值的行
            for line in lines:
                if 'total energy' in line.lower() or 'energy' in line.lower():
                    parts = line.split()
                    numeric_values = []
                    for part in parts:
                        try:
                            numeric_values.append(float(part))
                        except ValueError:
                            continue
                    if len(numeric_values) >= len(FOLDX_ENERGY_TERMS):
                        return np.array(numeric_values[:len(FOLDX_ENERGY_TERMS)], dtype=np.float32)
            
            return np.zeros(len(FOLDX_ENERGY_TERMS))
            
        except Exception as e:
            print(f"错误: 解析Average文件 {file_path} 时出错: {e}")
            return np.zeros(len(FOLDX_ENERGY_TERMS))
    
    def _parse_raw_file(self, file_path: Path) -> np.ndarray:
        """解析Raw文件"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            return self._search_energy_values(content)
            
        except Exception as e:
            print(f"错误: 解析Raw文件 {file_path} 时出错: {e}")
            return np.zeros(len(FOLDX_ENERGY_TERMS))
    
    def _search_energy_values(self, content: str) -> np.ndarray:
        """从文件内容中搜索能量值"""
        import re
        
        energy_values = []
        
        # 查找所有浮点数
        numbers = re.findall(r'[-+]?\d*\.\d+[eE]?[-+]?\d*', content)
        numeric_values = []
        
        for num in numbers:
            try:
                numeric_values.append(float(num))
            except ValueError:
                continue
        
        # 取前n个数值
        if len(numeric_values) >= len(FOLDX_ENERGY_TERMS):
            energy_values = numeric_values[:len(FOLDX_ENERGY_TERMS)]
        else:
            # 如果数值不够，用零填充
            energy_values = numeric_values + [0.0] * (len(FOLDX_ENERGY_TERMS) - len(numeric_values))
        
        return np.array(energy_values, dtype=np.float32)


class MultiESMModelLoader:
    """多ESM模型加载器 - 支持5个ESM-1v模型集成"""
    
    def __init__(self, weights_base_path: str = "/home/chengwang/weights/esm2"):
        self.weights_base_path = Path(weights_base_path)
        self.models = []
        self.tokenizers = []
        
        self._load_all_models()
    
    def _load_all_models(self):
        """使用transformers库加载所有5个ESM-1v模型"""
        
        model_names = [
            "esm1v_t33_650M_UR90S_1",
            "esm1v_t33_650M_UR90S_2", 
            "esm1v_t33_650M_UR90S_3",
            "esm1v_t33_650M_UR90S_4",
            "esm1v_t33_650M_UR90S_5"
        ]
        
        successful_models = 0
        
        for i, model_name in enumerate(model_names, 1):
            model_path = self.weights_base_path / model_name
            
            if model_path.exists() and model_path.is_dir():
                print(f"加载ESM模型 {i}/5: {model_name}")

                snapshots_path = model_path / "snapshots"
                snapshot_dirs = list(snapshots_path.iterdir())
                model_dir = snapshot_dirs[0]
                
                # 使用transformers加载
                tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                model = AutoModelForMaskedLM.from_pretrained(str(model_dir))  
                
                model.eval()
                
                # 保存到相应的列表中
                self.models.append(model)
                self.tokenizers.append(tokenizer)
                
                successful_models += 1
                print(f"  ✓ 模型 {i} 加载成功")
            else:
                print(f"  ✗ 模型文件夹不存在: {model_path}")

        
        print(f"成功加载 {successful_models}/5 个ESM模型")
        
        if successful_models == 0:
            print("警告: 没有成功加载任何ESM模型")
    
    def get_ensemble_embedding(self, sequence: str) -> np.ndarray:
        """获取集成嵌入 - 多个模型的平均"""
        if not self.models:
            return self._get_fallback_embedding(sequence)
        
        all_embeddings = []
        
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            try:
                # 准备数据 - 使用transformers的tokenizer
                inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
                
                # 获取嵌入
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    # ESM-1v通常使用第33层的隐藏状态
                    embeddings = outputs.hidden_states[-1]  # 最后一层
                
                # 平均池化得到序列级表示
                # embeddings形状: [batch_size, seq_len, hidden_size]
                sequence_embedding = embeddings.mean(dim=1).squeeze().cpu().numpy()
                all_embeddings.append(sequence_embedding)
                
                print(f"  模型 {i+1} 嵌入生成成功: 形状 {sequence_embedding.shape}")
                
            except Exception as e:
                print(f"模型 {i+1} 嵌入生成失败: {e}")
        
        if not all_embeddings:
            return self._get_fallback_embedding(sequence)
        
        # 计算所有模型嵌入的平均值
        ensemble_embedding = np.mean(all_embeddings, axis=0)
        return ensemble_embedding.astype(np.float32)
    
    def _get_fallback_embedding(self, sequence: str) -> np.ndarray:
        """备用嵌入生成方法"""
        print("使用备用嵌入生成方法")
        # 返回一个合适的维度，ESM-1v的输出维度通常是1280
        return np.random.randn(1280).astype(np.float32) * 0.01
    
    def get_model_count(self) -> int:
        """获取加载的模型数量"""
        return len(self.models)


class RealSequenceFeatureExtractor:
    """真实序列特征提取器 - 支持多ESM模型集成"""
    
    def __init__(self, use_esm: bool = True, weights_path: str = "/home/chengwang/weights/esm2"):
        self.use_esm = use_esm
        self.multi_esm_loader = None
        
        if use_esm:
            self._load_esm_models(weights_path)
    
    def _load_esm_models(self, weights_path: str):
        """加载ESM模型"""
        try:
            print("初始化多ESM模型加载器...")
            self.multi_esm_loader = MultiESMModelLoader(weights_path)
            model_count = self.multi_esm_loader.get_model_count()
            print(f"多ESM模型加载完成，共 {model_count} 个模型")
            
        except Exception as e:
            print(f"加载ESM模型失败: {e}")
            self.use_esm = False
    
    def extract_sequence_embedding(self, pdb_id: str, chain: str, pdb_file_path: str) -> np.ndarray:
        """从真实PDB文件提取序列特征"""
        try:
            # 从PDB文件提取序列
            sequence = self._extract_sequence_from_pdb(pdb_file_path, chain)
            if not sequence:
                print(f"警告: 无法从 {pdb_file_path} 提取序列，尝试备用方法")
                set_trace()
            
            # print(f"提取到序列长度: {len(sequence)}")
            
            if self.use_esm and self.multi_esm_loader:
                # 使用多ESM模型集成生成嵌入
                embedding = self.multi_esm_loader.get_ensemble_embedding(sequence)
                print(f"ESM嵌入形状: {embedding.shape}")
                return embedding
            else:
                # 使用简单的序列特征
                return self._get_simple_sequence_features(sequence)
            
                
        except Exception as e:
            print(f"错误: 提取 {pdb_id}_{chain} 序列特征时出错: {e}")
            # 返回默认特征作为备用
            return self._get_simple_sequence_features("")
    
    def _extract_sequence_from_pdb(self, pdb_file: str, chain: str) -> str:
        """从PDB文件提取序列"""
        try:
            # 使用Biopython提取序列
            from Bio.PDB import PDBParser
            from Bio.PDB.Polypeptide import is_aa, three_to_one
            
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_file)
            
            sequence = ""
            for model in structure:
                for chain_obj in model:
                    if chain_obj.id == chain:
                        for residue in chain_obj:
                            if is_aa(residue, standard=True):
                                try:
                                    resname = residue.get_resname()
                                    sequence += three_to_one(resname)
                                except Exception:
                                    # 跳过非标准氨基酸
                                    continue
            
            return sequence
            
        except ImportError:
            print("警告: Biopython未安装，使用简单PDB解析")
            # 如果没有安装Biopython，使用简单解析
            return self._simple_pdb_sequence_extraction(pdb_file, chain)
        except Exception as e:
            print(f"错误: 解析PDB文件 {pdb_file} 时出错: {e}")
            return ""
    
    def _simple_pdb_sequence_extraction(self, pdb_file: str, chain: str) -> str:
        """简单的PDB序列提取（不依赖Biopython）"""
        sequence = ""
        try:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        # 检查链标识符
                        if line[21] == chain:
                            resname = line[17:20].strip()
                            resnum = line[22:26].strip()
                            
                            # 简单的氨基酸三字母到单字母映射
                            aa_map = {
                                'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
                                'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
                                'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
                                'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
                                'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
                            }
                            
                            if resname in aa_map:
                                sequence += aa_map[resname]
        except Exception as e:
            print(f"简单PDB解析失败: {e}")
        
        return sequence
    
    def _get_simple_sequence_features(self, sequence: str) -> np.ndarray:
        """获取简单的序列特征（ESM不可用时使用）"""
        # 创建1280维的特征向量，包含序列长度、氨基酸组成等信息
        features = np.zeros(1280, dtype=np.float32)
        
        if not sequence:
            return features
        
        # 序列长度特征（归一化到0-1）
        features[0] = min(len(sequence) / 1000.0, 1.0)
        
        # 氨基酸组成（20种标准氨基酸）
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_counts = {aa: sequence.count(aa) for aa in amino_acids}
        total_aa = len(sequence)
        
        if total_aa > 0:
            for i, aa in enumerate(amino_acids):
                features[1 + i] = aa_counts.get(aa, 0) / total_aa
        
        # 添加一些序列属性
        if len(sequence) > 0:
            # 疏水性（基于简化的Kyte-Doolittle尺度）
            hydrophobicity = {
                'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5,
                'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
                'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
                'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
                'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
            }
            
            avg_hydrophobicity = sum(hydrophobicity.get(aa, 0) for aa in sequence) / len(sequence)
            features[21] = (avg_hydrophobicity + 5) / 10  # 归一化到0-1
        
        # 其他位置用随机值填充（模拟ESM维度）
        if features.shape[0] > 22:
            features[22:] = np.random.randn(features.shape[0] - 22).astype(np.float32) * 0.01
        
        return features


class RealFeatureExtractor:
    """真实特征提取器 - 整合序列和能量项提取"""
    
    def __init__(self, pdb_base_path: str, cache_dir: str = "./feature_cache", 
                 use_esm: bool = True, weights_path: str = "/home/chengwang/weights/esm2"):
        self.cache = FeatureCache(cache_dir)
        self.energy_extractor = FoldXEnergyExtractor(pdb_base_path)
        self.seq_extractor = RealSequenceFeatureExtractor(use_esm, weights_path)
        self.pdb_base_path = Path(pdb_base_path)
        
        print(f"初始化真实特征提取器:")
        print(f"  - PDB基础路径: {pdb_base_path}")
        print(f"  - 缓存目录: {cache_dir}")
        print(f"  - 使用ESM: {use_esm}")
        print(f"  - 权重路径: {weights_path}")
        print("-" * 50)
    
    def extract_features(self, pdb_id: str, chain: str, mutation: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        从真实数据中提取序列特征和能量项特征
        
        Returns:
            tuple: (sequence_embedding, energy_features)
        """
        print(f"\n提取特征: {pdb_id}_{chain}_{mutation}")
        
        # 检查缓存，需要更新错误缓存
        seq_feature = self.cache.get_sequence_feature(pdb_id, chain)
        energy_feature = self.cache.get_energy_feature(pdb_id, mutation)
        mutation_str = mutation[0] + chain + mutation[1:]
        
        # 提取序列特征（如果不在缓存中）
        if seq_feature is None:
            pdb_file = self._find_pdb_file(pdb_id, mutation_str)
            if pdb_file:
                seq_feature = self.seq_extractor.extract_sequence_embedding(pdb_id, chain, pdb_file)
                self.cache.save_sequence_feature(pdb_id, chain, seq_feature)
                # print(f"  提取序列特征: {pdb_id}_{chain} -> 形状: {seq_feature.shape}")
            else:
                # print(f"  警告: 未找到PDB文件 {pdb_id}，使用默认序列特征")
                set_trace
        
        # 提取能量项特征（如果不在缓存中）
        if energy_feature is None:
            energy_feature = self.energy_extractor.extract_energy_features(pdb_id, mutation_str)
            self.cache.save_energy_feature(pdb_id, mutation, energy_feature)
            # print(f"  提取能量特征: {pdb_id}_{mutation} -> 值: {energy_feature[:3]}... (总和: {np.sum(energy_feature):.2f})")
        
        return seq_feature, energy_feature
    
    def _find_pdb_file(self, pdb_id: str, mutation: str) -> Optional[str]:
        """查找PDB文件"""
        # 突变文件夹路径
        mutation_folder = self.pdb_base_path / f"{pdb_id}_{mutation}"
        if not mutation_folder.exists():
            print(f"  警告: 突变文件夹不存在: {mutation_folder}")
            return None
        
        # 可能的PDB文件名
        possible_files = [
            mutation_folder / f"{pdb_id}_Repair_1.pdb",
            mutation_folder / f"WT_{pdb_id}_Repair_1.pdb",
            mutation_folder / f"{pdb_id}_Repair.pdb",
            mutation_folder / f"{pdb_id}.pdb",
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                # print(f"  找到PDB文件: {file_path}")
                return str(file_path)
        
        # 如果没有找到标准文件，尝试列出所有PDB文件
        print(f"  在 {mutation_folder} 中搜索PDB文件...")
        pdb_files = list(mutation_folder.glob("*.pdb"))
        if pdb_files:
            print(f"  找到PDB文件: {pdb_files[0]}")
            return str(pdb_files[0])
        
        print(f"  警告: 在 {mutation_folder} 中未找到PDB文件")
        return None
    
    def preprocess_dataset(self, data_csv_path: str):
        """预处理整个数据集"""
        print("开始预处理真实数据集...")
        
        # 加载数据
        try:
            data = pd.read_csv(data_csv_path, sep='\t')
        except Exception as e:
            print(f"错误: 无法读取数据文件 {data_csv_path}: {e}")
            return
        
        # 检查必要的列是否存在
        required_columns = ['#Pdb_origin', 'Partner1', 'Mutation(s)_cleaned']
        for col in required_columns:
            if col not in data.columns:
                print(f"错误: 数据文件缺少必要列 '{col}'")
                return
        
        # 获取所有唯一的PDB和突变
        unique_combinations = data[['#Pdb_origin', 'Partner1', 'Mutation(s)_cleaned']].drop_duplicates()
        
        total = len(unique_combinations)
        success_count = 0
        
        print(f"需要处理 {total} 个独特组合")
        
        for idx, (_, row) in enumerate(unique_combinations.iterrows()):
            pdb_id = row['#Pdb_origin']
            chain = row['Partner1']
            mutation = row['Mutation(s)_cleaned']
            
            print(f"\n处理 {idx+1}/{total}: {pdb_id}_{chain}_{mutation}")
            
            try:
                # 提取特征（会自动缓存）
                seq_feat, energy_feat = self.extract_features(pdb_id, chain, mutation)
                success_count += 1
                
                # 打印进度
                if (idx + 1) % 10 == 0:
                    print(f"进度: {idx+1}/{total} (成功率: {success_count/(idx+1)*100:.1f}%)")
                    
            except Exception as e:
                print(f"错误: 处理 {pdb_id}_{chain}_{mutation} 时出错: {e}")
        
        # 打印缓存统计
        stats = self.cache.get_cache_stats()
        print(f"\n预处理完成!")
        print(f"序列特征缓存: {stats['sequence_features_count']} 个")
        print(f"能量项特征缓存: {stats['energy_features_count']} 个")
        print(f"缓存目录: {stats['cache_dir']}")
