import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Optional, Any, Union
from torch_cluster import radius
from feature_extractor import RealFeatureExtractor
# 导入几何特征模块
from geometry import InterfaceGraphData, SimplifiedGeometricGNN 
from pdb import set_trace


class CHYModelWithGeometric(nn.Module):    
    def __init__(self, 
                 esm_dim: int = 1280,
                 foldx_dim: int = 22,
                 hidden_dim: int = 512):
        super(CHYModelWithGeometric, self).__init__()
        
        self.esm_dim = esm_dim
        self.foldx_dim = foldx_dim
        
        # FoldX特征投影 - 输出128维
        self.foldx_projection = nn.Sequential(
            nn.Linear(foldx_dim, hidden_dim // 4),  # 直接输出128维
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )
        
        # 几何特征处理 - 使用单个共享的GNN实例
        self.geometric_gnn = SimplifiedGeometricGNN(
            node_feat_dim=96,   # 96维原子特征输入
            edge_feat_dim=96,   # 96维边特征
            hidden_dim=96,       # 96维隐藏层
            num_heads=4
        )
        
        # FoldX特征处理MLP（原来是seq_foldx_mlp，现在只处理FoldX）
        self.foldx_mlp = nn.Sequential(
            nn.Linear(
                hidden_dim // 4,  # 只处理FoldX特征 = 128维
                hidden_dim // 2  # 中间层
            ),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # 输出128维，与几何特征同维度
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
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
        
        # 特征融合注意力权重（只有FoldX和反对称几何特征两个权重）
        self.feature_fusion = nn.Sequential(
            nn.Linear(
                hidden_dim // 4 + hidden_dim // 4,  # foldx + antisymmetric = 128 + 128 = 256
                hidden_dim // 2  # 中间层维度
            ),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # 输出2个权重
            nn.Softmax(dim=-1)
        )
        
        # 最终特征融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(
                hidden_dim // 4,  # 加权融合后的特征维度保持128维
                hidden_dim // 4
            ),
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
        
        # 特别初始化特征融合层的偏置，让FoldX特征权重稍高一些
        if hasattr(self, 'feature_fusion'):
            # feature_fusion的最后一层Linear层输出2个权重
            # 设置偏置让第二个权重（反对称特征）初始为0.7，第一个（FoldX）为0.3
            fusion_layers = list(self.feature_fusion.children())
            for layer in fusion_layers:
                if isinstance(layer, nn.Linear) and layer.out_features == 2:
                    with torch.no_grad():
                        layer.bias.data = torch.tensor([0.3, 0.7])  # [foldx_weight, antisymmetric_weight]
    
    def forward(self, 
                esm_embeddings: torch.Tensor,
                foldx_features: torch.Tensor,
                wt_graph_data: InterfaceGraphData,
                mt_graph_data: InterfaceGraphData,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            esm_embeddings: [batch_size, seq_len, esm_dim] ESM嵌入（不使用）
            foldx_features: [batch_size, foldx_dim] FoldX能量项
            wt_graph_data: 野生型界面图数据
            mt_graph_data: 突变型界面图数据
            attention_mask: [batch_size, seq_len] 注意力掩码（不使用）
            
        Returns:
            ddg_predictions: [batch_size, 1] ΔΔG预测值
        """
        forward_start_time = time.time()
        batch_size = esm_embeddings.shape[0]
        
        # 1. 处理FoldX特征
        foldx_start = time.time()
        foldx_proj = self.foldx_projection(foldx_features)  # [batch_size, hidden_dim//4=128]
        foldx_time = time.time() - foldx_start
        
        # 2. 处理几何特征（WT和MT使用共享的GNN实例分别处理）
        geom_start = time.time()
        wt_geom_rep = self.geometric_gnn(wt_graph_data)  # [2, hidden_dim//4] (WT=0, MT=1)
        mt_geom_rep = self.geometric_gnn(mt_graph_data)  # [2, hidden_dim//4] (WT=0, MT=1)
        geom_time = time.time() - geom_start
        
        # 确保所有特征都有相同的batch size
        batch_size = foldx_proj.shape[0]
        
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
        
        # 1. FoldX特征处理
        foldx_features_processed = self.foldx_mlp(foldx_proj)  # [batch_size, hidden_dim//4=128]
        
        # 2. 几何特征反对称性处理
        # 创建两种组合用于反对称计算
        geom_mt_wt = torch.cat([mt_geom_rep, wt_geom_rep], dim=-1)  # [batch_size, 256]
        geom_wt_mt = torch.cat([wt_geom_rep, mt_geom_rep], dim=-1)  # [batch_size, 256]
        
        # 通过MLP处理两种几何组合
        mlp_geom_mt_wt = self.geometric_antisymmetric_mlp(geom_mt_wt)  # [batch_size, hidden_dim//4=128]
        mlp_geom_wt_mt = self.geometric_antisymmetric_mlp(geom_wt_mt)  # [batch_size, hidden_dim//4=128]
        
        # 反对称性计算: geom_antisymmetric = mlp([MT, WT]) - mlp([WT, MT])
        geometric_antisymmetric = mlp_geom_mt_wt - mlp_geom_wt_mt  # [batch_size, hidden_dim//4=128]
        
        # 3. 特征融合（FoldX特征和反对称几何特征）
        # 拼接FoldX特征和反对称几何特征
        fusion_input = torch.cat([
            foldx_features_processed,  # [batch_size, 128] FoldX特征
            geometric_antisymmetric  # [batch_size, 128] 几何反对称特征
        ], dim=-1)  # [batch_size, 256]
        
        # 计算注意力权重
        fusion_weights = self.feature_fusion(fusion_input)  # [batch_size, 2]
        
        # 应用权重进行加权融合
        # 现在两个特征都是128维，可以直接加权融合
        weighted_features = (
            fusion_weights[:, 0:1] * foldx_features_processed +  # FoldX特征权重
            fusion_weights[:, 1:2] * geometric_antisymmetric  # 几何反对称权重
        )  # [batch_size, 128]
        
        # 4. 最终特征融合
        final_features = self.final_fusion(weighted_features)  # [batch_size, hidden_dim//4=128]
        
        # 5. 回归预测
        ddg_pred = self.regression_head(final_features)  # [batch_size, 1]
        
        # 打印各部分处理时间
        total_forward_time = time.time() - forward_start_time
        # print(f"前向传播时间统计:")
        # print(f"  - FoldX特征处理: {foldx_time:.3f}秒")
        # print(f"  - 几何特征处理: {geom_time:.3f}秒")
        # print(f"  - 总前向传播时间: {total_forward_time:.3f}秒")
        
        return ddg_pred