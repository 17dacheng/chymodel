import torch
import torch.nn as nn
import time
# 导入几何特征模块
from geometry import InterfaceGraphData, SimplifiedGeometricGNN
from pdb import set_trace


class CHYModelWithGeometric(nn.Module):    
    def __init__(self, hidden_dim: int = 512):
        super(CHYModelWithGeometric, self).__init__()
        
        
        # 几何特征处理 - 使用单个共享的GNN实例
        self.geometric_gnn = SimplifiedGeometricGNN(
            node_feat_dim=128,   # 128维原子特征输入
            edge_feat_dim=128,   # 128维边特征
            hidden_dim=128,      # 128维隐藏层
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