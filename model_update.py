import torch
import torch.nn as nn
import time
from geometry_update import InterfaceGraphData, EnhancedGeometricGNN
from pdb import set_trace


class EnhancedCHYModelWithGeometric(nn.Module):    
    def __init__(self, hidden_dim: int = 512):
        super(EnhancedCHYModelWithGeometric, self).__init__()
        
        # 增强的几何特征处理 - 使用更大的隐藏维度
        self.geometric_gnn = EnhancedGeometricGNN(
            node_feat_dim=128,     # 128维原子特征输入
            edge_feat_dim=128,     # 128维边特征
            hidden_dim=256,        # 增强到256维隐藏层
            num_heads=8            # 8个注意力头
        )
        
        # 多层反对称几何特征处理MLP - 更深的网络
        self.geometric_antisymmetric_mlp = nn.Sequential(
            nn.Linear(
                256 + 256,  # mt_geom_rep + wt_geom_rep = 512 (现在都是256维)
                hidden_dim,  # 512维 -> 512维，充分提取反对称信息
            ),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Linear(hidden_dim, hidden_dim // 2),  # 512维 -> 256维
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),  # 256维 -> 256维
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # 增强的最终特征融合层 - 多层结构
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),  # 256维 -> 256维
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # 256维 -> 128维
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )
        
        # 增强的回归头 - 更深的网络
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1)
        )
        
        # 特征正则化
        self.feature_norm = nn.LayerNorm(hidden_dim // 4)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """增强的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用更好的初始化方法
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def enhanced_antisymmetric_processing(self, mt_feat, wt_feat):
        """
        增强的反对称处理 - 包含多种反对称组合
        """
        # 标准反对称: [MT, WT] - [WT, MT]
        mt_wt = torch.cat([mt_feat, wt_feat], dim=-1)
        wt_mt = torch.cat([wt_feat, mt_feat], dim=-1)
        
        mlp_mt_wt = self.geometric_antisymmetric_mlp(mt_wt)
        mlp_wt_mt = self.geometric_antisymmetric_mlp(wt_mt)
        
        standard_antisymmetric = mlp_mt_wt - mlp_wt_mt
        
        # 额外的特征交互 (element-wise差异)
        element_diff = mt_feat - wt_feat
        
        # 特征乘积 (捕获非线性交互)
        element_product = mt_feat * wt_feat
        
        # 组合所有反对称特征
        enhanced_features = torch.cat([
            standard_antisymmetric,
            element_diff,
            element_product
        ], dim=-1)
        
        return enhanced_features
    
    def forward(self, 
                wt_graph_data: InterfaceGraphData,
                mt_graph_data: InterfaceGraphData) -> torch.Tensor:
        """
        增强版前向传播
        
        Args:
            wt_graph_data: 野生型界面图数据
            mt_graph_data: 突变型界面图数据
            
        Returns:
            ddg_predictions: [batch_size, 1] ΔΔG预测值
        """
        forward_start_time = time.time()
        
        # 1. 处理几何特征（WT和MT使用共享的增强GNN）
        geom_start = time.time()
        wt_geom_rep = self.geometric_gnn(wt_graph_data)  # [batch_size, 256]
        mt_geom_rep = self.geometric_gnn(mt_graph_data)  # [batch_size, 256]
        geom_time = time.time() - geom_start
        
        # 确保batch维度匹配
        batch_size = max(wt_geom_rep.shape[0], mt_geom_rep.shape[0])
        
        if wt_geom_rep.shape[0] < batch_size:
            wt_geom_rep = wt_geom_rep[0:1].expand(batch_size, -1)
        if mt_geom_rep.shape[0] < batch_size:
            mt_geom_rep = mt_geom_rep[0:1].expand(batch_size, -1)
        
        # 2. 增强的反对称性处理
        antisymmetric_start = time.time()
        enhanced_antisymmetric = self.enhanced_antisymmetric_processing(mt_geom_rep, wt_geom_rep)
        
        # 调整维度以匹配融合层输入
        # enhanced_antisymmetric 维度应该是 [batch_size, hidden_dim // 4]
        current_dim = enhanced_antisymmetric.shape[-1]
        target_dim = self.final_fusion[0].in_features  # 获取目标维度
        
        if current_dim != target_dim:
            # 添加维度适配层
            if not hasattr(self, 'dim_adapter'):
                self.dim_adapter = nn.Linear(current_dim, target_dim).to(enhanced_antisymmetric.device)
                self._init_weights()  # 重新初始化新增层
            enhanced_antisymmetric = self.dim_adapter(enhanced_antisymmetric)
        
        antisymmetric_time = time.time() - antisymmetric_start
        
        # 3. 增强的特征融合
        fusion_start = time.time()
        final_features = self.final_fusion(enhanced_antisymmetric)
        
        # 特征正则化
        final_features = self.feature_norm(final_features)
        fusion_time = time.time() - fusion_start
        
        # 4. 回归预测
        regression_start = time.time()
        ddg_pred = self.regression_head(final_features)
        regression_time = time.time() - regression_start
        
        # 性能统计
        total_forward_time = time.time() - forward_start_time
        # print(f"增强版前向传播时间统计:")
        # print(f"  - 几何特征处理: {geom_time:.3f}秒")
        # print(f"  - 反对称处理: {antisymmetric_time:.3f}秒")
        # print(f"  - 特征融合: {fusion_time:.3f}秒")
        # print(f"  - 回归预测: {regression_time:.3f}秒")
        # print(f"  - 总前向传播时间: {total_forward_time:.3f}秒")
        
        return ddg_pred
    
    def get_feature_importance(self, wt_graph_data, mt_graph_data):
        """
        获取特征重要性分析（用于模型解释）
        """
        with torch.no_grad():
            # 获取中间特征
            wt_geom_rep = self.geometric_gnn(wt_graph_data)
            mt_geom_rep = self.geometric_gnn(mt_graph_data)
            
            # 分析各部分贡献
            mt_wt = torch.cat([mt_geom_rep, wt_geom_rep], dim=-1)
            wt_mt = torch.cat([wt_geom_rep, mt_geom_rep], dim=-1)
            
            mlp_mt_wt = self.geometric_antisymmetric_mlp(mt_wt)
            mlp_wt_mt = self.geometric_antisymmetric_mlp(wt_mt)
            
            # 返回特征重要性指标
            importance = {
                'wt_magnitude': torch.norm(wt_geom_rep, dim=-1).mean(),
                'mt_magnitude': torch.norm(mt_geom_rep, dim=-1).mean(),
                'antisymmetric_magnitude': torch.norm(mlp_mt_wt - mlp_wt_mt, dim=-1).mean(),
                'element_diff': torch.norm(mt_geom_rep - wt_geom_rep, dim=-1).mean()
            }
            
            return importance


# 兼容性别名 - 保持与原模型的接口一致
CHYModelWithGeometric = EnhancedCHYModelWithGeometric


# 测试函数
def test_model_compatibility():
    """测试模型兼容性和性能"""
    print("测试增强版模型...")
    
    # 创建测试数据
    batch_size = 2
    num_nodes = 100
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟图数据
    wt_graph = InterfaceGraphData(
        node_features=torch.randn(num_nodes, 128),
        edge_index=torch.randint(0, num_nodes, (2, 200)),
        edge_features=torch.randn(200, 128),
        edge_types=torch.randint(0, 3, (200,)),
        node_positions=torch.randn(num_nodes, 3),
        batch=torch.zeros(num_nodes, dtype=torch.long),
        atom_names=['CA'] * num_nodes,
        is_mutation=torch.zeros(num_nodes, dtype=torch.bool),
        residue_indices=[i // 5 for i in range(num_nodes)]
    ).to(device)
    
    mt_graph = InterfaceGraphData(
        node_features=torch.randn(num_nodes, 128),
        edge_index=torch.randint(0, num_nodes, (2, 200)),
        edge_features=torch.randn(200, 128),
        edge_types=torch.randint(0, 3, (200,)),
        node_positions=torch.randn(num_nodes, 3),
        batch=torch.zeros(num_nodes, dtype=torch.long),
        atom_names=['CA'] * num_nodes,
        is_mutation=torch.ones(num_nodes, dtype=torch.bool),
        residue_indices=[i // 5 for i in range(num_nodes)]
    ).to(device)
    
    # 创建模型
    model = EnhancedCHYModelWithGeometric().to(device)
    
    # 前向传播测试
    with torch.no_grad():
        output = model(wt_graph, mt_graph)
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # 特征重要性分析
        importance = model.get_feature_importance(wt_graph, mt_graph)
        print("特征重要性:")
        for k, v in importance.items():
            print(f"  {k}: {v.item():.3f}")
    
    print("测试完成！")
    return model


if __name__ == "__main__":
    test_model_compatibility()