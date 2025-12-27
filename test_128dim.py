#!/usr/bin/env python3
"""
测试128维修改是否正确
"""
import sys
import torch
import numpy as np
from pathlib import Path

# 添加当前目录到路径
sys.path.append('/home/chengwang/code/chymodel')

def test_model_initialization():
    """测试模型初始化"""
    print("=" * 50)
    print("测试模型初始化...")
    print("=" * 50)
    
    try:
        from model_wo_esm_foldx import CHYModelWithGeometric
        
        # 初始化模型
        model = CHYModelWithGeometric()
        print("✓ CHYModelWithGeometric 初始化成功")
        
        # 检查各层维度
        print(f"  - geometric_gnn参数:")
        print(f"    node_feat_dim: {model.geometric_gnn.node_proj[0].in_features}")
        print(f"    edge_feat_dim: {model.geometric_gnn.edge_proj[0].in_features}")
        print(f"    hidden_dim: {model.geometric_gnn.hidden_dim}")
        
        print(f"  - 几何反对称MLP第一层输入维度: {model.geometric_antisymmetric_mlp[0].in_features}")
        print(f"  - 几何反对称MLP第一层输出维度: {model.geometric_antisymmetric_mlp[0].out_features}")
        
        return True
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        return False

def test_geometry_components():
    """测试几何组件"""
    print("\n" + "=" * 50)
    print("测试几何组件...")
    print("=" * 50)
    
    try:
        from geometry import (
            UnifiedGeometricProcessor, 
            UnifiedResidueGeometry, 
            SimplifiedGeometricGNN
        )
        
        # 测试几何处理器
        processor = UnifiedGeometricProcessor(hidden_dim=128)
        print("✓ UnifiedGeometricProcessor (128维) 初始化成功")
        print(f"  - hidden_dim: {processor.hidden_dim}")
        
        # 测试残基几何处理器
        residue_geom = UnifiedResidueGeometry(hidden_dim=128)
        print("✓ UnifiedResidueGeometry (128维) 初始化成功")
        print(f"  - hidden_dim: {residue_geom.hidden_dim}")
        
        # 测试几何GNN
        gnn = SimplifiedGeometricGNN(
            node_feat_dim=128, 
            edge_feat_dim=128, 
            hidden_dim=128
        )
        print("✓ SimplifiedGeometricGNN (128维) 初始化成功")
        print(f"  - 节点特征投影输入维度: {gnn.node_proj[0].in_features}")
        print(f"  - 边特征投影输入维度: {gnn.edge_proj[0].in_features}")
        print(f"  - 隐藏层维度: {gnn.hidden_dim}")
        
        return True
    except Exception as e:
        print(f"✗ 几何组件测试失败: {e}")
        return False

def test_edge_feature_creation():
    """测试边特征创建"""
    print("\n" + "=" * 50)
    print("测试边特征创建...")
    print("=" * 50)
    
    try:
        from model_wo_esm_foldx import DDGModelTesterGeometric
        
        # 创建测试器实例
        tester = DDGModelTesterGeometric(
            pdb_base_path="/home/chengwang/data/SKEMPI/PDBs_fixed",
            cache_dir="./test_cache"
        )
        
        # 测试边特征创建函数
        num_edges = 10
        row = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        col = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        distances = torch.rand(num_edges) * 5.0
        node_positions = np.random.rand(20, 3).astype(np.float32)
        residue_indices = [f"A_{i}" for i in range(20)]
        
        edge_features = tester._create_edge_features_vectorized(
            row, col, distances, node_positions, residue_indices
        )
        
        print(f"✓ 边特征创建成功")
        print(f"  - 边数量: {num_edges}")
        print(f"  - 边特征维度: {edge_features.shape}")
        print(f"  - 期望维度: ({num_edges}, 128)")
        
        if edge_features.shape == (num_edges, 128):
            print("✓ 边特征维度正确！")
            return True
        else:
            print(f"✗ 边特征维度错误！期望 ({num_edges}, 128)，实际 {edge_features.shape}")
            return False
            
    except Exception as e:
        print(f"✗ 边特征创建测试失败: {e}")
        return False

def test_forward_pass():
    """测试前向传播"""
    print("\n" + "=" * 50)
    print("测试前向传播...")
    print("=" * 50)
    
    try:
        from model_wo_esm_foldx import CHYModelWithGeometric, InterfaceGraphData
        
        model = CHYModelWithGeometric()
        
        # 创建模拟图数据
        batch_size = 2
        num_nodes = 50
        num_edges = 100
        
        # 创建WT图数据
        wt_graph = InterfaceGraphData(
            node_features=torch.randn(num_nodes, 128),
            edge_index=torch.randint(0, num_nodes, (2, num_edges)),
            edge_features=torch.randn(num_edges, 128),
            edge_types=torch.randint(0, 3, (num_edges,)),
            node_positions=torch.randn(num_nodes, 3),
            batch=torch.zeros(num_nodes, dtype=torch.long),
            atom_names=[f"CA" for _ in range(num_nodes)],
            is_mutation=torch.zeros(num_nodes, dtype=torch.bool),
            residue_indices=[f"A_{i}" for i in range(num_nodes)]
        )
        
        # 创建MT图数据
        mt_graph = InterfaceGraphData(
            node_features=torch.randn(num_nodes, 128),
            edge_index=torch.randint(0, num_nodes, (2, num_edges)),
            edge_features=torch.randn(num_edges, 128),
            edge_types=torch.randint(0, 3, (num_edges,)),
            node_positions=torch.randn(num_nodes, 3),
            batch=torch.ones(num_nodes, dtype=torch.long),
            atom_names=[f"CA" for _ in range(num_nodes)],
            is_mutation=torch.zeros(num_nodes, dtype=torch.bool),
            residue_indices=[f"A_{i}" for i in range(num_nodes)]
        )
        
        # 前向传播
        with torch.no_grad():
            output = model(wt_graph, mt_graph)
        
        print(f"✓ 前向传播成功")
        print(f"  - 输出形状: {output.shape}")
        print(f"  - 期望形状: ({batch_size}, 1)")
        
        if output.shape == (batch_size, 1):
            print("✓ 输出维度正确！")
            return True
        else:
            print(f"✗ 输出维度错误！期望 ({batch_size}, 1)，实际 {output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试128维修改...")
    
    tests = [
        test_model_initialization,
        test_geometry_components,
        test_edge_feature_creation,
        test_forward_pass
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    print("=" * 50)
    
    if passed == total:
        print("🎉 所有测试通过！128维修改成功！")
        return True
    else:
        print("❌ 部分测试失败，需要进一步调试")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)