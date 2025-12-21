#!/usr/bin/env python3
import torch
from model import CHYModelWithGeometric, InterfaceGraphData

def test_model():
    """测试修改后的模型架构"""
    print("=== 测试原子级图卷积 + 残基级注意力模型 ===")
    
    # 创建模型
    model = CHYModelWithGeometric()
    print("✓ 模型创建成功")
    
    # 创建测试数据
    num_residues = 10  # 模拟10个残基
    atoms_per_residue = 4  # 每个残基4个原子（N, CA, C, CB）
    total_atoms = num_residues * atoms_per_residue
    
    print(f"创建测试数据: {total_atoms}个原子, {num_residues}个残基")
    
    # 创建原子名称和残基索引
    atom_names = []
    residue_indices = []
    for i in range(num_residues):
        for atom in ['N', 'CA', 'C', 'CB']:
            atom_names.append(atom)
            residue_indices.append(f'A_{i}')
    
    # 创建图数据
    wt_graph = InterfaceGraphData(
        node_features=torch.randn(total_atoms, 64),
        edge_index=torch.randint(0, total_atoms, (2, total_atoms * 3)),
        edge_features=torch.randn(total_atoms * 3, 64),
        edge_types=torch.randint(0, 3, (total_atoms * 3,)),
        node_positions=torch.randn(total_atoms, 3),
        batch=torch.zeros(total_atoms, dtype=torch.long),
        atom_names=atom_names,
        is_mutation=torch.zeros(total_atoms, dtype=torch.bool),
        residue_indices=residue_indices
    )
    
    mt_graph = InterfaceGraphData(
        node_features=torch.randn(total_atoms, 64),
        edge_index=torch.randint(0, total_atoms, (2, total_atoms * 3)),
        edge_features=torch.randn(total_atoms * 3, 64),
        edge_types=torch.randint(0, 3, (total_atoms * 3,)),
        node_positions=torch.randn(total_atoms, 3),
        batch=torch.ones(total_atoms, dtype=torch.long),
        atom_names=atom_names,
        is_mutation=torch.zeros(total_atoms, dtype=torch.bool),
        residue_indices=residue_indices
    )
    
    # 创建其他输入
    batch_size = 2
    seq_len = 64
    esm_dim = 1280
    foldx_dim = 22
    
    esm_embeddings = torch.randn(batch_size, seq_len, esm_dim)
    foldx_features = torch.randn(batch_size, foldx_dim)
    
    print("✓ 测试数据创建完成")
    
    # 测试前向传播
    print("\n开始前向传播测试...")
    try:
        with torch.no_grad():
            output = model(esm_embeddings, foldx_features, wt_graph, mt_graph)
            print(f"✓ 前向传播成功! 输出形状: {output.shape}")
            
            # 测试不同规模的数据
            print("\n=== 测试不同规模的显存占用 ===")
            test_sizes = [50, 100, 200, 256]  # 残基数量
            
            for num_res in test_sizes:
                total_atoms_test = num_res * atoms_per_residue
                
                # 创建对应规模的图
                atom_names_test = []
                residue_indices_test = []
                for i in range(num_res):
                    for atom in ['N', 'CA', 'C', 'CB']:
                        atom_names_test.append(atom)
                        residue_indices_test.append(f'A_{i}')
                
                test_graph = InterfaceGraphData(
                    node_features=torch.randn(total_atoms_test, 64),
                    edge_index=torch.randint(0, total_atoms_test, (2, total_atoms_test * 3)),
                    edge_features=torch.randn(total_atoms_test * 3, 64),
                    edge_types=torch.randint(0, 3, (total_atoms_test * 3,)),
                    node_positions=torch.randn(total_atoms_test, 3),
                    batch=torch.zeros(total_atoms_test, dtype=torch.long),
                    atom_names=atom_names_test,
                    is_mutation=torch.zeros(total_atoms_test, dtype=torch.bool),
                    residue_indices=residue_indices_test
                )
                
                try:
                    # 只测试几何GNN部分
                    geom_output = model.geometric_gnn_wt(test_graph)
                    print(f"✓ {num_res}残基({total_atoms_test}原子): 几何特征形状={geom_output.shape}")
                except Exception as e:
                    print(f"✗ {num_res}残基({total_atoms_test}原子): 失败 - {str(e)[:50]}...")
                
                # 清理显存
                del test_graph
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()